"""設定存檔換了翻譯模型後：馬上在背景載入並預熱，告訴使用者「模型更換中」，載入期間監控不送翻譯。"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import AppConfig
from app.models import CaptureInfo
from services.ocr.base import OCRLine
from services.translation.base import Translator
from tests.fake_llama import offline_translator
from workers.translation_worker import PipelineTask

QWEN3 = "qwen3-1.7b-gguf-q8_0"


class _SlowTranslator(Translator):
    """warmup 會停在 ``release`` 事件上，測試可以看「載入中」的樣子；``ready`` 決定開始時算不算已載入。"""

    name = "offline"

    def __init__(self, label="offline/3b", ready=False, error="") -> None:
        super().__init__({})
        self.label = label
        self.ready = ready
        self.error = error
        self.release = threading.Event()
        self.started = threading.Event()
        self.device = "gpu"

    def _translate_batch(self, texts, source_language, target_language):
        return list(texts)

    def model_label(self) -> str:
        return self.label

    def display_model_name(self) -> str:
        return self.label.split("/", 1)[1].upper()

    def is_ready(self) -> bool:
        return self.ready and not (self.started.is_set() and not self.release.is_set())

    def warmup(self) -> None:
        self.started.set()
        assert self.release.wait(5), "測試沒有放行預熱"
        if self.error:
            raise RuntimeError(self.error)
        self.ready = True


@pytest.fixture
def controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("SCREEN_TRANSLATOR_NO_WARMUP", raising=False)
    monkeypatch.delenv("SCREEN_TRANSLATOR_SELFTEST", raising=False)
    instance = Application(app)
    instance.statuses = []
    monkeypatch.setattr(instance, "set_status", instance.statuses.append)
    yield instance
    instance.floating_status.hide_immediate()
    instance.floating_status.deleteLater()


def _use(controller, translator):
    controller.translator = translator
    controller._translator_label = translator.model_label  # 標記直接用假翻譯器的（正式的會算 offline/<實際模型>）
    return translator


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return condition()


def _finish(controller, translator):
    translator.release.set()
    assert _wait_until(lambda: not controller._translator_warming())


# ----------------------------------------------------------------------------- 翻譯器
def test_an_offline_translator_is_not_ready_until_its_model_is_loaded():
    unloaded = offline_translator(lambda prompt: "譯")[0]
    unloaded._llama = None
    loaded = offline_translator(lambda prompt: "譯")[0]

    assert not unloaded.is_ready()
    assert loaded.is_ready()


def test_an_offline_translator_is_not_ready_while_it_warms_up():
    instance, _engine = offline_translator(lambda prompt: "譯")
    seen: list[bool] = []
    original = instance._generate

    def generate(texts, language):
        seen.append(instance.is_ready())  # 預熱翻的那兩句：算固定前綴的時候還不能算就緒
        return original(texts, language)

    instance._generate = generate

    instance.warmup()

    assert seen == [False, False]
    assert instance.is_ready()  # 預熱結束（含出錯時）就恢復


def test_a_failing_warmup_does_not_leave_the_translator_stuck_as_warming():
    instance, _engine = offline_translator(lambda prompt: "譯")

    def boom(texts, language):
        raise RuntimeError("壞掉了")

    instance._generate = boom
    with pytest.raises(RuntimeError):
        instance.warmup()

    assert instance.is_ready()


def test_the_display_name_is_the_short_model_name():
    instance, _engine = offline_translator(lambda prompt: "譯", offline={"model_name": QWEN3})

    assert instance.display_model_name() == "Qwen3 1.7B Q8_0"


def test_online_translators_are_always_ready():
    from services.translation.mock_translator import MockTranslator

    assert MockTranslator({}).is_ready()


# ----------------------------------------------------------------------------- 應用程式
def test_the_first_load_says_so_in_the_status_but_does_not_pop_up_the_bubble(controller):
    translator = _use(controller, _SlowTranslator("offline/3b"))

    note = controller._warmup()

    assert note == "正在載入翻譯模型 3B，完成後自動繼續翻譯"
    assert controller._translator_warming()
    assert not controller.floating_status.isVisible()  # 啟動時不跳氣泡，只有「換模型」才跳
    _finish(controller, translator)
    assert controller.statuses[-1].startswith("翻譯模型已就緒：3B（GPU，")


def test_changing_the_model_says_so_in_the_status_and_in_the_bubble(controller):
    first = _use(controller, _SlowTranslator("offline/3b", ready=True))
    controller._warmup()  # 先記下目前的模型（已載入，不吵使用者）
    _finish(controller, first)
    second = _use(controller, _SlowTranslator("offline/qwen3-1.7b"))

    note = controller._warmup()

    assert note == "翻譯模型更換中：正在載入 QWEN3-1.7B，完成後自動繼續翻譯"
    assert controller.floating_status.isVisible() and controller._model_bubble
    assert controller.floating_status._text == "模型更換中，請稍候…"
    _finish(controller, second)
    assert controller.statuses[-1].startswith("翻譯模型已就緒：QWEN3-1.7B（GPU，")
    assert controller.floating_status._text == "模型已就緒" and not controller._model_bubble


def test_a_model_that_is_already_in_memory_is_warmed_without_bothering_the_user(controller):
    translator = _use(controller, _SlowTranslator("offline/3b", ready=True))
    translator.release.set()  # 預熱馬上結束

    assert controller._warmup() == ""
    assert _wait_until(lambda: not controller._translator_warming())

    assert controller.statuses == [] and not controller.floating_status.isVisible()


def test_a_failed_load_is_reported_with_its_first_line_only(controller):
    translator = _use(controller, _SlowTranslator("offline/3b", error="找不到離線翻譯模型「3b」。\n  D:\\ocr\\models\\translation\\3b"))
    controller._warmup()

    _finish(controller, translator)

    assert controller.statuses[-1] == "翻譯模型載入失敗：找不到離線翻譯模型「3b」。"
    assert not controller._translator_warming()  # 之後的翻譯照常執行，錯誤由翻譯本身報給使用者


def test_the_bubble_is_not_taken_from_a_capture_in_progress(controller):
    first = _use(controller, _SlowTranslator("offline/3b", ready=True))
    controller._warmup()
    _finish(controller, first)
    controller._busy = True  # 正在擷取：氣泡是擷取流程的
    second = _use(controller, _SlowTranslator("offline/qwen3-1.7b"))

    controller._warmup()

    assert not controller.floating_status.isVisible() and not controller._model_bubble
    _finish(controller, second)
    controller._busy = False


def test_the_loading_notice_is_shown_only_for_the_translator_that_is_in_use(controller):
    stale = _use(controller, _SlowTranslator("offline/3b"))
    controller._warmup()
    current = _use(controller, _SlowTranslator("offline/qwen3-1.7b"))  # 載入途中又換了一個
    controller._warmup()

    stale.release.set()
    assert _wait_until(lambda: not any(item is stale for item in controller._warming))

    assert controller._translator_warming()  # 現在的翻譯器還在載
    assert not any(text.startswith("翻譯模型已就緒") for text in controller.statuses)
    _finish(controller, current)
    assert controller.statuses[-1].startswith("翻譯模型已就緒：QWEN3-1.7B")


def test_online_translators_have_nothing_to_load(controller):
    from services.translation.mock_translator import MockTranslator

    _use(controller, MockTranslator({}))
    controller.translator.model_label = lambda: "mock"

    assert controller._warmup() == "" and not controller._translator_warming()


# ----------------------------------------------------------------------------- 監控
def _monitor(controller, monkeypatch):
    controller._monitor_active = True
    controller._monitor_request = {"mode": "region", "bbox": (0, 0, 120, 40), "hwnd": 0}
    started = []
    monkeypatch.setattr(controller, "_start_pipeline", started.append)
    monkeypatch.setattr(controller.screenshot_service, "capture_bbox", lambda bbox: CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=bbox, mode="region"))
    return started


def test_the_monitor_sends_nothing_to_the_translator_while_the_model_loads(controller, monkeypatch):
    started = _monitor(controller, monkeypatch)
    translator = _use(controller, _SlowTranslator("offline/3b"))
    controller._warmup()

    controller._monitor_cycle()

    assert started == [] and controller._monitor_timer.isActive()  # 不送翻譯，下一輪再看
    assert controller._monitor_frame_hash == ""  # 這一格畫面沒有被記成「看過了」
    controller._monitor_timer.stop()
    _finish(controller, translator)
    controller._monitor_cycle()
    assert len(started) == 1  # 載好了：照常翻譯
    controller._monitor_active = False
    controller._monitor_timer.stop()


def test_refresh_while_the_model_loads_waits_instead_of_starting_a_task(controller, monkeypatch):
    started = _monitor(controller, monkeypatch)
    translator = _use(controller, _SlowTranslator("offline/3b"))
    controller._warmup()
    controller._monitor_stabilizer = None

    controller._refresh_monitor()
    controller._monitor_cycle()

    assert controller.statuses[-1] == "翻譯模型載入中，完成後自動重新整理"
    assert started == []  # 不會因為舊任務卡在載入上而跳出「任務仍在執行」
    _finish(controller, translator)
    controller._monitor_active = False
    controller._monitor_timer.stop()


# ----------------------------------------------------------------------------- 設定存檔
def test_saving_with_a_new_model_puts_the_loading_notice_in_the_saved_status(controller, monkeypatch):
    for name in ("_refresh_appearance", "refresh_monitor_map", "_apply_hotkeys", "_apply_autostart"):
        monkeypatch.setattr(controller, name, lambda *a, **k: None)
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    monkeypatch.setattr(controller, "_rebuild_translator", lambda: 0)
    monkeypatch.setattr(controller, "_warmup", lambda: "翻譯模型更換中：正在載入 3B Q4_K_M，完成後自動繼續翻譯")

    controller._apply_settings()

    assert controller.statuses[-1] == "設定已儲存；翻譯模型更換中：正在載入 3B Q4_K_M，完成後自動繼續翻譯"


def test_the_real_save_path_starts_the_load_right_away(controller, monkeypatch):
    """不是等第一次翻譯才載：存檔那一刻就開始背景載入。"""
    for name in ("_refresh_appearance", "refresh_monitor_map", "_apply_hotkeys", "_apply_autostart"):
        monkeypatch.setattr(controller, name, lambda *a, **k: None)
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    translator = _SlowTranslator("offline/qwen3-1.7b")
    _use(controller, translator)
    monkeypatch.setattr(controller, "_rebuild_translator", lambda: 0)

    controller._apply_settings()

    assert translator.started.wait(2), "存檔後沒有馬上開始載入模型"
    assert controller.statuses[-1].startswith("設定已儲存；正在載入翻譯模型 QWEN3-1.7B")
    _finish(controller, translator)


# ----------------------------------------------------------------------------- 單次擷取
class _OneLineOCR:
    def recognize(self, _image, lang="ch"):
        return [OCRLine("こんにちは", (10, 10, 100, 24), 0.95)]


def _run_worker(tmp_path, translator):
    app = QApplication.instance() or QApplication([])
    capture = CaptureInfo(image=np.zeros((60, 140, 3), dtype=np.uint8), bbox=(0, 0, 140, 60), mode="region")
    task = PipelineTask(capture=capture, ocr_engine=_OneLineOCR(), translator=translator, config=AppConfig(tmp_path / "config.json"))
    statuses: list[str] = []
    task.status.connect(statuses.append)
    task.run()
    app.processEvents()
    task.deleteLater()
    return statuses


def test_a_capture_while_the_model_loads_says_it_is_waiting_for_the_model(tmp_path):
    translator = _SlowTranslator("offline/3b")  # 沒載入
    translator.release.set()

    statuses = _run_worker(tmp_path, translator)

    assert "翻譯模型載入中…" in statuses and not any(text.startswith("正在翻譯") for text in statuses)


def test_a_capture_with_a_ready_model_just_translates(tmp_path):
    translator = _SlowTranslator("offline/3b", ready=True)

    statuses = _run_worker(tmp_path, translator)

    assert any(text.startswith("正在翻譯 1 個文本塊") for text in statuses) and "翻譯模型載入中…" not in statuses
