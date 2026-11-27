"""辨識與翻譯分開進行：worker 拆成兩段、控制器只顯示最新的一句、舊的放棄，以及設定開關。"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtWidgets import QApplication, QCheckBox

from app.application import Application
from app.config import DEFAULTS, AppConfig
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCRLine
from services.translation.base import TranslationError, TranslationInterrupted
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.stability import SAME, TextStabilizer, signature
from workers.translation_worker import PipelineTask, TranslationJob, TranslationTask


# ----------------------------------------------------------------------------- 假的辨識與翻譯
class _OCR:
    def recognize(self, _image, lang="auto"):
        return [OCRLine("紡ぐたったひとつ絆", (332, 29, 432, 62), 0.95)]


class _Translator:
    han_is_source = True
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        return ["只有一個羈絆" for _ in texts]


class _BlockingTranslator:
    """翻譯會一直等到 ``release`` 才完成；每個翻譯都記錄開始、完成、被放棄。"""

    han_is_source = True
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.abandoned: list[str] = []
        self.fail_with: Exception | None = None
        self._local = threading.local()

    def reset_context(self) -> None:
        pass

    @contextmanager
    def interruptible(self, should_stop):
        self._local.should_stop = should_stop
        try:
            yield
        finally:
            self._local.should_stop = None

    def translate(self, texts, source, target):
        text = texts[0]
        self.started.append(text)
        should_stop = getattr(self._local, "should_stop", None)
        deadline = time.monotonic() + 10
        while not self.release.is_set() and time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                self.abandoned.append(text)
                raise TranslationInterrupted()
            time.sleep(0.002)
        if self.fail_with is not None:
            raise self.fail_with
        self.finished.append(text)
        return [f"譯:{text}" for _ in texts]


def _capture() -> CaptureInfo:
    return CaptureInfo(image=np.full((40, 120, 3), 255, dtype=np.uint8), bbox=(0, 0, 120, 40), mode="region")


def _job(text: str) -> TranslationJob:
    region = TextRegion(text=text, x=5, y=5, width=60, height=15)
    return TranslationJob(
        capture=_capture(),
        regions=[region],
        texts=[text],
        unique_texts=[text],
        translatable=[text],
        source="auto",
        target="zh-hant",
        merge_original=False,
        key=signature([text]),
    )


# ----------------------------------------------------------------------------- worker
def _pipeline(tmp_path, translator, **kwargs):
    capture = CaptureInfo(
        image=np.full((200, 800, 3), 255, dtype=np.uint8), bbox=(0, 0, 800, 200), monitor_indices=[0], mode="region"
    )
    return PipelineTask(capture, _OCR(), translator, AppConfig(tmp_path / "config.json"), **kwargs)


def test_a_deferred_pipeline_hands_over_the_job_and_does_not_translate(tmp_path):
    translator = _Translator()
    task = _pipeline(tmp_path, translator, defer_translation=True)
    jobs, results = [], []
    task.job.connect(jobs.append)
    task.result.connect(results.append)

    task.run()

    assert translator.calls == []
    assert results == []
    assert len(jobs) == 1
    assert jobs[0].translatable == ["紡ぐたったひとつ絆"]
    assert jobs[0].key == signature(["紡ぐたったひとつ絆"])
    assert jobs[0].regions[0].background_color  # 顏色在辨識這一側就偵測好了，翻譯那一側只管翻


def test_a_deferred_pipeline_still_asks_the_gate_and_stays_silent_when_it_says_same(tmp_path):
    translator = _Translator()
    task = _pipeline(tmp_path, translator, defer_translation=True, gate=lambda texts: SAME)
    jobs, results = [], []
    task.job.connect(jobs.append)
    task.result.connect(results.append)

    task.run()

    assert jobs == [] and translator.calls == []
    assert results[0]["verdict"] == SAME


def test_translating_the_handed_over_job_gives_the_same_result_as_one_pass(tmp_path):
    one_pass, split = _Translator(), _Translator()
    sequential = _pipeline(tmp_path, one_pass)
    expected = []
    sequential.result.connect(expected.append)
    sequential.run()

    deferred = _pipeline(tmp_path, split, defer_translation=True)
    jobs = []
    deferred.job.connect(jobs.append)
    deferred.run()
    translation = TranslationTask(jobs[0], split)
    actual = []
    translation.result.connect(actual.append)
    translation.run()

    assert [r.translated_text for r in actual[0]["regions"]] == [r.translated_text for r in expected[0]["regions"]]
    for key in ("recognized_count", "translated_count", "failed_count", "translation_candidate_count"):
        assert actual[0][key] == expected[0][key]
    assert actual[0]["gate_texts"] == ["紡ぐたったひとつ絆"]
    assert actual[0]["job_key"] == jobs[0].key
    assert split.calls == one_pass.calls


def test_a_cancelled_translation_task_sends_nothing(tmp_path):
    translator = _Translator()
    task = TranslationTask(_job("こんにちは"), translator)
    results, errors = [], []
    task.result.connect(results.append)
    task.error.connect(errors.append)

    task.cancel()
    task.run()

    assert translator.calls == [] and results == [] and errors == []


def test_an_abandoned_translation_is_neither_a_result_nor_an_error():
    translator = _BlockingTranslator()
    task = TranslationTask(_job("こんにちは"), translator)
    results, errors = [], []
    task.result.connect(results.append)
    task.error.connect(errors.append)
    timer = threading.Timer(0.05, task.cancel)
    timer.start()

    task.run()
    timer.join()

    assert translator.abandoned == ["こんにちは"]
    assert results == [] and errors == []


def test_a_finished_translation_that_became_stale_meanwhile_is_not_sent():
    translator = _BlockingTranslator()
    translator.release.set()
    task = TranslationTask(_job("こんにちは"), translator)
    results = []
    task.result.connect(results.append)
    original = translator.translate

    def translate_then_cancel(texts, source, target):
        out = original(texts, source, target)
        task.cancel()  # 翻完的那一刻才發現過時
        return out

    translator.translate = translate_then_cancel
    task.run()

    assert results == []


def test_a_translation_error_is_reported_as_an_error():
    translator = _BlockingTranslator()
    translator.release.set()
    translator.fail_with = TranslationError("模型壞了")
    task = TranslationTask(_job("こんにちは"), translator)
    errors = []
    task.error.connect(errors.append)

    task.run()

    assert errors == ["模型壞了"]


# ----------------------------------------------------------------------------- 控制器
class _FakeOverlay:
    def __init__(self) -> None:
        self.shown: list[list[TextRegion]] = []
        self.cleared = 0

    def show_regions(self, capture, regions, animate_new_only=False):
        self.shown.append(list(regions))
        return True

    def clear_all(self):
        self.cleared += 1

    def hide_all(self, animate=False):
        pass

    def is_visible(self):
        return True

    def follow_region(self, bbox):
        pass


@pytest.fixture
def controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(app)
    instance.overlay_manager = _FakeOverlay()
    instance.translator = _BlockingTranslator()
    instance.config.set("translation.decouple_ocr", True)
    instance._monitor_active = True
    instance._monitor_stabilizer = TextStabilizer(rounds=1)
    instance._capture_session = None
    instance._pipeline_succeeded = False
    yield instance
    instance.translator.release.set()
    instance._cancel_translation_jobs()
    _pump(lambda: not instance._translation_task_running(), timeout=5)
    instance.floating_status.deleteLater()


def _pump(condition, timeout: float = 5.0) -> bool:
    """跑事件迴圈直到條件成立（背景執行緒的信號要靠事件迴圈送回主執行緒）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if condition():
            return True
        QThread.msleep(2)
    QCoreApplication.processEvents()
    return condition()


def _shown_texts(controller) -> list[str]:
    return [regions[0].translated_text for regions in controller.overlay_manager.shown]


def test_the_setting_is_off_by_default_and_only_applies_to_monitoring(controller):
    assert DEFAULTS["translation"]["decouple_ocr"] is False
    assert controller._decoupled_enabled() is True

    controller.config.set("translation.decouple_ocr", False)
    assert controller._decoupled_enabled() is False

    controller.config.set("translation.decouple_ocr", True)
    controller._monitor_active = False
    assert controller._decoupled_enabled() is False  # 單次擷取照舊一次做完


def test_a_subtitle_is_translated_in_the_background_shown_and_committed(controller):
    controller.translator.release.set()

    controller._on_translation_job(_job("こんにちは"))

    assert _pump(lambda: _shown_texts(controller) == ["譯:こんにちは"])
    assert _pump(lambda: not controller._translation_busy())
    assert controller._monitor_stabilizer.observe(["こんにちは"]) == SAME  # 顯示出來的那一組已記為「已顯示」


def test_a_newer_subtitle_abandons_the_one_being_translated_and_only_the_newest_is_shown(controller):
    translator = controller.translator

    controller._on_translation_job(_job("一つ目"))
    assert _pump(lambda: translator.started == ["一つ目"])
    controller._on_translation_job(_job("二つ目"))
    assert _pump(lambda: translator.abandoned == ["一つ目"])  # 舊的在翻到一半時被放棄
    assert _pump(lambda: translator.started == ["一つ目", "二つ目"])
    translator.release.set()

    assert _pump(lambda: _shown_texts(controller) == ["譯:二つ目"])
    assert translator.finished == ["二つ目"]
    assert _pump(lambda: not controller._translation_busy())


def test_of_several_waiting_subtitles_only_the_latest_is_ever_translated(controller):
    translator = controller.translator

    controller._on_translation_job(_job("一"))
    assert _pump(lambda: translator.started == ["一"])
    controller._on_translation_job(_job("二"))
    controller._on_translation_job(_job("三"))
    translator.release.set()

    assert _pump(lambda: _shown_texts(controller) == ["譯:三"])
    assert translator.started == ["一", "三"]  # 「二」從來沒進過模型


def test_the_same_subtitle_seen_in_every_round_is_translated_once(controller):
    translator = controller.translator

    controller._on_translation_job(_job("こんにちは"))
    controller._on_translation_job(_job("こんにちは"))
    controller._on_translation_job(_job("こんにちは"))
    translator.release.set()

    assert _pump(lambda: _shown_texts(controller) == ["譯:こんにちは"])
    assert translator.started == ["こんにちは"]


def test_a_same_verdict_means_the_screen_went_back_and_cancels_the_translation(controller):
    translator = controller.translator
    controller._on_translation_job(_job("こんにちは"))
    assert _pump(lambda: translator.started == ["こんにちは"])

    controller._present_result({"capture": _capture(), "regions": [], "verdict": SAME})

    assert _pump(lambda: translator.abandoned == ["こんにちは"])
    translator.release.set()
    assert _pump(lambda: not controller._translation_busy())
    assert controller.overlay_manager.shown == []


def test_a_result_that_arrives_after_the_text_was_cancelled_is_not_shown(controller):
    translator = controller.translator
    translator.release.set()
    controller._on_translation_job(_job("こんにちは"))
    deadline = time.monotonic() + 5
    while translator.finished != ["こんにちは"] and time.monotonic() < deadline:
        QThread.msleep(2)  # 不跑事件迴圈：翻好了，但結果的信號還排在佇列裡（跑了就可能在取消之前被送達，測試會時好時壞）
    assert translator.finished == ["こんにちは"]
    controller._cancel_translation_jobs()  # 結果的信號還在排隊時畫面就作廢了

    _pump(lambda: False, timeout=0.2)

    assert controller.overlay_manager.shown == []


def test_stopping_the_monitor_abandons_the_translation(controller):
    translator = controller.translator
    controller._on_translation_job(_job("こんにちは"))
    assert _pump(lambda: translator.started == ["こんにちは"])

    controller.stop_monitor()

    assert _pump(lambda: translator.abandoned == ["こんにちは"])
    assert _pump(lambda: not controller._translation_task_running())
    assert controller.overlay_manager.shown == []


def test_the_hide_after_no_text_timer_abandons_a_translation_that_is_still_running(controller):
    translator = controller.translator
    controller._on_translation_job(_job("こんにちは"))
    assert _pump(lambda: translator.started == ["こんにちは"])

    controller._close_translation_after_no_text()

    assert _pump(lambda: translator.abandoned == ["こんにちは"])


def test_repeated_translation_failures_stop_the_monitor_but_one_success_resets_the_count(controller):
    translator = controller.translator
    translator.release.set()
    translator.fail_with = TranslationError("模型壞了")

    controller._on_translation_job(_job("一"))
    assert _pump(lambda: controller._translation_errors == 1)
    assert _pump(lambda: not controller._translation_busy())
    translator.fail_with = None
    controller._on_translation_job(_job("二"))
    assert _pump(lambda: _shown_texts(controller) == ["譯:二"])
    assert controller._translation_errors == 0

    translator.fail_with = TranslationError("模型壞了")
    for text in ("三", "四", "五"):
        controller._on_translation_job(_job(text))
        assert _pump(lambda: not controller._translation_busy())

    assert controller._monitor_active is False  # 連續三次失敗：停止監控


def test_an_error_of_an_outdated_subtitle_does_not_count(controller):
    translator = controller.translator
    translator.fail_with = TranslationError("模型壞了")
    controller._on_translation_job(_job("舊"))
    assert _pump(lambda: translator.started == ["舊"])
    controller._on_translation_job(_job("新"))  # 舊的過時了（被叫停，不會有錯誤）

    assert _pump(lambda: translator.started == ["舊", "新"])
    translator.fail_with = None
    translator.release.set()

    assert _pump(lambda: _shown_texts(controller) == ["譯:新"])
    assert controller._translation_errors == 0


def test_shutdown_waits_for_the_translation_thread_to_end(controller):
    translator = controller.translator
    controller._on_translation_job(_job("こんにちは"))
    assert _pump(lambda: translator.started == ["こんにちは"])
    controller._shutting_down = True

    controller._wait_for_translation_before_shutdown()

    assert _pump(lambda: translator.abandoned == ["こんにちは"])
    assert _pump(lambda: controller._translation_task is None)
    assert controller._shutdown_finalized is True  # 翻譯結束後才真的收尾


def test_model_release_waits_while_a_translation_is_running(controller):
    translator = controller.translator
    controller._on_translation_job(_job("こんにちは"))
    assert _pump(lambda: translator.started == ["こんにちは"])
    controller._monitor_active = False  # 監控停了，但翻譯的執行緒還在跑

    assert controller._model_is_idle() is False


def test_the_same_frame_is_not_recognised_again_while_its_translation_is_running(controller, monkeypatch):
    class _Screens:
        def capture_bbox(self, bbox):
            return CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=tuple(bbox), mode="region")

    started = []
    controller.screenshot_service = _Screens()
    controller._monitor_request = {"mode": "region", "bbox": (0, 0, 120, 40), "hwnd": None}
    monkeypatch.setattr(controller, "_start_pipeline", lambda capture: started.append(capture))
    monkeypatch.setattr(controller, "_monitor_schedule_next", lambda: None)

    controller._monitor_cycle()
    assert len(started) == 1
    controller._on_translation_job(_job("こんにちは"))  # 辨識完、翻譯途中，譯文還沒出來
    assert _pump(lambda: controller.translator.started == ["こんにちは"])

    controller._monitor_cycle()
    assert len(started) == 1  # 同一格畫面：不必再辨識

    controller.translator.release.set()
    assert _pump(lambda: not controller._translation_busy())
    controller._monitor_signature = ""
    controller._monitor_cycle()
    assert len(started) == 2  # 翻完卻沒有顯示出來（例如失敗）：照舊重試


# ----------------------------------------------------------------------------- 設定頁
def test_the_toggle_is_in_the_translation_tab_and_round_trips_through_the_config(tmp_path):
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    try:
        checkbox = dialog.chk_decouple_ocr
        assert isinstance(checkbox, QCheckBox)
        assert checkbox.isChecked() is False
        assert "辨識與翻譯分開進行" in checkbox.text()
        assert dialog.pages.widget(3).widget().isAncestorOf(checkbox)  # 「翻譯」頁

        checkbox.setChecked(True)
        dialog._commit_accept(dialog._validated_hotkeys())
        dialog._save_close_timer.stop()
        assert AppConfig(tmp_path / "config.json").get("translation.decouple_ocr") is True

        reopened = SettingsDialog(AppConfig(tmp_path / "config.json"))
        try:
            assert reopened.chk_decouple_ocr.isChecked() is True
        finally:
            reopened.deleteLater()
    finally:
        dialog.deleteLater()

