"""即時監控的穩定化：新字幕連續辨識到才顯示、與已顯示的相同不重翻、關閉監控時清掉字幕。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import DEFAULTS, AppConfig
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCRLine
from ui.main_window import MainWindow
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.image_utils import frame_fingerprint
from utils.stability import CONFIRMED, PENDING, SAME, TextStabilizer, signature
from workers.translation_worker import PipelineTask


# ----------------------------------------------------------------------------- TextStabilizer
class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_new_text_needs_a_second_sighting_then_the_same_text_is_never_translated_again():
    stabilizer = TextStabilizer(rounds=2)

    assert stabilizer.observe(["こんにちは"]) == PENDING
    assert stabilizer.has_pending
    assert stabilizer.observe(["こんにちは"]) == CONFIRMED
    stabilizer.commit()

    assert not stabilizer.has_pending
    assert stabilizer.observe(["こんにちは"]) == SAME
    assert stabilizer.observe(["こんにちは"]) == SAME


def test_fast_mode_confirms_on_the_first_sighting_but_still_skips_repeats():
    stabilizer = TextStabilizer(rounds=1)

    assert stabilizer.observe(["こんにちは"]) == CONFIRMED
    stabilizer.commit()

    assert stabilizer.observe(["こんにちは"]) == SAME


def test_a_transitional_frame_that_changes_next_round_never_gets_translated():
    stabilizer = TextStabilizer(rounds=2)

    assert stabilizer.observe(["紡ぐた"]) == PENDING  # 淡入到一半被辨識成殘句
    assert stabilizer.observe(["紡ぐたったひとつ絆"]) == PENDING  # 下一輪已經變了：重新計數
    assert stabilizer.observe(["紡ぐたったひとつ絆"]) == CONFIRMED


def test_going_back_to_the_displayed_text_cancels_the_pending_one():
    stabilizer = TextStabilizer(rounds=2)
    stabilizer.observe(["A"])
    stabilizer.observe(["A"])
    stabilizer.commit()
    assert stabilizer.observe(["B"]) == PENDING

    assert stabilizer.observe(["A"]) == SAME
    assert not stabilizer.has_pending
    assert stabilizer.observe(["B"]) == PENDING  # B 要重新累積，不會因為先前那一次直接放行


def test_unstable_text_is_released_after_the_maximum_wait():
    clock = _Clock()
    stabilizer = TextStabilizer(rounds=2, max_wait=2.5, clock=clock)

    assert stabilizer.observe(["a"]) == PENDING
    clock.now += 1.0
    assert stabilizer.observe(["b"]) == PENDING
    clock.now += 1.0
    assert stabilizer.observe(["c"]) == PENDING
    clock.now += 0.6  # 距離第一次偏離已經 2.6 秒
    assert stabilizer.observe(["d"]) == CONFIRMED


def test_text_that_was_confirmed_but_never_shown_is_retried():
    stabilizer = TextStabilizer(rounds=2)
    stabilizer.observe(["A"])
    assert stabilizer.observe(["A"]) == CONFIRMED  # 翻譯途中失敗或被取消：沒有 commit

    assert stabilizer.observe(["A"]) == CONFIRMED  # 下一輪直接再放行，不會當成「已顯示」而不再翻譯


def test_commit_only_applies_to_what_was_just_confirmed():
    stabilizer = TextStabilizer(rounds=2)
    stabilizer.observe(["A"])
    stabilizer.observe(["A"])
    stabilizer.observe(["B"])  # 放行後被取消，緊接著看到別的字

    stabilizer.commit()

    assert stabilizer.observe(["A"]) == PENDING  # A 沒有被當成已顯示


def test_whitespace_and_width_differences_are_not_changes():
    assert signature(["きっと 星座も掴める", "ＡＢＣ"]) == signature(["きっと星座も掴める", "ABC"])
    stabilizer = TextStabilizer(rounds=1)
    stabilizer.observe(["Hello world"])
    stabilizer.commit()

    assert stabilizer.observe(["Hello  world "]) == SAME


def test_configure_changes_the_rounds_while_running_and_reset_forgets_the_display():
    stabilizer = TextStabilizer(rounds=2)
    stabilizer.configure(1, 2.5)
    assert stabilizer.observe(["A"]) == CONFIRMED
    stabilizer.commit()

    stabilizer.reset()

    assert stabilizer.observe(["A"]) == CONFIRMED


# ----------------------------------------------------------------------------- 逐像素指紋
def test_frame_fingerprint_detects_a_single_changed_pixel():
    frame = np.zeros((230, 1100, 3), dtype=np.uint8)
    other = frame.copy()
    other[100, 500, 1] = 1

    assert frame_fingerprint(frame) == frame_fingerprint(frame.copy())
    assert frame_fingerprint(frame) != frame_fingerprint(other)
    assert frame_fingerprint(frame) != frame_fingerprint(np.zeros((115, 2200, 3), dtype=np.uint8))
    assert frame_fingerprint(None) == "" and frame_fingerprint(np.zeros((0, 0, 3), dtype=np.uint8)) == ""


# ----------------------------------------------------------------------------- worker
class _OCR:
    def recognize(self, _image, lang="auto"):
        return [
            OCRLine("紡ぐたったひとつ絆", (332, 29, 432, 62), 0.95),
            OCRLine("tsumugu tatta hitotsu kizuna", (352, 119, 395, 46), 1.0),
        ]


class _Translator:
    han_is_source = True
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self) -> None:
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        return ["只有一個羈絆" for _ in texts]


def _run_worker(tmp_path, gate):
    capture = CaptureInfo(
        image=np.full((200, 800, 3), 255, dtype=np.uint8), bbox=(0, 0, 800, 200), monitor_indices=[0], mode="region"
    )
    translator = _Translator()
    task = PipelineTask(capture, _OCR(), translator, AppConfig(tmp_path / "config.json"), gate=gate)
    results = []
    task.result.connect(results.append)
    task.run()
    return translator, results


def test_worker_does_not_translate_when_the_gate_says_same_or_pending(tmp_path):
    for verdict in (SAME, PENDING):
        translator, results = _run_worker(tmp_path, lambda texts, verdict=verdict: verdict)

        assert translator.calls == []
        assert results == [{"capture": results[0]["capture"], "regions": [], "verdict": verdict}]


def test_worker_translates_when_the_gate_confirms_and_only_asks_about_translatable_text(tmp_path):
    asked = []

    def gate(texts):
        asked.append(list(texts))
        return CONFIRMED

    translator, results = _run_worker(tmp_path, gate)

    assert asked == [["紡ぐたったひとつ絆"]]  # 羅馬拼音那行不翻，也不參與穩定判斷
    assert translator.calls == [["紡ぐたったひとつ絆"]]
    assert results[0]["regions"][0].translated_text == "只有一個羈絆"
    assert "verdict" not in results[0] or results[0]["verdict"] == CONFIRMED


def test_worker_without_a_gate_behaves_as_before(tmp_path):
    translator, results = _run_worker(tmp_path, None)

    assert translator.calls == [["紡ぐたったひとつ絆"]]
    assert results[0]["regions"]


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


def _payload(texts=("こんにちは",)):
    capture = CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=(0, 0, 120, 40), mode="region")
    regions = [
        TextRegion(text=text, translated_text=f"譯:{text}", x=5, y=5 + index * 20, width=60, height=15)
        for index, text in enumerate(texts)
    ]
    return {"capture": capture, "regions": regions, "recognized_count": len(regions), "translated_count": len(regions), "failed_count": 0}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(app)
    instance.overlay_manager = _FakeOverlay()
    instance._monitor_active = True
    instance._monitor_stabilizer = TextStabilizer(rounds=2)
    instance._capture_session = None
    instance._pipeline_succeeded = False
    yield instance
    instance.floating_status.deleteLater()


def test_skipped_results_leave_the_screen_alone_and_count_as_success(controller):
    for verdict in (SAME, PENDING):
        controller._pipeline_succeeded = False

        controller.on_pipeline_result({"capture": _payload()["capture"], "regions": [], "verdict": verdict})

        assert controller._pipeline_succeeded is True
        assert controller.overlay_manager.shown == []


def test_displaying_a_confirmed_result_commits_it(controller):
    stabilizer = controller._monitor_stabilizer
    stabilizer.observe(["こんにちは"])
    assert stabilizer.observe(["こんにちは"]) == CONFIRMED

    controller.on_pipeline_result(_payload())

    assert len(controller.overlay_manager.shown) == 1
    assert stabilizer.observe(["こんにちは"]) == SAME  # 已顯示：之後辨識到一樣的字不再翻


def test_stopping_the_monitor_removes_the_translation_from_the_screen(controller):
    overlay = controller.overlay_manager
    controller.on_pipeline_result(_payload())
    assert len(overlay.shown) == 1

    controller.stop_monitor()

    assert overlay.cleared == 1
    assert controller._overlay_visible is False
    assert controller._monitor_active is False
    assert controller._monitor_stabilizer is None


def test_stopping_after_failures_also_clears(controller):
    controller.stop_monitor(reason="連續翻譯失敗，已停止即時監控")

    assert controller.overlay_manager.cleared == 1


def test_stop_does_nothing_when_the_monitor_was_not_running(controller):
    controller._monitor_active = False

    controller.stop_monitor()

    assert controller.overlay_manager.cleared == 0


def test_stability_setting_picks_the_number_of_rounds(controller):
    controller.config.set("monitor.stability", "fast")
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 1

    controller.config.set("monitor.stability", "stable")
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 2

    controller.config.set("monitor.stability", "fast")  # 監控中途切換，下一輪就生效
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 1


# ----------------------------------------------------------------------------- 監控每一輪的畫面門控
class _Screens:
    def __init__(self) -> None:
        self.frame = np.zeros((40, 120, 3), dtype=np.uint8)

    def capture_bbox(self, bbox):
        return CaptureInfo(image=self.frame.copy(), bbox=tuple(bbox), mode="region")


@pytest.fixture
def cycling(controller, monkeypatch):
    screens = _Screens()
    controller.screenshot_service = screens
    controller._monitor_request = {"mode": "region", "bbox": (0, 0, 120, 40), "hwnd": None}
    started, scheduled = [], []
    monkeypatch.setattr(controller, "_start_pipeline", lambda capture: started.append(capture))
    monkeypatch.setattr(controller, "_monitor_schedule_next", lambda: scheduled.append(True))
    controller.screens, controller.started, controller.scheduled = screens, started, scheduled
    return controller


def test_an_identical_frame_is_skipped_once_something_was_shown(cycling):
    cycling._monitor_cycle()
    assert len(cycling.started) == 1
    cycling._monitor_signature = "shown"  # 第一輪的結果已經顯示

    cycling._monitor_cycle()

    assert len(cycling.started) == 1
    assert cycling.scheduled  # 只是排下一輪


def test_an_identical_frame_is_processed_again_while_a_new_text_waits_for_confirmation(cycling):
    cycling._monitor_cycle()
    cycling._monitor_signature = "shown"
    cycling._monitor_stabilizer.observe(["新的字幕"])  # 等第二次確認

    cycling._monitor_cycle()

    assert len(cycling.started) == 2


def test_one_changed_pixel_is_enough_to_process_the_frame_again(cycling):
    cycling._monitor_cycle()
    cycling._monitor_signature = "shown"
    cycling.screens.frame[10, 10, 0] = 255

    cycling._monitor_cycle()

    assert len(cycling.started) == 2


def test_an_identical_frame_with_no_text_is_not_recognised_again(cycling):
    cycling._monitor_cycle()
    cycling._monitor_idle_hash = cycling._monitor_frame_hash  # 這一格辨識不到文字

    cycling._monitor_cycle()

    assert len(cycling.started) == 1


# ----------------------------------------------------------------------------- 設定
def test_defaults_to_the_fast_mode():
    assert DEFAULTS["monitor"]["stability"] == "fast"
    assert DEFAULTS["monitor"]["stable_rounds"] == 2


def test_an_invalid_stability_value_falls_back_to_fast(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"monitor": {"stability": "turbo"}}), encoding="utf-8")

    assert AppConfig(path).get("monitor.stability") == "fast"


def test_an_old_config_without_the_setting_gets_the_fast_mode(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"monitor": {"interval_ms": 400}}), encoding="utf-8")

    assert AppConfig(path).get("monitor.stability") == "fast"


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def test_settings_dialog_explains_and_saves_the_stability_mode_on_the_general_page(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    assert dialog.combo_stability.currentData() == "fast"
    assert "穩定" in dialog.combo_stability.toolTip() and "快速" in dialog.combo_stability.toolTip()
    assert dialog.pages.widget(0).widget().isAncestorOf(dialog.combo_stability)  # 第一頁「通用」，不用往下捲

    dialog.combo_stability.setCurrentIndex(dialog.combo_stability.findData("stable"))
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("monitor.stability") == "stable"
    dialog.deleteLater()


class _Controller:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_main_window_has_a_stable_display_checkbox_that_is_off_by_default(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    assert window.check_stable.text() == "穩定顯示"
    assert not window.check_stable.isChecked()
    assert "快速" in window.check_stable.toolTip()
    window.deleteLater()


def test_main_window_checkbox_switches_the_mode_immediately(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    window.check_stable.setChecked(True)
    assert AppConfig(tmp_path / "config.json").get("monitor.stability") == "stable"
    window.check_stable.setChecked(False)
    assert AppConfig(tmp_path / "config.json").get("monitor.stability") == "fast"
    window.deleteLater()


def test_main_window_checkbox_follows_the_saved_setting_and_stays_usable_while_monitoring(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("monitor.stability", "stable")
    config.save()
    window = MainWindow(_Controller(config))
    assert window.check_stable.isChecked()

    window.set_busy(True)

    assert window.check_stable.isEnabled()  # 即時監控中也能切換
    config.set("monitor.stability", "fast")
    window.reload_values()
    assert not window.check_stable.isChecked()
    assert AppConfig(tmp_path / "config.json").get("monitor.stability") == "stable"  # 同步介面不會反過來寫設定檔
    window.deleteLater()
