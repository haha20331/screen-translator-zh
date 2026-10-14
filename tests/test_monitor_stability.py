"""即時監控的穩定化：新字幕連續辨識到才顯示、與已顯示的相同不重翻、關閉監控時清掉字幕。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel

from app.application import Application
from app.config import DEFAULTS, AppConfig
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCRLine
from ui.main_window import MainWindow
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils import stability
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


def test_confirm_rounds_setting_is_used_and_can_change_while_monitoring(controller):
    controller.config.set("monitor.confirm_rounds", 1)
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 1

    controller.config.set("monitor.confirm_rounds", 2)
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 2

    controller.config.set("monitor.confirm_rounds", 5)  # 使用者自訂：監控中途改，下一輪就生效
    controller.config.set("monitor.max_wait_seconds", 4.0)
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 5
    assert controller._monitor_stabilizer.max_wait == 4.0


def test_out_of_range_or_broken_values_are_clamped_not_fatal(controller):
    controller.config.set("monitor.confirm_rounds", 99)
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 10

    controller.config.set("monitor.confirm_rounds", "abc")
    controller.config.set("monitor.max_wait_seconds", None)
    controller._configure_stabilizer()
    assert controller._monitor_stabilizer.rounds == 1
    assert controller._monitor_stabilizer.max_wait == 2.5


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
def test_defaults_are_the_recommended_fast_values():
    assert DEFAULTS["monitor"]["confirm_rounds"] == stability.RECOMMENDED_FAST_ROUNDS == 1
    assert DEFAULTS["monitor"]["max_wait_seconds"] == stability.RECOMMENDED_MAX_WAIT == 2.5
    assert stability.RECOMMENDED_STABLE_ROUNDS == 2
    assert "stability" not in DEFAULTS["monitor"] and "stable_rounds" not in DEFAULTS["monitor"]


def _load(tmp_path, monitor):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"monitor": monitor}), encoding="utf-8")
    return AppConfig(path)


@pytest.mark.parametrize(
    ("saved", "rounds"),
    [
        ({}, 1),
        ({"stability": "fast"}, 1),  # 舊版的選項換算成次數
        ({"stability": "stable"}, 2),
        ({"stability": "stable", "stable_rounds": 2}, 2),
        ({"stability": "stable", "confirm_rounds": 4}, 4),  # 使用者新輸入的次數優先
        ({"stability": "turbo"}, 1),
        ({"confirm_rounds": 3}, 3),
        ({"confirm_rounds": 0}, 1),
        ({"confirm_rounds": 50}, 10),
        ({"confirm_rounds": "many"}, 1),
    ],
)
def test_confirm_rounds_are_migrated_and_clamped(tmp_path, saved, rounds):
    config = _load(tmp_path, saved)

    assert config.get("monitor.confirm_rounds") == rounds
    assert config.get("monitor.stability") is None and config.get("monitor.stable_rounds") is None


@pytest.mark.parametrize(("saved", "wait"), [({}, 2.5), ({"max_wait_seconds": 5}, 5.0), ({"max_wait_seconds": 0}, 0.5), ({"max_wait_seconds": 999}, 30.0), ({"max_wait_seconds": "x"}, 2.5)])
def test_max_wait_is_clamped(tmp_path, saved, wait):
    assert _load(tmp_path, saved).get("monitor.max_wait_seconds") == wait


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def test_settings_dialog_lets_the_user_type_the_threshold_and_lists_the_recommendations(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    assert dialog.pages.widget(0).widget().isAncestorOf(dialog.spin_confirm_rounds)  # 第一頁「通用」，不用往下捲
    assert (dialog.spin_confirm_rounds.minimum(), dialog.spin_confirm_rounds.maximum()) == (1, 10)
    assert dialog.spin_confirm_rounds.value() == 1 and dialog.spin_max_wait.value() == 2.5
    texts = [label.text() for label in dialog.pages.widget(0).widget().findChildren(QLabel)]
    recommendation = next(text for text in texts if "系統推薦" in text)
    assert "快速：確認 1 次" in recommendation and "穩定：確認 2 次" in recommendation
    assert "2.5 秒" in recommendation
    assert not hasattr(dialog, "combo_stability")
    dialog.deleteLater()


def test_preset_buttons_fill_in_the_recommended_values(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    dialog.spin_confirm_rounds.setValue(7)
    dialog.spin_max_wait.setValue(9.0)

    dialog.btn_preset_stable.click()
    assert (dialog.spin_confirm_rounds.value(), dialog.spin_max_wait.value()) == (2, 2.5)
    dialog.spin_max_wait.setValue(9.0)
    dialog.btn_preset_fast.click()
    assert (dialog.spin_confirm_rounds.value(), dialog.spin_max_wait.value()) == (1, 2.5)
    dialog.deleteLater()


def test_settings_dialog_saves_the_typed_values(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    dialog.spin_confirm_rounds.setValue(3)
    dialog.spin_max_wait.setValue(4.5)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("monitor.confirm_rounds") == 3
    assert saved.get("monitor.max_wait_seconds") == 4.5
    reopened = SettingsDialog(saved)
    assert reopened.spin_confirm_rounds.value() == 3 and reopened.spin_max_wait.value() == 4.5
    dialog.deleteLater()
    reopened.deleteLater()


class _Controller:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_main_window_footer_no_longer_shows_stability_or_context(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    assert not hasattr(window, "check_stable") and not hasattr(window, "check_context")
    assert hasattr(window, "check_lyrics")  # 歌詞勾選留在主視窗
    texts = {box.text() for box in window.findChildren(QCheckBox)}
    assert "穩定顯示" not in texts and "上下文" not in texts and "歌詞" in texts
    window.deleteLater()


# ----------------------------------------------------------------------------- 辨識間隔（使用者可調）
def test_default_scan_interval_is_the_recommended_value():
    assert DEFAULTS["monitor"]["interval_ms"] == stability.RECOMMENDED_INTERVAL_MS == 100
    assert (stability.MIN_INTERVAL_MS, stability.MAX_INTERVAL_MS) == (100, 10_000)


@pytest.mark.parametrize(
    ("saved", "interval"),
    [
        ({}, 100),
        ({"interval_ms": 150}, 150),  # 小數兩位的 0.15 秒
        ({"interval_ms": 800, "interval_default_version": 2}, 800),
        ({"interval_ms": 2500}, 2500),
        ({"interval_ms": 50}, 100),  # 太短：夾到最小 0.1 秒
        ({"interval_ms": 0}, 100),
        ({"interval_ms": -300}, 100),
        ({"interval_ms": 99999}, 10_000),
        ({"interval_ms": "fast"}, 100),
        ({"interval_ms": None}, 100),
    ],
)
def test_scan_interval_is_clamped_to_the_allowed_range(tmp_path, saved, interval):
    assert _load(tmp_path, saved).get("monitor.interval_ms") == interval


@pytest.mark.parametrize(
    ("saved", "interval"),
    [
        ({"interval_ms": 1200}, 100),  # 上游的舊預設：換成新預設
        ({"interval_ms": 400}, 100),  # 前一版的預設：換成新預設
        ({"interval_ms": 400, "interval_migrated": True}, 100),  # 前一版設定畫面存的預設值也一樣
        ({"interval_ms": 1200, "interval_migrated": True}, 1200),  # 前一版設定畫面存的 1.2 秒是使用者自己選的
        ({"interval_ms": 1200, "interval_default_version": 2}, 1200),  # 換過之後，任何值都照使用者的
        ({"interval_ms": 400, "interval_default_version": 2}, 400),
    ],
)
def test_old_defaults_are_replaced_once_but_what_the_user_chose_is_kept(tmp_path, saved, interval):
    config = _load(tmp_path, saved)

    assert config.get("monitor.interval_ms") == interval
    assert config.get("monitor.interval_default_version") == 2  # 標記存進檔案，之後就不會再換
    assert config.get("monitor.interval_migrated") is None


def test_after_the_replacement_is_saved_a_typed_value_survives_restarts(tmp_path):
    config = _load(tmp_path, {"interval_ms": 400})
    assert config.get("monitor.interval_ms") == 100
    config.set("monitor.interval_ms", 400)  # 使用者自己選了 0.4 秒
    config.save()

    assert AppConfig(tmp_path / "config.json").get("monitor.interval_ms") == 400


def test_the_scheduler_uses_the_configured_interval(controller):
    controller._monitor_timer.stop()
    controller.config.set("monitor.interval_ms", 800)
    controller._monitor_schedule_next()
    assert controller._monitor_timer.interval() == 800 and controller._monitor_timer.isActive()

    controller.config.set("monitor.interval_ms", 150)  # 小數兩位的 0.15 秒
    controller._monitor_schedule_next()
    assert controller._monitor_timer.interval() == 150

    controller.config.set("monitor.interval_ms", 50)  # 太短：夾到最小值，避免整台電腦被辨識佔滿
    controller._monitor_schedule_next()
    assert controller._monitor_timer.interval() == 100

    controller.config.set("monitor.interval_ms", "abc")
    controller._monitor_schedule_next()
    assert controller._monitor_timer.interval() == 100
    controller._monitor_timer.stop()


def test_changing_the_interval_in_settings_applies_to_the_round_already_waiting(controller):
    controller.config.set("monitor.interval_ms", 9000)
    controller._monitor_schedule_next()  # 正在等 9 秒後的下一輪
    controller.config.set("monitor.interval_ms", 500)

    controller._reschedule_monitor_after_settings()
    assert controller._monitor_timer.isActive() and controller._monitor_timer.interval() == 500
    controller._monitor_timer.stop()


def test_changing_the_interval_does_not_start_a_round_that_is_still_running(controller):
    controller._monitor_timer.stop()  # 這一輪正在辨識／翻譯：計時器沒在跑，結束時才會排下一輪
    controller.config.set("monitor.interval_ms", 500)

    controller._reschedule_monitor_after_settings()
    assert not controller._monitor_timer.isActive()

    controller._monitor_active = False  # 監控已關閉時更不會被重新啟動
    controller._monitor_timer.start(1000)
    controller._reschedule_monitor_after_settings()
    assert controller._monitor_timer.interval() == 1000
    controller._monitor_timer.stop()


def test_settings_dialog_lets_the_user_set_the_scan_interval(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    assert dialog.pages.widget(0).widget().isAncestorOf(dialog.spin_scan_interval)  # 「通用」頁，不用往下捲
    assert (dialog.spin_scan_interval.minimum(), dialog.spin_scan_interval.maximum()) == (0.1, 10.0)
    assert dialog.spin_scan_interval.decimals() == 2  # 小數下兩位
    assert dialog.spin_scan_interval.value() == 0.1  # 預設 0.1 秒
    assert dialog.lbl_scan_rate.text() == "約每秒檢查 10.0 次畫面"
    texts = [label.text() for label in dialog.pages.widget(0).widget().findChildren(QLabel)]
    assert "辨識間隔：0.1 秒" in next(text for text in texts if "系統推薦" in text)

    dialog.spin_scan_interval.setValue(0.5)
    assert dialog.lbl_scan_rate.text() == "約每秒檢查 2.0 次畫面"
    dialog.spin_scan_interval.setValue(1.2)  # 剛好是舊預設的值也要存得住
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("monitor.interval_ms") == 1200
    assert AppConfig(tmp_path / "config.json").get("monitor.interval_ms") == 1200  # 重新載入不被舊預設遷移蓋掉
    reopened = SettingsDialog(saved)
    assert reopened.spin_scan_interval.value() == 1.2
    dialog.deleteLater()
    reopened.deleteLater()


def _type_into_the_interval_box(dialog, text):
    dialog.spin_scan_interval.lineEdit().setText(text)  # 像使用者輸入
    dialog.spin_scan_interval.interpretText()


@pytest.mark.parametrize(
    ("typed", "milliseconds"),
    [("0.15", 150), ("0.25", 250), ("0.29", 290), ("0.1", 100), ("0.10", 100), ("0.35 秒", 350), ("1.2", 1200), ("10", 10_000)],
)
def test_typed_values_are_accepted_to_two_decimals(qapp, tmp_path, typed, milliseconds):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    _type_into_the_interval_box(dialog, typed)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("monitor.interval_ms") == milliseconds
    assert AppConfig(tmp_path / "config.json").get("monitor.interval_ms") == milliseconds  # 再載入一次，仍是同一個值
    dialog.deleteLater()


@pytest.mark.parametrize("typed", ["0.07", "0.05", "0", "-1", "12", "0.123", "abc", ""])
def test_values_the_box_cannot_take_leave_the_previous_value(qapp, tmp_path, typed):
    """低於 0.1、超過 10、第三位小數、不是數字：輸入框直接拒絕，維持原本的值（不會存進奇怪的數字）。"""
    config = AppConfig(tmp_path / "config.json")
    config.set("monitor.interval_ms", 300)
    dialog = SettingsDialog(config)

    _type_into_the_interval_box(dialog, typed)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("monitor.interval_ms") == 300
    dialog.deleteLater()


# ----------------------------------------------------------------------------- 抓不到字幕一段時間後自動關閉譯文
NO_TEXT = "沒有識別到文字（檢查 OCR 引擎設定或置信度閾值）"


@pytest.mark.parametrize(
    ("saved", "seconds"),
    [({}, 1.0), ({"hide_after_no_text_seconds": 2.5}, 2.5), ({"hide_after_no_text_seconds": 0}, 0.0),
     ({"hide_after_no_text_seconds": -3}, 0.0), ({"hide_after_no_text_seconds": 999}, 30.0), ({"hide_after_no_text_seconds": "x"}, 1.0)],
)
def test_hide_after_no_text_is_clamped(tmp_path, saved, seconds):
    assert _load(tmp_path, saved).get("monitor.hide_after_no_text_seconds") == seconds


def test_no_text_for_a_while_closes_the_translation_after_one_second_by_default(controller):
    controller._no_text_timer.stop()
    stabilizer = controller._monitor_stabilizer
    stabilizer.observe(["こんにちは"])
    stabilizer.observe(["こんにちは"])
    stabilizer.commit()
    controller._monitor_signature = "shown"

    controller._on_worker_error(NO_TEXT)

    assert controller._no_text_timer.isActive() and controller._no_text_timer.interval() == 1000
    assert controller.overlay_manager.cleared == 0  # 還沒滿 1 秒：先不關

    controller._close_translation_after_no_text()  # 計時器到了
    assert controller.overlay_manager.cleared == 1
    assert controller._monitor_signature == ""
    assert stabilizer.observe(["こんにちは"]) != SAME  # 同一句字幕之後再出現，會重新顯示
    controller._no_text_timer.stop()


def test_finding_text_again_cancels_the_pending_close(controller):
    controller._on_worker_error(NO_TEXT)
    assert controller._no_text_timer.isActive()

    controller.on_pipeline_result({"capture": _payload()["capture"], "regions": [], "verdict": SAME})  # 又抓到字（相同字幕）

    assert not controller._no_text_timer.isActive()
    assert controller.overlay_manager.cleared == 0


def test_a_new_subtitle_cancels_the_pending_close_too(controller):
    controller._on_worker_error(NO_TEXT)
    controller._monitor_stabilizer.observe(["こんにちは"])
    controller._monitor_stabilizer.observe(["こんにちは"])

    controller.on_pipeline_result(_payload())

    assert not controller._no_text_timer.isActive()
    assert len(controller.overlay_manager.shown) == 1


def test_consecutive_empty_rounds_do_not_restart_the_countdown(controller):
    controller._no_text_timer.stop()
    controller._on_worker_error(NO_TEXT)
    controller._no_text_timer.start(200)  # 已經倒數到一半

    controller._on_worker_error(NO_TEXT)  # 下一輪也沒字：不能重新倒數，否則畫面一直在動時永遠關不掉

    assert controller._no_text_timer.interval() == 200
    controller._no_text_timer.stop()


def test_the_close_delay_is_configurable_and_zero_turns_it_off(controller):
    controller._no_text_timer.stop()
    controller.config.set("monitor.hide_after_no_text_seconds", 2.5)
    controller._on_worker_error(NO_TEXT)
    assert controller._no_text_timer.interval() == 2500
    controller._no_text_timer.stop()

    controller.config.set("monitor.hide_after_no_text_seconds", 0)
    controller._on_worker_error(NO_TEXT)
    assert not controller._no_text_timer.isActive()  # 0 = 不自動關閉


def test_it_never_fires_outside_monitoring(controller):
    controller._monitor_active = False
    controller._arm_no_text_timer()
    assert not controller._no_text_timer.isActive()

    controller._close_translation_after_no_text()  # 計時器剛好在停止監控前到期
    assert controller.overlay_manager.cleared == 0


def test_stopping_the_monitor_cancels_the_countdown(controller):
    controller._on_worker_error(NO_TEXT)

    controller.stop_monitor()

    assert not controller._no_text_timer.isActive()


def test_settings_dialog_edits_the_no_text_delay(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.pages.widget(0).widget().isAncestorOf(dialog.spin_hide_no_text)
    assert dialog.spin_hide_no_text.value() == 1.0

    dialog.spin_hide_no_text.setValue(0)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("monitor.hide_after_no_text_seconds") == 0.0
    dialog.deleteLater()


def test_the_main_window_no_longer_has_a_hide_translation_button(qapp, tmp_path):
    from tests.test_translation_context_lyrics import _Controller

    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    assert not hasattr(window, "btn_toggle") and not hasattr(window, "set_overlay_checked")
    labels = [button.text() for button in window.findChildren(__import__("PySide6.QtWidgets", fromlist=["QPushButton"]).QPushButton)]
    assert "隱藏譯文" not in labels and "顯示譯文" not in labels
    window.deleteLater()


# ----------------------------------------------------------------------------- 主視窗預設高度
def test_the_main_window_default_size_is_980_by_580_and_shrinks_to_fit_small_screens():
    from ui.main_window import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, MIN_WINDOW_SIZE, default_window_size

    # 曾經放大到 1.3 倍（754）；刪掉主視窗的範例卡之後內容變少，使用者把高度改回 580
    assert (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) == (980, 580)
    assert default_window_size() == (980, 580)
    assert default_window_size(1080) == (980, 580)  # 螢幕夠高：照預設
    assert default_window_size(600) == (980, MIN_WINDOW_SIZE[1])  # 放不下：縮到螢幕可用高度內，但不低於最小尺寸
    assert default_window_size(300) == (980, MIN_WINDOW_SIZE[1])


def test_a_new_main_window_uses_the_default_size(qapp, tmp_path):
    from tests.test_translation_context_lyrics import _Controller
    from ui.main_window import default_window_size

    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))
    screen = qapp.primaryScreen()

    assert (window.width(), window.height()) == default_window_size(screen.availableGeometry().height())
    assert window.minimumHeight() == 540
    window.deleteLater()
