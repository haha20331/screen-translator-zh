"""預設使用即時監控：翻譯完成後自動進入監控；監控中設定、隱藏譯文與新的擷取都不被擋住。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import DEFAULTS, AppConfig
from app.models import CaptureInfo
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.stability import CONFIRMED, TextStabilizer


class _FakeOverlay:
    def __init__(self) -> None:
        self.cleared = 0

    def clear_all(self):
        self.cleared += 1

    def hide_all(self, animate=False):
        pass

    def is_visible(self):
        return True


class _Worker:
    def deleteLater(self) -> None:
        pass


@pytest.fixture
def controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(app)
    instance.overlay_manager = _FakeOverlay()
    instance._last_capture = CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=(0, 0, 120, 40), mode="region")
    instance.started = []
    monkeypatch.setattr(instance, "start_monitor", lambda: instance.started.append(True))
    yield instance
    instance.floating_status.deleteLater()


def _finish_one_shot(controller, succeeded=True):
    controller.worker = _Worker()
    controller._monitor_active = False
    controller._capture_session = None
    controller._pipeline_succeeded = succeeded
    controller._pipeline_error = "" if succeeded else "失敗"
    controller._on_worker_finished()


# ----------------------------------------------------------------------------- 自動開始
def test_monitoring_starts_by_default_after_a_successful_translation(controller):
    assert DEFAULTS["monitor"]["auto_start"] is True

    _finish_one_shot(controller)

    assert controller.started == [True]
    assert controller._busy is False


def test_a_failed_translation_does_not_start_monitoring(controller):
    _finish_one_shot(controller, succeeded=False)

    assert controller.started == []


@pytest.mark.parametrize("setting", ["monitor.auto_start", "monitor.enabled"])
def test_monitoring_can_be_switched_off(controller, setting):
    controller.config.set(setting, False)

    _finish_one_shot(controller)

    assert controller.started == []


def test_no_second_monitor_is_started_while_one_is_running(controller):
    controller._monitor_active = True

    controller._auto_start_monitor()

    assert controller.started == []


def test_nothing_starts_without_a_capture_or_during_shutdown(controller):
    controller._last_capture = None
    controller._auto_start_monitor()
    assert controller.started == []

    controller._last_capture = CaptureInfo(image=np.zeros((4, 4, 3), dtype=np.uint8), bbox=(0, 0, 4, 4), mode="region")
    controller._shutting_down = True
    controller._auto_start_monitor()
    assert controller.started == []


# ----------------------------------------------------------------------------- 監控中不再卡住其他功能
def test_monitoring_does_not_hold_the_app_busy(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(app)
    instance.overlay_manager = _FakeOverlay()
    instance.monitor_map = None  # 不建立框選區域視窗
    instance._last_capture = CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=(0, 0, 120, 40), mode="region")
    monkeypatch.setattr(instance, "_monitor_cycle", lambda: None)

    instance.start_monitor()
    assert instance._monitor_active and instance._busy is False  # 設定、隱藏譯文、新擷取都不會被擋

    instance.stop_monitor()
    assert instance._monitor_active is False and instance._busy is False
    instance.floating_status.deleteLater()


def test_a_new_capture_replaces_the_running_monitor(controller, monkeypatch):
    controller.window = MagicMock()
    controller._monitor_active = True
    controller._monitor_stabilizer = TextStabilizer()
    begun = []
    monkeypatch.setattr(controller, "_begin_capture_session", lambda *args, **kwargs: begun.append(args))

    controller.start_region_capture()

    assert controller._monitor_active is False
    assert controller.overlay_manager.cleared == 1  # 舊的譯文隨監控一起清掉
    assert [args[0] for args in begun] == ["region"]


@pytest.mark.parametrize("mode", ["fullscreen"])
def test_fullscreen_capture_also_replaces_the_monitor(controller, monkeypatch, mode):
    controller.window = MagicMock()
    controller._monitor_active = True
    controller._monitor_stabilizer = TextStabilizer()
    begun = []
    monkeypatch.setattr(controller, "_begin_capture_session", lambda *args, **kwargs: begun.append(args))

    controller.start_capture(mode)

    assert controller._monitor_active is False
    assert [args[0] for args in begun] == [mode]


def test_refresh_while_monitoring_forgets_the_display_and_rescans_immediately(controller):
    controller._monitor_active = True
    controller._monitor_stabilizer = TextStabilizer(rounds=1)
    controller._monitor_stabilizer.observe(["A"])
    controller._monitor_stabilizer.commit()
    controller._monitor_signature = "shown"
    controller._monitor_frame_hash = "frame"

    controller.refresh()

    assert controller._monitor_signature == "" and controller._monitor_frame_hash == ""
    assert controller._monitor_stabilizer.observe(["A"]) == CONFIRMED  # 已顯示的記憶清掉：同一句會重新翻譯並顯示
    assert controller._monitor_timer.isActive()
    controller._monitor_timer.stop()


# ----------------------------------------------------------------------------- 設定
def test_settings_dialog_has_the_auto_monitor_checkbox_on_the_first_page(tmp_path):
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.chk_auto_monitor.isChecked()
    assert dialog.pages.widget(0).widget().isAncestorOf(dialog.chk_auto_monitor)

    dialog.chk_auto_monitor.setChecked(False)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("monitor.auto_start") is False
    dialog.deleteLater()
