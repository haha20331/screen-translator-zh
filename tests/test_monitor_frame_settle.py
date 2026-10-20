"""移動或調整監視框時不辨識、不翻譯：按著的期間一直等，放開之後再等設定的秒數（設定 → 即時監控）。"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import DEFAULTS, AppConfig
from ui import monitor_region
from ui.monitor_region import GRIP, GRIP_INSET, MonitorRegionSelector
from ui.settings_dialog import SettingsDialog
from utils.dpi_utils import MonitorInfo

MONITORS = [MonitorInfo(index=0, physical=(0, 0, 3840, 2160), logical_origin=(0, 0), dpr=1.0)]


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------------- 設定值
def test_the_default_wait_is_one_second():
    assert DEFAULTS["monitor"]["frame_settle_seconds"] == 1.0


@pytest.mark.parametrize("stored, expected", [(2.5, 2.5), (0, 0.0), (-3, 0.0), (99, 10.0), ("abc", 1.0), (None, 1.0)])
def test_the_wait_is_limited_to_zero_to_ten_seconds(tmp_path, stored, expected):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"monitor": {"frame_settle_seconds": stored}}), encoding="utf-8")

    assert AppConfig(path).get("monitor.frame_settle_seconds") == expected


def test_the_settings_page_offers_the_wait_and_saves_it(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.spin_frame_settle.value() == 1.0 and dialog.spin_frame_settle.maximum() == 10.0

    dialog.spin_frame_settle.setValue(2.5)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("monitor.frame_settle_seconds") == 2.5
    reopened = SettingsDialog(saved)
    assert reopened.spin_frame_settle.value() == 2.5
    dialog.deleteLater()
    reopened.deleteLater()


# ----------------------------------------------------------------------------- 框自己：什麼時候算「正在調整」
class _Cursor:
    where = QPoint(-5000, -5000)

    @classmethod
    def pos(cls):
        return cls.where


def _send(widget, kind, global_point: QPoint, buttons=Qt.MouseButton.LeftButton):
    local = QPointF(global_point - widget.geometry().topLeft())
    button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    QApplication.sendEvent(widget, QMouseEvent(kind, local, QPointF(global_point), button, buttons, Qt.KeyboardModifier.NoModifier))


def test_the_frame_is_adjusting_from_the_press_until_the_release(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    frame = MonitorRegionSelector(MONITORS, (100, 100, 500, 300), idle_ms=0)
    frame.show()
    grip = frame.geometry().topLeft() + QPoint(GRIP_INSET + GRIP // 2, GRIP_INSET + GRIP // 2)
    assert not frame.is_adjusting()

    _send(frame, QEvent.Type.MouseButtonPress, grip)
    assert frame.is_adjusting()
    _send(frame, QEvent.Type.MouseMove, grip + QPoint(20, 20))
    assert frame.is_adjusting()  # 按著不放（就算沒在動）還是在調整
    _send(frame, QEvent.Type.MouseButtonRelease, grip + QPoint(20, 20), buttons=Qt.MouseButton.NoButton)

    assert not frame.is_adjusting()
    frame.close()
    frame.deleteLater()


# ----------------------------------------------------------------------------- 控制器：監控輪次要等
class _Selector:
    def __init__(self) -> None:
        self.adjusting = False

    def is_adjusting(self) -> bool:
        return self.adjusting


class _Screens:
    def __init__(self) -> None:
        self.captured: list[tuple] = []

    def capture_bbox(self, bbox):
        self.captured.append(tuple(bbox))
        return SimpleNamespace(image=np.zeros((20, 40, 3), dtype=np.uint8), mode="region", bbox=tuple(bbox))


@pytest.fixture
def controller(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(qapp)
    instance.screenshot_service = _Screens()
    instance._monitor_active = True
    instance._monitor_request = {"mode": "region", "bbox": (10, 10, 300, 100), "hwnd": None}
    instance._monitor_region_selector = _Selector()
    instance.started = []
    monkeypatch.setattr(instance, "_start_pipeline", lambda capture: instance.started.append(capture))
    monkeypatch.setattr(instance, "_translator_warming", lambda: False)
    yield instance
    instance._monitor_region_selector = None
    instance._monitor_timer.stop()
    instance.floating_status.deleteLater()


def test_a_round_does_nothing_while_the_frame_is_being_moved(controller, monkeypatch):
    statuses = []
    monkeypatch.setattr(controller, "set_status", statuses.append)
    controller._monitor_region_selector.adjusting = True

    controller._monitor_cycle()
    controller._monitor_cycle()

    assert controller.screenshot_service.captured == [] and controller.started == []  # 不截圖、不辨識、不翻譯
    assert controller._monitor_timer.isActive() and controller._monitor_timer.interval() == 100  # 一小段時間後再看
    assert len(statuses) == 1 and "停下來" in statuses[0]  # 狀態列只提示一次，不會每隔 0.1 秒洗一次


def test_a_round_waits_the_set_seconds_after_the_release(controller):
    controller.config.set("monitor.frame_settle_seconds", 1.0)

    controller._on_monitor_region_committed((20, 20, 310, 110))  # 放開
    controller._monitor_cycle()

    assert controller.screenshot_service.captured == []
    interval = controller._monitor_timer.interval()
    assert 900 <= interval <= 1100  # 還要等約 1 秒
    assert controller._monitor_request["bbox"] == (20, 20, 310, 110)  # 新位置照樣記下來


def test_after_the_wait_the_round_runs_on_the_new_position(controller):
    controller.config.set("monitor.frame_settle_seconds", 1.0)
    controller._on_monitor_region_committed((20, 20, 310, 110))
    controller._monitor_quiet_until = time.monotonic() - 0.01  # 時間到了

    controller._monitor_cycle()

    assert controller.screenshot_service.captured == [(20, 20, 310, 110)] and len(controller.started) == 1


def test_moving_the_frame_again_during_the_wait_starts_the_wait_over(controller):
    controller.config.set("monitor.frame_settle_seconds", 2.0)
    controller._on_monitor_region_committed((20, 20, 310, 110))
    first = controller._monitor_quiet_until
    time.sleep(0.05)

    controller._on_monitor_region_committed((30, 30, 320, 120))

    assert controller._monitor_quiet_until > first


def test_zero_seconds_means_no_waiting_at_all_even_while_dragging(controller):
    controller.config.set("monitor.frame_settle_seconds", 0)
    controller._monitor_region_selector.adjusting = True

    controller._monitor_cycle()

    assert len(controller.screenshot_service.captured) == 1  # 舊行為：拖曳中照常辨識
    controller._on_monitor_region_committed((20, 20, 310, 110))
    assert controller._monitor_timer.interval() == 80 and controller._monitor_adjust_wait_ms() == 0


def test_the_release_schedules_the_next_round_after_the_wait_not_80_ms(controller):
    controller.config.set("monitor.frame_settle_seconds", 1.5)
    controller.worker = None

    controller._on_monitor_region_committed((20, 20, 310, 110))

    assert controller._monitor_timer.isActive() and 1400 <= controller._monitor_timer.interval() <= 1600


def test_starting_the_monitor_again_clears_a_leftover_wait(controller):
    controller._monitor_quiet_until = time.monotonic() + 60

    controller._monitor_active = False
    controller._last_capture = SimpleNamespace(mode="region", bbox=(10, 10, 300, 100), image=np.zeros((5, 5, 3), dtype=np.uint8))
    controller._monitor_region_selector = None
    controller._create_monitor_region_selector = lambda: None
    controller._ensure_model_loaded = lambda: ""
    controller.start_monitor()

    assert controller._monitor_quiet_until == 0.0
    controller._monitor_timer.stop()


def test_the_translation_panel_still_follows_the_frame_while_waiting(controller):
    followed = []
    controller.overlay_manager = SimpleNamespace(follow_region=lambda bbox: followed.append(tuple(bbox)))

    controller._on_monitor_region_changed((50, 50, 340, 140))

    assert followed == [(50, 50, 340, 140)]
