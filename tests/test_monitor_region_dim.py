"""常駐框：沒被動到超過 2 秒就變得更透明，滑鼠移進框內恢復。"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from ui import monitor_region
from ui.monitor_region import DIM_OPACITY, IDLE_DIM_MS, MonitorRegionSelector
from utils.dpi_utils import MonitorInfo


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _Cursor:
    """假的滑鼠游標：測試決定它在哪裡。"""

    where = QPoint(-5000, -5000)  # 預設在框外

    @classmethod
    def pos(cls):
        return cls.where


@pytest.fixture
def frame(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)  # 不要動畫：結果立刻看得到
    _Cursor.where = QPoint(-5000, -5000)
    monitors = [MonitorInfo(index=0, physical=(0, 0, 1920, 1080), logical_origin=(0, 0), dpr=1.0)]
    widget = MonitorRegionSelector(monitors, (100, 100, 500, 300), idle_ms=40)
    yield widget
    widget.close()
    widget.deleteLater()


def _wait(qapp, seconds):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


def _enter(widget):
    center = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(widget, QEnterEvent(center, center, center))


def _leave(widget):
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


def _mouse(widget, kind, x=20, y=20):  # (20, 20) 在左上角的移動符號裡面：框的中間不收滑鼠
    point = QPointF(x, y)
    button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    QApplication.sendEvent(widget, QMouseEvent(kind, point, point, button, button, Qt.KeyboardModifier.NoModifier))


def test_the_defaults_are_two_seconds_and_clearly_more_transparent():
    assert IDLE_DIM_MS == 2000 and 0 < DIM_OPACITY < 0.6


def test_an_untouched_frame_becomes_more_transparent_after_the_idle_time(qapp, frame):
    frame.show()
    assert frame.windowOpacity() == 1.0 and not frame.is_dimmed()

    _wait(qapp, 0.25)

    assert frame.is_dimmed() and frame.windowOpacity() == pytest.approx(monitor_region.DIM_OPACITY, abs=0.01)  # Qt 存成 8 位元


def test_it_does_not_dim_before_the_idle_time_is_up(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monitors = [MonitorInfo(index=0, physical=(0, 0, 1920, 1080), logical_origin=(0, 0), dpr=1.0)]
    widget = MonitorRegionSelector(monitors, (100, 100, 500, 300))  # 預設 2 秒
    widget.show()

    _wait(qapp, 0.3)

    assert not widget.is_dimmed() and widget.windowOpacity() == 1.0
    widget.close()


def test_moving_the_mouse_into_the_frame_restores_it_and_keeps_it_clear(qapp, frame):
    frame.show()
    _wait(qapp, 0.25)
    assert frame.is_dimmed()

    _Cursor.where = frame.geometry().center()
    _enter(frame)
    assert not frame.is_dimmed() and frame.windowOpacity() == 1.0

    _wait(qapp, 0.25)  # 滑鼠一直在框內：不會再變透明
    assert frame.windowOpacity() == 1.0


def test_leaving_the_frame_starts_the_countdown_again(qapp, frame):
    frame.show()
    _Cursor.where = frame.geometry().center()
    _enter(frame)
    _wait(qapp, 0.15)
    assert frame.windowOpacity() == 1.0

    _Cursor.where = QPoint(-5000, -5000)
    _leave(frame)
    _wait(qapp, 0.25)

    assert frame.is_dimmed()


def test_a_cursor_that_is_inside_at_the_deadline_keeps_the_frame_clear(qapp, frame):
    """enter／leave 事件沒送到（例如框剛顯示在靜止的游標底下）時，以游標的實際位置為準。"""
    _Cursor.where = frame.geometry().center()
    frame.show()

    _wait(qapp, 0.25)

    assert not frame.is_dimmed() and frame.windowOpacity() == 1.0


def test_pressing_and_dragging_the_frame_never_dims_it(qapp, frame):
    frame.show()
    _wait(qapp, 0.25)
    assert frame.is_dimmed()

    _mouse(frame, QEvent.Type.MouseButtonPress)
    assert frame.windowOpacity() == 1.0

    _Cursor.where = QPoint(-5000, -5000)  # 拖曳中游標暫時不在框的位置上
    _leave(frame)
    _wait(qapp, 0.25)
    assert not frame.is_dimmed()

    _mouse(frame, QEvent.Type.MouseButtonRelease)
    _wait(qapp, 0.25)  # 放開之後游標在框外：重新倒數、變透明
    assert frame.is_dimmed()


def test_moving_inside_a_dimmed_frame_restores_it_even_without_an_enter_event(qapp, frame):
    frame.show()
    _wait(qapp, 0.25)
    assert frame.is_dimmed()

    _mouse(frame, QEvent.Type.MouseMove)

    assert not frame.is_dimmed() and frame.windowOpacity() == 1.0


def test_an_idle_time_of_zero_turns_the_dimming_off(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monitors = [MonitorInfo(index=0, physical=(0, 0, 1920, 1080), logical_origin=(0, 0), dpr=1.0)]
    widget = MonitorRegionSelector(monitors, (100, 100, 500, 300), idle_ms=0)
    widget.show()

    _wait(qapp, 0.2)

    assert not widget.is_dimmed() and widget.windowOpacity() == 1.0
    widget.close()


def test_with_animations_on_the_opacity_still_ends_at_the_right_values(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 40)
    monitors = [MonitorInfo(index=0, physical=(0, 0, 1920, 1080), logical_origin=(0, 0), dpr=1.0)]
    widget = MonitorRegionSelector(monitors, (100, 100, 500, 300), idle_ms=30)
    widget.show()
    _wait(qapp, 0.4)
    assert widget.windowOpacity() == pytest.approx(monitor_region.DIM_OPACITY, abs=0.01)

    _Cursor.where = widget.geometry().center()
    _enter(widget)
    _wait(qapp, 0.2)

    assert widget.windowOpacity() == pytest.approx(1.0, abs=0.01)
    widget.close()
