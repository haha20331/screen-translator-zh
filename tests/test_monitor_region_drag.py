"""常駐框：拖曳、縮放要跟著游標、不抽搐；框內全透明、框線不變；框變淡時游標移進來照樣恢復。"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from ui import monitor_region
from ui.monitor_region import BAND, GRIP, GRIP_INSET, MIN_FRAME_SIZE, MonitorRegionSelector
from utils.dpi_utils import MonitorInfo

MONITORS = [MonitorInfo(index=0, physical=(0, 0, 3840, 2160), logical_origin=(0, 0), dpr=1.0)]
START = (100, 100, 500, 300)  # 螢幕座標：400 x 200 的框


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _Cursor:
    where = QPoint(-5000, -5000)

    @classmethod
    def pos(cls):
        return cls.where


@pytest.fixture
def frame(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    _Cursor.where = QPoint(-5000, -5000)
    widget = MonitorRegionSelector(MONITORS, START, idle_ms=0)  # 不變淡：這裡只看拖曳
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()


def _wait(qapp, seconds):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.003)


def _top_left_grip(widget) -> QPoint:
    """左上角移動符號的中心（螢幕座標）：移動框要按住它拖。"""
    return widget.geometry().topLeft() + QPoint(GRIP_INSET + GRIP // 2, GRIP_INSET + GRIP // 2)


def _bottom_right_grip(widget) -> QPoint:
    return widget.geometry().bottomRight() - QPoint(GRIP_INSET + GRIP // 2 - 1, GRIP_INSET + GRIP // 2 - 1)


def _send(widget, kind, global_point: QPoint, buttons=Qt.MouseButton.LeftButton):
    """像真的滑鼠事件：區域座標是「游標的螢幕座標減去視窗『現在』的位置」，視窗移動之後同一個游標位置的區域座標就變了。"""
    local = QPointF(global_point - widget.geometry().topLeft())
    button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    QApplication.sendEvent(widget, QMouseEvent(kind, local, QPointF(global_point), button, buttons, Qt.KeyboardModifier.NoModifier))


def test_dragging_the_frame_follows_the_cursor_exactly_at_every_step(qapp, frame):
    base = frame.geometry()
    press = _top_left_grip(frame)
    _send(frame, QEvent.Type.MouseButtonPress, press)

    for step in range(1, 21):
        cursor = press + QPoint(7 * step, 4 * step)
        _send(frame, QEvent.Type.MouseMove, cursor)
        assert frame.geometry() == base.translated(cursor - press), f"第 {step} 步：框跳到別的位置了"  # 以前每一步在兩個位置之間來回跳

    _send(frame, QEvent.Type.MouseButtonRelease, cursor, buttons=Qt.MouseButton.NoButton)
    assert frame.geometry() == base.translated(cursor - press)


def test_holding_the_cursor_still_keeps_the_frame_still(qapp, frame):
    press = _top_left_grip(frame)
    _send(frame, QEvent.Type.MouseButtonPress, press)
    _send(frame, QEvent.Type.MouseMove, press + QPoint(60, 30))
    settled = frame.geometry()

    for _ in range(10):  # 游標沒動、事件還是一直來（例如框移動後系統補送的事件）
        _send(frame, QEvent.Type.MouseMove, press + QPoint(60, 30))
        assert frame.geometry() == settled


def test_resizing_from_the_right_edge_and_the_bottom_right_corner_is_steady_too(qapp, frame):
    base = frame.geometry()
    edge = QPoint(base.right() - 2, base.center().y())  # 右邊緣（把手範圍內）
    _send(frame, QEvent.Type.MouseButtonPress, edge)
    for step in range(1, 11):
        _send(frame, QEvent.Type.MouseMove, edge + QPoint(10 * step, 0))
        assert (frame.width(), frame.height(), frame.x(), frame.y()) == (base.width() + 10 * step, base.height(), base.x(), base.y())
    _send(frame, QEvent.Type.MouseButtonRelease, edge + QPoint(100, 0), buttons=Qt.MouseButton.NoButton)

    grown = frame.geometry()
    corner = QPoint(grown.right() - 2, grown.bottom() - 2)
    _send(frame, QEvent.Type.MouseButtonPress, corner)
    for step in range(1, 11):
        _send(frame, QEvent.Type.MouseMove, corner + QPoint(-6 * step, -3 * step))
        assert (frame.width(), frame.height()) == (grown.width() - 6 * step, grown.height() - 3 * step)
    _send(frame, QEvent.Type.MouseButtonRelease, corner, buttons=Qt.MouseButton.NoButton)


def test_resizing_from_the_left_and_top_edges_moves_the_frame_without_shaking(qapp, frame):
    """左邊、上邊縮放時框的左上角會跟著動，以前的算法這時候抖得最厲害。"""
    base = frame.geometry()
    left = QPoint(base.left() + 2, base.center().y())
    _send(frame, QEvent.Type.MouseButtonPress, left)
    for step in range(1, 11):
        _send(frame, QEvent.Type.MouseMove, left + QPoint(-8 * step, 0))
        assert (frame.x(), frame.width(), frame.y(), frame.height()) == (base.x() - 8 * step, base.width() + 8 * step, base.y(), base.height())
    _send(frame, QEvent.Type.MouseButtonRelease, left, buttons=Qt.MouseButton.NoButton)

    grown = frame.geometry()
    top_left = QPoint(grown.left() + 2, grown.top() + 2)
    _send(frame, QEvent.Type.MouseButtonPress, top_left)
    for step in range(1, 11):
        _send(frame, QEvent.Type.MouseMove, top_left + QPoint(5 * step, 3 * step))
        assert (frame.x(), frame.y(), frame.width(), frame.height()) == (
            grown.x() + 5 * step, grown.y() + 3 * step, grown.width() - 5 * step, grown.height() - 3 * step)
    _send(frame, QEvent.Type.MouseButtonRelease, top_left, buttons=Qt.MouseButton.NoButton)


def test_the_frame_never_gets_smaller_than_the_minimum_whatever_the_cursor_does(qapp, frame):
    base = frame.geometry()
    edge = QPoint(base.right() - 2, base.center().y())
    _send(frame, QEvent.Type.MouseButtonPress, edge)

    _send(frame, QEvent.Type.MouseMove, edge + QPoint(-5000, 0))

    assert frame.width() >= 32 and frame.x() == base.x()
    _send(frame, QEvent.Type.MouseButtonRelease, edge, buttons=Qt.MouseButton.NoButton)


def test_the_region_updates_for_the_panel_are_coalesced_but_the_last_one_is_exact(qapp, frame):
    changed, committed = [], []
    frame.region_changed.connect(changed.append)
    frame.region_committed.connect(committed.append)
    press = _top_left_grip(frame)
    _send(frame, QEvent.Type.MouseButtonPress, press)

    for step in range(1, 201):  # 兩百個滑鼠事件擠在一起
        _send(frame, QEvent.Type.MouseMove, press + QPoint(step, step // 2))
    _send(frame, QEvent.Type.MouseButtonRelease, press + QPoint(200, 100), buttons=Qt.MouseButton.NoButton)
    _wait(qapp, 0.1)

    assert len(changed) < 60  # 不是每個滑鼠事件都重排一次譯文面板
    expected = (300, 200, 700, 400)  # 原本 (100, 100, 500, 300) 往右 200、往下 100
    assert changed[-1] == expected and committed == [expected]  # 最後停下來的位置一定送出去


def test_a_finished_drag_leaves_no_pending_update_behind(qapp, frame):
    changed = []
    frame.region_changed.connect(changed.append)
    press = _top_left_grip(frame)
    _send(frame, QEvent.Type.MouseButtonPress, press)
    _send(frame, QEvent.Type.MouseMove, press + QPoint(40, 40))
    _send(frame, QEvent.Type.MouseButtonRelease, press + QPoint(40, 40), buttons=Qt.MouseButton.NoButton)
    count = len(changed)

    _wait(qapp, 0.15)

    assert len(changed) == count  # 放開之後不會再有遲到的「變更」


# ----------------------------------------------------------------------------- 外觀
def test_the_inside_of_the_frame_is_invisible_but_the_border_and_handles_are_unchanged(qapp, frame):
    image = frame.grab().toImage()
    w, h = image.width(), image.height()

    inside = image.pixelColor(w // 2, h // 2)
    assert inside.alpha() <= 1  # 肉眼看是全透明（留 1 是為了讓滑鼠還能抓住框的裡面，見 paintEvent 的說明）
    border = image.pixelColor(1, h // 2)
    assert border.alpha() >= 150  # 框線和以前一樣
    corner = image.pixelColor(4, 4)
    assert corner.alpha() >= 200  # 四個角的把手也和以前一樣


# ----------------------------------------------------------------------------- 變淡之後游標移進來
def test_a_dimmed_frame_comes_back_when_the_cursor_moves_in_even_without_any_mouse_event(qapp, monkeypatch):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    _Cursor.where = QPoint(-5000, -5000)
    widget = MonitorRegionSelector(MONITORS, START, idle_ms=40, poll_ms=20)
    widget.show()
    _wait(qapp, 0.25)
    assert widget.is_dimmed()

    _Cursor.where = widget.geometry().center()  # 變淡後框裡面幾乎不收滑鼠事件，要靠定時檢查游標的位置
    _wait(qapp, 0.2)

    assert not widget.is_dimmed() and widget.windowOpacity() == 1.0
    _Cursor.where = QPoint(-5000, -5000)
    _wait(qapp, 0.3)
    assert widget.is_dimmed()  # 游標離開之後照樣倒數、再變淡
    widget.close()
    widget.deleteLater()


def test_the_frame_geometry_helper_is_a_plain_rect(frame):
    assert isinstance(frame.geometry(), QRect)


# ----------------------------------------------------------------------------- 中空的框與移動符號
def _local(widget, point: QPoint) -> QPoint:
    return point - widget.geometry().topLeft()


def test_only_the_border_ring_and_the_two_grips_belong_to_the_window(qapp, frame):
    shape = frame.mask()
    width, height = frame.width(), frame.height()

    assert not shape.contains(QPoint(width // 2, height // 2))  # 中間：滑鼠穿過去，點得到下面真正的畫面
    assert not shape.contains(QPoint(BAND + 1, GRIP_INSET + GRIP + 6))  # 框內、緊貼邊框那一圈、不在移動符號上
    for point in (QPoint(1, height // 2), QPoint(width - 2, height // 2), QPoint(width // 2, 1), QPoint(width // 2, height - 2)):
        assert shape.contains(point)  # 邊框那一圈（縮放用）
    top_left, bottom_right = frame._grip_rects()
    assert shape.contains(top_left.center()) and shape.contains(bottom_right.center())  # 兩個移動符號


def test_the_two_grips_sit_inside_the_top_left_and_bottom_right_corners_and_never_overlap(qapp, frame):
    top_left, bottom_right = frame._grip_rects()

    assert top_left.left() >= BAND and top_left.top() >= BAND
    assert bottom_right.right() <= frame.width() - BAND and bottom_right.bottom() <= frame.height() - BAND
    assert top_left.center().x() < frame.width() / 2 < bottom_right.center().x()
    assert top_left.center().y() < frame.height() / 2 < bottom_right.center().y()
    frame.setGeometry(100, 100, *MIN_FRAME_SIZE)  # 最小的框
    assert not frame._grip_rects()[0].intersects(frame._grip_rects()[1])


def test_the_hit_test_tells_resizing_from_moving_from_nothing(qapp, frame):
    width, height = frame.width(), frame.height()
    top_left, bottom_right = frame._grip_rects()

    assert frame._hit_test(top_left.center()) == 0 and frame._hit_test(bottom_right.center()) == 0
    assert frame._hit_test(QPoint(2, 2)) == 1 and frame._hit_test(QPoint(width - 2, height - 2)) == 4
    assert frame._hit_test(QPoint(width - 2, height // 2)) == 6
    assert frame._hit_test(QPoint(width // 2, height // 2)) == -1  # 框的中間：不處理


def test_pressing_the_middle_of_the_frame_does_not_start_anything(qapp, frame):
    base = frame.geometry()

    _send(frame, QEvent.Type.MouseButtonPress, base.center())
    _send(frame, QEvent.Type.MouseMove, base.center() + QPoint(80, 40))

    assert frame._dragging is False and frame.geometry() == base


def test_the_bottom_right_grip_moves_the_frame_too(qapp, frame):
    base = frame.geometry()
    press = _bottom_right_grip(frame)
    _send(frame, QEvent.Type.MouseButtonPress, press)

    for step in range(1, 11):
        _send(frame, QEvent.Type.MouseMove, press + QPoint(-9 * step, 5 * step))
        assert frame.geometry() == base.translated(QPoint(-9 * step, 5 * step))
    _send(frame, QEvent.Type.MouseButtonRelease, press, buttons=Qt.MouseButton.NoButton)


def test_resizing_stops_at_the_minimum_size_so_the_grips_never_collide(qapp, frame):
    base = frame.geometry()
    right = QPoint(base.right() - 2, base.center().y())
    _send(frame, QEvent.Type.MouseButtonPress, right)
    _send(frame, QEvent.Type.MouseMove, right + QPoint(-5000, 0))
    _send(frame, QEvent.Type.MouseButtonRelease, right, buttons=Qt.MouseButton.NoButton)
    assert frame.width() == MIN_FRAME_SIZE[0]

    bottom = QPoint(frame.geometry().center().x(), frame.geometry().bottom() - 2)
    _send(frame, QEvent.Type.MouseButtonPress, bottom)
    _send(frame, QEvent.Type.MouseMove, bottom + QPoint(0, -5000))
    _send(frame, QEvent.Type.MouseButtonRelease, bottom, buttons=Qt.MouseButton.NoButton)
    assert frame.height() == MIN_FRAME_SIZE[1]


def test_the_painting_leaves_the_middle_empty_and_keeps_the_ring_almost_invisible(qapp, frame):
    image = frame.grab().toImage()
    width, height = image.width(), image.height()
    top_left, bottom_right = frame._grip_rects()

    assert image.pixelColor(width // 2, height // 2).alpha() == 0  # 全透明
    assert image.pixelColor(BAND - 3, height // 2).alpha() == 1  # 邊框那一圈沒畫線的地方：看不見，但收得到滑鼠
    assert image.pixelColor(1, height // 2).alpha() >= 150  # 框線和以前一樣
    assert image.pixelColor(4, 4).alpha() >= 200  # 四個角的把手和以前一樣
    for grip in (top_left, bottom_right):
        assert image.pixelColor(grip.left() + 2, grip.top() + 2).alpha() >= 200  # 移動符號看得見


# ----------------------------------------------------------------------------- 用系統座標對準位置
class _FakeNative:
    """假的系統視窗：記下被要求放到哪裡，之後問位置就回答放過去的位置。"""

    def __init__(self, rect):
        self.rect = tuple(rect)
        self.placed = []

    @staticmethod
    def available():
        return True

    def window_rect(self, hwnd):
        return self.rect

    def set_window_rect(self, hwnd, rect):
        self.placed.append(tuple(rect))
        self.rect = tuple(rect)
        return True


def _native_frame(qapp, monkeypatch, initial):
    fake = _FakeNative(initial)
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    monkeypatch.setattr(monitor_region, "native_window", fake)
    widget = MonitorRegionSelector(MONITORS, START, idle_ms=0)
    return widget, fake


def test_a_frame_that_shows_up_in_the_wrong_place_is_put_where_the_capture_is(qapp, monkeypatch):
    widget, fake = _native_frame(qapp, monkeypatch, initial=(130, 90, 530, 290))  # 偏了 (30, -10)
    widget.show()

    _wait(qapp, 0.3)

    assert fake.placed and fake.placed[-1] == START and fake.rect == START
    widget.close()


def test_a_frame_that_is_already_right_is_left_alone(qapp, monkeypatch):
    widget, fake = _native_frame(qapp, monkeypatch, initial=START)
    widget.show()

    _wait(qapp, 0.3)

    assert fake.placed == []  # 單螢幕、100% 縮放的電腦上本來就是對的，什麼都不必做
    widget.close()


def test_one_pixel_of_rounding_is_not_worth_correcting(qapp, monkeypatch):
    widget, fake = _native_frame(qapp, monkeypatch, initial=(100, 101, 500, 300))
    widget.show()

    _wait(qapp, 0.3)

    assert fake.placed == []
    widget.close()


def test_the_position_is_corrected_again_if_the_system_moves_the_window_right_after_it_appears(qapp, monkeypatch):
    widget, fake = _native_frame(qapp, monkeypatch, initial=START)
    widget.show()
    _wait(qapp, 0.05)
    fake.rect = (140, 60, 540, 260)  # 剛顯示後系統（例如螢幕縮放變了）把它搬走

    _wait(qapp, 1.8)

    assert fake.rect == START
    widget.close()


def test_once_the_user_grabs_the_frame_it_is_never_pulled_back(qapp, monkeypatch):
    widget, fake = _native_frame(qapp, monkeypatch, initial=START)
    widget.show()
    _wait(qapp, 0.05)
    _send(widget, QEvent.Type.MouseButtonPress, _top_left_grip(widget))
    fake.rect = (400, 400, 800, 600)

    _wait(qapp, 1.8)

    assert fake.placed == [] and fake.rect == (400, 400, 800, 600)
    widget.close()


def test_the_region_sent_to_the_app_is_where_the_window_really_is(qapp, monkeypatch):
    widget, fake = _native_frame(qapp, monkeypatch, initial=START)
    widget.show()
    _wait(qapp, 0.05)
    changed, committed = [], []
    widget.region_changed.connect(changed.append)
    widget.region_committed.connect(committed.append)
    press = _top_left_grip(widget)
    _send(widget, QEvent.Type.MouseButtonPress, press)
    _send(widget, QEvent.Type.MouseMove, press + QPoint(30, 10))
    fake.rect = (777, 555, 1177, 755)  # 系統回報的實際位置（和 Qt 自己算的不一樣）

    _send(widget, QEvent.Type.MouseButtonRelease, press + QPoint(30, 10), buttons=Qt.MouseButton.NoButton)

    assert committed == [(777, 555, 1177, 755)]  # 要擷取的範圍就是使用者看到的框的位置
    widget.close()
