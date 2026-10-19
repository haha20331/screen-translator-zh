"""框選一定要建出監視框：OCR 沒抓到字也建、拉得太小也建；框選的位置用系統座標，不被 Qt 的縮放換算帶偏。"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.capture_session import _CaptureSession
from ui import selection_overlay
from ui.monitor_region import MIN_FRAME_SIZE
from ui.selection_overlay import SelectionOverlay
from utils import dpi_utils
from utils.dpi_utils import MonitorInfo

ONE = [MonitorInfo(0, (0, 0, 1920, 1080), (0, 0), 1.0)]
TWO_MIXED = [
    MonitorInfo(0, (0, 0, 1920, 1080), (0, 0), 1.5),  # 主螢幕 150%：邏輯 1280 x 720
    MonitorInfo(1, (1920, 0, 3840, 1080), (1920, 0), 1.0),  # 副螢幕 100%
]


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------------- 最小尺寸
def test_a_frame_smaller_than_the_minimum_is_enlarged_around_its_centre():
    box = dpi_utils.ensure_min_size((500, 400, 503, 402), ONE, MIN_FRAME_SIZE)

    assert (box[2] - box[0], box[3] - box[1]) == MIN_FRAME_SIZE
    assert ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) == (501, 401)


def test_the_minimum_is_in_logical_pixels_so_it_grows_with_the_scaling():
    box = dpi_utils.ensure_min_size((500, 400, 501, 401), TWO_MIXED, MIN_FRAME_SIZE)

    assert (box[2] - box[0], box[3] - box[1]) == (96, 60)  # 150%：64 x 40 邏輯像素 = 96 x 60 物理像素


def test_a_big_enough_frame_is_left_exactly_as_it_is():
    assert dpi_utils.ensure_min_size((100, 100, 700, 300), ONE, MIN_FRAME_SIZE) == (100, 100, 700, 300)


def test_an_enlarged_frame_stays_on_the_screen_it_was_dragged_on():
    corner = dpi_utils.ensure_min_size((0, 0, 2, 2), ONE, MIN_FRAME_SIZE)
    far = dpi_utils.ensure_min_size((1918, 1078, 1920, 1080), ONE, MIN_FRAME_SIZE)
    second = dpi_utils.ensure_min_size((3800, 500, 3801, 501), TWO_MIXED, MIN_FRAME_SIZE)

    assert corner == (0, 0, 64, 40) and far == (1856, 1040, 1920, 1080)
    assert second[0] >= 1920 and second[2] <= 3840  # 在副螢幕上拉的，放大後還在副螢幕


# ----------------------------------------------------------------------------- 框選層
def _mouse(widget, kind, x, y):
    button = Qt.MouseButton.LeftButton
    QApplication.sendEvent(widget, QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, button, Qt.KeyboardModifier.NoModifier))


def _select(monkeypatch, start, end, native=None):
    monkeypatch.setattr(selection_overlay, "motion_duration", lambda *args, **kwargs: 0)
    cursor = iter(native or [])
    monkeypatch.setattr(selection_overlay.native_window, "cursor_position", lambda: next(cursor, None))
    overlay = SelectionOverlay(QRect(0, 0, 800, 600))
    done, cancelled = [], []
    overlay.selection_done.connect(done.append)
    overlay.cancelled.connect(cancelled.append)
    _mouse(overlay, QEvent.Type.MouseButtonPress, *start)
    _mouse(overlay, QEvent.Type.MouseButtonRelease, *end)
    return overlay, done, cancelled


def test_a_drag_is_reported_with_the_cursor_positions_of_the_system(qapp, monkeypatch):
    overlay, done, cancelled = _select(monkeypatch, (100, 100), (400, 250), native=[(310, 210), (910, 510)])

    assert done == [(100, 100, 400, 250)] and cancelled == []
    assert overlay.native_bbox == (310, 210, 910, 510)  # 系統座標（物理像素）：框選的範圍以它為準
    overlay.deleteLater()


def test_the_system_positions_are_sorted_whatever_direction_the_user_dragged(qapp, monkeypatch):
    overlay, _, _ = _select(monkeypatch, (400, 250), (100, 100), native=[(910, 510), (310, 210)])

    assert overlay.native_bbox == (310, 210, 910, 510)
    overlay.deleteLater()


def test_without_system_positions_the_logical_ones_are_all_there_is(qapp, monkeypatch):
    overlay, done, _ = _select(monkeypatch, (100, 100), (400, 250))

    assert overlay.native_bbox is None and done == [(100, 100, 400, 250)]
    overlay.deleteLater()


@pytest.mark.parametrize("end", [(102, 101), (100, 100), (160, 101)])
def test_a_tiny_drag_or_a_plain_click_is_not_thrown_away(qapp, monkeypatch, end):
    overlay, done, cancelled = _select(monkeypatch, (100, 100), end, native=[(100, 100), end])

    assert cancelled == [] and len(done) == 1  # 以前這裡直接取消，什麼都沒有；現在交給程式放大到最小尺寸
    left, top, right, bottom = done[0]
    assert right - left >= 1 and bottom - top >= 1
    overlay.deleteLater()


def test_escape_still_cancels(qapp, monkeypatch):
    monkeypatch.setattr(selection_overlay, "motion_duration", lambda *args, **kwargs: 0)
    overlay = SelectionOverlay(QRect(0, 0, 800, 600))
    cancelled = []
    overlay.cancelled.connect(lambda: cancelled.append(True))

    overlay.cancel()

    assert cancelled == [True]
    overlay.deleteLater()


# ----------------------------------------------------------------------------- 控制器：框選結果 → 擷取範圍
@pytest.fixture
def controller(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(qapp)
    instance.monitor_map = list(ONE)
    instance.calls = SimpleNamespace(scheduled=[], aborted=[])
    monkeypatch.setattr(instance, "_schedule_capture", lambda mode, **kw: instance.calls.scheduled.append((mode, kw)))
    monkeypatch.setattr(instance, "_abort_capture", lambda status, **kw: instance.calls.aborted.append((status, kw)))
    yield instance
    instance.floating_status.deleteLater()


def _selection(native):
    return SimpleNamespace(native_bbox=native, dismiss=lambda: None, deleteLater=lambda: None)


def test_the_capture_area_is_where_the_cursor_was_even_if_qt_converted_it_differently(controller):
    controller.selection = _selection((600, 300, 1200, 500))

    controller._on_selection_done((400, 100, 1000, 300))  # Qt 邏輯座標換出來是別的位置（像是縮放換算偏掉）

    assert controller.calls.scheduled[0][1]["bbox"] == (600, 300, 1200, 500) and controller.calls.aborted == []


def test_without_system_positions_the_qt_conversion_is_used_as_before(controller):
    controller.selection = _selection(None)

    controller._on_selection_done((400, 100, 1000, 300))

    assert controller.calls.scheduled[0][1]["bbox"] == (400, 100, 1000, 300)


def test_a_tiny_selection_becomes_a_frame_of_the_minimum_size(controller):
    controller.selection = _selection((700, 500, 702, 501))

    controller._on_selection_done((700, 500, 702, 501))

    left, top, right, bottom = controller.calls.scheduled[0][1]["bbox"]
    assert (right - left, bottom - top) == MIN_FRAME_SIZE and controller.calls.aborted == []


def test_a_click_that_the_qt_conversion_cannot_use_still_makes_a_frame_through_the_system_positions(controller):
    controller.selection = _selection((900, 400, 901, 401))

    controller._on_selection_done((5000, 5000, 5001, 5001))  # Qt 的座標在螢幕外

    assert controller.calls.scheduled and controller.calls.aborted == []


def test_a_selection_off_every_screen_is_still_refused_without_system_positions(controller):
    controller.selection = _selection(None)

    controller._on_selection_done((5000, 5000, 5100, 5100))

    assert controller.calls.scheduled == [] and controller.calls.aborted


def test_a_selection_across_screens_with_different_scaling_is_still_refused(controller):
    controller.monitor_map = [MonitorInfo(0, (0, 0, 1920, 1080), (0, 0), 1.0), MonitorInfo(1, (1920, 0, 3840, 540), (1920, 0), 1.0)]
    controller.selection = _selection((1800, 100, 2100, 900))  # 右邊的螢幕比較矮：跨過去的範圍不是矩形

    controller._on_selection_done((1800, 100, 2100, 900))

    assert controller.calls.scheduled == [] and controller.calls.aborted[0][1].get("failed")


# ----------------------------------------------------------------------------- 沒抓到字也建監視框
def _finish(controller, *, error, mode="region", auto_start=True, enabled=True, starts=True):
    controller.config.set("monitor.auto_start", auto_start)
    controller.config.set("monitor.enabled", enabled)
    shown = []
    controller.window = SimpleNamespace(finish_capture=lambda m: shown.append(m), set_busy=lambda busy: None, isVisible=lambda: True)
    controller._capture_session = _CaptureSession(mode=mode, window_was_visible=True, window_was_minimized=False, overlay_was_visible=False)
    controller._last_capture = SimpleNamespace(mode=mode, bbox=(100, 100, 400, 200))
    controller.worker = SimpleNamespace(deleteLater=lambda: None)
    controller._pipeline_succeeded = False
    controller._pipeline_error = error
    started = []
    controller.start_monitor = lambda: (started.append(True), setattr(controller, "_monitor_active", starts))
    statuses = []
    controller.set_status = statuses.append
    controller._on_worker_finished()
    return started, statuses, shown


def test_a_region_without_any_text_still_gets_its_monitor_frame(controller):
    started, statuses, shown = _finish(controller, error="沒有識別到文字（檢查 OCR 引擎設定或置信度閾值）")

    assert started == [True] and shown == ["region"]
    assert controller.calls.aborted == [] and controller._capture_session is None
    assert "已建立監視框" in statuses[-1]  # 告訴使用者：框已經建好了，可以先調整位置


def test_text_that_needs_no_translation_also_keeps_the_frame(controller):
    started, _, _ = _finish(controller, error="沒有識別到可翻譯的文字")

    assert started == [True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error": "處理失敗：模型載入失敗"},  # 真的出錯：照原本的方式告訴使用者
        {"error": ""},  # 被取消
        {"error": "沒有識別到文字", "mode": "fullscreen"},  # 全螢幕、視窗擷取沒有「框」
        {"error": "沒有識別到文字", "auto_start": False},  # 使用者關掉了「翻譯完成後自動進入即時監控」
        {"error": "沒有識別到文字", "enabled": False},  # 整個即時監控都關了
    ],
)
def test_everything_else_keeps_the_old_failure_handling(controller, kwargs):
    started, _, _ = _finish(controller, **kwargs)

    assert started == [] and len(controller.calls.aborted) == 1


def test_if_the_monitor_cannot_start_the_failure_is_reported_instead_of_silence(controller):
    started, statuses, _ = _finish(controller, error="沒有識別到文字", starts=False)

    assert started == [True]  # 試過了，但監控沒有開成
    assert statuses and "沒有識別到" in statuses[-1] and controller.calls.aborted == []


# ----------------------------------------------------------------------------- 框選層鋪滿所有螢幕
class _FakeNative:
    def __init__(self, rect):
        self.rect = tuple(rect)
        self.placed = []

    @staticmethod
    def available():
        return True

    @staticmethod
    def cursor_position():
        return None

    def window_rect(self, hwnd):
        return self.rect

    def set_window_rect(self, hwnd, rect):
        self.placed.append(tuple(rect))
        self.rect = tuple(rect)
        return True


def _wait(qapp, seconds):
    import time

    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


DESKTOP = (-2560, 0, 1920, 1600)  # 筆電 200% 螢幕在左邊、一般螢幕在右邊的整個桌面（物理像素）


def test_the_selection_layer_is_stretched_over_every_screen_in_system_coordinates(qapp, monkeypatch):
    fake = _FakeNative((-2560, 0, 1920, 1200))  # Qt 用邏輯座標放的結果：比桌面矮，筆電螢幕的下面一截蓋不到
    monkeypatch.setattr(selection_overlay, "native_window", fake)
    monkeypatch.setattr(selection_overlay, "motion_duration", lambda *args, **kwargs: 0)
    overlay = SelectionOverlay(QRect(-2560, 0, 4480, 1200), native_rect=DESKTOP)

    overlay.show()
    _wait(qapp, 0.2)

    assert fake.placed == [DESKTOP] and fake.rect == DESKTOP
    overlay.dismiss()
    overlay.deleteLater()


def test_a_layer_that_already_covers_the_desktop_is_left_alone(qapp, monkeypatch):
    fake = _FakeNative(DESKTOP)
    monkeypatch.setattr(selection_overlay, "native_window", fake)
    monkeypatch.setattr(selection_overlay, "motion_duration", lambda *args, **kwargs: 0)
    overlay = SelectionOverlay(QRect(0, 0, 800, 600), native_rect=DESKTOP)

    overlay.show()
    _wait(qapp, 0.2)

    assert fake.placed == []
    overlay.dismiss()
    overlay.deleteLater()


def test_without_a_system_rect_nothing_is_done(qapp, monkeypatch):
    fake = _FakeNative((0, 0, 10, 10))
    monkeypatch.setattr(selection_overlay, "native_window", fake)
    monkeypatch.setattr(selection_overlay, "motion_duration", lambda *args, **kwargs: 0)
    overlay = SelectionOverlay(QRect(0, 0, 800, 600))

    overlay.show()
    _wait(qapp, 0.2)

    assert fake.placed == []
    overlay.dismiss()
    overlay.deleteLater()


def test_the_size_badge_shows_the_real_pixels_that_will_be_captured(qapp, monkeypatch):
    overlay = SelectionOverlay(QRect(0, 0, 800, 600))
    assert overlay._size_text(QRect(10, 10, 300, 100)) == "300 × 100"  # 沒有系統座標：用 Qt 的

    overlay._press_native, overlay._current_native = (1000, 1000), (1600, 1200)

    assert overlay._size_text(QRect(10, 10, 300, 100)) == "600 × 200"  # 200% 縮放的視窗裡，邏輯像素只有實際的一半
    overlay.deleteLater()
