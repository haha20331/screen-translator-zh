"""監視框的移動方式（設定 → 即時監控）：預設拖左上、右下的符號；也可以改成整個框的中央都能拖（但中間不再穿透）。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import AppConfig
from ui import monitor_region
from ui.monitor_region import BAND, DRAG_CENTER, DRAG_GRIPS, GRIP_INSET, MonitorRegionSelector
from ui.settings_dialog import SettingsDialog
from utils.dpi_utils import MonitorInfo

MONITORS = [MonitorInfo(index=0, physical=(0, 0, 3840, 2160), logical_origin=(0, 0), dpr=1.0)]
START = (100, 100, 500, 300)  # 400 x 200


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _Cursor:
    where = QPoint(-5000, -5000)

    @classmethod
    def pos(cls):
        return cls.where


def _make(monkeypatch, mode):
    monkeypatch.setattr(monitor_region, "QCursor", _Cursor)
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    _Cursor.where = QPoint(-5000, -5000)
    widget = MonitorRegionSelector(MONITORS, START, idle_ms=0, drag_mode=mode)
    widget.show()
    return widget


@pytest.fixture
def center_frame(qapp, monkeypatch):
    widget = _make(monkeypatch, DRAG_CENTER)
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.fixture
def grip_frame(qapp, monkeypatch):
    widget = _make(monkeypatch, DRAG_GRIPS)
    yield widget
    widget.close()
    widget.deleteLater()


def _send(widget, kind, global_point: QPoint, buttons=Qt.MouseButton.LeftButton):
    local = QPointF(global_point - widget.geometry().topLeft())
    button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    QApplication.sendEvent(widget, QMouseEvent(kind, local, QPointF(global_point), button, buttons, Qt.KeyboardModifier.NoModifier))


def _middle(widget) -> QPoint:
    return widget.geometry().center()


# ----------------------------------------------------------------------------- 預設與無效值
def test_the_default_is_the_corner_grips_and_an_unknown_mode_falls_back_to_it(qapp, monkeypatch):
    assert MonitorRegionSelector(MONITORS, START).drag_mode() == DRAG_GRIPS
    assert MonitorRegionSelector(MONITORS, START, drag_mode="nonsense").drag_mode() == DRAG_GRIPS


def test_the_setting_defaults_to_the_grips_and_a_bad_value_in_the_file_is_corrected(tmp_path):
    path = tmp_path / "config.json"
    assert AppConfig(path).get("monitor.drag_mode") == "grips"

    path.write_text('{"monitor": {"drag_mode": "sideways"}}', encoding="utf-8")
    assert AppConfig(path).get("monitor.drag_mode") == "grips"

    path.write_text('{"monitor": {"drag_mode": "center"}}', encoding="utf-8")
    assert AppConfig(path).get("monitor.drag_mode") == "center"


# ----------------------------------------------------------------------------- 整個框的中央都能拖
def test_in_center_mode_the_whole_frame_receives_the_mouse(center_frame):
    assert center_frame.mask().isEmpty()  # 沒有挖洞：整個視窗都收滑鼠，中間不再穿透
    assert center_frame._hit_test(QPoint(center_frame.width() // 2, center_frame.height() // 2)) == 0
    assert center_frame._hit_test(QPoint(BAND + 3, BAND + 3)) == 0  # 以前這裡（符號旁邊的空地）收不到
    assert center_frame._hit_test(QPoint(2, 2)) == 1 and center_frame._hit_test(QPoint(1, center_frame.height() // 2)) == 5  # 邊框、四個角照樣縮放


def test_in_center_mode_the_middle_of_the_frame_is_drawn_so_that_windows_hit_tests_it(center_frame):
    image = center_frame.grab().toImage()

    assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() == 1  # 肉眼看不見，但不是 0：Windows 對 0 的像素讓滑鼠穿過去


def test_in_center_mode_no_move_symbols_are_drawn(center_frame):
    image = center_frame.grab().toImage()
    top_left, bottom_right = center_frame._grip_rects()

    for grip in (top_left, bottom_right):
        assert image.pixelColor(grip.center()).alpha() <= 1  # 沒有符號：框裡面不畫任何東西，免得被辨識進去


def test_dragging_from_the_middle_moves_the_frame_by_exactly_the_cursor_distance(qapp, center_frame):
    base = center_frame.geometry()
    press = _middle(center_frame)
    _send(center_frame, QEvent.Type.MouseButtonPress, press)

    for step in range(1, 11):
        _send(center_frame, QEvent.Type.MouseMove, press + QPoint(7 * step, 4 * step))
        assert center_frame.geometry().topLeft() == base.topLeft() + QPoint(7 * step, 4 * step)
        assert center_frame.size() == base.size()
    _send(center_frame, QEvent.Type.MouseButtonRelease, press + QPoint(70, 40), buttons=Qt.MouseButton.NoButton)


def test_the_cursor_over_the_middle_is_the_move_cursor_in_center_mode(center_frame):
    center_frame._apply_cursor(QPoint(center_frame.width() // 2, center_frame.height() // 2))

    assert center_frame.cursor().shape() == Qt.CursorShape.SizeAllCursor


def test_the_tooltip_explains_the_current_way_to_move_the_frame(grip_frame, center_frame):
    assert "符號" in grip_frame.toolTip()
    assert "中央" in center_frame.toolTip() and "符號" not in center_frame.toolTip()


# ----------------------------------------------------------------------------- 預設的拖曳符號沒有變
def test_in_grip_mode_the_middle_still_lets_the_mouse_through(grip_frame):
    assert not grip_frame.mask().isEmpty()
    assert not grip_frame.mask().contains(QPoint(grip_frame.width() // 2, grip_frame.height() // 2))
    assert grip_frame._hit_test(QPoint(grip_frame.width() // 2, grip_frame.height() // 2)) == -1
    assert grip_frame._hit_test(QPoint(GRIP_INSET + 3, GRIP_INSET + 3)) == 0


def test_pressing_in_the_middle_does_nothing_in_grip_mode(grip_frame):
    base = grip_frame.geometry()
    press = _middle(grip_frame)

    _send(grip_frame, QEvent.Type.MouseButtonPress, press)
    _send(grip_frame, QEvent.Type.MouseMove, press + QPoint(30, 30))

    assert grip_frame.geometry() == base


# ----------------------------------------------------------------------------- 框在畫面上時換方式
def test_switching_the_mode_while_the_frame_is_on_screen_takes_effect_at_once(grip_frame):
    assert not grip_frame.mask().isEmpty() and grip_frame._hit_test(QPoint(grip_frame.width() // 2, 40)) == -1

    grip_frame.set_drag_mode(DRAG_CENTER)
    assert grip_frame.drag_mode() == DRAG_CENTER and grip_frame.mask().isEmpty()
    assert grip_frame._hit_test(QPoint(grip_frame.width() // 2, 40)) == 0

    grip_frame.set_drag_mode(DRAG_GRIPS)
    assert not grip_frame.mask().isEmpty() and grip_frame._hit_test(QPoint(grip_frame.width() // 2, 40)) == -1
    assert "符號" in grip_frame.toolTip()

    grip_frame.set_drag_mode("nonsense")  # 無效值當成預設
    assert grip_frame.drag_mode() == DRAG_GRIPS


def test_a_resize_keeps_the_right_shape_in_center_mode(center_frame):
    center_frame.resize(300, 150)

    assert center_frame.mask().isEmpty()


# ----------------------------------------------------------------------------- 設定頁與程式
def test_the_settings_page_offers_both_methods_and_saves_the_choice(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.combo_drag_mode.currentData() == "grips"  # 預設是拖曳符號
    assert [dialog.combo_drag_mode.itemData(i) for i in range(dialog.combo_drag_mode.count())] == ["grips", "center"]

    dialog.combo_drag_mode.setCurrentIndex(dialog.combo_drag_mode.findData("center"))
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("monitor.drag_mode") == "center"
    reopened = SettingsDialog(saved)
    assert reopened.combo_drag_mode.currentData() == "center"
    dialog.deleteLater()
    reopened.deleteLater()


@pytest.fixture
def controller(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(qapp)
    instance.monitor_map = list(MONITORS)
    yield instance
    instance._dispose_monitor_region()
    instance.floating_status.deleteLater()


def test_a_new_frame_is_made_with_the_chosen_method(controller, monkeypatch):
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    controller._monitor_request = {"bbox": START}

    controller._create_monitor_region_selector()
    assert controller._monitor_region_selector.drag_mode() == DRAG_GRIPS

    controller.config.set("monitor.drag_mode", "center")
    controller._create_monitor_region_selector()
    assert controller._monitor_region_selector.drag_mode() == DRAG_CENTER


def test_saving_the_settings_changes_the_frame_that_is_already_showing(controller, monkeypatch):
    monkeypatch.setattr(monitor_region, "motion_duration", lambda ms, **kwargs: 0)
    controller._monitor_request = {"bbox": START}
    controller._create_monitor_region_selector()
    selector = controller._monitor_region_selector

    controller.config.set("monitor.drag_mode", "center")
    controller._apply_monitor_frame_settings()
    assert selector.drag_mode() == DRAG_CENTER

    controller.config.set("monitor.drag_mode", "grips")
    controller._apply_monitor_frame_settings()
    assert selector.drag_mode() == DRAG_GRIPS


def test_applying_the_settings_without_a_frame_is_harmless(controller):
    controller._monitor_region_selector = None

    controller._apply_monitor_frame_settings()
