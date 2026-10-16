"""設定畫面的數值輸入：滾輪只捲頁面，不會改數值。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QDoubleSpinBox, QLabel, QSlider, QSpinBox

from app.config import AppConfig
from ui.main_window import MainWindow
from ui.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider, NoWheelSpinBox
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _wheel(widget, delta: int = 120) -> QWheelEvent:
    position = QPointF(widget.width() / 2, widget.height() / 2)
    return QWheelEvent(
        position, widget.mapToGlobal(position), QPoint(0, 0), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )


@pytest.mark.parametrize(
    ("factory", "value"),
    [(NoWheelSpinBox, 5), (NoWheelDoubleSpinBox, 0.5), (lambda: NoWheelSlider(Qt.Orientation.Horizontal), 50)],
)
def test_the_wheel_does_not_change_the_value(qapp, factory, value):
    widget = factory()
    widget.setRange(0, 100)
    widget.setValue(value)
    widget.show()

    for delta in (120, -120, 240):
        event = _wheel(widget, delta)
        QApplication.sendEvent(widget, event)
        # 事件標成「未處理」：真實的滑鼠滾輪輸入遇到這種事件，Qt 會交給外層（捲動區域）處理，頁面照常捲動。
        # （程式手動送出的事件不會自動往外層傳，所以這裡只能驗證「未處理」這個前提。）
        assert not event.isAccepted()

    assert widget.value() == value
    widget.deleteLater()


def test_every_number_input_in_the_settings_dialog_ignores_the_wheel(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    spins = dialog.findChildren(QAbstractSpinBox)
    sliders = dialog.findChildren(QSlider)

    assert len(spins) >= 14 and len(sliders) >= 2  # 沒有找漏
    assert all(isinstance(widget, (NoWheelSpinBox, NoWheelDoubleSpinBox)) for widget in spins), "有數值框沒有關閉滾輪"
    assert all(isinstance(widget, NoWheelSlider) for widget in sliders), "有滑桿沒有關閉滾輪"
    assert not [w for w in spins if type(w) in (QSpinBox, QDoubleSpinBox)]
    dialog.deleteLater()


def test_keyboard_and_arrow_buttons_still_change_the_value(qapp):
    box = NoWheelSpinBox()
    box.setRange(0, 10)
    box.setValue(3)

    box.stepUp()
    assert box.value() == 4
    box.setValue(9)  # 直接輸入
    assert box.value() == 9
    box.deleteLater()


# ----------------------------------------------------------------------------- 下拉選單（輸入語言、目標語言…）
def test_the_wheel_does_not_change_the_selection_of_a_combo_box(qapp):
    box = NoWheelComboBox()
    box.addItems(["自動檢測", "日文", "英文"])
    box.setCurrentIndex(1)
    box.show()

    for delta in (120, -120, 360):
        event = _wheel(box, delta)
        QApplication.sendEvent(box, event)
        assert not event.isAccepted()  # 交還給外層（捲動區域）處理，頁面照常捲動

    assert box.currentIndex() == 1
    box.setCurrentIndex(2)  # 程式與點選仍然可以改
    assert box.currentText() == "英文"
    box.deleteLater()


def test_every_combo_box_in_the_settings_dialog_ignores_the_wheel(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    combos = dialog.findChildren(QComboBox)

    assert len(combos) >= 8  # 沒有找漏（輸入語言、目標語言、離線模型、OCR…）
    assert all(isinstance(widget, NoWheelComboBox) for widget in combos), "有下拉選單沒有關閉滾輪"
    assert dialog.combo_source.currentData() is not None
    before = (dialog.combo_source.currentIndex(), dialog.combo_target.currentIndex())
    for combo in (dialog.combo_source, dialog.combo_target):
        QApplication.sendEvent(combo, _wheel(combo, -120))
    assert (dialog.combo_source.currentIndex(), dialog.combo_target.currentIndex()) == before
    dialog.deleteLater()


class _Controller:
    def __init__(self, config) -> None:
        self.config = config

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_the_combo_boxes_on_the_main_window_ignore_the_wheel_too(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    combos = window.findChildren(QComboBox)

    assert len(combos) == 3 and all(isinstance(widget, NoWheelComboBox) for widget in combos)  # 輸入語言、目標語言、OCR
    window.deleteLater()


def test_the_source_language_is_called_input_language_everywhere_it_is_shown(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    window = MainWindow(_Controller(config))
    dialog = SettingsDialog(config)

    texts = {label.text() for label in window.findChildren(QLabel)} | {label.text() for label in dialog.findChildren(QLabel)}

    assert "輸入語言" in texts and "目標語言" in texts
    assert not [text for text in texts if "源語言" in text]
    window.deleteLater()
    dialog.deleteLater()
