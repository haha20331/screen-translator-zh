"""設定畫面的數值輸入：滾輪只捲頁面，不會改數值。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QDoubleSpinBox, QSlider, QSpinBox

from app.config import AppConfig
from ui.no_wheel import NoWheelDoubleSpinBox, NoWheelSlider, NoWheelSpinBox
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

    assert len(spins) >= 15 and len(sliders) >= 2  # 沒有找漏
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
