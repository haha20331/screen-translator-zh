"""下拉清單的選項在 Windows 深色模式下曾變成白底白字（看不見）。

樣式表把清單底色設成淺色，但沒指定文字色，文字改吃系統調色盤（深色模式 = 白色）。
修正後：樣式表明確指定清單文字色，且整組調色盤由主題決定、不隨系統深淺色改變。
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QComboBox

from ui.appearance import current_appearance, resolve_tokens
from ui.style import apply_style, build_palette, build_stylesheet


def _luminance(color: QColor) -> float:
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(color.redF()) + 0.7152 * channel(color.greenF()) + 0.0722 * channel(color.blueF())


def _contrast(a: QColor, b: QColor) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    previous = current_appearance()
    original_palette = instance.palette()
    yield instance
    instance.setPalette(original_palette)
    apply_style(instance, previous)


def _dark_system_palette() -> QPalette:
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#202020"),
        (QPalette.ColorRole.WindowText, "#FFFFFF"),
        (QPalette.ColorRole.Base, "#2B2B2B"),
        (QPalette.ColorRole.Text, "#FFFFFF"),
        (QPalette.ColorRole.Button, "#333333"),
        (QPalette.ColorRole.ButtonText, "#FFFFFF"),
    ):
        palette.setColor(role, QColor(color))
    return palette


def test_popup_rules_set_their_own_text_color():
    sheet = build_stylesheet(resolve_tokens({}))
    popup_rule = sheet.split("QComboBox QAbstractItemView {", 1)[1].split("}", 1)[0]
    assert "color:" in popup_rule.replace("selection-color:", "").replace("background-color:", "")
    # 清單一律用一般清單檢視，不讓平台樣式改成選單式彈出（那條路徑不吃這些規則）
    assert "combobox-popup: 0" in sheet


@pytest.mark.parametrize("palette_name", ["warm_paper", "mist", "midnight"])
def test_popup_text_is_readable_when_the_system_is_in_dark_mode(app, palette_name):
    app.setPalette(_dark_system_palette())
    tokens = apply_style(app, {"palette": palette_name})

    combo = QComboBox()
    combo.addItem("英文", "en")
    combo.addItem("日文", "ja")
    combo.show()
    combo.showPopup()
    app.processEvents()
    view = combo.view()
    try:
        text = view.palette().color(QPalette.ColorRole.Text)
        base = view.palette().color(QPalette.ColorRole.Base)
        assert _contrast(text, base) >= 4.5, f"{palette_name}: 清單文字 {text.name()} 在 {base.name()} 上看不清楚"
        assert base.name().upper() == QColor(tokens.surface).name().upper()
    finally:
        combo.hidePopup()
        combo.close()


def test_application_palette_follows_the_theme_not_the_system(app):
    app.setPalette(_dark_system_palette())
    tokens = apply_style(app, {"palette": "warm_paper"})
    palette = app.palette()
    assert palette.color(QPalette.ColorRole.Text).name().upper() == tokens.ink.upper()
    assert _contrast(palette.color(QPalette.ColorRole.Text), palette.color(QPalette.ColorRole.Base)) >= 4.5
    assert _contrast(palette.color(QPalette.ColorRole.WindowText), palette.color(QPalette.ColorRole.Window)) >= 4.5


def test_build_palette_dims_disabled_text():
    tokens = resolve_tokens({})
    palette = build_palette(tokens)
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
    assert disabled.name().upper() == tokens.disabled.upper()
