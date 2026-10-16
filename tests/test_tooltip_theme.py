"""浮動說明（滑鼠移到設定上跳出來的文字）要跟著個性化主題換色，任何主題都看得清楚。"""

from __future__ import annotations

import os
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

from app.config import DEFAULTS
from ui.style import apply_style, build_stylesheet

PALETTES = ("warm_paper", "mist", "midnight")


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _luminance(color: QColor) -> float:
    def channel(value: int) -> float:
        value = value / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) + 0.0722 * channel(color.blue())


def _contrast(a: QColor, b: QColor) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _appearance(palette: str, **extra) -> dict:
    return {**DEFAULTS["appearance"], "palette": palette, **extra}


@pytest.mark.parametrize("palette", PALETTES)
def test_the_tooltip_style_uses_the_theme_colors_with_a_strong_contrast(qapp, palette):
    tokens = apply_style(qapp, _appearance(palette))

    rule = re.search(r"QToolTip\s*\{([^}]*)\}", qapp.styleSheet()).group(1)
    background = re.search(r"background:\s*(#[0-9A-Fa-f]{6})", rule).group(1)
    text = re.search(r"(?<![-\w])color:\s*(#[0-9A-Fa-f]{6})", rule).group(1)

    assert (background.upper(), text.upper()) == (tokens.ink.upper(), tokens.root.upper())
    assert _contrast(QColor(background), QColor(text)) >= 7  # 樣式表曾經寫死白字：深色主題的淺底就變成白底白字
    tooltip = qapp.palette()
    assert tooltip.color(QPalette.ColorRole.ToolTipBase).name().upper() == background.upper()
    assert tooltip.color(QPalette.ColorRole.ToolTipText).name().upper() == text.upper()


def test_there_is_exactly_one_tooltip_rule_so_nothing_can_override_the_theme(qapp):
    tokens = apply_style(qapp, _appearance("midnight"))

    assert len(re.findall(r"QToolTip\s*\{", build_stylesheet(tokens))) == 1


@pytest.mark.parametrize("palette", PALETTES)
def test_a_tooltip_that_is_really_shown_is_readable(qapp, palette):
    """真的把說明顯示出來、截下畫面，量底色與文字顏色的對比。"""
    tokens = apply_style(qapp, _appearance(palette))
    anchor = QWidget()
    anchor.resize(300, 100)
    anchor.show()
    tip = None
    for _ in range(5):
        QToolTip.showText(QPoint(100, 100), "Tooltip text 說明文字", anchor)
        for _ in range(10):
            qapp.processEvents()
            time.sleep(0.01)
        tips = [widget for widget in qapp.topLevelWidgets() if widget.metaObject().className() == "QTipLabel" and widget.isVisible()]
        if tips:
            tip = tips[0]
            break
    assert tip is not None, "說明沒有顯示出來"

    image = tip.grab().toImage()
    background = QColor(image.pixel(3, 3))
    farthest = max(
        (QColor(image.pixel(x, y)) for x in range(image.width()) for y in range(image.height())),
        key=lambda color: abs(color.red() - background.red()) + abs(color.green() - background.green()) + abs(color.blue() - background.blue()),
    )

    assert background.name().upper() == tokens.ink.upper()
    assert _contrast(background, farthest) >= 7, (palette, background.name(), farthest.name())
    QToolTip.hideText()
    anchor.deleteLater()


def test_switching_the_theme_while_running_changes_the_tooltip_colors_too(qapp):
    def colors():
        rule = re.search(r"QToolTip\s*\{([^}]*)\}", qapp.styleSheet()).group(1)
        return re.search(r"background:\s*(#\w{6})", rule).group(1), re.search(r"(?<![-\w])color:\s*(#\w{6})", rule).group(1)

    apply_style(qapp, _appearance("warm_paper"))
    light = colors()
    apply_style(qapp, _appearance("midnight"))
    dark = colors()

    assert light[0] != dark[0] and light[1] != dark[1]  # 換主題之後說明的底色與字色都跟著換
