"""譯文面板：框選區域時，把譯文放在框的下方，原文保持原樣可見。"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QWidget

from ui.translation_overlay import font_families

_MARGIN = 6  # 與螢幕邊緣、與框之間的最小間距（邏輯像素）
_PAD_X, _PAD_Y = 14, 8
_MIN_WIDTH = 420  # 框很窄時，面板最寬可以放大到這個寬度，避免譯文被擠成一直排
_TEXT_FLAGS = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap


def place_panel(anchor: QRect, size: QSize, bounds: QRect, gap: int = _MARGIN) -> QRect:
    """面板放在框 ``anchor`` 的下方、水平置中並夾在 ``bounds``（螢幕）內。

    下方放不下就放框的上方；上下都放不下（例如框占滿整個螢幕）才退到框內的底部。
    """
    width, height = size.width(), size.height()
    x = anchor.center().x() - width // 2
    x = max(bounds.left(), min(x, bounds.right() + 1 - width))

    below = anchor.bottom() + 1 + gap
    above = anchor.top() - gap - height
    if below + height <= bounds.bottom() + 1:
        y = below
    elif above >= bounds.top():
        y = above
    else:
        y = bounds.bottom() + 1 - height - gap
    y = max(bounds.top(), y)
    return QRect(QPoint(x, y), size)


class TranslationPanelWindow(QWidget):
    """透明置頂、滑鼠穿透的小視窗，只畫一塊半透明底的譯文。"""

    def __init__(self, config, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        super().__init__(parent, flags)
        self._config = config
        self._lines: list[str] = []
        self._anchor = QRect()
        self._bounds = QRect()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    # ------------------------------------------------------------- public
    def has_content(self) -> bool:
        return bool(self._lines)

    def lines(self) -> list[str]:
        return list(self._lines)

    def set_content(self, lines: list[str], anchor: QRect, bounds: QRect) -> None:
        """``anchor`` 是被翻譯的框、``bounds`` 是它所在的螢幕，都是 Qt 全域邏輯座標。"""
        self._lines = list(lines)
        self.set_anchor(anchor, bounds)

    def set_anchor(self, anchor: QRect, bounds: QRect) -> None:
        self._anchor, self._bounds = QRect(anchor), QRect(bounds)
        self._relayout()

    def clear(self) -> None:
        self._lines = []
        self.hide()

    def apply_style(self) -> None:
        self._relayout()

    # ------------------------------------------------------------- layout
    def _font(self) -> QFont:
        font = QFont()
        font.setFamilies(font_families(self._config))
        font.setPointSize(max(6, int(self._config.get("overlay.font_size", 18))))
        return font

    def _panel_size(self, metrics: QFontMetrics) -> QSize:
        """面板貼著文字大小：框寬（至少 _MIN_WIDTH）是上限，文字多就換行。"""
        text = "\n".join(self._lines)
        limit = min(max(self._anchor.width(), _MIN_WIDTH), max(1, self._bounds.width() - 2 * _MARGIN))
        text_limit = max(1, limit - 2 * _PAD_X)
        natural = max((metrics.horizontalAdvance(line) for line in self._lines), default=0)
        text_width = max(1, min(natural + 2, text_limit))  # +2：避免四捨五入把最後一個字擠到下一行
        text_height = metrics.boundingRect(0, 0, text_width, 100000, int(_TEXT_FLAGS), text).height()
        height = min(text_height + 2 * _PAD_Y, max(1, self._bounds.height() - 2 * _MARGIN))
        return QSize(text_width + 2 * _PAD_X, height)

    def _relayout(self) -> None:
        if not self._lines or self._bounds.isEmpty():
            self.update()
            return
        size = self._panel_size(QFontMetrics(self._font()))
        self.setGeometry(place_panel(self._anchor, size, self._bounds))
        self.update()

    # ------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        if not self._lines:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        background = QColor(str(self._config.get("overlay.background_color", "#000000")))
        background.setAlpha(max(0, min(255, int(self._config.get("overlay.panel_alpha", 215)))))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        radius = int(self._config.get("overlay.border_radius", 4)) + 4
        painter.drawRoundedRect(self.rect(), radius, radius)

        painter.setFont(self._font())
        painter.setPen(QColor(str(self._config.get("overlay.text_color", "#FFFFFF"))))
        painter.drawText(
            self.rect().adjusted(_PAD_X, _PAD_Y, -_PAD_X, -_PAD_Y), int(_TEXT_FLAGS), "\n".join(self._lines)
        )
