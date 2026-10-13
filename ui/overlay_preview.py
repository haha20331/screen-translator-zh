"""設定畫面的譯文範例：用目前（還沒儲存）的控制項數值，畫出實際會看到的譯文。

直接重用真正顯示譯文的視窗（框下方的面板、蓋在原文上的覆蓋層）來畫，所以範例與實際顯示一致。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ui.translation_overlay import Block, TranslationOverlayWindow, font_families
from ui.translation_panel import TranslationPanelWindow
from utils.image_utils import ensure_text_contrast

SAMPLE_SOURCE = "こんにちは、お元気ですか？"
SAMPLE_TRANSLATION = "你好，你還好嗎？"
# 模擬畫面上字幕所在位置的底色：自動背景會取這種顏色，半透明時也看得出底下的畫面
_SCREEN_TOP, _SCREEN_BOTTOM = QColor("#27456F"), QColor("#7FA3CF")
_SAMPLE_BACKGROUND = "#3A5C8E"


class OverlayOptions:
    """預覽用的設定覆寫層：沒覆寫的項目沿用實際設定。與 ``AppConfig.get`` 相同的呼叫方式。"""

    def __init__(self, base, overrides: dict) -> None:
        self._base = base
        self._overrides = dict(overrides)

    def get(self, dotted: str, default=None):
        if dotted in self._overrides:
            return self._overrides[dotted]
        return self._base.get(dotted, default) if self._base is not None else default


class OverlayPreview(QWidget):
    """一格模擬畫面：原文字幕 + 目前設定下的譯文。"""

    def __init__(self, base_config=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base = base_config
        self._overrides: dict = {}
        self.setFixedSize(QSize(330, 200))
        self.setAccessibleName("譯文外觀範例")

    def set_options(self, overrides: dict) -> None:
        """``overrides`` 是 ``overlay.*`` 的設定值（包含 ``overlay.placement``）。"""
        self._overrides = dict(overrides)
        self.update()

    def options(self) -> OverlayOptions:
        return OverlayOptions(self._base, self._overrides)

    # ------------------------------------------------------------- painting
    def _source_font(self) -> QFont:
        font = QFont()
        font.setFamilies(font_families(self.options()))
        font.setPixelSize(20)
        font.setBold(True)
        return font

    def _render_widget(self, widget: QWidget, size: QSize) -> QPixmap:
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        widget.render(pixmap)
        return pixmap

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        gradient = QLinearGradient(QPointF(0, 0), QPointF(0, rect.height()))
        gradient.setColorAt(0.0, _SCREEN_TOP)
        gradient.setColorAt(1.0, _SCREEN_BOTTOM)
        painter.fillRect(self.rect(), gradient)

        options = self.options()
        placement = str(options.get("overlay.placement", "below"))
        # 原文字幕（白字黑邊），框選區域就是它周圍那一圈
        text_rect = QRect(14, 28 if placement == "below" else 70, self.width() - 28, 36)
        painter.setFont(self._source_font())
        painter.setPen(QColor("#000000"))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            painter.drawText(text_rect.translated(dx, dy), Qt.AlignmentFlag.AlignCenter, SAMPLE_SOURCE)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, SAMPLE_SOURCE)

        if placement == "below":
            frame = text_rect.adjusted(-6, -4, 6, 4)
            painter.setPen(QPen(QColor("#2878E8"), 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(frame)
            panel = TranslationPanelWindow(options)
            panel.set_content([SAMPLE_TRANSLATION], frame, QRect(self.rect()))
            painter.drawPixmap(panel.geometry().topLeft(), self._render_widget(panel, panel.size()))
        else:
            overlay = TranslationOverlayWindow(options)
            overlay.resize(self.size())
            color = (
                QColor(ensure_text_contrast("#FFFFFF", _SAMPLE_BACKGROUND))
                if options.get("overlay.use_auto_text_color", True)
                else QColor(str(options.get("overlay.text_color", "#FFFFFF")))
            )
            overlay.set_blocks_static(
                [Block(QRectF(text_rect), SAMPLE_TRANSLATION, color, _SAMPLE_BACKGROUND, float(text_rect.height()), 1)]
            )
            painter.drawPixmap(0, 0, self._render_widget(overlay, self.size()))
