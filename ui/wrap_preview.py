"""設定畫面「多行合併」的範例：用目前（還沒儲存）的行距與對齊誤差，畫出上下兩行字幕最遠能離多遠、
還會被併成同一句一起翻譯，以及再遠一點就會分開的第三行。

距離的算法與真正合併（``utils.layout_utils.merge_wrapped_labels``）相同，直接呼叫它的
``wrap_gap_limit``／``wrap_align_limit``。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.preview_style import FRAME, MEASURE, MERGED, SCREEN, SEPARATE, TEXT, draw_label
from ui.translation_overlay import font_families
from utils.layout_utils import (
    MIN_WRAP_ALIGN,
    MIN_WRAP_GAP,
    WRAP_ALIGN_RATIO,
    WRAP_GAP_RATIO,
    wrap_align_limit,
    wrap_gap_limit,
)

SAMPLE_LINE_HEIGHT = 24  # 範例字的高度（像素）；所有距離都跟行高成正比
SAMPLE_TEXTS = ("こんにちは、", "お元気ですか？", "また明日ね")  # 一句字幕被拆成上下兩行，第三行是下一句
SEPARATE_EXTRA = 12  # 第三行放在「剛好超過」行距上限再多 12 像素的地方，示範會被分開
MAX_RATIO = 2.0  # 設定畫面兩個比例的上限（版面按這個預留空間，調整時第一行不會跳動）
CENTER = QColor("#8FD3FF")  # 置中對齊的中心點範圍

_TEXT_SCALE = 0.7  # 字的像素大小 = 行高 × 這個值
_TOP_AREA = 38  # 第一行上面：左緣誤差的括號與標籤
_BOTTOM_AREA = 22  # 第三行下面：置中的標籤
_RIGHT_AREA = 66  # 行距標示與標籤
_LEFT_MARGIN = MAX_RATIO * SAMPLE_LINE_HEIGHT + 10  # 左緣誤差的範圍可能往左延伸


def wrap_limits(
    gap_ratio: float, align_ratio: float, line_height: float = SAMPLE_LINE_HEIGHT
) -> tuple[int, int]:
    """(上下行距最大像素, 左緣最大誤差像素)。和真正合併用同一組函式，所以會套用最小像素數。"""
    return wrap_gap_limit(line_height, gap_ratio), wrap_align_limit(line_height, align_ratio)


def wrap_summary(gap_ratio: float, align_ratio: float, centered: bool) -> str:
    """範例底下的文字說明：把幾個比例換算成範例字的像素距離。"""
    gap, align = wrap_limits(gap_ratio, align_ratio)
    gap_note = f"{gap_ratio:g} 倍行高" + (f"，最少 {MIN_WRAP_GAP}" if gap_ratio * SAMPLE_LINE_HEIGHT < MIN_WRAP_GAP else "")
    align_note = f"{align_ratio:g} 倍行高" + (f"，最少 {MIN_WRAP_ALIGN}" if align_ratio * SAMPLE_LINE_HEIGHT < MIN_WRAP_ALIGN else "")
    target = "左緣或中心點" if centered else "左緣"
    return (
        f"以 {SAMPLE_LINE_HEIGHT} 像素高的字為例，上下兩行的空隙 ≤ {gap} 像素（{gap_note}）、"
        f"{target}相差 ≤ {align} 像素（{align_note}）時，併成同一句。"
    )


@dataclass(frozen=True)
class WrapLayout:
    """範例的版面（設計座標，還沒縮放）。``first``／``second`` 會被併成同一句，``third`` 太遠被分開。"""

    first: QRectF
    second: QRectF
    third: QRectF
    gap: int
    align: int
    width: float
    height: float


class WrapPreview(QWidget):
    """一格模擬畫面：被拆成上下兩行的字幕，以及目前設定下能併成同一句的最遠位置。"""

    def __init__(self, base_config=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base = base_config
        self._gap_ratio = WRAP_GAP_RATIO
        self._align_ratio = WRAP_ALIGN_RATIO
        self._centered = True
        self.setFixedSize(QSize(330, 190))
        self.setAccessibleName("多行合併範例")

    def set_values(self, gap_ratio: float, align_ratio: float, centered: bool) -> None:
        self._gap_ratio = float(gap_ratio)
        self._align_ratio = float(align_ratio)
        self._centered = bool(centered)
        self.update()

    def values(self) -> tuple[float, float, bool]:
        return self._gap_ratio, self._align_ratio, self._centered

    # ------------------------------------------------------------- layout
    def _font(self, pixel_size: float) -> QFont:
        font = QFont()
        if self._base is not None:
            font.setFamilies(font_families(self._base))
        font.setPixelSize(max(1, round(pixel_size)))
        return font

    def layout_info(self) -> WrapLayout:
        height = float(SAMPLE_LINE_HEIGHT)
        gap, align = wrap_limits(self._gap_ratio, self._align_ratio)
        metrics = QFontMetricsF(self._font(height * _TEXT_SCALE))
        widths = [metrics.horizontalAdvance(text) + 6 for text in SAMPLE_TEXTS]
        first = QRectF(max(_LEFT_MARGIN, align + 10.0), _TOP_AREA, widths[0], height)  # 設定畫面的上限內第一行不會動
        second = QRectF(first.left() + align, first.bottom() + gap, widths[1], height)
        third = QRectF(first.left(), second.bottom() + gap + SEPARATE_EXTRA, widths[2], height)
        right = max(first.right(), second.right())
        return WrapLayout(
            first, second, third, gap, align,
            width=right + _RIGHT_AREA,
            height=third.bottom() + _BOTTOM_AREA,
        )

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), SCREEN)

        info = self.layout_info()
        scale = min(1.0, self.width() / info.width, self.height() / info.height)
        painter.scale(scale, scale)

        text_font = self._font(SAMPLE_LINE_HEIGHT * _TEXT_SCALE)
        label_font = self._font(11)
        first, second, third = info.first, info.second, info.third
        band_top, band_bottom = first.top() - 6, third.bottom() + 6

        # 對齊的範圍：第一行左緣（置中時另外還有中心點）前後 align 像素；第二行放在這個範圍的邊緣
        painter.setPen(Qt.PenStyle.NoPen)
        left_band = QColor(MEASURE)
        left_band.setAlpha(46)
        painter.setBrush(left_band)
        painter.drawRect(QRectF(first.left() - info.align, band_top, info.align * 2, band_bottom - band_top))
        if self._centered:
            centre = first.center().x()
            centre_band = QColor(CENTER)
            centre_band.setAlpha(46)
            painter.setBrush(centre_band)
            painter.drawRect(QRectF(centre - info.align, band_top, info.align * 2, band_bottom - band_top))

        # 併成同一句之後的範圍（OCR 合併後會變成一個框）
        merged = first.united(second).adjusted(-4, -4, 4, 4)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(MERGED, 1.4, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(merged, 4, 4)
        draw_label(painter, label_font, MERGED, "同一句", QRectF(merged.right() + 6, merged.top(), 56, 14))

        painter.setFont(text_font)
        for rect, text in ((first, SAMPLE_TEXTS[0]), (second, SAMPLE_TEXTS[1])):
            painter.setPen(QPen(FRAME, 1.2))
            painter.drawRect(rect)
            painter.setPen(TEXT)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        faded = QColor(TEXT)
        faded.setAlpha(150)
        painter.setPen(QPen(SEPARATE, 1.2, Qt.PenStyle.DashLine))
        painter.drawRect(third)
        painter.setPen(faded)
        painter.drawText(third, Qt.AlignmentFlag.AlignCenter, SAMPLE_TEXTS[2])
        draw_label(painter, label_font, SEPARATE, "分開翻譯", QRectF(third.right() + 6, third.top(), 60, 14))

        # 左緣誤差：第一行上方的括號
        bracket_y = first.top() - 14
        painter.setPen(QPen(MEASURE, 1.4))
        painter.drawLine(QPointF(first.left() - info.align, bracket_y), QPointF(first.left() + info.align, bracket_y))
        for x in (first.left() - info.align, first.left() + info.align):
            painter.drawLine(QPointF(x, bracket_y - 3), QPointF(x, bracket_y + 3))
        draw_label(
            painter, label_font, MEASURE, f"左緣 ±{info.align} px",
            QRectF(first.left() - info.align, bracket_y - 18, 120, 14),
        )
        if self._centered:
            draw_label(
                painter, label_font, CENTER, f"置中：中心點 ±{info.align} px",
                QRectF(first.center().x() - info.align, band_bottom - 2, 150, 14),
            )

        # 行距：右邊一條垂直量距，標在第一行與第二行之間
        bar_x = max(first.right(), second.right()) + 12
        painter.setPen(QPen(MEASURE, 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(first.right(), first.bottom()), QPointF(bar_x + 3, first.bottom()))
        painter.drawLine(QPointF(second.left(), second.top()), QPointF(bar_x + 3, second.top()))
        painter.setPen(QPen(MEASURE, 1.4))
        painter.drawLine(QPointF(bar_x, first.bottom()), QPointF(bar_x, second.top()))
        for y in (first.bottom(), second.top()):
            painter.drawLine(QPointF(bar_x - 3, y), QPointF(bar_x + 3, y))
        middle = (first.bottom() + second.top()) / 2
        draw_label(
            painter, label_font, MEASURE, f"行距\n≤ {info.gap} px",
            QRectF(bar_x + 6, middle - 14, 58, 28),
        )
