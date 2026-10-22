"""設定畫面「文本組合」的範例：用目前（還沒儲存）的行合併容差與橫向合併閾值，畫出被 OCR 拆開的幾塊字
最遠能離多遠、還會被接成同一句。

距離的算法與真正合併（``utils.layout_utils``）相同：橫向用 ``x_gap_limit``；上下錯開是「容差 × 行高」
（兩塊一樣高時，``group_into_rows`` 的重疊條件就是上下錯開不超過 容差 × 行高）。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.translation_overlay import font_families
from utils.layout_utils import CHAR_WIDTH_PER_HEIGHT, MIN_CHAR_WIDTH, x_gap_limit

SAMPLE_LINE_HEIGHT = 24  # 範例字的高度（像素）；所有距離都跟行高成正比
SAMPLE_TEXTS = ("こんにちは", "お元気", "ですか")  # 一句話被 OCR 拆成三塊
SEPARATE_EXTRA = 12  # 第三塊放在「剛好超過」閾值再多 12 像素的地方，示範會被分開
MAX_Y_RATIO = 2.0  # 設定畫面行合併容差的上限（版面按這個預留高度，調整時第一塊不會跳動）

_SCREEN = QColor("#27456F")  # 模擬畫面的底色：與譯文範例同一種藍，不跟系統的深淺色走
_TEXT = QColor("#FFFFFF")
_FRAME = QColor("#7CC4FF")
_MERGED = QColor("#6EE7A8")
_MEASURE = QColor("#FFC857")
_SEPARATE = QColor("#B7C3D6")

_LEFT_MARGIN = 66
_TEXT_SCALE = 0.7  # 字的像素大小 = 行高 × 這個值
_TOP_LABEL = 18
_BOTTOM_AREA = 40
_RIGHT_MARGIN = 8


def merge_limits(
    y_tolerance_ratio: float, x_gap_ratio: float, line_height: float = SAMPLE_LINE_HEIGHT
) -> tuple[float, float]:
    """(橫向最大空隙, 上下最多錯開)，單位都是像素。"""
    return x_gap_limit(line_height, x_gap_ratio), y_tolerance_ratio * line_height


def merge_summary(y_tolerance_ratio: float, x_gap_ratio: float) -> str:
    """範例底下的文字說明：把兩個比例換算成範例字的像素距離。"""
    gap, offset = merge_limits(y_tolerance_ratio, x_gap_ratio)
    char_width = max(MIN_CHAR_WIDTH, SAMPLE_LINE_HEIGHT * CHAR_WIDTH_PER_HEIGHT)
    if gap > 0:
        horizontal = f"橫向：空隙小於 {gap:.1f} 像素（{x_gap_ratio:g} 倍字寬）就接成同一句。"
    else:
        horizontal = "橫向：0 倍字寬，兩塊的框要互相重疊才會接成同一句。"
    return (
        f"以 {SAMPLE_LINE_HEIGHT} 像素高的字為例（1 倍字寬 ≈ {char_width:.1f} 像素）：\n"
        f"{horizontal}\n"
        f"上下：最多錯開 {offset:.1f} 像素（{y_tolerance_ratio:g} 倍行高）仍算同一行。"
    )


@dataclass(frozen=True)
class MergeLayout:
    """範例的版面（設計座標，還沒縮放）。``first``／``second`` 會被接成同一句，``third`` 太遠被分開。"""

    first: QRectF
    second: QRectF
    third: QRectF
    gap: float
    offset: float
    width: float
    height: float


class MergePreview(QWidget):
    """一格模擬畫面：被拆開的三塊字，以及目前設定下能被接成同一句的最遠位置。"""

    def __init__(self, base_config=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base = base_config
        self._y_ratio = 1.5
        self._x_ratio = 0.8
        self.setFixedSize(QSize(330, 150))
        self.setAccessibleName("文本組合範例")

    def set_values(self, y_tolerance_ratio: float, x_gap_ratio: float) -> None:
        self._y_ratio = float(y_tolerance_ratio)
        self._x_ratio = float(x_gap_ratio)
        self.update()

    def values(self) -> tuple[float, float]:
        return self._y_ratio, self._x_ratio

    # ------------------------------------------------------------- layout
    def _font(self, pixel_size: float) -> QFont:
        font = QFont()
        if self._base is not None:
            font.setFamilies(font_families(self._base))
        font.setPixelSize(max(1, round(pixel_size)))
        return font

    def layout_info(self) -> MergeLayout:
        height = float(SAMPLE_LINE_HEIGHT)
        gap, offset = merge_limits(self._y_ratio, self._x_ratio)
        offset = min(offset, MAX_Y_RATIO * height)
        metrics = QFontMetricsF(self._font(height * _TEXT_SCALE))
        widths = [metrics.horizontalAdvance(text) + 6 for text in SAMPLE_TEXTS]
        top = _TOP_LABEL + MAX_Y_RATIO * height + 6  # 第一塊固定在這一行，第二塊往上錯開
        first = QRectF(_LEFT_MARGIN, top, widths[0], height)
        second = QRectF(first.right() + gap, top - offset, widths[1], height)
        third = QRectF(second.right() + gap + SEPARATE_EXTRA, second.top(), widths[2], height)
        return MergeLayout(
            first, second, third, gap, offset,
            width=third.right() + _RIGHT_MARGIN,
            height=top + height + _BOTTOM_AREA,
        )

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _SCREEN)

        info = self.layout_info()
        scale = min(1.0, self.width() / info.width, self.height() / info.height)
        painter.scale(scale, scale)

        text_font = self._font(SAMPLE_LINE_HEIGHT * _TEXT_SCALE)
        label_font = self._font(11)
        first, second, third = info.first, info.second, info.third

        # 接成同一句之後的範圍（OCR 合併後會變成一個框）
        merged = first.united(second).adjusted(-4, -4, 4, 4)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_MERGED, 1.4, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(merged, 4, 4)
        self._label(painter, label_font, _MERGED, "同一句", QRectF(merged.left(), merged.top() - 16, 80, 14))

        painter.setFont(text_font)
        for rect, text in ((first, SAMPLE_TEXTS[0]), (second, SAMPLE_TEXTS[1])):
            painter.setPen(QPen(_FRAME, 1.2))
            painter.drawRect(rect)
            painter.setPen(_TEXT)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        faded = QColor(_TEXT)
        faded.setAlpha(150)
        painter.setPen(QPen(_SEPARATE, 1.2, Qt.PenStyle.DashLine))
        painter.drawRect(third)
        painter.setPen(faded)
        painter.drawText(third, Qt.AlignmentFlag.AlignCenter, SAMPLE_TEXTS[2])
        self._label(painter, label_font, _SEPARATE, "分開翻譯", QRectF(third.left(), third.top() - 16, 80, 14))

        # 橫向：第一塊右緣到第二塊左緣的最大空隙
        arrow_y = first.bottom() + 12
        painter.setPen(QPen(_MEASURE, 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(first.right(), first.bottom()), QPointF(first.right(), arrow_y + 4))
        painter.drawLine(QPointF(second.left(), second.bottom()), QPointF(second.left(), arrow_y + 4))
        painter.setPen(QPen(_MEASURE, 1.4))
        painter.drawLine(QPointF(first.right(), arrow_y), QPointF(second.left(), arrow_y))
        for x in (first.right(), second.left()):
            painter.drawLine(QPointF(x, arrow_y - 3), QPointF(x, arrow_y + 3))
        middle = (first.right() + second.left()) / 2
        self._label(
            painter, label_font, _MEASURE, f"空隙 < {info.gap:.1f} px" if info.gap > 0 else "框要重疊才接",
            QRectF(middle - 60, arrow_y + 6, 120, 14), Qt.AlignmentFlag.AlignHCenter,
        )

        # 上下：第一塊與第二塊上緣的最大落差
        bar_x = _LEFT_MARGIN - 8
        centre = first.top() + first.height() / 2
        painter.setPen(QPen(_MEASURE, 1.4))
        painter.drawLine(QPointF(bar_x, first.top()), QPointF(bar_x, second.top()))
        for y in (first.top(), second.top()):
            painter.drawLine(QPointF(bar_x - 3, y), QPointF(bar_x + 3, y))
        self._label(
            painter, label_font, _MEASURE, f"上下錯開\n≤ {info.offset:.1f} px",
            QRectF(0, centre - 14, bar_x - 4, 28), Qt.AlignmentFlag.AlignRight,
        )

    @staticmethod
    def _label(painter: QPainter, font: QFont, color: QColor, text: str, rect: QRectF, align=Qt.AlignmentFlag.AlignLeft) -> None:
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(rect, int(align | Qt.AlignmentFlag.AlignVCenter), text)
