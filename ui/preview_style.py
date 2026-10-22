"""設定畫面的距離範例（``merge_preview``、``wrap_preview``）共用的顏色與標籤繪製。

範例是一格模擬畫面，底色與文字顏色固定，不跟系統的深淺色走（和譯文範例同一種藍）。
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter

SCREEN = QColor("#27456F")  # 模擬畫面的底色
TEXT = QColor("#FFFFFF")
FRAME = QColor("#7CC4FF")  # OCR 文字框
MERGED = QColor("#6EE7A8")  # 會被接成同一句的範圍
MEASURE = QColor("#FFC857")  # 量距標示
SEPARATE = QColor("#B7C3D6")  # 太遠、會分開翻譯的那一塊


def draw_label(
    painter: QPainter,
    font: QFont,
    color: QColor,
    text: str,
    rect: QRectF,
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
) -> None:
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(rect, int(align | Qt.AlignmentFlag.AlignVCenter), text)
