"""設定畫面的數值輸入：滑鼠滾輪不會改變數值。

設定頁很長，使用者用滾輪捲頁面時，游標經過數值框或滑桿就會把數字改掉，而且常常沒發現。
這些元件把滾輪事件交還給外層（``ignore``），頁面照常捲動；要改數值請直接輸入、按上下箭頭或拖曳滑桿。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QDoubleSpinBox, QSlider, QSpinBox


class NoWheelSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # 滾輪不會讓它取得焦點

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class NoWheelSlider(QSlider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()
