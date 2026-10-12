"""讓程式自己的視窗不會被螢幕擷取到（Windows 10 2004 以上）。

監控框、框上的提示文字與譯文標籤都畫在被擷取的區域之上；若不排除，它們會被截進畫面，
下一輪 OCR 就把「拖拽調整 · 雙擊停止」之類的介面文字或自己的譯文當成原文讀進來。
``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` 讓視窗只顯示在螢幕上，截圖時完全不出現。
"""

from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QWidget

from app.logger import get_logger

log = get_logger("window_affinity")

WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11
# Windows 10 version 2004 (build 19041) 才支援 WDA_EXCLUDEFROMCAPTURE
_MIN_BUILD = 19041


def supported() -> bool:
    """目前系統是否支援把視窗排除在擷取之外。"""
    if sys.platform != "win32":
        return False
    try:
        return int(sys.getwindowsversion().build) >= _MIN_BUILD
    except Exception:
        return False


def exclude_from_capture(hwnd: int, user32: Any = None) -> bool:
    """對指定視窗設定「排除擷取」；成功回傳 True。不支援的系統上不做事並回傳 False。"""
    if not hwnd:
        return False
    if user32 is None:
        if not supported():
            return False
        import ctypes

        user32 = ctypes.windll.user32
    try:
        return bool(user32.SetWindowDisplayAffinity(int(hwnd), WDA_EXCLUDEFROMCAPTURE))
    except Exception as exc:
        log.debug("SetWindowDisplayAffinity 失敗：%s", exc)
        return False


class CaptureExclusionFilter(QObject):
    """應用程式層級的事件過濾器：每個頂層視窗顯示時，自動把它排除在擷取之外。"""

    def __init__(self, parent: QObject | None = None, apply: Any = exclude_from_capture) -> None:
        super().__init__(parent)
        self._apply = apply
        self._failed_logged = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
            try:
                ok = self._apply(int(obj.winId()))
            except Exception as exc:  # 過濾器內的例外不能影響視窗顯示
                log.debug("排除擷取失敗：%s", exc)
                ok = False
            if not ok and supported() and not self._failed_logged:
                self._failed_logged = True
                log.warning("無法把視窗排除在擷取之外：程式自己的視窗可能被截進畫面")
        return False  # 永遠不攔截事件
