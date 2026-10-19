"""用 Windows 自己的（物理像素）座標讀取游標位置、讀取與設定視窗位置。

Qt 的座標是「邏輯座標」：每個螢幕有自己的縮放倍率，視窗在建立、搬到另一個螢幕的當下，
Qt 用哪一個螢幕的倍率去換算，在多螢幕、縮放比例不同的電腦上有時會和我們預期的不一樣
（第一次顯示的視窗會偏掉，之後用滑鼠拖才正常）。截圖（mss）和這裡一樣是物理像素，
所以「框的位置」和「要擷取的範圍」直接用這裡的座標對齊，不經過 Qt 的換算。

程式本身是 Per-Monitor DPI aware（Qt 啟動時設定），``GetCursorPos``／``GetWindowRect``／
``SetWindowPos`` 因此都是物理像素。非 Windows 的平台一律回傳 None／False，呼叫端照舊用 Qt 的座標。
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010


def available() -> bool:
    return os.name == "nt"


def _user32():
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    return user32


def cursor_position() -> tuple[int, int] | None:
    """游標目前的位置（物理像素，虛擬桌面座標，可為負）。"""
    if not available():
        return None
    try:
        point = wintypes.POINT()
        if not _user32().GetCursorPos(ctypes.byref(point)):
            return None
        return int(point.x), int(point.y)
    except Exception:
        return None


def window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """視窗在螢幕上的位置 (left, top, right, bottom)，物理像素。"""
    if not available() or not hwnd:
        return None
    try:
        rect = wintypes.RECT()
        if not _user32().GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    except Exception:
        return None


def set_window_rect(hwnd: int, rect: tuple[int, int, int, int]) -> bool:
    """把視窗放到物理像素的 ``rect``（不改變前後順序、不搶焦點）。"""
    if not available() or not hwnd:
        return False
    left, top, right, bottom = (int(value) for value in rect)
    try:
        return bool(
            _user32().SetWindowPos(
                wintypes.HWND(hwnd), None, left, top, max(1, right - left), max(1, bottom - top),
                SWP_NOZORDER | SWP_NOACTIVATE,
            )
        )
    except Exception:
        return False
