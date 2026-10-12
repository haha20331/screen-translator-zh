"""簡繁中文轉換（OpenCC）。

專案要求中文輸出一律是繁體：離線翻譯模型的中文輸出可能是簡體，或簡繁混雜，
翻譯完成後固定經過 :func:`to_traditional`（``s2twp``：字形轉換並套用台灣慣用詞彙，
例如 软件→軟體、网络→網路），再顯示在螢幕上與寫入 txt。
目標語言選簡體中文時改用 :func:`to_simplified`（``tw2sp``）。
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

_TO_TRADITIONAL = "s2twp"
_TO_SIMPLIFIED = "tw2sp"
_lock = threading.Lock()
_converters: dict[str, Any] = {}


def available() -> bool:
    return importlib.util.find_spec("opencc") is not None


def _convert(config: str, text: str) -> str:
    if not text:
        return text
    with _lock:
        converter = _converters.get(config)
        if converter is None:
            try:
                from opencc import OpenCC
            except ImportError as exc:
                raise RuntimeError("缺少 OpenCC，無法輸出繁體中文：請安裝 opencc（pip install opencc）") from exc
            converter = _converters[config] = OpenCC(config)
        return converter.convert(text)


def to_traditional(text: str) -> str:
    """簡體轉台灣繁體。已經是繁體的字元不受影響。"""
    return _convert(_TO_TRADITIONAL, text)


def to_simplified(text: str) -> str:
    """繁體（台灣用語）轉簡體。"""
    return _convert(_TO_SIMPLIFIED, text)
