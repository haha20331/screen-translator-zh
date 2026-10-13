"""簡繁中文轉換（OpenCC）。

專案要求中文輸出一律是繁體：離線翻譯模型的中文輸出可能是簡體，或簡繁混雜，
翻譯完成後固定經過 :func:`to_traditional`（``s2twp``：字形轉換並套用台灣慣用詞彙，
例如 软件→軟體、网络→網路），再顯示在螢幕上與寫入 txt。
目標語言選簡體中文時改用 :func:`to_simplified`（``tw2sp``）。
"""

from __future__ import annotations

import importlib.util
import re
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


# 日文漢字文字幾乎不會出現、但中文很常見的字（繁體與簡體）
_CHINESE_MARKER_CHARS = frozenset("你妳她們们這这個个嗎吗吧呢沒没還还麼么裡里")


def looks_chinese(text: str) -> bool:
    """純漢字（沒有假名）的一行，是不是看得出是中文而不是日文。

    日文漢字與中文本來無法區分，這裡只找「正面證據」，沒有證據就回傳 False（當日文處理）：
    全形逗號、中文特有的字，以及 OpenCC 能分辨的字形：只有繁體中文會用的字（日文用新字體，
    ``t2jp`` 轉了會變）與只有簡體中文會用的字（``s2t`` 會變、而日文新字體 ``jp2t`` 不會）。
    """
    value = text or ""
    if "，" in value or any(char in _CHINESE_MARKER_CHARS for char in value):
        return True
    if not available():
        return False
    if _convert("t2jp", value) != value:
        return True
    return _convert("s2t", value) != value and _convert("jp2t", value) == value


def to_traditional(text: str) -> str:
    """簡體轉台灣繁體。已經是繁體的字元不受影響。"""
    return _convert(_TO_TRADITIONAL, text)


def to_simplified(text: str) -> str:
    """繁體（台灣用語）轉簡體。"""
    return _convert(_TO_SIMPLIFIED, text)


_ASCII_TO_FULLWIDTH = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；"}
# 前一個字元是中日文字或全形標點時，ASCII 標點才換成全形；「2.4.1」「Ctrl+S.」這類不受影響
_CJK_BEFORE = r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]"
_PUNCT_AFTER_CJK = re.compile(rf"(?<={_CJK_BEFORE})\s*([,.?!:;])")


def normalize_punctuation(text: str) -> str:
    """機器翻譯常輸出 ASCII 標點（``你好,世界.``）；接在中文後面的換成全形，並移除標點前的空白。

    同時移除 NLLB 詞表缺字時輸出的 ``⁇``。
    """
    if not text:
        return text
    text = text.replace("\u2047", "").replace("\u2047 ", "")
    text = _PUNCT_AFTER_CJK.sub(lambda match: _ASCII_TO_FULLWIDTH[match.group(1)], text).strip()
    # 小模型有時以孤零零的逗號起頭（「，我知道…」）；譯文不會以逗號、句號開始
    return text.lstrip(_LEADING_PUNCTUATION).strip()


_LEADING_PUNCTUATION = " ,.;:，、。；：\u3000"
_REPEATED_PHRASE_RE = re.compile(r"(.{6,}).*?\1")
_REPEATED_RUN_RE = re.compile(r"(.{3,}?)\1{2,}")
_REPEATED_SOURCE_RE = re.compile(r"(.{3,}).*?\1")


def has_runaway_repetition(output: str, source: str = "") -> bool:
    """譯文是不是陷入「同一段話重複」的迴圈（小型翻譯模型遇到雜訊輸入時的典型失敗）。

    同一段 6 字以上的話出現兩次，或 3 字以上的片段連續出現三次。原文本身就在重複
    （歌詞副歌、「哈哈哈」）時，譯文跟著重複是正常的，不算。
    """
    compact = re.sub(r"\s+", "", output or "")
    if not (_REPEATED_PHRASE_RE.search(compact) or _REPEATED_RUN_RE.search(compact)):
        return False
    return _REPEATED_SOURCE_RE.search(re.sub(r"\s+", "", source or "")) is None
