"""生成時就擋掉不該出現在中文譯文裡的詞，取代「翻完才發現不對、再翻一次」。

Qwen2.5 這類多語模型偶爾會在中文裡夾一個韓文字、俄文字、日文假名，或原文沒有的英文單字（``계속你的名字``、
``數 NIGHT SKY 中的星星``），3B 還會把中文字一個一個隔開（``淚 的 數 越``）。以前的做法是翻完檢查、不對就改用取樣重翻，
每次重翻多花一次完整的模型運算（3B 約 +0.6 秒，7B 約 +1.5 秒），而且重翻也不保證翻對。

這裡改成在選字的那一步，把這些詞的分數壓成負無限大，模型根本選不到它們：

- ``foreign``：日文假名、韓文、俄文、希臘、希伯來、阿拉伯、天城文、泰文的字（譯成中文時永遠不該出現）。
- ``latin``：英文字母。只有原文完全沒有英文字母時才擋（原文有 HP、Ctrl、Windows 的，譯文保留它們是對的）。
- ``spaces``：空白，以及「空白＋漢字」。同樣只有原文沒有英文字母時才擋（中文字之間不該有空格；中文與英文、數字之間的空格
  只會出現在原文有英文的句子）。換行不擋。

只擋「模型本來就不該選」的詞，所以原本翻得對的句子不受影響；擋掉之後，模型會選下一個最可能的詞。
詞彙表的分類在載入模型後第一次需要時建立一次（約 150,000 個詞，約一兩秒）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, Sequence

import numpy as np

FOREIGN = "foreign"
LATIN = "latin"
SPACES = "spaces"

# 與 utils/language_utils 的 has_kana、has_foreign_script 同一組範圍，擋掉的就是事後檢查會判為「不像中文」的字
_FOREIGN_RE = re.compile(
    "["
    "\u3040-\u30ff\uff66-\uff9f"  # 平假名、片假名（含半形）
    "\u0370-\u03ff"  # 希臘
    "\u0400-\u052f"  # 西里爾
    "\u0590-\u05ff"  # 希伯來
    "\u0600-\u06ff"  # 阿拉伯
    "\u0900-\u097f"  # 天城文
    "\u0e00-\u0e7f"  # 泰文
    "\u1100-\u11ff\u3130-\u318f\uac00-\ud7af"  # 韓文
    "]"
)
_LATIN_RE = re.compile(r"[A-Za-z]")
# 一個以上的空白後面接漢字或全形標點（「 的」「 ，」），或整個詞都是空白（不含換行）
_SPACED_RE = re.compile(r"^ +(?:[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]|$)")
_SPECIAL_RE = re.compile(r"^<\|.*\|>$")  # <|im_end|> 等控制記號：不分類，不然模型就停不下來了

_HAS_LATIN_RE = re.compile(r"[A-Za-z]")


def source_has_latin(text: str) -> bool:
    """原文有沒有英文字母（全形的 ＨＰ 也算）。"""
    return bool(_HAS_LATIN_RE.search(unicodedata.normalize("NFKC", text or "")))


def bans_for(source: str) -> frozenset[str]:
    """這一句原文，生成時要擋哪幾類詞。外文一律擋；原文沒有英文字母時，英文字母與空白也擋。"""
    if source_has_latin(source):
        return frozenset({FOREIGN})
    return frozenset({FOREIGN, LATIN, SPACES})


class TokenFilter:
    """依詞彙表建立「擋掉哪些詞」的遮罩。``pieces``：每個詞編號對應的文字（位元組）；``tokenize``：把文字切成詞編號。"""

    def __init__(self, pieces: Callable[[], Iterable[bytes]], tokenize: Callable[[str], Sequence[int]]) -> None:
        self._load_pieces = pieces
        self._tokenize = tokenize
        self._classes: dict[str, np.ndarray] | None = None
        self._masks: dict[tuple[frozenset[str], tuple[str, ...]], np.ndarray] = {}

    def _build(self) -> dict[str, np.ndarray]:
        foreign: list[int] = []
        latin: list[int] = []
        spaces: list[int] = []
        size = 0
        for index, piece in enumerate(self._load_pieces()):
            size = index + 1
            if not piece:
                continue
            text = piece.decode("utf-8", "ignore")  # 只有半個字的詞（位元組片段）解不出來，當成不用擋
            if not text or _SPECIAL_RE.match(text):
                continue
            if _FOREIGN_RE.search(text):
                foreign.append(index)
            if _LATIN_RE.search(text):
                latin.append(index)
            if _SPACED_RE.match(text):
                spaces.append(index)
        return {
            FOREIGN: np.asarray(foreign, dtype=np.int64),
            LATIN: np.asarray(latin, dtype=np.int64),
            SPACES: np.asarray(spaces, dtype=np.int64),
            "size": np.asarray([size], dtype=np.int64),
        }

    def ensure_loaded(self) -> None:
        if self._classes is None:
            self._classes = self._build()

    def counts(self) -> dict[str, int]:
        self.ensure_loaded()
        return {name: int(len(self._classes[name])) for name in (FOREIGN, LATIN, SPACES)}

    def mask(self, bans: Iterable[str], allow: Sequence[str] = ()) -> np.ndarray | None:
        """加在每個詞分數上的遮罩（要擋的是負無限大，其餘 0）。``allow``：這些文字切出來的詞一律不擋
        （例如 ``[SKIP]``：沒有英文的原文要擋英文字母，但「只有擬聲詞」的回報標記要放行）。沒有要擋的回傳 None。"""
        bans = frozenset(bans)
        if not bans:
            return None
        key = (bans, tuple(allow))
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        self.ensure_loaded()
        size = int(self._classes["size"][0])
        mask = np.zeros(size, dtype=np.float32)
        for name in bans:
            ids = self._classes.get(name)
            if ids is not None and len(ids):
                mask[ids] = -np.inf
        for text in allow:
            for candidate in (text, " " + text):
                for token in self._tokenize(candidate):
                    if 0 <= token < size:
                        mask[token] = 0.0
        self._masks[key] = mask
        return mask

    def processor(self, bans: Iterable[str], allow: Sequence[str] = ()):
        """給 llama-cpp-python 的 ``logits_processor``：每一步把遮罩加到所有詞的分數上。"""
        mask = self.mask(bans, allow)
        if mask is None:
            return None

        def apply(_input_ids, scores):
            if len(scores) == len(mask):
                scores = scores + mask
            return scores

        return apply
