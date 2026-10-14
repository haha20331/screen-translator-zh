"""測試用的假 llama.cpp 引擎：收到完整提示詞，回傳「模型會寫的字」，並記錄每次呼叫。"""

from __future__ import annotations

import re
from collections.abc import Callable

from services.translation.offline_translator import OfflineTranslator

_SOURCE_RE = re.compile(r"Translate this \w+ text:\n(.*?)<\|im_end\|>", re.S)


def source_of(prompt: str) -> str:
    """提示詞最後一個待翻譯的原文（前面的是示範與前文）。"""
    return _SOURCE_RE.findall(prompt)[-1]


def static_part_of(prompt: str) -> str:
    """提示詞開頭的固定前綴（系統提示 + 示範），到第一個「待翻譯」的使用者輪次之前。"""
    return prompt.split("<|im_start|>user", 1)[0]


class FakeLlamaEngine:
    """像 ``LlamaCppEngine``：``decode(prompt)`` 決定模型回什麼。"""

    gpu_layers = 0

    def __init__(self, decode: Callable[..., str]) -> None:
        self.decode = decode
        # decode 可以收第二個參數 seed（None = 貪婪解碼），用來模擬「取樣重翻會得到不同的結果」
        import inspect

        self._decode_takes_seed = len(inspect.signature(decode).parameters) >= 2
        self.seeds: list[int | None] = []
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []
        self.prefixes: list[str] = []
        self.bans: list[frozenset[str]] = []
        self.allows: list[tuple[str, ...]] = []
        self.filter_prepared = 0

    def prepare_token_filter(self) -> None:
        self.filter_prepared += 1

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        prefix: str = "",
        seed: int | None = None,
        bans: frozenset[str] = frozenset(),
        allow: tuple[str, ...] = (),
    ) -> str:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        self.prefixes.append(prefix)
        self.seeds.append(seed)
        self.bans.append(bans)
        self.allows.append(allow)
        return self.decode(prompt, seed) if self._decode_takes_seed else self.decode(prompt)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def offline_translator(decode: Callable[[str], str], **section) -> tuple[OfflineTranslator, FakeLlamaEngine]:
    """離線翻譯器，裝上假引擎（不載入任何模型）。``section`` 放在 translation 區段頂層（use_context 等）。"""
    instance = OfflineTranslator({"offline": {}, **section})
    engine = FakeLlamaEngine(decode)
    instance._llama = engine
    return instance, engine
