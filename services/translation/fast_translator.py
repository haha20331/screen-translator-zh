"""離線快速翻譯：NLLB-200-distilled-600M（CTranslate2 int8、貪婪解碼），近乎即時。

取捨：比標準翻譯（Qwen2.5）快 3–4 倍（CPU 單句約 0.2 秒、批次約 50 ms/句），
但品質較低 —— 一般句子與介面文字可讀，字幕、口語對白與成語式文案常誤譯，
NLLB 詞表缺字時會漏字。需要品質時請改用「離線翻譯（標準）」。

NLLB 以簡體輸出（``zho_Hans``）再經 OpenCC 轉台灣繁體：直接輸出 ``zho_Hant`` 的品質差很多。
授權：NLLB-200 權重是 CC-BY-NC-4.0（非商用）。
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

from app.logger import get_logger
from services.translation.base import TranslationError, register_translator
from services.translation.ct2_common import (
    ChineseTargetTranslator,
    default_threads,
    find_model_dir,
    model_search_paths,
    resolve_device,
)
from utils import chinese

log = get_logger("translation.fast")

DEFAULT_FAST_MODEL_NAME = "nllb-200-distilled-600m-ct2-int8"
_LANGUAGE_CODES = {"ja": "jpn_Jpan", "en": "eng_Latn"}
_TARGET_CODE = "zho_Hans"
_BATCH_SIZE = 16
_SPACE = "▁"  # sentencepiece 的詞首標記 ▁
# <unk> 是詞表缺字（NLLB 缺少「棕」「懶」這類字）；直接略過比顯示 <unk> 好
_SPECIAL_PIECES = frozenset({"</s>", "<s>", "<pad>", "<unk>"})


def decode_pieces(pieces: list[str]) -> str:
    """把 NLLB 的 sentencepiece 片段接回文字（等同 sentencepiece 的 decode）。"""
    return "".join(piece for piece in pieces if piece not in _SPECIAL_PIECES).replace(_SPACE, " ").strip()


class FastTranslator(ChineseTargetTranslator):
    name = "offline_fast"
    config_key = "offline_fast"

    def __init__(self, config_section: dict, cache=None, api_key: str = "") -> None:
        super().__init__(config_section, cache, api_key)
        self._model_lock = threading.Lock()
        self._translator: Any = None
        self._tokenizer: Any = None
        self.device = ""
        self.model_name = ""

    # ------------------------------------------------------------- availability / loading
    @classmethod
    def availability_reason(cls) -> str | None:
        for module, package in (("ctranslate2", "ctranslate2"), ("tokenizers", "tokenizers"), ("opencc", "opencc")):
            if importlib.util.find_spec(module) is None:
                return f"快速翻譯依賴不可用：請安裝 {package}（pip install {package}）"
        return None

    def _load(self) -> None:
        if self._translator is not None:
            return
        reason = self.availability_reason()
        if reason is not None:
            raise TranslationError(reason)
        if not chinese.available():
            raise TranslationError("缺少 OpenCC，無法輸出繁體中文：請安裝 opencc（pip install opencc）")

        settings = self._settings()
        name = str(settings.get("model_name") or DEFAULT_FAST_MODEL_NAME)
        model_dir = str(settings.get("model_dir") or "")
        path = find_model_dir(name, model_dir)
        if path is None:
            searched = "\n".join(f"  {p}" for p in model_search_paths(name, model_dir))
            raise TranslationError(
                f"找不到快速翻譯模型「{name}」。請依 DEVELOPMENT.md 取得模型"
                f"（scripts/fetch_translation_models.py --model {name}），放到下列任一位置：\n{searched}"
            )

        import ctranslate2
        from tokenizers import Tokenizer

        device = resolve_device(str(settings.get("device", "auto")), ctranslate2)
        compute_type = str(settings.get("compute_type", "auto"))
        if compute_type == "auto":
            compute_type = "int8_float16" if device == "cuda" else "int8"
        threads = default_threads(int(settings.get("threads", 0)))
        log.info("載入快速翻譯模型 %s（device=%s, compute=%s, threads=%d）", path, device, compute_type, threads)
        self._translator = ctranslate2.Translator(
            str(path), device=device, compute_type=compute_type, inter_threads=1, intra_threads=threads
        )
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.device, self.model_name = device, name

    def warmup(self) -> None:
        self._generate(["Hello"], "en")

    # ------------------------------------------------------------- inference
    def _generate(self, texts: list[str], language: str) -> list[str]:
        with self._model_lock:
            self._load()
            code = _LANGUAGE_CODES[language]
            batches = [
                [code] + self._tokenizer.encode(text, add_special_tokens=False).tokens + ["</s>"] for text in texts
            ]
            longest = max(len(batch) for batch in batches)
            results = self._translator.translate_batch(
                batches,
                target_prefix=[[_TARGET_CODE]] * len(batches),
                beam_size=max(1, int(self._settings().get("beam_size", 1))),
                max_decoding_length=min(256, max(32, longest * 3)),
                max_batch_size=_BATCH_SIZE,
            )
            outputs = []
            for result in results:
                hypothesis = list(result.hypotheses[0])
                if hypothesis and hypothesis[0] == _TARGET_CODE:
                    hypothesis = hypothesis[1:]
                outputs.append(decode_pieces(hypothesis))
            return outputs


register_translator(FastTranslator)
