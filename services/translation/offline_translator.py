"""離線翻譯（標準）：Qwen2.5-Instruct（CTranslate2 int8）把英文、日文翻成繁體中文，不連網。

為什麼用指令模型而不是專用翻譯模型（評估結果見 PLAN.md 4.5）：
- ``Helsinki-NLP/opus-mt-ja-zh`` 並不存在；經英文轉譯（ja→en→zh）會產生幻覺。
- NLLB-200（600M / 1.3B）對字幕、遊戲對白這類短句常誤譯，詞表缺字還會輸出 ``⁇``
  （它改作為「快速翻譯」，見 fast_translator.py）。
- Qwen2.5-1.5B-Instruct 加上示範（few-shot）後穩定且自然，CPU 約 0.5 秒/句。

依硬體自動選模型（``model_name = "auto"``，預設）：有 NVIDIA GPU（CUDA）時優先用較大的
7B 模型，載入失敗（例如顯示卡記憶體不足）就退到 1.5B，沒有 GPU 則用 CPU 的 1.5B。

流程：依文字判斷來源語言（假名→日文、拉丁字母→英文）→ 分組批次交給模型 →
整理輸出 → 以 OpenCC ``s2twp`` 轉成台灣繁體（目標為簡體 ``zh`` 時則轉簡體）。
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any, NamedTuple

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

log = get_logger("translation.offline")

DEFAULT_MODEL_NAME = "qwen2.5-1.5b-instruct-ct2-int8"
LARGE_MODEL_NAME = "qwen2.5-7b-instruct-ct2-int8"
# auto 模式的候選：(模型名稱, 是否需要 CUDA)。依序嘗試，載入失敗就換下一個。
MODEL_TIERS: list[tuple[str, bool]] = [(LARGE_MODEL_NAME, True), (DEFAULT_MODEL_NAME, False)]
_BATCH_SIZE = 8

_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into Traditional Chinese "
    "as used in Taiwan. Output only the translation, with no explanations."
)
_LANGUAGE_NAMES = {"ja": "Japanese", "en": "English"}
# 示範能大幅降低小模型回傳原文、加註解的機率；內容刻意涵蓋介面、遊戲、對白三類。
_SHOTS: dict[str, list[tuple[str, str]]] = {
    "ja": [
        ("今日はいい天気ですね。", "今天天氣真好呢。"),
        ("ファイルを保存しますか？", "要儲存檔案嗎？"),
        ("HPが50回復した！", "恢復了 50 點 HP！"),
    ],
    "en": [
        ("Press any key to continue", "按任意鍵繼續"),
        ("You found a rusty key.", "你找到了一把生鏽的鑰匙。"),
        ("Save changes before closing?", "關閉前要儲存變更嗎？"),
    ],
}


def static_prompt(language: str) -> str:
    """固定前綴（system + 示範），每次翻譯都一樣；結尾剛好在 ``<|im_end|>\\n`` 的分詞邊界上。

    交給 CTranslate2 的 ``static_prompt``，這段的注意力快取只算一次（實測每句快約 40%）。
    """
    name = _LANGUAGE_NAMES[language]
    parts = [f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"]
    for source, target in _SHOTS[language]:
        parts.append(f"<|im_start|>user\nTranslate this {name} text:\n{source}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{target}<|im_end|>\n")
    return "".join(parts)


def user_prompt(text: str, language: str) -> str:
    """每句不同的部分：待翻譯文字，並留下 assistant 開頭讓模型續寫。"""
    name = _LANGUAGE_NAMES[language]
    return f"<|im_start|>user\nTranslate this {name} text:\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_prompt(text: str, language: str) -> str:
    """完整的 Qwen ChatML 提示詞（固定前綴 + 待翻譯文字）。"""
    return static_prompt(language) + user_prompt(text, language)


def clean_output(text: str) -> str:
    """只留第一個非空行（小模型偶爾會在後面附註解），並去掉頭尾空白。"""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


class LoadPlan(NamedTuple):
    model_name: str
    device: str
    path: Path


# ----------------------------------------------------------------------------- translator
class OfflineTranslator(ChineseTargetTranslator):
    name = "offline"
    config_key = "offline"

    def __init__(self, config_section: dict, cache=None, api_key: str = "") -> None:
        super().__init__(config_section, cache, api_key)
        self._model_lock = threading.Lock()
        self._generator: Any = None
        self._tokenizer: Any = None
        self._static_tokens: dict[str, list[str]] = {}
        self.device = ""
        self.model_name = ""  # 實際載入的模型（auto 模式下可能和設定不同）

    # ------------------------------------------------------------- availability
    @classmethod
    def availability_reason(cls) -> str | None:
        for module, package in (("ctranslate2", "ctranslate2"), ("tokenizers", "tokenizers"), ("opencc", "opencc")):
            if importlib.util.find_spec(module) is None:
                return f"離線翻譯依賴不可用：請安裝 {package}（pip install {package}）"
        return None

    def _resolve_device(self, ctranslate2: Any) -> str:
        return resolve_device(str(self._settings().get("device", "auto")), ctranslate2)

    # ------------------------------------------------------------- model selection
    def load_plans(self, ctranslate2: Any) -> list[LoadPlan]:
        """要嘗試載入的 (模型, 裝置, 路徑) 清單，依優先順序；找不到任何模型就丟 TranslationError。"""
        settings = self._settings()
        name = str(settings.get("model_name") or "auto")
        model_dir = str(settings.get("model_dir") or "")
        device = self._resolve_device(ctranslate2)
        plans: list[LoadPlan] = []

        if name != "auto":
            path = find_model_dir(name, model_dir)
            if path is not None:
                plans.append(LoadPlan(name, device, path))
        else:
            for tier_name, needs_cuda in MODEL_TIERS:
                if needs_cuda and device != "cuda":
                    continue
                path = find_model_dir(tier_name, model_dir if tier_name == DEFAULT_MODEL_NAME else "")
                if path is not None:
                    plans.append(LoadPlan(tier_name, device, path))
            if device == "cuda":  # GPU 失敗（記憶體不足等）時最後退到 CPU 的標準模型
                path = find_model_dir(DEFAULT_MODEL_NAME, model_dir)
                if path is not None:
                    plans.append(LoadPlan(DEFAULT_MODEL_NAME, "cpu", path))

        if not plans:
            wanted = name if name != "auto" else DEFAULT_MODEL_NAME
            searched = "\n".join(f"  {path}" for path in model_search_paths(wanted, model_dir))
            raise TranslationError(
                f"找不到離線翻譯模型「{wanted}」。請依 DEVELOPMENT.md 取得模型（scripts/fetch_translation_models.py），"
                f"放到下列任一位置：\n{searched}"
            )
        return plans

    def _compute_type(self, device: str) -> str:
        setting = str(self._settings().get("compute_type", "auto"))
        if setting != "auto":
            return setting
        return "int8_float16" if device == "cuda" else "int8"

    # ------------------------------------------------------------- model loading
    def _load(self) -> None:
        if self._generator is not None:
            return
        reason = self.availability_reason()
        if reason is not None:
            raise TranslationError(reason)
        if not chinese.available():  # 繁體是必要需求，沒有 OpenCC 時不應該悄悄輸出簡體
            raise TranslationError("缺少 OpenCC，無法輸出繁體中文：請安裝 opencc（pip install opencc）")

        import ctranslate2
        from tokenizers import Tokenizer

        threads = default_threads(int(self._settings().get("threads", 0)))
        errors: list[str] = []
        for plan in self.load_plans(ctranslate2):
            compute_type = self._compute_type(plan.device)
            log.info(
                "載入離線翻譯模型 %s（device=%s, compute=%s, threads=%d）", plan.path, plan.device, compute_type, threads
            )
            try:
                self._generator = ctranslate2.Generator(
                    str(plan.path),
                    device=plan.device,
                    compute_type=compute_type,
                    inter_threads=1,
                    intra_threads=threads,
                )
                self._tokenizer = Tokenizer.from_file(str(plan.path / "tokenizer.json"))
            except Exception as exc:  # 顯示卡記憶體不足、CUDA 執行環境缺失等：換下一個候選
                log.warning("載入 %s（%s）失敗：%s", plan.model_name, plan.device, exc)
                errors.append(f"{plan.model_name}/{plan.device}: {exc}")
                self._generator = self._tokenizer = None
                continue
            self.device, self.model_name = plan.device, plan.model_name
            self._static_tokens = {}
            return
        raise TranslationError("離線翻譯模型載入失敗：" + "；".join(errors))

    def warmup(self) -> None:
        """載入模型並翻一句，讓第一次真正翻譯不必等模型載入。"""
        self._generate(["Hello"], "en")

    # ------------------------------------------------------------- inference
    def _generate(self, texts: list[str], language: str) -> list[str]:
        with self._model_lock:
            self._load()
            settings = self._settings()
            if language not in self._static_tokens:
                self._static_tokens[language] = self._tokenizer.encode(
                    static_prompt(language), add_special_tokens=False
                ).tokens
            batches = [
                self._tokenizer.encode(user_prompt(text, language), add_special_tokens=False).tokens for text in texts
            ]
            results = self._generator.generate_batch(
                batches,
                static_prompt=self._static_tokens[language],
                max_length=int(settings.get("max_new_tokens", 160)),
                max_batch_size=_BATCH_SIZE,
                beam_size=max(1, int(settings.get("beam_size", 1))),
                sampling_topk=1,
                include_prompt_in_result=False,
                end_token="<|im_end|>",
            )
            return [
                clean_output(self._tokenizer.decode(list(result.sequences_ids[0]), skip_special_tokens=True))
                for result in results
            ]


register_translator(OfflineTranslator)
