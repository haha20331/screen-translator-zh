"""離線翻譯：Qwen2.5-Instruct（CTranslate2 int8）把英文、日文翻成繁體中文，不連網。

為什麼用指令模型而不是專用翻譯模型（評估結果見 PLAN.md 4.5）：
- ``Helsinki-NLP/opus-mt-ja-zh`` 並不存在；經英文轉譯（ja→en→zh）會產生幻覺。
- NLLB-200（600M / 1.3B）對字幕、遊戲對白這類短句常誤譯，詞表缺字還會輸出 ``⁇``。
- Qwen2.5-1.5B-Instruct 加上示範（few-shot）後穩定且自然，CPU 約 0.7 秒/句。

流程：依文字判斷來源語言（假名→日文、拉丁字母→英文）→ 分組批次交給模型 →
整理輸出 → 以 OpenCC ``s2twp`` 轉成台灣繁體（目標為簡體 ``zh`` 時則轉簡體）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any

from app.logger import app_data_dir, get_logger
from services.translation.base import TranslationError, Translator, register_translator
from utils import chinese
from utils.language_utils import detect_source_language, has_han, has_kana

log = get_logger("translation.offline")

DEFAULT_MODEL_NAME = "qwen2.5-1.5b-instruct-ct2-int8"
# 在 Intel 混合核心 CPU（i7-13700H）上實測：6 執行緒最快，超過 8 反而慢 3 倍以上。
_DEFAULT_MAX_THREADS = 6
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


# ----------------------------------------------------------------------------- model location
def _app_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
        roots.append(Path(sys.executable).resolve().parent)
    else:
        roots.append(Path(__file__).resolve().parents[2])
    return roots


def model_search_paths(model_dir: str = "", model_name: str = DEFAULT_MODEL_NAME) -> list[Path]:
    """依優先順序列出可能放模型的位置：設定指定 → 程式資料夾 ``models/translation`` → 使用者資料夾。"""
    candidates: list[Path] = []
    if model_dir:
        configured = Path(model_dir).expanduser()
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.extend(root / configured for root in _app_roots())
    for root in _app_roots():
        candidates.append(root / "models" / "translation" / model_name)
    candidates.append(app_data_dir() / "models" / "translation" / model_name)
    return candidates


def find_model_dir(model_dir: str = "", model_name: str = DEFAULT_MODEL_NAME) -> Path | None:
    for candidate in model_search_paths(model_dir, model_name):
        if (candidate / "model.bin").is_file() and (candidate / "tokenizer.json").is_file():
            return candidate
    return None


# ----------------------------------------------------------------------------- translator
class OfflineTranslator(Translator):
    name = "offline"
    # 純漢字（沒有假名）的文字視為日文來源：日文漢字與中文無法區分，因此不可被當成「已是中文」而跳過。
    han_is_source = True
    # 結果與原文相同是正常的（數字、按鍵組合、日中同形的漢字），失敗由本類別自己標記。
    identical_means_failure = False

    def __init__(self, config_section: dict, cache=None, api_key: str = "") -> None:
        super().__init__(config_section, cache, api_key)
        self._model_lock = threading.Lock()
        self._generator: Any = None
        self._tokenizer: Any = None
        self._static_tokens: dict[str, list[str]] = {}
        self.device = ""

    # ------------------------------------------------------------- availability
    @classmethod
    def availability_reason(cls) -> str | None:
        for module, package in (("ctranslate2", "ctranslate2"), ("tokenizers", "tokenizers"), ("opencc", "opencc")):
            if importlib.util.find_spec(module) is None:
                return f"離線翻譯依賴不可用：請安裝 {package}（pip install {package}）"
        return None

    def _settings(self) -> dict:
        section = self.config.get("offline", {})
        return section if isinstance(section, dict) else {}

    def _locate_model(self) -> Path:
        settings = self._settings()
        name = str(settings.get("model_name") or DEFAULT_MODEL_NAME)
        found = find_model_dir(str(settings.get("model_dir") or ""), name)
        if found is None:
            searched = "\n".join(f"  {path}" for path in model_search_paths(str(settings.get("model_dir") or ""), name))
            raise TranslationError(
                f"找不到離線翻譯模型「{name}」。請依 DEVELOPMENT.md 取得模型（scripts/fetch_translation_models.py），"
                f"放到下列任一位置：\n{searched}"
            )
        return found

    # ------------------------------------------------------------- model loading
    def _resolve_device(self, ctranslate2: Any) -> str:
        setting = str(self._settings().get("device", "auto")).lower()
        if setting in ("cpu", "cuda"):
            return setting
        try:
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # 沒有 CUDA 執行環境時 CTranslate2 可能丟例外
            return "cpu"

    def _load(self) -> None:
        if self._generator is not None:
            return
        reason = self.availability_reason()
        if reason is not None:
            raise TranslationError(reason)
        model_path = self._locate_model()
        chinese_ready = chinese.available()
        if not chinese_ready:  # 繁體是必要需求，沒有 OpenCC 時不應該悄悄輸出簡體
            raise TranslationError("缺少 OpenCC，無法輸出繁體中文：請安裝 opencc（pip install opencc）")

        import ctranslate2
        from tokenizers import Tokenizer

        settings = self._settings()
        self.device = self._resolve_device(ctranslate2)
        threads = int(settings.get("threads", 0))
        if threads <= 0:
            threads = min(_DEFAULT_MAX_THREADS, os.cpu_count() or 1)
        compute_type = str(settings.get("compute_type", "auto"))
        if compute_type == "auto":
            compute_type = "int8_float16" if self.device == "cuda" else "int8"
        log.info("載入離線翻譯模型 %s（device=%s, compute=%s, threads=%d）", model_path, self.device, compute_type, threads)
        self._generator = ctranslate2.Generator(
            str(model_path),
            device=self.device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=threads,
        )
        self._tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))

    def warmup(self) -> None:
        """載入模型並翻一句，讓第一次真正翻譯不必等模型載入。"""
        with self._model_lock:
            self._load()
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

    def _to_target_script(self, text: str, target_language: str) -> str:
        if (target_language or "").lower() == "zh":
            return chinese.to_simplified(text)
        return chinese.to_traditional(text)

    # ------------------------------------------------------------- Translator API
    def _translate_batch(self, texts: list[str], source_language: str | None, target_language: str) -> list[str]:
        if not (target_language or "").lower().startswith("zh"):
            raise TranslationError("離線翻譯目前只支援翻成中文（目標語言請選「繁體中文」）")
        han_only_as = self._settings().get("han_only_as", "ja") or None
        explicit = source_language if source_language in _LANGUAGE_NAMES else None

        groups: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            language = explicit or detect_source_language(text, han_only_as=han_only_as)
            if language in _LANGUAGE_NAMES:
                groups.setdefault(language, []).append(index)

        results = list(texts)  # 無法判斷語言的文字原樣保留
        failed: set[int] = set()
        for language, indices in groups.items():
            outputs = self._generate([texts[i] for i in indices], language)
            for index, output in zip(indices, outputs):
                # 空輸出或只是把日文抄回來：保留原文，讓基底類別把它當作翻譯失敗（不寫入快取）
                if output and not self.output_looks_untranslated(texts[index], output):
                    results[index] = self._to_target_script(output, target_language)
                else:
                    failed.add(index)
        self.last_failed_indices = failed
        return results

    def output_looks_untranslated(self, source: str, output: str) -> bool:
        """日文來源的輸出不像中文：仍以假名為主（抄回原文），或完全沒有漢字（例如把「スキ」譯成英文 skip）。"""
        if not has_kana(source) or not output:
            return False
        kana = sum(has_kana(char) for char in output)
        if kana / max(1, len(output)) > 0.5:
            return True
        return not has_han(output)


register_translator(OfflineTranslator)
