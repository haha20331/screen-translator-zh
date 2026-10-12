"""CTranslate2 離線翻譯器共用的小工具：模型位置、裝置與執行緒、譯文整理。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from app.logger import app_data_dir
from services.translation.base import TranslationError, Translator
from utils import chinese
from utils.language_utils import detect_source_language, has_han, has_kana

# 在 Intel 混合核心 CPU（i7-13700H）上實測：6 執行緒最快，超過 8 反而慢 3 倍以上。
DEFAULT_MAX_THREADS = 6


# ----------------------------------------------------------------------------- 模型位置
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


def model_search_paths(model_name: str, model_dir: str = "") -> list[Path]:
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


def find_model_dir(model_name: str, model_dir: str = "", required: tuple[str, ...] = ("model.bin", "tokenizer.json")) -> Path | None:
    for candidate in model_search_paths(model_name, model_dir):
        if all((candidate / name).is_file() for name in required):
            return candidate
    return None


# ----------------------------------------------------------------------------- 裝置與執行緒
def cuda_available(ctranslate2: Any) -> bool:
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # 沒有 CUDA 執行環境時 CTranslate2 可能丟例外
        return False


def resolve_device(setting: str, ctranslate2: Any) -> str:
    """``cpu`` / ``cuda`` 依設定；``auto`` 有 NVIDIA GPU（CUDA）就用 GPU，否則 CPU。"""
    setting = (setting or "auto").lower()
    if setting in ("cpu", "cuda"):
        return setting
    return "cuda" if cuda_available(ctranslate2) else "cpu"


def default_threads(setting: int = 0) -> int:
    return setting if setting > 0 else min(DEFAULT_MAX_THREADS, os.cpu_count() or 1)


# ----------------------------------------------------------------------------- 譯文檢查
def looks_untranslated(source: str, output: str) -> bool:
    """日文來源的輸出不像中文：仍以假名為主（抄回原文），或完全沒有漢字（例如把「スキ」譯成英文 skip）。"""
    if not has_kana(source) or not output:
        return False
    kana = sum(has_kana(char) for char in output)
    if kana / max(1, len(output)) > 0.5:
        return True
    return not has_han(output)


# ----------------------------------------------------------------------------- 共用基底類別
class ChineseTargetTranslator(Translator):
    """本機翻譯器的共同流程：判斷來源語言 → 分組批次翻譯 → 整理標點 → 轉成目標中文字體。

    子類別實作 ``_generate(texts, language)``（同一種來源語言的一批文字 → 簡體或繁體中文的譯文）。
    """

    config_key = "offline"  # 在 translation 設定區段中的子區段名稱
    source_languages: tuple[str, ...] = ("ja", "en")
    # 純漢字（沒有假名）的文字視為日文來源：日文漢字與中文無法區分，因此不可被當成「已是中文」而跳過。
    han_is_source = True
    # 結果與原文相同是正常的（數字、按鍵組合、日中同形的漢字），失敗由本類別自己標記。
    identical_means_failure = False

    def _settings(self) -> dict:
        section = self.config.get(self.config_key, {})
        return section if isinstance(section, dict) else {}

    def _generate(self, texts: list[str], language: str) -> list[str]:  # pragma: no cover - 由子類別實作
        raise NotImplementedError

    def _to_target_script(self, text: str, target_language: str) -> str:
        if (target_language or "").lower() == "zh":
            return chinese.to_simplified(text)
        return chinese.to_traditional(text)

    def output_looks_untranslated(self, source: str, output: str) -> bool:
        return looks_untranslated(source, output)

    def _translate_batch(self, texts: list[str], source_language: str | None, target_language: str) -> list[str]:
        if not (target_language or "").lower().startswith("zh"):
            raise TranslationError("離線翻譯目前只支援翻成中文（目標語言請選「繁體中文」）")
        han_only_as = self._settings().get("han_only_as", "ja") or None
        explicit = source_language if source_language in self.source_languages else None

        groups: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            language = explicit or detect_source_language(text, han_only_as=han_only_as)
            if language in self.source_languages:
                groups.setdefault(language, []).append(index)

        results = list(texts)  # 無法判斷語言的文字原樣保留
        failed: set[int] = set()
        for language, indices in groups.items():
            outputs = self._generate([texts[i] for i in indices], language)
            for index, output in zip(indices, outputs):
                # 空輸出或只是把日文抄回來：保留原文，讓基底類別把它當作翻譯失敗（不寫入快取）
                if output and not self.output_looks_untranslated(texts[index], output):
                    results[index] = self._to_target_script(chinese.normalize_punctuation(output), target_language)
                else:
                    failed.add(index)
        self.last_failed_indices = failed
        return results
