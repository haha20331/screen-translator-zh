"""離線翻譯器共用的小工具：模型位置、執行緒、擬聲詞防呆、譯文整理。"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

from app.logger import app_data_dir
from services.translation import llama_backend
from services.translation.base import TranslationError, Translator
from utils import chinese
from utils.language_utils import (
    detect_source_language,
    has_foreign_script,
    has_han,
    has_kana,
    has_spaced_out_chars,
    has_symbol_junk,
    unexpected_latin_words,
    untranslated_english_words,
)

# 預設最多用幾個執行緒（設定 ``threads`` 可改）。在 Intel 混合核心 CPU（i7-13700H）上，執行緒開太多反而更慢。
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
    """依優先順序列出可能放模型的位置：設定指定的資料夾（先找它底下以模型名稱命名的子資料夾，再把它本身當成模型資料夾）
    → 程式資料夾 ``models/translation/<名稱>`` → 使用者資料夾。"""
    candidates: list[Path] = []
    if model_dir:
        configured = Path(model_dir).expanduser()
        bases = [configured] if configured.is_absolute() else [configured, *(root / configured for root in _app_roots())]
        for base in bases:
            candidates.extend([base / model_name, base])
    for root in _app_roots():
        candidates.append(root / "models" / "translation" / model_name)
    candidates.append(app_data_dir() / "models" / "translation" / model_name)
    return candidates


def custom_model_folder(model_dir: str) -> Path | None:
    """設定裡自訂的模型資料夾（沒設定回傳 None）；相對路徑從程式資料夾算起。"""
    if not model_dir:
        return None
    configured = Path(model_dir).expanduser()
    return configured if configured.is_absolute() else _app_roots()[-1] / configured


def model_folders(model_dir: str = "") -> list[Path]:
    """放模型的資料夾（每個模型一個子資料夾，裡面放 ``.gguf``），給使用者看、也給掃描用：
    程式資料夾的 ``models/translation``、使用者資料夾的 ``models/translation``、設定裡自訂的資料夾（有設定才有）。"""
    root = _app_roots()[-1]  # 安裝版是 exe 所在的資料夾（不是 _internal），從原始碼執行是專案資料夾
    folders = [root / "models" / "translation", app_data_dir() / "models" / "translation"]
    custom = custom_model_folder(model_dir)
    if custom is not None:
        folders.append(custom)
    unique: list[Path] = []
    for folder in folders:
        if all(str(folder).lower() != str(other).lower() for other in unique):
            unique.append(folder)
    return unique


class FoundModel(NamedTuple):
    name: str  # 資料夾名稱，也是 ``model_name`` 設定值
    folder: Path
    file: Path  # 資料夾裡的 .gguf（分片的取第一片）
    size: int  # 位元組；分片的只算找到的那一片


def discover_models(model_dir: str = "") -> list[FoundModel]:
    """掃描 ``model_folders`` 底下每個含 ``.gguf`` 的子資料夾；自訂的資料夾本身若直接放著 ``.gguf``，也算一個模型
    （名稱取資料夾名稱）。同名的只算先找到的那個。"""
    found: dict[str, FoundModel] = {}

    def add(name: str, folder: Path) -> None:
        gguf = llama_backend.find_gguf(folder)
        if gguf is not None and name not in found:
            try:
                size = gguf.stat().st_size
            except OSError:
                size = 0
            found[name] = FoundModel(name, folder, gguf, size)

    custom = custom_model_folder(model_dir)
    for folder in model_folders(model_dir):
        if not folder.is_dir():
            continue
        if folder == custom:
            add(folder.name, folder)  # 自訂資料夾本身就是一個模型資料夾（直接放著 .gguf）
        try:
            children = sorted(child for child in folder.iterdir() if child.is_dir())
        except OSError:
            continue
        for child in children:
            add(child.name, child)
    return list(found.values())


def find_model_file(model_name: str, model_dir: str = "") -> Path | None:
    """這個名稱的模型實際會讀的 GGUF 檔；找不到回傳 None。"""
    for directory in model_search_paths(model_name, model_dir):
        gguf = llama_backend.find_gguf(directory)
        if gguf is not None:
            return gguf
    return None


# ----------------------------------------------------------------------------- 執行緒
GPU_AUTO = "auto"
GPU_CPU = "cpu"


def gpu_choice(settings: dict | None) -> str:
    """使用者選的運算裝置：``auto``（有可用的 GPU 就用）、``cpu``（不用 GPU），或指定的 GPU 名稱（設定 ``gpu_device``）。

    舊版只有 ``gpu_layers``：0 = 只用 CPU。設定裡沒有 ``gpu_device`` 時還是照這個意思走（載入設定檔時會一次轉成 ``gpu_device``）。"""
    settings = settings or {}
    choice = str(settings.get("gpu_device") or "").strip()
    if not choice and "gpu_device" not in settings:
        try:
            return GPU_CPU if int(settings.get("gpu_layers", -1)) == 0 else GPU_AUTO
        except (TypeError, ValueError):
            return GPU_AUTO
    if not choice or choice.lower() == GPU_AUTO:
        return GPU_AUTO
    return GPU_CPU if choice.lower() == GPU_CPU else choice


def default_threads(setting: int = 0) -> int:
    return setting if setting > 0 else min(DEFAULT_MAX_THREADS, os.cpu_count() or 1)


# ----------------------------------------------------------------------------- 擬聲詞防呆
# 提示詞要求模型：原文只是音效、擬聲詞、笑聲、嘆息這類沒有實際詞義的聲音時，不要翻譯，只回這個標記。
SKIP_MARKER = "[SKIP]"
# 有括號的標記（[SKIP]、【skip】）不分大小寫；沒括號時只認全大寫的 SKIP，
# 因為小寫的 skip 可能是模型把「スキ」翻成英文的失敗結果，那要算翻譯失敗而不是略過。
_SKIP_BRACKETED_RE = re.compile(r"^[\[【(（<]\s*skip\s*[\]】)）>]$", re.IGNORECASE)
_SKIP_BARE_RE = re.compile(r"^SKIP$")


def is_skip_output(output: str) -> bool:
    """模型是否回報「這句只有擬聲詞，不翻」。"""
    value = re.sub(r"[.。!！\s]+$", "", (output or "").strip())
    return bool(_SKIP_BRACKETED_RE.match(value) or _SKIP_BARE_RE.match(value))


# 譯文只剩狀聲詞：小模型常把音效「翻」成中文的擬聲詞（砰、汪汪、哈哈、滴答滴答），這種譯文沒有資訊量。
# 只收擬聲用字；單獨一個「嗯」「啊」「哦」這類應答、感嘆詞是真的有意思（うん＝嗯），不在其中。
_SOUND_CHARS = frozenset(
    "汪喵砰啪咚噹叮滴答嘎咕嚕轟隆喀嚓哐啷嗖咻嘟嘀嗶啵咔叭咩哞吱喳啾嗡呸哧嗷吼噗嘩啦哈呵嘻嘿哼嗚呼噠唰沙嘰咯呱嘯鈴"
)
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def is_onomatopoeia_output(output: str) -> bool:
    """譯文是不是只有擬聲詞：去掉標點後空白、全是擬聲用字，或同一個字連續三次以上（啊啊啊啊）。"""
    letters = _NON_WORD_RE.sub("", output or "")
    if not output or not output.strip():
        return False
    if not letters:
        return True  # 只剩標點（原文只是「Ughh...」這類聲音）
    return all(char in _SOUND_CHARS for char in letters) or bool(_REPEATED_CHAR_RE.fullmatch(letters))


_JA_PUNCT_RE = re.compile(r"[\s!?！？、。，,.「」『』（）()]+")
_JA_SOUND_MARKS = frozenset("ーッっ〜～…ぁぃぅぇぉゃゅょァィゥェォャュョ")
_JA_REPEAT_RE = re.compile(r"(.{1,3})\1")


def looks_like_sound_ja(text: str) -> bool:
    """日文原文的長相像不像音效：不含漢字，而且有長音／促音／小假名／刪節號，或有重複（ドキドキ、ははは），
    或全是片假名。「またね」「ただいま」「愛してる」這類真正的話不符合，模型就算回報「只有擬聲詞」也不採信。"""
    value = _JA_PUNCT_RE.sub("", text or "")
    if not value:
        return True
    if has_han(value):
        return False
    if any(char in _JA_SOUND_MARKS for char in value) or _JA_REPEAT_RE.search(value):
        return True
    return all("\u30a0" <= char <= "\u30ff" for char in value)


# ----------------------------------------------------------------------------- 譯文檢查
def looks_untranslated(source: str, output: str) -> bool:
    """日文來源的輸出不像中文：仍以假名為主（抄回原文），或完全沒有漢字（例如把「スキ」譯成英文 skip）。"""
    if not has_kana(source) or not output:
        return False
    kana = sum(has_kana(char) for char in output)
    if kana / max(1, len(output)) > 0.5:
        return True
    return not has_han(output)


def looks_not_chinese(source: str, output: str) -> bool:
    """譯文夾雜了不該有的字：日文假名、韓文／俄文等外文、原文沒有的英文單字、每個字都隔著空格或 ``######``、``!");`` 這類亂碼。

    小模型偶爾會寫出「明天見ます」「계속你的名字」「數 NIGHT SKY 中的星星」「淚 的 數 越 總 勝 達」。
    這種結果值得再翻一次（改用取樣，貪婪解碼重翻會得到一模一樣的結果）；
    再翻還是不行就照原樣輸出，不算翻譯失敗。
    """
    return (
        has_kana(output)
        or has_foreign_script(output)
        or has_spaced_out_chars(output)
        or has_symbol_junk(output, source)
        or bool(unexpected_latin_words(source, output))
        or bool(untranslated_english_words(source, output))
        or chinese.is_meta_reply(output)
    )


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
        return looks_untranslated(source, output) or chinese.has_runaway_repetition(output, source) or chinese.is_meta_reply(output)

    # ----------------------------------------------------------- 擬聲詞防呆
    def sound_skip_enabled(self) -> bool:
        """設定 ``skip_sounds``（預設開）：只有擬聲詞（音效、笑聲、狀聲詞）的句子不翻譯也不顯示。"""
        return bool(self.config.get("skip_sounds", True))

    @staticmethod
    def sound_skip_allowed(text: str, language: str) -> bool:
        """這句原文長得像音效才允許略過；日文的一般短句（またね、はい）即使被模型誤判也照翻。"""
        return language != "ja" or looks_like_sound_ja(text)

    def cache_namespace(self, source: str | None) -> str | None:
        return source if self.sound_skip_enabled() else f"{source or ''}|nosound"

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
                if (
                    self.sound_skip_enabled()
                    and self.sound_skip_allowed(texts[index], language)
                    and (is_skip_output(output) or is_onomatopoeia_output(output))
                ):
                    # 只有擬聲詞的句子：原文原樣保留、不算失敗；譯文等於原文，所以不會顯示，也不會寫進上下文
                    continue
                if is_skip_output(output):  # 不該略過卻回了標記：不能把「[SKIP]」當譯文顯示
                    failed.add(index)
                    continue
                # 空輸出或只是把日文抄回來：保留原文，讓基底類別把它當作翻譯失敗（不寫入快取）
                if output and not self.output_looks_untranslated(texts[index], output):
                    tidy = chinese.normalize_punctuation(chinese.strip_skip_leak(output))
                    results[index] = self._to_target_script(tidy, target_language)
                else:
                    failed.add(index)
        self.last_failed_indices = failed
        return results
