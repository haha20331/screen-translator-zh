"""離線翻譯：Qwen2.5-Instruct（llama.cpp，GGUF）把英文、日文翻成繁體中文，不連網。

為什麼用指令模型而不是專用翻譯模型（評估結果見 PLAN.md 4.5）：
- ``Helsinki-NLP/opus-mt-ja-zh`` 並不存在；經英文轉譯（ja→en→zh）會產生幻覺。
- NLLB-200（600M / 1.3B）對字幕、遊戲對白這類短句常誤譯，詞表缺字還會輸出 ``⁇``，已不再使用。
- Qwen2.5-Instruct 加上示範（few-shot）後穩定且自然。

引擎是 llama.cpp（``llama_backend.py``）：CPU 上每句約 0.3 秒（1.5B Q4_K_M）、0.55 秒（3B Q4_K_M），
比先前的 CTranslate2 快 2～3 倍，品質沒有明顯差異（見 PLAN.md 9.2.7），所以只留這一個引擎。模型有大小之分
（0.5B／1.5B／3B／7B，見 ``OFFLINE_MODELS``），``model_name`` 設成其中一個；預設 ``auto``：有 GPU 版的 llama.cpp
且放了 7B 就用 7B，否則用 3B Q4_K_M（真實字幕上和 7B 差不多，只要一半的時間與記憶體），沒有下載 3B 時退回 1.5B Q4_K_M
（Apache-2.0，可以隨程式散布；3B 是 Qwen Research License，只能請使用者自己下載）。

三個可勾選的輔助功能（設定在 ``translation`` 區段，即時生效、不必重新載入模型）：
- ``use_context``：記住最近 ``context_size`` 句（預設 2）已翻好的原文與譯文，放在提示詞裡當前文，
  讓代名詞、省略的主語、同一首歌的語氣前後一致。
- ``lyrics_mode``：告訴模型這是歌詞，改用歌詞的系統提示與示範（較自然、有詩意，不逐字直譯）。
- ``retry_non_chinese``（預設開）：生成的當下就擋掉外文、原文沒有的英文字母與字間空格（``token_filter``），
  模型選不到這些字，大多數句子就不必重翻；仍然不是中文（殘留日文假名、不該略過的句子被回報略過）時，
  再改用取樣重翻；關閉就完全不干涉，只進模型一次，翻出什麼就輸出什麼（較快）。

流程：依文字判斷來源語言（假名→日文、拉丁字母→英文）→ 逐句交給模型 →
整理輸出 → 以 OpenCC ``s2twp`` 轉成台灣繁體（目標為簡體 ``zh`` 時則轉簡體）。
"""

from __future__ import annotations

import importlib.util
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

from app.logger import get_logger
from services.translation import llama_backend
from services.translation.base import TranslationError, register_translator
from services.translation.offline_common import (
    SKIP_MARKER,
    ChineseTargetTranslator,
    default_threads,
    is_skip_output,
    looks_not_chinese,
    model_search_paths,
)
from services.translation.token_filter import bans_for
from utils import chinese
from utils.language_utils import detect_source_language, has_kana, strip_symbol_junk

log = get_logger("translation.offline")


class OfflineModel(NamedTuple):
    name: str  # 資料夾名稱，也是 ``model_name`` 設定值
    label: str  # 設定畫面顯示的名稱


# 可選的模型，由小到大（同一個 Qwen2.5-Instruct 系列；取得方式見 scripts/fetch_translation_models.py）
OFFLINE_MODELS: tuple[OfflineModel, ...] = (
    OfflineModel("qwen2.5-0.5b-instruct-gguf-q8_0", "Qwen2.5 0.5B・Q8_0（最小最快，日文品質較差）"),
    OfflineModel("qwen2.5-1.5b-instruct-gguf-q4_k_m", "Qwen2.5 1.5B・Q4_K_M（最快的實用選擇，可隨程式散布）"),
    OfflineModel("qwen2.5-1.5b-instruct-gguf-q8_0", "Qwen2.5 1.5B・Q8_0（較準，約慢 1.4 倍）"),
    OfflineModel("qwen2.5-3b-instruct-gguf-q4_k_m", "Qwen2.5 3B・Q4_K_M（預設，較準，約慢 2 倍；非商用授權，需自行下載）"),
    OfflineModel("qwen2.5-7b-instruct-gguf-q4_k_m", "Qwen2.5 7B・Q4_K_M（最準，CPU 上很慢，建議有 GPU 版 llama.cpp 時用）"),
)
# 提示詞的版本：改了系統提示詞或示範，譯文就會不同；快取鍵帶著版本，舊提示詞留下的譯文不會被當成新的
PROMPT_VERSION = 3  # 3：生成時擋掉外文、原文沒有的英文字母與字間空格（見 token_filter），譯文可能與 2 不同
# 預設：真實字幕上和 7B 差不多準，速度與記憶體只要一半。授權是 Qwen Research License，不隨程式散布，使用者自己下載
DEFAULT_MODEL_NAME = "qwen2.5-3b-instruct-gguf-q4_k_m"
# 備用：Apache-2.0，可以隨程式散布；沒有下載預設的 3B 時用它
FALLBACK_MODEL_NAME = "qwen2.5-1.5b-instruct-gguf-q4_k_m"
LARGE_MODEL_NAME = "qwen2.5-7b-instruct-gguf-q4_k_m"
_MAX_HISTORY = 10
_SKIP_MARKERS = ("[SKIP]", "SKIP")  # 擬聲詞略過的回報標記；沒有英文的原文要擋英文字母，這個標記要放行
_SAMPLING_SEEDS = (1, 2)  # 重翻用的取樣種子：最多再翻兩次
_PLACEHOLDER_RE = re.compile(r"__ST_KEEP_\d+__")


# 防呆：只有音效、擬聲詞、笑聲、嘆息的句子沒有可翻的內容，硬翻只會得到亂寫的句子
_SKIP_RULE = (
    f" If the text is only a sound effect, onomatopoeia, laughter, sigh, scream or other meaningless "
    f"vocalization (for example ドン, ガシャーン, ワンワン, ははは, Boom, Woof, Haha), do not turn it into Chinese "
    f"sound words such as 砰 or 汪汪; output exactly {SKIP_MARKER} instead."
)
# 日文常省略主詞，直譯成中文時小模型愛自己補「我」「你」（会いたい→我想見你）；原文沒有主詞，譯文也不要加
_NO_SUBJECT_RULE = (
    " Do not add a subject or pronoun (such as 我, 你, 他, 我們) that the original text does not have: "
    "if the original leaves out the subject, leave it out of the translation too."
)
_BASE_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into Traditional Chinese "
    "as used in Taiwan. Output only the translation, with no explanations." + _NO_SUBJECT_RULE
)
_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + _SKIP_RULE
_LANGUAGE_NAMES = {"ja": "Japanese", "en": "English"}
# 示範能大幅降低小模型回傳原文、加註解的機率；內容刻意涵蓋介面、遊戲、對白三類。
_SHOTS: dict[str, list[tuple[str, str]]] = {
    "ja": [
        ("今日はいい天気ですね。", "今天天氣真好呢。"),
        ("ファイルを保存しますか？", "要儲存檔案嗎？"),
        ("HPが50回復した！", "恢復了 50 點 HP！"),
        # 原文沒有主詞，譯文也沒有；原文有主詞就照翻
        ("疲れたから早く帰りたい", "累了，想早點回家"),
        ("私は学生です。", "我是學生。"),
        ("ドーン！ガシャーン！", SKIP_MARKER),
        # 短而且只有假名的句子也可能是真的話（問候、告白）：用對照的示範避免被誤判成音效
        ("おかえり", "歡迎回來"),
        ("ワンワン", SKIP_MARKER),
        ("好きだよ", "喜歡喔"),
        ("ふふっ", SKIP_MARKER),
    ],
    "en": [
        ("Press any key to continue", "按任意鍵繼續"),
        ("You found a rusty key.", "你找到了一把生鏽的鑰匙。"),
        ("Want some coffee?", "要喝杯咖啡嗎？"),
        ("I am a student.", "我是學生。"),
        ("Save changes before closing?", "關閉前要儲存變更嗎？"),
        ("Boom! Crash!", SKIP_MARKER),
        ("See you soon", "待會見"),
        ("Woof woof", SKIP_MARKER),
        ("Haha", SKIP_MARKER),
    ],
}


_BASE_LYRICS_SYSTEM_PROMPT = (
    "You are a professional translator of song lyrics. The user's text is one line from a song. "
    "Translate it into natural, poetic Traditional Chinese as used in Taiwan, keeping the meaning, "
    "emotion and imagery instead of translating word by word. The line may be a fragment. "
    "Output only the translation, with no explanations." + _NO_SUBJECT_RULE
)
_LYRICS_SYSTEM_PROMPT = _BASE_LYRICS_SYSTEM_PROMPT + _SKIP_RULE
# 歌詞的示範：片段式、有意象的句子，讓模型知道可以意譯、不必補完整句
_LYRICS_SHOTS: dict[str, list[tuple[str, str]]] = {
    "ja": [
        ("君の声が聞こえる", "聽得見你的聲音"),
        ("星空の下で踊ろう", "在星空下共舞吧"),
        ("僕は君を忘れない", "我不會忘記你"),
        ("明日になれば忘れてしまうの？", "到了明天，就會忘記了嗎？"),
        ("ラララ…", SKIP_MARKER),
        ("ふふふっ", SKIP_MARKER),
    ],
    "en": [
        ("I will always love you", "我會永遠愛著你"),
        ("Hold me close and never let go", "緊緊抱住我，永遠不要放手"),
        ("We are dancing in the moonlight", "我們在月光下翩翩起舞"),
        ("Dancing in the moonlight", "在月光下起舞"),
        ("Na na na na", SKIP_MARKER),
    ],
}


def static_prompt(language: str, lyrics: bool = False, skip_sounds: bool = True) -> str:
    """固定前綴（system + 示範），每次翻譯都一樣；結尾剛好在 ``<|im_end|>\\n`` 的分詞邊界上。

    交給 CTranslate2 的 ``static_prompt``，這段的注意力快取只算一次（實測每句快約 40%）。
    ``lyrics`` 為真時換成歌詞用的系統提示與示範；``skip_sounds`` 為假時拿掉「只有擬聲詞就回報略過」的規則與示範。
    """
    name = _LANGUAGE_NAMES[language]
    if lyrics:
        system = _LYRICS_SYSTEM_PROMPT if skip_sounds else _BASE_LYRICS_SYSTEM_PROMPT
    else:
        system = _SYSTEM_PROMPT if skip_sounds else _BASE_SYSTEM_PROMPT
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for source, target in (_LYRICS_SHOTS if lyrics else _SHOTS)[language]:
        if target == SKIP_MARKER and not skip_sounds:
            continue
        parts.append(f"<|im_start|>user\nTranslate this {name} text:\n{source}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{target}<|im_end|>\n")
    return "".join(parts)


def user_prompt(text: str, language: str) -> str:
    """每句不同的部分：待翻譯文字，並留下 assistant 開頭讓模型續寫。"""
    name = _LANGUAGE_NAMES[language]
    return f"<|im_start|>user\nTranslate this {name} text:\n{text}<|im_end|>\n<|im_start|>assistant\n"


def llama_model_available(name: str, model_dir: str = "") -> bool:
    """llama.cpp 與這個 GGUF 模型都在（設定畫面用來決定選項能不能選）。"""
    return llama_backend.available() and any(
        llama_backend.find_gguf(directory) is not None for directory in model_search_paths(name, model_dir)
    )


def model_tag(name: str) -> str:
    """快取鍵用的短名稱：qwen2.5-1.5b-instruct-gguf-q4_k_m → 1.5b-q4_k_m。"""
    return name.removeprefix("qwen2.5-").replace("-instruct-gguf-", "-")

class ContextEntry(NamedTuple):
    source: str
    translation: str
    language: str
    timestamp: float


def history_prompt(entries: list[ContextEntry]) -> str:
    """前文：先前翻好的句子，照時間順序接在示範後面，當成更早的幾輪對話。

    用「已完成的對話輪」而不是「請參考以下前文」的說明文字：小模型看到說明容易連前文一起翻，
    對話輪則會自然接續同樣的格式，只翻最後一句。
    """
    parts = []
    for entry in entries:
        name = _LANGUAGE_NAMES[entry.language]
        parts.append(f"<|im_start|>user\nTranslate this {name} text:\n{entry.source}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{entry.translation}<|im_end|>\n")
    return "".join(parts)


def build_prompt(
    text: str, language: str, history: list[ContextEntry] | None = None, lyrics: bool = False, skip_sounds: bool = True
) -> str:
    """完整的 Qwen ChatML 提示詞（固定前綴 + 前文 + 待翻譯文字）。"""
    return static_prompt(language, lyrics, skip_sounds) + history_prompt(history or []) + user_prompt(text, language)


def clean_output(text: str) -> str:
    """只留第一個非空行（小模型偶爾會在後面附註解），並去掉頭尾空白。"""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


# ----------------------------------------------------------------------------- translator
class OfflineTranslator(ChineseTargetTranslator):
    name = "offline"
    config_key = "offline"

    def __init__(self, config_section: dict, cache=None, api_key: str = "") -> None:
        super().__init__(config_section, cache, api_key)
        self._model_lock = threading.Lock()
        self._llama: llama_backend.LlamaCppEngine | None = None
        # 最近翻好的句子（上下文功能）。保留比 context_size 多一些，調大設定時馬上有前文可用
        self._history: deque[ContextEntry] = deque(maxlen=_MAX_HISTORY)
        self._history_lock = threading.Lock()
        self._resolved_model: str | None = None
        self.device = ""
        self.model_name = ""  # 實際載入的模型（auto 模式下可能和設定不同）

    # ------------------------------------------------------------- availability
    @classmethod
    def availability_reason(cls) -> str | None:
        if importlib.util.find_spec("opencc") is None:
            return "離線翻譯依賴不可用：請安裝 opencc（pip install opencc）"
        if not llama_backend.available():
            return "離線翻譯依賴不可用：請安裝 llama-cpp-python（python -m pip install -r requirements-core.txt）"
        return None

    # ------------------------------------------------------------- 模型選擇
    def resolved_model_name(self) -> str:
        """實際要用的模型名稱。``auto``：有 GPU 版 llama.cpp 且放了 7B 就用 7B；否則用預設的 3B，
        沒有 3B 就退回 1.5B；兩個都沒有時回傳預設的 3B（找不到模型的錯誤訊息才會指到推薦的那個）。"""
        if self._resolved_model is not None:
            return self._resolved_model
        settings = self._settings()
        name = str(settings.get("model_name") or "auto")
        if name == "auto":
            model_dir = str(settings.get("model_dir") or "")
            if llama_backend.gpu_offload_supported() and llama_model_available(LARGE_MODEL_NAME, model_dir):
                name = LARGE_MODEL_NAME
            elif not llama_model_available(DEFAULT_MODEL_NAME, model_dir) and llama_model_available(FALLBACK_MODEL_NAME, model_dir):
                name = FALLBACK_MODEL_NAME
            else:
                name = DEFAULT_MODEL_NAME
        self._resolved_model = name
        return name

    def llama_model_path(self) -> Path:
        """GGUF 檔的位置；找不到就丟 TranslationError，並列出找過的資料夾與下載指令。"""
        name = self.resolved_model_name()
        model_dir = str(self._settings().get("model_dir") or "")
        searched = model_search_paths(name, model_dir)
        for directory in searched:
            found = llama_backend.find_gguf(directory)
            if found is not None:
                return found
        listing = "\n".join(f"  {path}" for path in searched)
        raise TranslationError(
            f"找不到離線翻譯模型「{name}」。請執行 scripts/fetch_translation_models.py --model {name}，"
            f"或把 .gguf 檔放到下列任一資料夾：\n{listing}"
        )

    def gpu_layers(self) -> int:
        """要交給 GPU 的層數：設定 -1（預設）= 有 GPU 版的 llama.cpp 就全部，否則 0（只用 CPU）。"""
        try:
            setting = int(self._settings().get("gpu_layers", -1))
        except (TypeError, ValueError):
            setting = -1
        if setting < 0:
            return -1 if llama_backend.gpu_offload_supported() else 0
        return setting

    # ------------------------------------------------------------- 上下文與歌詞模式
    def context_enabled(self) -> bool:
        return bool(self.config.get("use_context", False))

    def lyrics_enabled(self) -> bool:
        return bool(self.config.get("lyrics_mode", False))

    def retry_enabled(self) -> bool:
        """設定 ``retry_non_chinese``（預設開）：譯文不是中文時重翻一次。"""
        return bool(self.config.get("retry_non_chinese", True))

    def output_looks_untranslated(self, source: str, output: str) -> bool:
        if self.retry_enabled():
            return super().output_looks_untranslated(source, output)
        # 只進模型一次、翻出什麼就輸出什麼：不是中文也照顯示，只擋「原樣抄回原文」與失控重複
        echoed = has_kana(source) and output.strip() == source.strip()
        return echoed or chinese.has_runaway_repetition(output, source)

    def context_size(self) -> int:
        try:
            size = int(self.config.get("context_size", 2))
        except (TypeError, ValueError):
            size = 2
        return max(1, min(_MAX_HISTORY, size))

    def cache_namespace(self, source: str | None) -> str | None:
        """模型、歌詞模式、上下文、擬聲詞略過、不重翻都會改變譯文，快取要分開放，否則切換設定後看到的還是舊譯文。"""
        tags = [
            tag
            for tag, on in (
                ("lyrics", self.lyrics_enabled()),
                ("ctx", self.context_enabled()),
                ("nosound", not self.sound_skip_enabled()),
                ("noretry", not self.retry_enabled()),
            )
            if on
        ]
        tags.append(f"p{PROMPT_VERSION}")
        tags.append(model_tag(self.resolved_model_name()))  # 不同大小、不同量化的模型譯文不同
        return "|".join([source or "", *tags])

    def model_label(self) -> str:
        return f"{self.name}/{self.resolved_model_name()}"

    def reset_context(self) -> None:
        with self._history_lock:
            self._history.clear()

    def _context_for(self, text: str) -> list[ContextEntry]:
        """要放進提示詞的前文：最近 N 句，略過太舊的與句子本身。"""
        if not self.context_enabled():
            return []
        ttl = float(self.config.get("context_ttl_seconds", 120))
        now = time.time()
        with self._history_lock:
            recent = [e for e in self._history if now - e.timestamp <= ttl and e.source != text]
        return recent[-self.context_size():]

    def _remember(self, texts: list[str], results: list[str]) -> None:
        """把這一批翻好的句子記下來，給下一批當前文；翻失敗（譯文等於原文）的不記。"""
        han_only_as = self._settings().get("han_only_as", "ja") or None
        now = time.time()
        with self._history_lock:
            for text, result in zip(texts, results):
                source = _PLACEHOLDER_RE.sub("", text).strip()
                translation = _PLACEHOLDER_RE.sub("", result).strip()
                language = detect_source_language(source, han_only_as=han_only_as)
                if not source or not translation or translation == source or language not in self.source_languages:
                    continue
                if self._history and self._history[-1].source == source:
                    self._history.pop()  # 同一句字幕每輪重新辨識：只留最新一份，不重複塞進前文
                self._history.append(ContextEntry(source, translation, language, now))

    def translate(self, texts: list[str], source_language: str | None, target_language: str) -> list[str]:
        results = super().translate(texts, source_language, target_language)
        if self.context_enabled():
            self._remember(texts, results)
        return results

    # ------------------------------------------------------------- model loading
    def _load(self) -> None:
        if self._llama is not None:
            return
        if not chinese.available():  # 繁體是必要需求，沒有 OpenCC 時不應該悄悄輸出簡體
            raise TranslationError("缺少 OpenCC，無法輸出繁體中文：請安裝 opencc（pip install opencc）")
        reason = self.availability_reason()
        if reason is not None:
            raise TranslationError(reason)
        path = self.llama_model_path()
        threads = default_threads(int(self._settings().get("threads", 0)))
        layers = self.gpu_layers()
        log.info("載入離線翻譯模型 %s（threads=%d, gpu_layers=%d）", path, threads, layers)
        try:
            self._llama = llama_backend.LlamaCppEngine(path, threads, gpu_layers=layers)
        except Exception as exc:  # 檔案損毀、記憶體不足等
            raise TranslationError(f"載入 {path.name} 失敗：{exc}") from exc
        self.device = "gpu" if self._llama.gpu_layers else "cpu"
        self.model_name = path.parent.name

    def warmup(self) -> None:
        """載入模型、分類詞彙表並翻一句，讓第一次真正翻譯不必等模型載入。"""
        with self._model_lock:
            self._load()
            prepare = getattr(self._llama, "prepare_token_filter", None)
            if callable(prepare) and self.retry_enabled():
                prepare()
        # 日文與英文各翻一句：兩種語言的固定前綴（系統提示 + 示範，約 300 個 token）各要算一次，
        # 3B 約 3 秒、7B 約 7 秒；在這裡（背景預熱）先算好，第一句字幕才不必等
        self._generate(["Hello"], "en")
        self._generate(["こんにちは"], "ja")

    # ------------------------------------------------------------- inference
    def _generate(self, texts: list[str], language: str) -> list[str]:
        with self._model_lock:
            self._load()
            lyrics = self.lyrics_enabled()
            skip = self.sound_skip_enabled()
            contexts = [self._context_for(text) for text in texts]
            outputs = self._run(texts, language, lyrics, contexts, skip)
            if not self.retry_enabled():
                return outputs  # 只進模型一次：翻出什麼就輸出什麼
            if skip:
                # 模型回報「只有擬聲詞」但原文看起來是一般的話（またね、ただいま）：改用沒有略過規則的提示重翻
                vetoed = [
                    index
                    for index, output in enumerate(outputs)
                    if is_skip_output(output) and not self.sound_skip_allowed(texts[index], language)
                ]
                if vetoed:
                    redone = self._run([texts[i] for i in vetoed], language, lyrics, [contexts[i] for i in vetoed], False)
                    for index, output in zip(vetoed, redone):
                        outputs[index] = output
            if lyrics or any(contexts):
                # 歌詞提示或前文偶爾讓小模型把日文原樣留在譯文裡（「強さ」「進むと」）；
                # 這幾句改用一般提示重翻，重翻的結果沒有假名才採用。
                leaked = [index for index, output in enumerate(outputs) if has_kana(output)]
                if leaked:
                    plain = self._run([texts[index] for index in leaked], language, False, [[] for _ in leaked], skip)
                    for index, output in zip(leaked, plain):
                        if not has_kana(output):
                            outputs[index] = output
            # 還是夾雜假名、韓文／俄文等外文、原文沒有的英文單字：貪婪解碼重翻只會得到一樣的結果，
            # 改用取樣（固定種子，可重現）再翻，最多兩次，採用第一個像中文的；都不行就照原樣輸出。
            for seed in _SAMPLING_SEEDS:
                bad = [
                    index
                    for index, output in enumerate(outputs)
                    if output and not is_skip_output(output) and looks_not_chinese(texts[index], output)
                ]
                if not bad:
                    break
                redone = self._run([texts[i] for i in bad], language, lyrics, [contexts[i] for i in bad], skip, seed=seed)
                for index, output in zip(bad, redone):
                    if output and not is_skip_output(output) and not looks_not_chinese(texts[index], output):
                        outputs[index] = output
            # 重翻也救不回來的亂碼符號（!");）：至少不要顯示出來。「只有擬聲詞」的回報標記 [SKIP] 本身有方括號，不能動
            return [output if is_skip_output(output) else strip_symbol_junk(output, text) for text, output in zip(texts, outputs)]

    def _run(
        self,
        texts: list[str],
        language: str,
        lyrics: bool,
        contexts: list[list[ContextEntry]],
        skip: bool = True,
        seed: int | None = None,
    ) -> list[str]:
        """逐句交給 llama.cpp：固定前綴（系統提示 + 示範）的 KV 快取由引擎保管，換語言時不必重算。
        ``seed`` 不是 None 時用取樣解碼（重翻用），否則貪婪解碼。

        「譯文不是中文時處理」（``retry_non_chinese``）開著時，生成當下就擋掉外文、原文沒有的英文字母與字間空格
        （``token_filter``），模型選不到這些詞，大多數句子就不必事後重翻；關著時完全不干涉模型的輸出。"""
        max_tokens = int(self._settings().get("max_new_tokens", 160))
        prefix = static_prompt(language, lyrics, skip)
        extra = {} if seed is None else {"seed": seed}
        guard = self.retry_enabled()
        if guard:
            extra["allow"] = _SKIP_MARKERS if skip else ()  # 「只有擬聲詞」的回報標記 [SKIP] 含英文字母，不能擋
        return [
            clean_output(
                self._llama.complete(
                    prefix + history_prompt(context) + user_prompt(text, language),
                    max_tokens,
                    prefix,
                    **extra,
                    **({"bans": bans_for(text)} if guard else {}),
                )
            )
            for text, context in zip(texts, contexts)
        ]


register_translator(OfflineTranslator)
