"""離線翻譯：Qwen2.5-Instruct（llama.cpp，GGUF）把英文、日文翻成繁體中文，不連網。

為什麼用指令模型而不是專用翻譯模型（評估結果見 PLAN.md 4.5）：
- ``Helsinki-NLP/opus-mt-ja-zh`` 並不存在；經英文轉譯（ja→en→zh）會產生幻覺。
- NLLB-200（600M / 1.3B）對字幕、遊戲對白這類短句常誤譯，詞表缺字還會輸出 ``⁇``，已不再使用。
- Qwen2.5-Instruct 加上示範（few-shot）後穩定且自然。

引擎是 llama.cpp（``llama_backend.py``）：CPU 上每句約 0.4 秒（1.5B Q8_0）、0.55 秒（3B Q4_K_M），
比先前的 CTranslate2 快 2～3 倍，品質沒有明顯差異（見 PLAN.md 9.2.7），所以只留這一個引擎。模型有大小之分
（1.5B／3B／4B／7B，見 ``OFFLINE_MODELS``），``model_name`` 設成其中一個；預設 ``auto``：有獨立顯示卡（可用顯示卡記憶體
至少 5.5 GB）且放了 7B 就用 7B，否則用 3B Q4_K_M（真實字幕上和 7B 差不多，只要一半的時間與記憶體），沒有下載 3B 時退回 1.5B Q8_0
（Apache-2.0，可以隨程式散布；3B 是 Qwen Research License，只能請使用者自己下載）。

三個可勾選的輔助功能（設定在 ``translation`` 區段，即時生效、不必重新載入模型）：
- ``use_context``（預設開）：記住最近 ``context_size`` 句（預設 2）已翻好的原文與譯文，放在提示詞裡當前文，
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
from services.translation.model_names import (
    answer_prefix,
    display_name,
    how_to_get,
    model_tag,
    prompt_style,
    strip_thinking,
)
from services.translation.offline_common import (
    GPU_AUTO,
    GPU_CPU,
    SKIP_MARKER,
    ChineseTargetTranslator,
    default_threads,
    gpu_choice,
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


# 可選的模型，由小到大：Qwen2.5-Instruct 的 1.5B Q8_0／3B Q4_K_M／7B Q4_K_M 與 Qwen3 4B Q4_K_M（取得方式見 scripts/fetch_translation_models.py）
OFFLINE_MODELS: tuple[OfflineModel, ...] = (
    OfflineModel("qwen2.5-1.5b-instruct-gguf-q8_0", "Qwen2.5 1.5B（約 1.9 GB）：最小最快，翻得還可以，電腦比較慢時用這個"),
    OfflineModel("qwen2.5-3b-instruct-gguf-q4_k_m", "Qwen2.5 3B（約 2.1 GB）：推薦，翻得準、速度也不慢，一般電腦夠用"),
    OfflineModel("qwen3-4b-gguf-q4_k_m", "Qwen3 4B（約 2.5 GB）：比 3B 再大一點，更準但更慢"),
    OfflineModel("qwen2.5-7b-instruct-gguf-q4_k_m", "Qwen2.5 7B（約 4.7 GB）：最準，但很慢、要約 7 GB 記憶體，有獨立顯示卡時才建議用"),
)
# 提示詞的版本：改了系統提示詞或示範，譯文就會不同；快取鍵帶著版本，舊提示詞留下的譯文不會被當成新的
PROMPT_VERSION = 5  # 5：Qwen2.5 系列改用中文提示詞（Qwen3 維持英文）、譯文後處理（頓號、新字體、觸摸、SKIP 混入、客套回覆）；4：歌詞提示詞要求用白話文、不要文言文；3：生成時擋掉外文、原文沒有的英文字母與字間空格（見 token_filter）
# 預設：真實字幕上和 7B 差不多準，速度與記憶體只要一半。授權是 Qwen Research License，不隨程式散布，使用者自己下載
DEFAULT_MODEL_NAME = "qwen2.5-3b-instruct-gguf-q4_k_m"
# 備用：Apache-2.0，可以隨程式散布；沒有下載預設的 3B 時用它
FALLBACK_MODEL_NAME = "qwen2.5-1.5b-instruct-gguf-q8_0"
LARGE_MODEL_NAME = "qwen2.5-7b-instruct-gguf-q4_k_m"
# auto 自動改用 7B 的條件之一：獨立顯示卡的可用顯示卡記憶體至少這麼多（7B Q4_K_M 約 4.7 GB，再加上快取與緩衝）。
# 內顯與 CPU 共用系統記憶體，7B 要 6～7 GB、又不會變快，所以內顯不算
LARGE_MODEL_MIN_GPU_BYTES = 5_500_000_000
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
    "emotion and imagery instead of translating word by word. "
    "Write in modern vernacular Chinese (白話文), the way people speak and write today; "
    "do not use classical or literary Chinese (文言文) and avoid archaic words such as 之, 乃, 吾, 汝, 爾, 兮, 矣, 焉. "
    "The line may be a fragment. "
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


# ----------------------------------------------------------------------------- 兩種提示詞風格
# 中文版：系統提示詞與每句使用者輪（「翻譯這句日文：」）都用中文，示範同英文版。Qwen2.5 是以中文為主的模型，
# 盲測（PROMPT_TUNING.md）：3B 比英文版好（歌詞 62:31 與 55:28，通順度 +0.4～0.6、保真度 +0.15～0.3），台詞持平；
# Qwen3（盲測用的是 1.7B）的台詞反而變差、歌詞稍好，整體沒有差別，所以 Qwen3 維持英文版。
_ZH_SKIP_RULE = (
    f" 如果文字只有音效、擬聲詞、笑聲、嘆息、尖叫這類沒有意義的聲音（例如 ドン、ガシャーン、ワンワン、ははは、Boom、Woof、Haha），"
    f"不要翻成「砰」「汪汪」之類的中文擬聲詞，直接輸出 {SKIP_MARKER}。"
)
_ZH_NO_SUBJECT_RULE = "原文沒有主詞時，譯文也不要加主詞（例如不要自己補「我」「你」「他」）。"
_ZH_BASE_SYSTEM_PROMPT = (
    "你是專業的翻譯。請把使用者給的文字翻譯成自然流暢的台灣繁體中文，完整保留原意，只輸出譯文，不要任何說明。" + _ZH_NO_SUBJECT_RULE
)
_ZH_BASE_LYRICS_SYSTEM_PROMPT = (
    "你是專業的歌詞翻譯。使用者給的是一句歌詞（可能只是片段）。請翻成自然、有詩意的台灣繁體中文，保留原意、情感與意象，"
    "不要逐字直譯；用現代白話文，不要用文言文，避免「之、乃、吾、汝、爾、兮、矣、焉」這類古字。只輸出譯文，不要任何說明。"
    + _ZH_NO_SUBJECT_RULE
)


class PromptStyle(NamedTuple):
    system: str  # 一般字幕的系統提示詞（不含略過擬聲詞的規則）
    system_lyrics: str  # 歌詞模式
    skip_rule: str  # 開著「略過擬聲詞」時接在系統提示詞後面
    user_format: str  # 每句使用者輪的格式（示範、前文、待翻譯句都用同一個）；{name} 是語言名稱
    names: dict[str, str]


PROMPT_STYLES: dict[str, PromptStyle] = {
    "en": PromptStyle(
        _BASE_SYSTEM_PROMPT, _BASE_LYRICS_SYSTEM_PROMPT, _SKIP_RULE, "Translate this {name} text:\n{text}", _LANGUAGE_NAMES
    ),
    "zh": PromptStyle(
        _ZH_BASE_SYSTEM_PROMPT,
        _ZH_BASE_LYRICS_SYSTEM_PROMPT,
        _ZH_SKIP_RULE,
        "翻譯這句{name}：\n{text}",
        {"ja": "日文", "en": "英文"},
    ),
}
DEFAULT_PROMPT_STYLE = "zh"


def _user_turn(style: PromptStyle, text: str, language: str) -> str:
    return style.user_format.format(name=style.names[language], text=text)


def static_prompt(language: str, lyrics: bool = False, skip_sounds: bool = True, style: str = DEFAULT_PROMPT_STYLE) -> str:
    """固定前綴（system + 示範），每次翻譯都一樣；結尾剛好在 ``<|im_end|>\\n`` 的分詞邊界上。

    交給 CTranslate2 的 ``static_prompt``，這段的注意力快取只算一次（實測每句快約 40%）。
    ``lyrics`` 為真時換成歌詞用的系統提示與示範；``skip_sounds`` 為假時拿掉「只有擬聲詞就回報略過」的規則與示範。
    """
    chosen = PROMPT_STYLES[style]
    system = chosen.system_lyrics if lyrics else chosen.system
    if skip_sounds:
        system += chosen.skip_rule
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for source, target in (_LYRICS_SHOTS if lyrics else _SHOTS)[language]:
        if target == SKIP_MARKER and not skip_sounds:
            continue
        parts.append(f"<|im_start|>user\n{_user_turn(chosen, source, language)}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{target}<|im_end|>\n")
    return "".join(parts)


def user_prompt(text: str, language: str, answer_prefix: str = "", style: str = DEFAULT_PROMPT_STYLE) -> str:
    """每句不同的部分：待翻譯文字，並留下 assistant 開頭讓模型續寫。
    ``answer_prefix``：先幫模型寫好的回答開頭（Qwen3 非思考模式的空思考區塊）。"""
    turn = _user_turn(PROMPT_STYLES[style], text, language)
    return f"<|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n{answer_prefix}"


def llama_model_available(name: str, model_dir: str = "") -> bool:
    """llama.cpp 與這個 GGUF 模型都在（設定畫面用來決定選項能不能選）。"""
    return llama_backend.available() and any(
        llama_backend.find_gguf(directory) is not None for directory in model_search_paths(name, model_dir)
    )



class ContextEntry(NamedTuple):
    source: str
    translation: str
    language: str
    timestamp: float


def history_prompt(entries: list[ContextEntry], style: str = DEFAULT_PROMPT_STYLE) -> str:
    """前文：先前翻好的句子，照時間順序接在示範後面，當成更早的幾輪對話。

    用「已完成的對話輪」而不是「請參考以下前文」的說明文字：小模型看到說明容易連前文一起翻，
    對話輪則會自然接續同樣的格式，只翻最後一句。
    """
    chosen = PROMPT_STYLES[style]
    parts = []
    for entry in entries:
        parts.append(f"<|im_start|>user\n{_user_turn(chosen, entry.source, entry.language)}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{entry.translation}<|im_end|>\n")
    return "".join(parts)


def build_prompt(
    text: str,
    language: str,
    history: list[ContextEntry] | None = None,
    lyrics: bool = False,
    skip_sounds: bool = True,
    style: str = DEFAULT_PROMPT_STYLE,
) -> str:
    """完整的 Qwen ChatML 提示詞（固定前綴 + 前文 + 待翻譯文字）。"""
    return (
        static_prompt(language, lyrics, skip_sounds, style)
        + history_prompt(history or [], style)
        + user_prompt(text, language, style=style)
    )


def clean_output(text: str) -> str:
    """只留第一個非空行（小模型偶爾會在後面附註解），並去掉頭尾空白。Qwen3 自己寫出來的思考區塊也先去掉。"""
    for line in strip_thinking(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


# ----------------------------------------------------------------------------- translator
class OfflineTranslator(ChineseTargetTranslator):
    name = "offline"
    config_key = "offline"

    def __init__(self, config_section: dict, cache=None) -> None:
        super().__init__(config_section, cache)
        self._model_lock = threading.Lock()
        self._llama: llama_backend.LlamaCppEngine | None = None
        # 最近翻好的句子（上下文功能）。保留比 context_size 多一些，調大設定時馬上有前文可用
        self._history: deque[ContextEntry] = deque(maxlen=_MAX_HISTORY)
        self._history_lock = threading.Lock()
        self._resolved_model: str | None = None
        self.device = ""
        self.device_name = ""  # 使用者指定的 GPU 載入成功時，是哪一顆
        self.model_name = ""  # 實際載入的模型（auto 模式下可能和設定不同）
        self._warmups_running = 0  # 背景預熱的執行緒數（預熱期間 is_ready 是 False）
        self._warmup_count_lock = threading.Lock()

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
        """實際要用的模型名稱。``auto``：有獨立顯示卡（可用顯示卡記憶體夠）且放了 7B 就用 7B；否則用預設的 3B，
        沒有 3B 就退回 1.5B Q8_0；兩個都沒有時回傳預設的 3B（找不到模型的錯誤訊息才會指到推薦的那個）。"""
        if self._resolved_model is not None:
            return self._resolved_model
        settings = self._settings()
        name = str(settings.get("model_name") or "auto")
        if name == "auto":
            model_dir = str(settings.get("model_dir") or "")
            if llama_backend.discrete_gpu_free_bytes(self.gpu_choice()) >= LARGE_MODEL_MIN_GPU_BYTES and llama_model_available(LARGE_MODEL_NAME, model_dir):
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
            f"找不到離線翻譯模型「{name}」。{how_to_get(name)}，"
            f"或把 .gguf 檔放到下列任一資料夾：\n{listing}"
        )

    def gpu_choice(self) -> str:
        """設定 ``gpu_device``：``auto``、``cpu``，或指定的 GPU 名稱（見 ``offline_common.gpu_choice``）。"""
        return gpu_choice(self._settings())

    def gpu_layers(self) -> int:
        """要交給 GPU 的層數：選「只用 CPU」是 0；否則設定 -1（預設）= 有 GPU 版的 llama.cpp 就全部，沒有 GPU 版也是 0。"""
        if self.gpu_choice() == GPU_CPU:
            return 0
        try:
            setting = int(self._settings().get("gpu_layers", -1))
        except (TypeError, ValueError):
            setting = -1
        if setting <= 0:  # 0 是舊版「只用 CPU」的寫法；現在選 GPU 又留著 0 的話，當成全部
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
        return echoed or chinese.has_runaway_repetition(output, source) or chinese.is_meta_reply(output)

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

    def display_model_name(self) -> str:
        """給使用者看的短名稱（3B Q4_K_M、Qwen3 4B Q4_K_M），顯示載入進度時用。"""
        return display_name(self.resolved_model_name())

    def is_ready(self) -> bool:
        """模型已載入、而且沒有在預熱：現在翻譯不必等。"""
        with self._warmup_count_lock:
            warming = self._warmups_running > 0
        return self._llama is not None and not warming

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
        choice = self.gpu_choice()
        log.info(
            "載入離線翻譯模型 %s（threads=%d, gpu_layers=%d, 運算裝置=%s, llama.cpp=%s）",
            path, threads, layers, choice, llama_backend.library_description(),
        )
        extra = {} if bool(self._settings().get("cpu_fallback", True)) else {"cpu_fallback": False}
        if layers and choice not in (GPU_AUTO, GPU_CPU):
            extra["gpu_device"] = choice  # 使用者指定了哪一顆 GPU；沒指定就不傳，交給 llama.cpp 選
        try:
            self._llama = llama_backend.LlamaCppEngine(path, threads, gpu_layers=layers, **extra)
        except Exception as exc:  # 檔案損毀、記憶體不足等
            raise TranslationError(f"載入 {path.name} 失敗：{exc}") from exc
        # 只有安裝的 llama-cpp-python 真的支援 GPU，才算在 GPU 上（CPU 版收到 gpu_layers > 0 只是忽略）
        self.device = "gpu" if self._llama.gpu_layers and llama_backend.gpu_offload_supported() else "cpu"
        self.device_name = str(getattr(self._llama, "device_name", "") or "") if self.device == "gpu" else ""
        self.model_name = path.parent.name
        fell_back = " （要求用 GPU 但載入失敗，已改用 CPU）" if layers and self.device == "cpu" and llama_backend.gpu_offload_supported() else ""
        shown = "CPU" if self.device == "cpu" else f"GPU（{self.device_name}）" if self.device_name else "GPU"
        log.info("離線翻譯模型已載入：%s，裝置＝%s%s", path.name, shown, fell_back)

    def close(self) -> None:
        """釋放模型佔用的記憶體（速度測試一個接一個載入不同的模型時需要）；之後再翻譯會重新載入。"""
        with self._model_lock:
            engine, self._llama = self._llama, None
            if engine is not None:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()

    def warmup(self) -> None:
        """載入模型、分類詞彙表並翻一句，讓第一次真正翻譯不必等模型載入。"""
        with self._warmup_count_lock:
            self._warmups_running += 1
        try:
            with self._model_lock:
                self._load()
                prepare = getattr(self._llama, "prepare_token_filter", None)
                if callable(prepare) and self.retry_enabled():
                    prepare()
            # 日文與英文各翻一句：兩種語言的固定前綴（系統提示 + 示範，約 300 個 token）各要算一次，
            # 3B 約 3 秒、7B 約 7 秒；在這裡（背景預熱）先算好，第一句字幕才不必等
            self._generate(["Hello"], "en")
            self._generate(["こんにちは"], "ja")
        finally:
            with self._warmup_count_lock:
                self._warmups_running -= 1

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
        model = self.resolved_model_name()
        style = prompt_style(model)  # Qwen2.5 系列：中文提示詞；Qwen3：英文提示詞
        prefix = static_prompt(language, lyrics, skip, style)
        opening = answer_prefix(model)  # Qwen3：空的思考區塊，直接寫答案
        extra = {} if seed is None else {"seed": seed}
        guard = self.retry_enabled()
        if guard:
            extra["allow"] = _SKIP_MARKERS if skip else ()  # 「只有擬聲詞」的回報標記 [SKIP] 含英文字母，不能擋
        return [
            clean_output(
                self._llama.complete(
                    prefix + history_prompt(context, style) + user_prompt(text, language, opening, style),
                    max_tokens,
                    prefix,
                    **extra,
                    **({"bans": bans_for(text)} if guard else {}),
                )
            )
            for text, context in zip(texts, contexts)
        ]


register_translator(OfflineTranslator)
