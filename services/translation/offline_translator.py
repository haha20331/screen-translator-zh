"""離線翻譯（標準）：Qwen2.5-Instruct（CTranslate2 int8）把英文、日文翻成繁體中文，不連網。

為什麼用指令模型而不是專用翻譯模型（評估結果見 PLAN.md 4.5）：
- ``Helsinki-NLP/opus-mt-ja-zh`` 並不存在；經英文轉譯（ja→en→zh）會產生幻覺。
- NLLB-200（600M / 1.3B）對字幕、遊戲對白這類短句常誤譯，詞表缺字還會輸出 ``⁇``
  （它改作為「快速翻譯」，見 fast_translator.py）。
- Qwen2.5-1.5B-Instruct 加上示範（few-shot）後穩定且自然，CPU 約 0.5 秒/句。

依硬體自動選模型（``model_name = "auto"``，預設）：有 NVIDIA GPU（CUDA）時優先用較大的
7B 模型，載入失敗（例如顯示卡記憶體不足）就退到 1.5B，沒有 GPU 則用 CPU 的 1.5B。

兩個可勾選的輔助功能（設定在 ``translation`` 區段，即時生效、不必重新載入模型）：
- ``use_context``：記住最近 ``context_size`` 句（預設 2）已翻好的原文與譯文，放在提示詞裡當前文，
  讓代名詞、省略的主語、同一首歌的語氣前後一致。
- ``lyrics_mode``：告訴模型這是歌詞，改用歌詞的系統提示與示範（較自然、有詩意，不逐字直譯）。

流程：依文字判斷來源語言（假名→日文、拉丁字母→英文）→ 分組批次交給模型 →
整理輸出 → 以 OpenCC ``s2twp`` 轉成台灣繁體（目標為簡體 ``zh`` 時則轉簡體）。
"""

from __future__ import annotations

import importlib.util
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

from app.logger import get_logger
from services.translation.base import TranslationError, register_translator
from services.translation.ct2_common import (
    SKIP_MARKER,
    ChineseTargetTranslator,
    default_threads,
    find_model_dir,
    is_skip_output,
    model_search_paths,
    resolve_device,
)
from utils import chinese
from utils.language_utils import detect_source_language, has_kana

log = get_logger("translation.offline")

DEFAULT_MODEL_NAME = "qwen2.5-1.5b-instruct-ct2-int8"
LARGE_MODEL_NAME = "qwen2.5-7b-instruct-ct2-int8"
# auto 模式的候選：(模型名稱, 是否需要 CUDA)。依序嘗試，載入失敗就換下一個。
MODEL_TIERS: list[tuple[str, bool]] = [(LARGE_MODEL_NAME, True), (DEFAULT_MODEL_NAME, False)]
_BATCH_SIZE = 8
_MAX_HISTORY = 10
_PLACEHOLDER_RE = re.compile(r"__ST_KEEP_\d+__")

# 防呆：只有音效、擬聲詞、笑聲、嘆息的句子沒有可翻的內容，硬翻只會得到亂寫的句子
_SKIP_RULE = (
    f" If the text is only a sound effect, onomatopoeia, laughter, sigh, scream or other meaningless "
    f"vocalization (for example ドン, ガシャーン, ワンワン, ははは, Boom, Woof, Haha), do not turn it into Chinese "
    f"sound words such as 砰 or 汪汪; output exactly {SKIP_MARKER} instead."
)
_BASE_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into Traditional Chinese "
    "as used in Taiwan. Output only the translation, with no explanations."
)
_SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + _SKIP_RULE
_LANGUAGE_NAMES = {"ja": "Japanese", "en": "English"}
# 示範能大幅降低小模型回傳原文、加註解的機率；內容刻意涵蓋介面、遊戲、對白三類。
_SHOTS: dict[str, list[tuple[str, str]]] = {
    "ja": [
        ("今日はいい天気ですね。", "今天天氣真好呢。"),
        ("ファイルを保存しますか？", "要儲存檔案嗎？"),
        ("HPが50回復した！", "恢復了 50 點 HP！"),
        ("ドーン！ガシャーン！", SKIP_MARKER),
        # 短而且只有假名的句子也可能是真的話（問候、告白）：用對照的示範避免被誤判成音效
        ("おかえり", "歡迎回來"),
        ("ワンワン", SKIP_MARKER),
        ("好きだよ", "我喜歡你"),
        ("ふふっ", SKIP_MARKER),
    ],
    "en": [
        ("Press any key to continue", "按任意鍵繼續"),
        ("You found a rusty key.", "你找到了一把生鏽的鑰匙。"),
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
    "Output only the translation, with no explanations."
)
_LYRICS_SYSTEM_PROMPT = _BASE_LYRICS_SYSTEM_PROMPT + _SKIP_RULE
# 歌詞的示範：片段式、有意象的句子，讓模型知道可以意譯、不必補完整句
_LYRICS_SHOTS: dict[str, list[tuple[str, str]]] = {
    "ja": [
        ("君の声が聞こえる", "我聽見了你的聲音"),
        ("星空の下で踊ろう", "讓我們在星空下共舞吧"),
        ("明日になれば忘れてしまうの？", "到了明天，就會忘記了嗎？"),
        ("ラララ…", SKIP_MARKER),
        ("ふふふっ", SKIP_MARKER),
    ],
    "en": [
        ("I will always love you", "我會永遠愛著你"),
        ("Hold me close and never let go", "緊緊抱住我，永遠不要放手"),
        ("We are dancing in the moonlight", "我們在月光下翩翩起舞"),
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
        self._static_tokens: dict[tuple[str, bool, bool], list[str]] = {}
        # 最近翻好的句子（上下文功能）。保留比 context_size 多一些，調大設定時馬上有前文可用
        self._history: deque[ContextEntry] = deque(maxlen=_MAX_HISTORY)
        self._history_lock = threading.Lock()
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

    # ------------------------------------------------------------- 上下文與歌詞模式
    def context_enabled(self) -> bool:
        return bool(self.config.get("use_context", False))

    def lyrics_enabled(self) -> bool:
        return bool(self.config.get("lyrics_mode", False))

    def context_size(self) -> int:
        try:
            size = int(self.config.get("context_size", 2))
        except (TypeError, ValueError):
            size = 2
        return max(1, min(_MAX_HISTORY, size))

    def cache_namespace(self, source: str | None) -> str | None:
        """歌詞模式、上下文、擬聲詞略過都會改變譯文，快取要分開放，否則切換設定後看到的還是舊譯文。"""
        tags = [
            tag
            for tag, on in (
                ("lyrics", self.lyrics_enabled()),
                ("ctx", self.context_enabled()),
                ("nosound", not self.sound_skip_enabled()),
            )
            if on
        ]
        return "|".join([source or "", *tags]) if tags else source

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
            lyrics = self.lyrics_enabled()
            skip = self.sound_skip_enabled()
            contexts = [self._context_for(text) for text in texts]
            outputs = self._run(texts, language, lyrics, contexts, skip)
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
            return outputs

    def _run(
        self, texts: list[str], language: str, lyrics: bool, contexts: list[list[ContextEntry]], skip: bool = True
    ) -> list[str]:
        settings = self._settings()
        key = (language, lyrics, skip)
        if key not in self._static_tokens:
            self._static_tokens[key] = self._tokenizer.encode(
                static_prompt(language, lyrics, skip), add_special_tokens=False
            ).tokens
        batches = [
            self._tokenizer.encode(history_prompt(context) + user_prompt(text, language), add_special_tokens=False).tokens
            for text, context in zip(texts, contexts)
        ]
        results = self._generator.generate_batch(
            batches,
            static_prompt=self._static_tokens[key],
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
