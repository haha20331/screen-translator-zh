"""语言代码与显示名、各服务语言参数映射。"""

from __future__ import annotations

import re
import unicodedata

LANGUAGES = [
    ("auto", "自動檢測"),
    ("zh", "簡體中文"),
    ("zh-hant", "繁體中文"),
    ("en", "英語"),
    ("ja", "日語"),
    ("ko", "韓語"),
    ("fr", "法語"),
    ("de", "德語"),
    ("es", "西班牙語"),
    ("ru", "俄語"),
    ("pt", "葡萄牙語"),
    ("it", "義大利語"),
    ("vi", "越南語"),
    ("th", "泰語"),
    ("ar", "阿拉伯語"),
]

LANGUAGE_CODES = [code for code, _ in LANGUAGES]


_TECHNICAL_PATH_RE = re.compile(r"^[\w.@:+~-]+(?:[/\\][\w.@:+~-]+)+$")
_TECHNICAL_FILE_RE = re.compile(
    r"^[\w@+-]+\.(?:py|pyw|md|json|toml|ini|cfg|conf|spec|txt|ya?ml|"
    r"exe|dll|so|dylib|zip|7z|rar|png|jpe?g|gif|svg|html?|css|js|ts)$",
    re.IGNORECASE,
)
_HEX_ID_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_BRAND_LOGO_RE = re.compile(
    r"^(?:[0O]\s*)?(?:Claude|Cursor|GitHub|Stripe|Spotify|Bridgewater|Banner\s+Health)"
    r"(?:[®™])?$",
    re.IGNORECASE,
)
_BENCHMARK_NAME_RE = re.compile(
    r"(?:bench(?:mark)?|GDP(?:val)?|OSWorld|FrontierCode|"
    r"Humanity['’]?s Last Exam|Codex CLI|Gemini CLI)",
    re.IGNORECASE,
)


def has_kana(text: str) -> bool:
    """文字中是否含有日文假名（平假名、片假名，含半形片假名）。"""
    return any("\u3040" <= char <= "\u30ff" or "\uff66" <= char <= "\uff9f" for char in text)


def has_han(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


# 譯成中文時不該出現的文字：韓文、西里爾字母（俄文）、希臘、希伯來、阿拉伯、天城文（印地文）、泰文。
# Qwen2.5 這類多語模型偶爾會在中文裡夾一個韓文字或俄文字（「계속呼唤著你的名字」）。
_FOREIGN_SCRIPT_RANGES = (
    (0x0370, 0x03FF),  # 希臘
    (0x0400, 0x052F),  # 西里爾
    (0x0590, 0x05FF),  # 希伯來
    (0x0600, 0x06FF),  # 阿拉伯
    (0x0900, 0x097F),  # 天城文
    (0x0E00, 0x0E7F),  # 泰文
    (0x1100, 0x11FF),  # 韓文字母
    (0x3130, 0x318F),  # 韓文相容字母
    (0xAC00, 0xD7AF),  # 韓文音節
)
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def has_foreign_script(text: str) -> bool:
    """文字中有沒有韓文、俄文、阿拉伯文、泰文等不該出現在中文譯文裡的書寫系統。"""
    return any(low <= ord(char) <= high for char in text for low, high in _FOREIGN_SCRIPT_RANGES)


# 每個字中間都隔一個空格的中文（「淚 的 數 越 總 勝 達」）：小模型解碼失控時的典型輸出，正常的譯文不會這樣
_SPACED_OUT_RE = re.compile(r"(?:[\u3400-\u9fff] ){3,}[\u3400-\u9fff]")


def has_spaced_out_chars(text: str) -> bool:
    """有連續四個以上的漢字、每個之間都隔著一個空格（失控的輸出）。「恢復了 50 點 HP」這種中文與數字間的空格不算。"""
    return bool(_SPACED_OUT_RE.search(text or ""))


def unexpected_latin_words(source: str, output: str) -> list[str]:
    """譯文裡原文沒有的英文單字（3 個字母以上）。原文本來就有的（HP、Ctrl、Windows）不算；
    「HP」「MP」這種兩個字母的縮寫也不算。"""
    known = {word.lower() for word in _LATIN_WORD_RE.findall(source or "")}
    return [word for word in _LATIN_WORD_RE.findall(output or "") if word.lower() not in known]


def script_class(text: str) -> str:
    """文字的書寫系統：``ja``（含假名）、``han``（只有漢字，沒有假名）、``latin``、``other``（數字、符號）。"""
    value = text or ""
    if has_kana(value):
        return "ja"
    if has_han(value):
        return "han"
    if re.search(r"[A-Za-z]", value):
        return "latin"
    return "other"


def same_script_class(first: str, second: str) -> bool:
    """兩行文字是否屬於同一種書寫系統（數字、符號可以接在任何一種後面）。

    雙語、三語字幕常把日文、羅馬拼音、中文譯文疊成幾行；把它們併成一段交給翻譯，
    模型會把羅馬拼音音譯成一串怪字，還可能整段胡亂重複。
    日文被換行時，下一行可能沒有假名（「20年連続」）：含假名的行與純漢字行仍可合併，
    除非純漢字那行看得出是中文（那是附帶的中文譯文）。
    """
    first_class, second_class = script_class(first), script_class(second)
    if first_class == second_class or "other" in (first_class, second_class):
        return True
    if {first_class, second_class} == {"ja", "han"}:
        from utils import chinese

        return not chinese.looks_chinese(first if first_class == "han" else second)
    return False


# 羅馬拼音（Hepburn / 訓令式）的音節：（子音）+ 母音，或單獨的 n。促音的雙子音與 tch 先壓成單子音再比對。
_ROMAJI_SYLLABLE = r"(?:(?:ch|sh|ts|ky|gy|ny|hy|by|py|my|ry|j|[kgsztdnhbpmryw])?[aiueo]|n(?![aiueoy]))"
_ROMAJI_WORD_RE = re.compile(rf"(?:{_ROMAJI_SYLLABLE})+")
# 羅馬拼音的特徵：日文獨有的拼法（sh/ch/ts/拗音），或常見助詞；英文單字很少同時整句都符合音節規則又帶有這些特徵
_ROMAJI_CUE_RE = re.compile(r"sh|ch|ts|ky|gy|ny|hy|my|ry|by|py|(?<![a-z])(?:wa|ga|wo|ni|ka)(?![a-z])")


def is_romanized_japanese(text: str) -> bool:
    """這行拉丁字母是不是日文的羅馬拼音（歌詞字幕常附在日文下方）。

    羅馬拼音只是日文的讀音，翻成中文只會得到一串音譯的怪字，應該略過。
    條件偏保守，避免把英文句子（例如 ``Take me home``）誤判：
    至少兩個單字、幾乎每個單字都能拆成羅馬拼音音節，而且帶有日文獨有的拼法或助詞。
    """
    decomposed = unicodedata.normalize("NFD", text or "")
    # ō / ū 等長音符號：去掉組合用的長音符號，視為有特徵
    macron = "\u0304" in decomposed
    folded = "".join(char for char in decomposed if not unicodedata.combining(char)).lower()
    folded = folded.replace("'", "").replace("\u2019", "")
    if re.search(r"[^a-z\s,.!?~\-]", folded):
        return False
    words = re.findall(r"[a-z]+", folded)
    if len(words) < 2:
        return False

    def parses(word: str) -> bool:
        word = re.sub(r"tch", "ch", word)
        word = re.sub(r"([kgsztdbpjhmrfw])\1", r"\1", word)  # 促音：kk、tt、ss…
        return _ROMAJI_WORD_RE.fullmatch(word) is not None

    parsed = sum(parses(word) for word in words)
    # OCR 偶爾認錯一個字母（hakanai → hakanal）：單字夠多時容許一個壞字
    if parsed < len(words) - (1 if len(words) >= 4 else 0):
        return False
    return macron or _ROMAJI_CUE_RE.search(" ".join(words)) is not None


def detect_source_language(text: str, han_only_as: str | None = "ja") -> str | None:
    """判斷一段 OCR 文字的來源語言：``"ja"``、``"en"``，無法判斷（純數字、符號等）回傳 ``None``。

    - 含假名 → 日文（即使混有英文與漢字）。
    - 只有漢字、沒有假名 → 日文漢字與中文無法區分，依 ``han_only_as`` 決定
      （預設視為日文；傳入 ``None`` 表示不處理）。
    - 其餘有兩個以上連續拉丁字母 → 英文。
    """
    value = text or ""
    if has_kana(value):
        return "ja"
    if has_han(value):
        return han_only_as
    if re.search(r"[A-Za-z]{2,}", value):
        return "en"
    return None


def needs_translation(text: str, target_language: str, han_is_source: bool = False) -> bool:
    """Reject UI glyphs, technical identifiers and text already in the target script.

    ``han_is_source``: 翻譯器會把純漢字文字當成日文來源（日文漢字與中文無法區分）時為 True，
    此時漢字不再被視為「已經是中文」。含假名的文字不論此參數一律需要翻譯成中文。
    """

    value = (text or "").strip()
    if not value:
        return False
    letters = [char for char in value if char.isalpha()]
    if not letters or (len(letters) == 1 and letters[0].isascii()):
        return False
    if _TECHNICAL_PATH_RE.fullmatch(value) or _TECHNICAL_FILE_RE.fullmatch(value):
        return False
    if _HEX_ID_RE.fullmatch(value):
        return False
    if _BRAND_LOGO_RE.fullmatch(value):
        return False
    if value.casefold() == "xhigh":
        return False
    if is_romanized_japanese(value):
        return False
    # 表格里的基准/产品标识不是自然语言。翻它们既浪费请求，又容易把
    # SWE-Bench、GDPval-AA 之类弄成一坨假中文。
    latin_words = re.findall(r"[A-Za-z0-9]+", value)
    if len(value) <= 64 and len(latin_words) <= 5 and _BENCHMARK_NAME_RE.search(value):
        return False

    target = (target_language or "").lower()
    if target.startswith("zh") and has_kana(value):
        return True
    if target.startswith("zh") and han_is_source and has_han(value):
        return True
    if target.startswith("zh") and has_han(value):
        # 刷新残缺覆盖层时可能拿到“少量中文 + 一整段残留英文”。旧逻辑
        # 只要看见一个汉字就整块跳过，导致半中半英永远修不回来。
        remaining_latin_words = re.findall(r"[A-Za-z]{2,}", value)
        latin_count = sum(len(word) for word in remaining_latin_words)
        han_count = sum("\u3400" <= char <= "\u9fff" for char in value)
        return len(remaining_latin_words) >= 4 and latin_count >= max(16, han_count * 2)
    if target.startswith("ja") and any(
        "\u3040" <= char <= "\u30ff" for char in value
    ):
        return False
    if target.startswith("ko") and any(
        "\uac00" <= char <= "\ud7af" for char in value
    ):
        return False
    if target.startswith("ru") and any(
        "\u0400" <= char <= "\u04ff" for char in value
    ):
        return False
    if target.startswith("ar") and any(
        "\u0600" <= char <= "\u06ff" for char in value
    ):
        return False
    if target.startswith("th") and any(
        "\u0e00" <= char <= "\u0e7f" for char in value
    ):
        return False
    return True


def display_name(code: str) -> str:
    for c, name in LANGUAGES:
        if c == code:
            return name
    return code


def to_openai_lang(code: str) -> str:
    return {
        "auto": "auto",
        "zh": "簡體中文 (zh)",
        "zh-hant": "繁體中文 (zh-Hant)",
        "en": "English (en)",
        "ja": "日本語 (ja)",
        "ko": "한국어 (ko)",
        "fr": "français (fr)",
        "de": "Deutsch (de)",
        "es": "español (es)",
        "ru": "русский (ru)",
        "pt": "português (pt)",
        "it": "italiano (it)",
        "vi": "tiếng Việt (vi)",
        "th": "ไทย (th)",
        "ar": "العربية (ar)",
    }.get(code, code)


def to_deepl_lang(code: str) -> str:
    return {
        "auto": "",
        "zh": "ZH",
        "zh-hant": "ZH",
        "en": "EN-US",
        "ja": "JA",
        "ko": "KO",
        "fr": "FR",
        "de": "DE",
        "es": "ES",
        "ru": "RU",
        "pt": "PT-BR",
        "it": "IT",
        "vi": "VI",
        "th": "TH",
        "ar": "AR",
    }.get(code, code.upper())


def to_google_lang(code: str) -> str:
    return {
        "auto": "",
        "zh": "zh-CN",
        "zh-hant": "zh-TW",
    }.get(code, code)


def to_windows_lang(code: str) -> str:
    return {
        "auto": "zh-Hans-CN",
        "zh": "zh-Hans-CN",
        "zh-hant": "zh-Hant-TW",
        "en": "en-US",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
        "ru": "ru-RU",
        "pt": "pt-BR",
        "it": "it-IT",
        "vi": "vi-VN",
        "th": "th-TH",
        "ar": "ar-SA",
    }.get(code, "zh-Hans-CN")


def to_mymemory_lang(code: str) -> str:
    """MyMemory 使用 ISO 639-1；中文需带地区后缀。auto 默认按英文处理。"""
    return {
        "auto": "en",
        "zh": "zh-CN",
        "zh-hant": "zh-TW",
    }.get(code, code)


def to_baidu_lang(code: str) -> str:
    return {
        "auto": "auto",
        "zh": "zh",
        "zh-hant": "zh",
        "en": "en",
        "ja": "jp",
        "ko": "kor",
        "fr": "fra",
        "de": "de",
        "es": "spa",
        "ru": "ru",
        "pt": "pt",
        "it": "it",
        "vi": "vie",
        "th": "th",
        "ar": "ar",
    }.get(code, code)


def to_youdao_lang(code: str) -> str:
    return {
        "auto": "auto",
        "zh": "zh-CHS",
        "zh-hant": "zh-CHT",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "ru": "ru",
        "pt": "pt",
        "it": "it",
        "vi": "vi",
        "th": "th",
        "ar": "ar",
    }.get(code, code)


def to_azure_lang(code: str) -> str:
    # Azure 源语言省略即自动检测；目标语言用标准区域代码。
    return {
        "auto": "",
        "zh": "zh-Hans",
        "zh-hant": "zh-Hant",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "ru": "ru",
        "pt": "pt-BR",
        "it": "it",
        "vi": "vi",
        "th": "th",
        "ar": "ar",
    }.get(code, code)
