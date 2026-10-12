"""语言代码与显示名、各服务语言参数映射。"""

from __future__ import annotations

import re

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
