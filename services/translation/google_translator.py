"""Google Cloud Translation v2 适配器。"""

from __future__ import annotations

import html

import requests

from services.translation.base import TranslationError, Translator, register_translator
from services.translation.http_errors import raise_for_status
from utils.language_utils import to_google_lang


class GoogleTranslator(Translator):
    name = "google"

    def __init__(self, config_section: dict, cache=None, api_key: str = "") -> None:
        super().__init__(config_section, cache)
        google_cfg = config_section.get("google", {}) if config_section else {}
        self.api_key = api_key or str(google_cfg.get("api_key", ""))
        self.base_url = str(
            google_cfg.get("base_url", "https://translation.googleapis.com/language/translate/v2")
        ).rstrip("/")

    def _translate_batch(self, texts: list[str], source_language: str | None, target_language: str) -> list[str]:
        if not self.api_key:
            raise TranslationError("缺少 Google API Key：請設定 GOOGLE_TRANSLATE_API_KEY 或在設定中填寫")
        target = to_google_lang(target_language)
        params = {
            "key": self.api_key,
            "q": texts,
            "target": target,
        }
        source = to_google_lang(source_language) if source_language and source_language != "auto" else None
        if source:
            params["source"] = source
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=float(self.config.get("timeout_seconds", 30)),
            )
        except requests.RequestException as exc:
            raise TranslationError("Google 翻譯請求失敗（網路連線異常）") from exc
        response.encoding = "utf-8"
        if response.status_code != 200:
            raise_for_status(response, "Google")
        try:
            items = response.json()["data"]["translations"]
            return [html.unescape(item["translatedText"]) for item in items]
        except (KeyError, ValueError) as exc:
            raise TranslationError("Google 響應格式異常") from exc


register_translator(GoogleTranslator)
