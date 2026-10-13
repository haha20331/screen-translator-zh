"""翻译器工厂。"""

from __future__ import annotations

from typing import Callable

from services.translation.base import _REGISTRY, Translator, list_translators

_SERVICE_DISPLAY = {
    "offline": "離線翻譯（llama.cpp，本機執行，可選不同大小的模型）",
    "mock": "Mock（本地詞典演示）",
    "mymemory": "MyMemory（免費線上）",
    "google_free": "Google 免費（無 Key）",
    "openai": "OpenAI",
    "deepl": "DeepL",
    "google": "Google 翻譯",
    "baidu": "百度翻譯",
    "youdao": "有道翻譯",
    "azure": "Azure 翻譯",
}


def service_display_name(name: str) -> str:
    return _SERVICE_DISPLAY.get(name, name)


def create_translator(
    name: str,
    config_section: dict,
    cache=None,
    api_key_resolver: Callable[[str], str] | None = None,
) -> Translator:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知翻譯服務：{name}")
    api_key = api_key_resolver(name) if api_key_resolver else ""
    return cls(config_section, cache, api_key=api_key)


__all__ = ["create_translator", "list_translators", "service_display_name"]
