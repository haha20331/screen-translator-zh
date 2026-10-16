"""翻译器工厂。"""

from __future__ import annotations

from services.translation.base import _REGISTRY, Translator, list_translators

# 只有離線翻譯（llama.cpp，本機執行）。Mock 是測試與自我檢查用的替身，介面上不提供。
_SERVICE_DISPLAY = {
    "offline": "離線翻譯（llama.cpp，本機執行，可選不同大小的模型）",
    "mock": "Mock（測試用的本地詞典）",
}


def service_display_name(name: str) -> str:
    return _SERVICE_DISPLAY.get(name, name)


def create_translator(name: str, config_section: dict, cache=None) -> Translator:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知翻譯服務：{name}")
    return cls(config_section, cache)


__all__ = ["create_translator", "list_translators", "service_display_name"]
