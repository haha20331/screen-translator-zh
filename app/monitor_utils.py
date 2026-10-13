"""连续监控模式的纯逻辑工具（不依赖 Qt，便于单测）。"""

from __future__ import annotations

import hashlib
from typing import Iterable

from app.models import TextRegion


def region_signature(capture, regions: Iterable[TextRegion]) -> str:
    """内容指纹：同一批 (坐标, 原文, 译文) 算同一哈希。

    监控模式下内容没变就跳过覆盖层重绘，避免整屏闪烁。
    """
    items = []
    for region in regions:
        items.append(
            (
                region.x,
                region.y,
                region.width,
                region.height,
                region.text,
                region.translated_text,
            )
        )
    digest = hashlib.sha1(repr(items).encode("utf-8")).hexdigest()
    return digest


def _without_kept_original(text: str, translated: str) -> str:
    """「保留原文」會把譯文顯示成「原文\n譯文」；寫回快取的只能是譯文，否則下一輪又會多疊一份原文。"""
    prefix = text.strip() + "\n"
    translated = translated.strip()
    while text.strip() and translated.startswith(prefix):
        translated = translated[len(prefix):].lstrip()
    return translated


def known_translations(regions: Iterable[TextRegion]) -> dict[str, str]:
    """把本轮的 (原文 -> 译文) 收成字典，供下一轮预热翻译缓存。"""
    known = {}
    for region in regions:
        if not region.text or not region.translated_text or not region.translated_text.strip():
            continue
        translated = _without_kept_original(region.text, region.translated_text)
        if translated:
            known[region.text] = translated
    return known
