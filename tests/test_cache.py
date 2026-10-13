"""翻译缓存测试。"""

import json
import time

from services.translation.cache import TranslationCache


def test_cache_set_get(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json")
    cache.set("en", "zh", "hello", "你好")
    assert cache.get("en", "zh", "hello") == "你好"
    assert cache.get("en", "zh", "other") is None


def test_cache_ttl_expiry(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json", ttl_days=1e-9)
    cache.set("en", "zh", "hello", "你好")
    time.sleep(0.01)
    assert cache.get("en", "zh", "hello") is None


def test_cache_max_entries(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json", ttl_days=30, max_entries=3)
    for i in range(10):
        cache.set("en", "zh", f"text{i}", f"译文{i}")
    # 只保留最近 3 条
    assert len(cache._data) <= 3
    assert cache.get("en", "zh", "text0") is None
    assert cache.get("en", "zh", "text9") == "译文9"


def test_cache_persist_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    cache = TranslationCache(path)
    cache.set("en", "zh", "hello", "你好")
    cache.flush()
    cache2 = TranslationCache(path)
    assert cache2.get("en", "zh", "hello") == "你好"


def test_cache_corrupt_file(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{broken json", encoding="utf-8")
    cache = TranslationCache(path)
    cache.set("en", "zh", "a", "b")
    assert cache.get("en", "zh", "a") == "b"


def test_cache_hit_counts_as_recent_use(tmp_path):
    """筆數超過上限時淘汰最久沒用的：反覆出現的句子（歌詞副歌）不會因為最早翻譯就被擠掉。"""
    cache = TranslationCache(tmp_path / "cache.json", ttl_days=30, max_entries=3)
    cache.set("ja", "zh", "副歌", "chorus")
    time.sleep(0.01)
    cache.set("ja", "zh", "a", "A")
    time.sleep(0.01)
    cache.set("ja", "zh", "b", "B")
    time.sleep(0.01)
    assert cache.get("ja", "zh", "副歌") == "chorus"  # 副歌又出現了
    time.sleep(0.01)

    cache.set("ja", "zh", "c", "C")  # 超過上限：該淘汰的是最久沒用的 a，不是剛用過的副歌

    assert cache.get("ja", "zh", "副歌") == "chorus"
    assert cache.get("ja", "zh", "a") is None


def test_default_cache_limit_is_large_enough_to_keep_a_whole_song():
    from app.config import DEFAULTS

    assert DEFAULTS["translation"]["cache_max_entries"] >= 1000
