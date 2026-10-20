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


# ----------------------------------------------------------------------------- 新增一筆的時間不隨快取變大
def test_over_the_limit_a_quarter_of_the_capacity_is_evicted_at_once(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json", ttl_days=30, max_entries=100)
    for i in range(100):
        cache.set("ja", "zh", f"t{i}", f"譯{i}")
    assert cache.count() == 100  # 剛好滿，還沒超過

    cache.set("ja", "zh", "overflow", "多一筆")

    assert cache.count() == 75  # 超過的那一筆一次淘汰總容量的四分之一（25 筆），留下 75 筆
    assert cache.get("ja", "zh", "t0") is None and cache.get("ja", "zh", "t25") is None  # 最久沒用的先走
    assert cache.get("ja", "zh", "t26") == "譯26" and cache.get("ja", "zh", "overflow") == "多一筆"


def test_after_a_bulk_eviction_the_next_new_entries_need_no_cleanup(tmp_path, monkeypatch):
    cache = TranslationCache(tmp_path / "cache.json", ttl_days=30, max_entries=100)
    for i in range(101):
        cache.set("ja", "zh", f"t{i}", "x")
    removed = []
    original = cache._data.popitem
    monkeypatch.setattr(cache._data, "popitem", lambda last=True: (removed.append(1), original(last=last))[1], raising=False)

    for i in range(20):  # 還有 25 筆空間：這 20 筆都不會觸發淘汰
        cache.set("ja", "zh", f"new{i}", "y")

    assert removed == [] and cache.count() == 95


def test_expired_entries_are_swept_at_most_once_a_minute(tmp_path, monkeypatch):
    cache = TranslationCache(tmp_path / "cache.json", ttl_days=1, max_entries=1000)
    sweeps = []
    original = cache._sweep_expired_locked
    monkeypatch.setattr(cache, "_sweep_expired_locked", lambda: (sweeps.append(1), original())[1])
    for i in range(50):
        cache.set("ja", "zh", f"t{i}", "x")
        cache._data[cache._key("ja", "zh", f"t{i}")]["ts"] = time.time() - 2 * 86400  # 兩天前就過期了

    for i in range(10):
        cache.set("ja", "zh", f"more{i}", "y")

    assert sweeps == [] and cache.count() == 60  # 一分鐘內不整理
    assert cache.get("ja", "zh", "t0") is None  # 但過期的項目查詢時照樣當作沒有
    cache._last_sweep -= 61  # 過了一分鐘
    cache.set("ja", "zh", "later", "z")
    assert sweeps == [1]
    assert cache.count() == 11  # 前面那些過期的都清掉了：剩下 10 筆沒過期的加上剛新增的


def test_the_sweep_only_looks_at_the_front_so_it_costs_the_number_of_expired_entries(tmp_path):
    import time as _time

    cache = TranslationCache(tmp_path / "cache.json", ttl_days=1, max_entries=1000)
    now = _time.time()
    for i in range(10):
        cache.set("ja", "zh", f"old{i}", "x")
        cache._data[cache._key("ja", "zh", f"old{i}")]["ts"] = now - 2 * 86400  # 兩天前
    for i in range(10):
        cache.set("ja", "zh", f"new{i}", "y")

    cache._sweep_expired_locked()

    assert cache.count() == 10 and cache.get("ja", "zh", "new0") == "y"


def test_a_file_written_in_insertion_order_is_loaded_in_last_used_order(tmp_path):
    path = tmp_path / "cache.json"
    now = time.time()
    entries = {f"k{i}": {"text": f"t{i}", "translated": f"譯{i}", "ts": now - (10 - i)} for i in range(10)}
    entries["k0"]["ts"] = now  # 檔案裡排第一、其實是最近才用過的
    path.write_text(json.dumps(entries), encoding="utf-8")

    cache = TranslationCache(path, ttl_days=30, max_entries=8)  # 載入時超過上限：最久沒用的先走

    assert "k0" in cache._data and "k1" not in cache._data and "k2" not in cache._data
    assert list(cache._data)[-1] == "k0"


def test_broken_entries_in_the_file_do_not_crash_the_loading(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"a": {"text": "t", "translated": "譯", "ts": "yesterday"}, "b": "not a dict", "c": {"text": "u", "translated": "x", "ts": time.time()}}), encoding="utf-8")

    cache = TranslationCache(path)

    assert cache.count() == 1 and "b" not in cache._data  # 不是字典的丟掉
    assert "a" not in cache._data and "c" in cache._data  # 壞的時間戳記當成很久以前：視為過期，載入時就清掉


def test_adding_an_entry_takes_the_same_time_whether_the_cache_is_small_or_big(tmp_path):
    def per_set(entries: int) -> float:
        cache = TranslationCache(tmp_path / f"c{entries}.json", ttl_days=30, max_entries=entries)
        for i in range(entries):
            cache.set("ja", "zh", f"t{i}", "x")
        start = time.perf_counter()
        for i in range(2000):  # 滿了之後繼續新增（含整批淘汰）
            cache.set("ja", "zh", f"more{i}", "y")
        return (time.perf_counter() - start) / 2000

    small, big = per_set(2000), per_set(60000)

    assert big < small * 5 + 0.0005  # 以前是 O(筆數)：6 萬筆比 2 千筆慢 30 倍以上；現在差不多（容許量測雜訊）


def test_the_default_limit_is_100000_and_a_new_cache_object_uses_it(tmp_path):
    from app.config import DEFAULTS

    assert DEFAULTS["translation"]["cache_max_entries"] == 100000
    assert TranslationCache(tmp_path / "cache.json").max_entries == 100000
