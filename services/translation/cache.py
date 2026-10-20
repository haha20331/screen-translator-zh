"""翻译结果缓存（JSON 落地，可配 TTL 与容量上限）。

資料放在 ``OrderedDict`` 裡，順序就是「最近用到的在最後面」：查到、寫入都把那一筆移到最後面。這樣

- 淘汰最久沒用的，是從最前面拿，不必排序；
- 過期的也一定在最前面（時間戳記由前往後遞增），清理只看最前面幾筆，不必掃過整個快取；
- 新增一筆的時間和快取有多大無關（以前每新增一筆都要掃過全部、滿了還要整個排序，十萬筆時一筆要 70 毫秒）。

另外兩個減少整理次數的做法：過期項目最多每分鐘整理一次（沒整理到的過期項目，查詢時也會當作沒有）；
超過容量上限時一次淘汰總容量的四分之一，而不是每多一筆就淘汰一筆。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from app.logger import get_logger

log = get_logger("translation.cache")

# 過期項目最多隔這麼久整理一次（秒）
SWEEP_INTERVAL_SECONDS = 60.0
# 超過容量上限時，一次淘汰總容量的幾分之一（最久沒用的先走）
EVICT_FRACTION = 4


class TranslationCache:
    def __init__(
        self,
        path: Path | str,
        ttl_days: float = 30,
        max_entries: int = 2000,
    ) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_days * 86400
        self.max_entries = max_entries
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()
        self._load()

    @staticmethod
    def _key(source: str | None, target: str, text: str) -> str:
        raw = f"{source or ''}|{target}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, source: str | None, target: str, text: str) -> str | None:
        key = self._key(source, target, text)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            now = time.time()
            if now - entry.get("ts", 0) > self.ttl_seconds:
                self._data.pop(key, None)
                return None
            # 用到就算「最近用過」：移到最後面，筆數超過上限時最先淘汰最久沒用的，歌詞副歌這種反覆出現的句子不會被擠掉
            entry["ts"] = now
            self._data.move_to_end(key)
            return entry.get("translated")

    def set(self, source: str | None, target: str, text: str, translated: str) -> None:
        key = self._key(source, target, text)
        with self._lock:
            self._data[key] = {"text": text, "translated": translated, "ts": time.time()}
            self._data.move_to_end(key)
            self._prune_locked()

    def count(self) -> int:
        """目前存了幾句（含還沒整理掉的過期項目）。"""
        with self._lock:
            return len(self._data)

    def clear(self) -> int:
        """刪除所有已儲存的翻譯（記憶體與檔案），回傳刪了幾句。"""
        with self._lock:
            removed = len(self._data)
            self._data.clear()
        self.flush()
        return removed

    def flush(self) -> None:
        with self._lock:
            self._save_locked()

    def set_ttl(self, ttl_days: float, max_entries: int) -> None:
        with self._lock:
            self.ttl_seconds = ttl_days * 86400
            self.max_entries = max_entries
            self._sweep_expired_locked()
            self._evict_locked()

    # ------------------------------------------------------------------
    def _sweep_expired_locked(self) -> None:
        """過期的一定在最前面：從最前面一路拿，遇到沒過期的就停（只花「過期筆數」的時間）。"""
        self._last_sweep = time.monotonic()
        now = time.time()
        data = self._data
        while data:
            key = next(iter(data))
            if now - data[key].get("ts", 0) <= self.ttl_seconds:
                break
            del data[key]

    def _evict_locked(self) -> None:
        """超過容量上限：一次淘汰總容量的四分之一（至少一筆），最久沒用的先走，之後的新增就不必每次都整理。"""
        limit = max(1, int(self.max_entries))
        if len(self._data) <= limit:
            return
        target = limit - max(1, limit // EVICT_FRACTION)
        data = self._data
        while len(data) > target:
            data.popitem(last=False)

    def _prune_locked(self) -> None:
        if time.monotonic() - self._last_sweep >= SWEEP_INTERVAL_SECONDS:
            self._sweep_expired_locked()
        self._evict_locked()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                entries = [(key, value) for key, value in loaded.items() if isinstance(value, dict)]
                for _, value in entries:
                    if not isinstance(value.get("ts"), (int, float)):
                        value["ts"] = 0  # 壞掉的時間戳記當成很久以前：最先被淘汰，不要讓排序出錯
                # 檔案裡的順序不一定是「最近用到的在最後面」（舊版是依寫入順序存的）：依時間戳記排好
                entries.sort(key=lambda item: item[1]["ts"])
                self._data = OrderedDict(entries)
                self._sweep_expired_locked()
                self._evict_locked()
        except (json.JSONDecodeError, OSError) as exc:
            try:
                self.path.rename(self.path.with_suffix(".json.bak"))
            except OSError:
                pass
            log.warning("翻譯快取損壞，已重置：%s", exc)

    def _save_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(self._data, fp, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("翻譯快取儲存失敗：%s", exc)
