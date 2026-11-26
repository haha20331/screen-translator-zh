"""辨識與翻譯分開進行時的排程：一次只翻一筆，只保留最新的一筆，過時的就放棄。

即時監控預設是「辨識 → 翻譯」排成一串：翻譯途中不看畫面。開啟「辨識與翻譯分開進行」之後，
辨識照常一直看畫面，確認有新字幕就交給這裡排隊翻譯。翻譯同一時間只有一筆（模型一次只能翻一句），
所以翻譯途中又出現更新的字幕時：

* 還沒開始翻的舊字幕直接被蓋掉（只保留最新的一筆）；
* 正在翻的舊字幕已經過時，叫它盡快停（由呼叫端中斷），翻完的結果也不顯示。

這個類別只管「誰該翻、誰過時」，不碰執行緒與畫面，所以可以單獨測試。
``key`` 是這組文字的指紋（``utils.stability.signature``）。
"""

from __future__ import annotations

from typing import Any, Hashable

# submit 的回傳值
START = "start"  # 現在沒有在翻：呼叫端馬上開始翻這一筆
QUEUED = "queued"  # 已經排在後面：目前那筆翻完就接著翻
IGNORED = "ignored"  # 同一組文字已經在翻或在排隊：什麼都不用做


class TranslationScheduler:
    def __init__(self) -> None:
        self._latest: Hashable | None = None  # 畫面上最新確認的那組文字
        self._inflight: Hashable | None = None  # 正在翻的那一組
        self._inflight_stopped = False  # 正在翻的那筆已經被要求停止
        self._pending: tuple[Hashable, Any] | None = None  # 排在後面的最新一筆

    # ------------------------------------------------------------------ 狀態
    @property
    def busy(self) -> bool:
        """有一筆正在翻，或排著隊。"""
        return self._inflight is not None or self._pending is not None

    @property
    def running(self) -> bool:
        return self._inflight is not None

    def is_current(self, key: Hashable) -> bool:
        """這組文字還是畫面上最新的嗎？翻完的結果只有「是」才顯示。"""
        return key is not None and key == self._latest

    # ------------------------------------------------------------------ 流程
    def submit(self, key: Hashable, job: Any) -> str:
        """辨識確認了一組新字幕。回傳 START／QUEUED／IGNORED；回傳 QUEUED 時，看 ``should_stop_running``。"""
        if key == self._latest and (
            (key == self._inflight and not self._inflight_stopped)
            or (self._pending is not None and self._pending[0] == key)
        ):
            return IGNORED  # 同一組文字還在處理，辨識每一輪都會再確認一次，不重複送
        self._latest = key
        if self._inflight is None:
            self._inflight = key
            self._inflight_stopped = False
            return START
        if key == self._inflight and not self._inflight_stopped:
            self._pending = None  # 翻到一半的舊字幕又回來了（中途閃過別的字幕）：繼續翻它就好
            return IGNORED
        self._pending = (key, job)
        return QUEUED

    def should_stop_running(self) -> bool:
        """正在翻的那一筆已經不是最新的了嗎（呼叫端據此中斷它）？只回報一次，之後視為已經要求過。"""
        if self._inflight is None or self._inflight_stopped or self._inflight == self._latest:
            return False
        self._inflight_stopped = True
        return True

    def finished(self) -> tuple[Hashable, Any] | None:
        """正在翻的那一筆結束了（翻完、被中斷或失敗）。回傳接著要翻的 (key, job)，沒有就是 None。"""
        self._inflight = None
        self._inflight_stopped = False
        if self._pending is None:
            return None
        key, job = self._pending
        self._pending = None
        self._inflight = key
        return key, job

    def cancel_all(self) -> bool:
        """畫面上的內容已經作廢（停止監控、重新整理、字幕消失、換設定）：排隊的丟掉，正在翻的過時。

        回傳正在翻的那一筆是否需要中斷。"""
        self._latest = None
        self._pending = None
        if self._inflight is not None and not self._inflight_stopped:
            self._inflight_stopped = True
            return True
        return False
