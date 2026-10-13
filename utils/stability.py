"""即時監控的文字穩定化：辨識結果連續相同才翻譯並顯示，與已顯示的相同就完全不翻。

字幕淡入淡出、換字幕的過渡畫面常被辨識成殘缺的句子，翻出來就是一句怪話。穩定化只在
「同一組文字連續出現 N 次」之後才放行（N = 確認次數，使用者可調）：N = 1 是快速（辨識到就顯示），
N = 2 是穩定（系統推薦值見 ``RECOMMENDED_*``）。
"""

from __future__ import annotations

import threading
import time
import unicodedata
from typing import Callable, Sequence

SAME = "same"  # 與目前顯示的相同：不必翻譯、不必重畫
PENDING = "pending"  # 新的文字，但還沒穩定：先不翻譯
CONFIRMED = "confirmed"  # 穩定了（或等太久）：翻譯並顯示

# 系統推薦值：連續幾次辨識到相同結果才翻譯並顯示（確認次數），以及新文字一直在變時最久等幾秒。
# 設定畫面、說明文字與預設值都從這裡取，避免說明和實際行為不一致。
RECOMMENDED_FAST_ROUNDS = 1
RECOMMENDED_STABLE_ROUNDS = 2
RECOMMENDED_MAX_WAIT = 2.5
MIN_ROUNDS, MAX_ROUNDS = 1, 10
MIN_MAX_WAIT, MAX_MAX_WAIT = 0.5, 30.0


def signature(texts: Sequence[str]) -> tuple[str, ...]:
    """一組文字的指紋：全形半形、空白差異不算不同（OCR 在這些地方會抖動）。"""
    return tuple("".join(unicodedata.normalize("NFKC", text).split()) for text in texts)


class TextStabilizer:
    """判斷這一輪辨識到的文字該不該翻譯。

    ``rounds``：同一組新文字要連續出現幾輪才放行（穩定 = 2，快速 = 1）。
    ``max_wait``：新文字一直在變（辨識抖動、影片背景雜訊）時，最久等幾秒就放行最新的一組，避免畫面卡住不更新。

    ``observe`` 只做判斷；真正顯示出來之後由呼叫端 ``commit``，所以翻譯途中失敗、被取消時
    下一輪還會再試，不會誤以為已經顯示過了。執行緒安全（背景執行緒判斷、主執行緒 commit）。
    """

    def __init__(self, rounds: int = 2, max_wait: float = 2.5, clock: Callable[[], float] = time.monotonic) -> None:
        self.rounds = max(1, int(rounds))
        self.max_wait = float(max_wait)
        self._clock = clock
        self._lock = threading.Lock()
        self._displayed: tuple[str, ...] | None = None
        self._candidate: tuple[str, ...] | None = None
        self._count = 0
        self._waiting_since = 0.0
        self._confirmed: tuple[str, ...] | None = None

    def configure(self, rounds: int, max_wait: float) -> None:
        with self._lock:
            self.rounds = max(1, int(rounds))
            self.max_wait = float(max_wait)

    @property
    def has_pending(self) -> bool:
        """有一組新文字正在等第二次確認（此時畫面沒變也要再辨識一次，不能跳過）。"""
        with self._lock:
            return self._candidate is not None

    def observe(self, texts: Sequence[str]) -> str:
        current = signature(texts)
        with self._lock:
            if current == self._displayed:
                self._candidate, self._count, self._confirmed = None, 0, None
                return SAME
            now = self._clock()
            self._confirmed = None  # 只有最近一次判斷放行的那組文字才可以被 commit
            if current == self._candidate:
                self._count += 1
            else:
                if self._candidate is None:
                    self._waiting_since = now
                self._candidate, self._count = current, 1
            if self._count >= self.rounds or now - self._waiting_since >= self.max_wait:
                self._confirmed = current
                return CONFIRMED
            return PENDING

    def commit(self) -> None:
        """剛放行的那組文字已經顯示出來了。"""
        with self._lock:
            if self._confirmed is not None:
                self._displayed = self._confirmed
            self._candidate, self._count, self._confirmed = None, 0, None

    def reset(self) -> None:
        with self._lock:
            self._displayed = self._candidate = self._confirmed = None
            self._count = 0
