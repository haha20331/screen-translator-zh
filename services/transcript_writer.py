"""把輸出的 txt 寫到磁碟：專用的寫入執行緒，主執行緒與翻譯執行緒只把整理好的文字丟進佇列。

為什麼不在呼叫端直接寫：Windows 上「開檔→寫→關檔」每次約 10 ms（防毒即時掃描會讓最慢的一次到 70 ms 以上），
檔案保持開著再 flush 只要 0.02 ms。所以由這條執行緒負責 I/O，並讓檔案開著寫；
安靜 ``idle_close_seconds`` 秒就關檔，使用者才刪得掉、改得了當天的檔案。

寫入不批次等待：佇列裡有多少就一次寫完（自然合併），寫完立刻 flush，不為了湊批次而延後，
強制結束時最多只丟「還在佇列裡」的毫秒級資料。停止監控與離開程式時 ``close`` 會把佇列寫完並 fsync。

寫入失敗（檔案被其他程式鎖住、磁碟滿）：重試 ``retries`` 次，仍失敗就改寫到同資料夾的備援檔
``YYYY-MM-DD.part-HHMMSS.txt``，並只通知一次（``on_problem``），不會每句都跳錯誤。
改寫備援檔之後不再每句重試被鎖住的檔案，每隔 ``fallback_recheck_seconds`` 秒才試一次，放開了就回到原檔。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from app.logger import get_logger

log = get_logger("transcript")

BOM = "\ufeff"
HEADER = (
    "# ScreenTranslator 輸出（UTF-8，欄位以 TAB 分隔；# 開頭的是註解；一句字幕結束時才寫入一次，所以依結束順序排列）\n"
    "# 開始時間\t結束時間\t原文\t譯文\t狀態\t模型\t四角點（左上 右上 右下 左下，螢幕絕對座標，物理像素）\n"
)


class _Flush:
    def __init__(self) -> None:
        self.done = threading.Event()


_STOP = object()


class TranscriptWriter:
    def __init__(
        self,
        directory: Path | str,
        *,
        idle_close_seconds: float = 5.0,
        retries: int = 3,
        retry_delay: float = 0.5,
        fallback_recheck_seconds: float = 60.0,
        on_problem: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._directory = Path(directory)
        self._idle_close = max(0.0, float(idle_close_seconds))
        self._retries = max(0, int(retries))
        self._retry_delay = max(0.0, float(retry_delay))
        self._recheck = max(0.0, float(fallback_recheck_seconds))
        self._clock = clock
        self._on_problem = on_problem
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._closing = threading.Event()
        # 以下只由寫入執行緒存取
        self._handle = None
        self._handle_path: Path | None = None
        self._fallback: dict[tuple[Path, str], tuple[Path, float]] = {}  # (資料夾, 日期) -> (備援檔, 上次檢查原檔的時間)
        self._problem_reported = False

    # ------------------------------------------------------------------ 呼叫端（任何執行緒）
    @property
    def directory(self) -> Path:
        return self._directory

    def set_directory(self, directory: Path | str) -> None:
        """之後的寫入改到新的資料夾（已經在佇列裡的也照新資料夾寫）。"""
        self._directory = Path(directory)

    def submit(self, day: str, lines: Sequence[str]) -> bool:
        """把幾行文字（不含換行）排進 ``day``（YYYY-MM-DD）那一天的檔案。立即返回；關閉後回傳 False。"""
        if self._closing.is_set() or not lines:
            return False
        self._ensure_thread()
        self._queue.put((day, tuple(lines)))
        return True

    def flush(self, timeout: float = 5.0) -> bool:
        """等佇列裡現有的資料寫完（測試與「開啟輸出資料夾」前使用）；逾時回傳 False。"""
        if self._thread is None or not self._thread.is_alive():
            return True
        marker = _Flush()
        self._queue.put(marker)
        return marker.done.wait(timeout)

    def close(self, timeout: float = 3.0) -> bool:
        """寫完佇列、fsync、關檔並結束執行緒。逾時（例如正在重試被鎖住的檔案）回傳 False，執行緒是 daemon，不會卡住結束。"""
        if self._closing.is_set():
            return True
        self._closing.set()
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        self._queue.put(_STOP)
        thread.join(timeout)
        return not thread.is_alive()

    # ------------------------------------------------------------------ 寫入執行緒
    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="transcript-writer", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        stopping = False
        while not stopping:
            try:
                first = self._queue.get(timeout=self._idle_close if self._handle is not None else None)
            except queue.Empty:
                self._close_handle()  # 安靜了一陣子：關檔，使用者才能刪除或改名
                continue
            items = [first]
            while True:  # 有多少寫多少，不等待
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            entries: list[tuple[str, tuple[str, ...]]] = []
            markers: list[_Flush] = []
            for item in items:
                if item is _STOP:
                    stopping = True
                elif isinstance(item, _Flush):
                    markers.append(item)
                else:
                    entries.append(item)
            try:
                self._write(entries)
            except Exception:  # 寫入執行緒絕不能因為一次意外而死掉
                log.exception("輸出檔寫入發生未預期的錯誤")
            for marker in markers:
                marker.done.set()
        self._close_handle()

    def _write(self, entries: list[tuple[str, tuple[str, ...]]]) -> None:
        index = 0
        while index < len(entries):
            day = entries[index][0]
            text = ""
            while index < len(entries) and entries[index][0] == day:
                text += "".join(line + "\n" for line in entries[index][1])
                index += 1
            self._append(day, text)

    def _append(self, day: str, text: str) -> None:
        directory = self._directory
        primary = directory / f"{day}.txt"
        key = (directory, day)
        side = self._fallback.get(key)
        if side is not None and self._clock() - side[1] >= self._recheck:
            if self._try_write(primary, text, retries=0) is None:  # 被鎖住的檔案放開了：回到原檔
                del self._fallback[key]
                self._problem_reported = False
                return
            side = self._fallback[key] = (side[0], self._clock())
        if side is None:
            error = self._try_write(primary, text)
            if error is None:
                self._problem_reported = False
                return
            side = self._fallback[key] = (directory / f"{day}.part-{time.strftime('%H%M%S')}.txt", self._clock())
            notice = f"輸出檔 {primary.name} 無法寫入（{_reason(error)}），改寫到 {side[0].name}"
            error = self._try_write(side[0], text)
        else:
            notice = None
            error = self._try_write(side[0], text, retries=0)
        if error is None:
            if notice:
                self._report(notice)
            return
        log.warning("輸出檔寫入失敗，這 %d 行沒有寫入：%s", text.count("\n"), error)
        self._report(f"輸出檔無法寫入（{_reason(error)}），這段字幕沒有存進 txt")

    def _try_write(self, path: Path, text: str, retries: int | None = None) -> OSError | None:
        attempts = 1 + (self._retries if retries is None else retries)
        error: OSError | None = None
        for attempt in range(attempts):
            try:
                handle = self._open(path)
                handle.write(text)
                handle.flush()
                return None
            except OSError as exc:
                error = exc
                self._close_handle(sync=False)  # 壞掉的檔案代碼不能再用
                if attempt + 1 < attempts and self._closing.wait(self._retry_delay):
                    break  # 正在結束程式：不要為了重試拖住離開
        return error

    def _open(self, path: Path):
        if self._handle is not None and self._handle_path == path:
            return self._handle
        self._close_handle()
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size == 0
        handle = open(path, "a", encoding="utf-8", newline="\r\n")
        try:
            if fresh:
                handle.write(BOM + HEADER)
                handle.flush()
        except OSError:
            handle.close()
            raise
        self._handle, self._handle_path = handle, path
        return handle

    def _close_handle(self, sync: bool = True) -> None:
        handle, self._handle, self._handle_path = self._handle, None, None
        if handle is None:
            return
        try:
            if sync:
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass

    def _report(self, message: str) -> None:
        if self._problem_reported or self._on_problem is None:
            return
        self._problem_reported = True
        try:
            self._on_problem(message)
        except Exception:
            log.exception("通知輸出問題時發生錯誤")


def _reason(error: OSError) -> str:
    return (error.strerror or str(error)).strip() or type(error).__name__
