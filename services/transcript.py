"""字幕輸出的內容：一句字幕「從顯示到結束」記成一筆，結束時才寫（寫入由 ``TranscriptWriter`` 負責）。

為什麼等字幕結束才寫：``monitor.confirm_rounds`` 預設 1，字幕淡入淡出時殘缺的辨識結果也會顯示，
而 txt 是附加寫入、寫了就留著。等到字幕被下一句取代（或消失、停止監控、離開程式）才寫，
寫下的就是最後、最完整的那一版，並且帶有開始與結束時間（像 SRT）。

一行一個文字區塊，欄位以 TAB 分隔::

    開始時間  結束時間  原文  譯文  狀態  模型  四角點

- 狀態：``正常``、``原樣``（譯文與原文相同：已是中文、不需翻譯或翻譯失敗保留原文）、``略過``（擬聲詞等不顯示的句子），
  顯示不到 ``min_seconds`` 秒就被下一句取代的加上 ``短暫``（多半是淡入淡出的殘句，方便事後過濾，不丟掉）。
- 四角點：左上、右上、右下、左下，螢幕絕對座標（物理像素）。
- 原文、譯文裡的換行與 TAB 換成 `` / ``，維持一行一筆。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from utils.stability import signature

STATUS_OK = "正常"
STATUS_SAME = "原樣"
STATUS_SKIPPED = "略過"
STATUS_BRIEF = "短暫"

DEFAULT_MIN_SECONDS = 0.3

_BREAKS = re.compile(r"\s*[\r\n\t\v\f\x85\u2028\u2029]+\s*")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class LineSink(Protocol):
    def submit(self, day: str, lines: Sequence[str]) -> bool: ...


@dataclass(frozen=True)
class Row:
    """一個文字區塊：原文、顯示出來的譯文、四角點。"""

    text: str
    translation: str
    corners: tuple[tuple[int, int], ...]

    @classmethod
    def from_box(cls, text: str, translation: str, x: int, y: int, width: int, height: int) -> "Row":
        right, bottom = x + width, y + height
        return cls(text, translation, ((x, y), (right, y), (right, bottom), (x, bottom)))

    @property
    def status(self) -> str:
        translation = self.translation.strip()
        if not translation:
            return STATUS_SKIPPED
        if translation == self.text.strip():
            return STATUS_SAME
        return STATUS_OK


@dataclass
class _Event:
    started: float
    rows: tuple[Row, ...]
    model: str
    key: tuple
    single_shot: bool = False


def resolve_directory(raw: str, default: Path) -> Path:
    """設定的輸出資料夾：空白 = 預設；支援 ``~`` 與 ``%VAR%``；相對路徑視為相對於預設資料夾的上一層（程式資料夾）。"""
    raw = (raw or "").strip()
    if not raw:
        return default
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    return path if path.is_absolute() else default.parent / path


def clean_field(text: str) -> str:
    """一個欄位的文字：換行與 TAB 換成 `` / ``，去掉其他控制字元與無法編碼的字元。"""
    text = _BREAKS.sub(" / ", str(text or ""))
    text = _CONTROL.sub("", text)
    return text.encode("utf-8", "replace").decode("utf-8").strip()


def format_time(timestamp: float) -> str:
    milliseconds = int(round(timestamp * 1000)) % 1000
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) + f".{milliseconds:03d}"


def format_corners(corners: Sequence[tuple[int, int]]) -> str:
    return "".join(f"({int(x)},{int(y)})" for x, y in corners)


def format_line(started: float, ended: float | None, row: Row, status: str, model: str) -> str:
    fields = (
        format_time(started),
        format_time(ended) if ended is not None else "",
        clean_field(row.text),
        clean_field(row.translation),
        status,
        clean_field(model),
        format_corners(row.corners),
    )
    return "\t".join(fields)


@dataclass
class TranscriptRecorder:
    """追蹤「目前顯示中的字幕」，字幕結束時把它寫成幾行交給 ``sink``。只在主執行緒使用。"""

    sink: LineSink
    min_seconds: float = DEFAULT_MIN_SECONDS
    clock: Callable[[], float] = time.time
    _current: _Event | None = field(default=None, init=False, repr=False)
    _gone_at: float | None = field(default=None, init=False, repr=False)

    @property
    def is_open(self) -> bool:
        return self._current is not None

    def show(self, rows: Sequence[Row], model: str, *, single_shot: bool = False) -> None:
        """畫面上顯示了一組新的字幕。和目前的是同一句（文字與模型都相同）就視為同一筆的延續，不另外記一筆；
        否則先結束目前這一筆（結束時間 = 現在），再開始新的一筆。"""
        if not rows:
            return
        now = self.clock()
        key = (signature([row.text for row in rows]), model)
        current = self._current
        self._gone_at = None
        if current is not None and current.key == key:
            current.single_shot = current.single_shot and single_shot  # 監控接手了單次截圖的那句：之後有真正的結束時間
            return
        if current is not None:
            self._close(current, now, replaced=True)
        self._current = _Event(now, tuple(rows), model, key, single_shot)

    def note_text_gone(self) -> None:
        """OCR 這一輪沒抓到字：字幕大概從這一刻就消失了（要等「無字幕自動關閉」的計時到了才真的結束）。"""
        if self._current is not None and self._gone_at is None:
            self._gone_at = self.clock()

    def cancel_text_gone(self) -> None:
        self._gone_at = None

    def end(self, reason: str = "stop") -> None:
        """目前這一筆結束了（``gone`` 字幕消失、``stop`` 停止監控、``shutdown`` 離開程式）。沒有進行中的就什麼都不做。"""
        current, self._current = self._current, None
        gone_at, self._gone_at = self._gone_at, None
        if current is None:
            return
        ended = gone_at if gone_at is not None else self.clock()  # OCR 已經抓不到字了：字幕在那一刻就結束
        self._close(current, ended, replaced=False)

    def finish_single_shot(self) -> None:
        """單次截圖翻譯後沒有接著監控：不知道字幕何時結束，結束時間留白，立刻寫出。"""
        current = self._current
        if current is None or not current.single_shot:
            return
        self._current = None
        self._write(current, None, brief=False)

    def comment(self, text: str) -> None:
        """寫一行註解（# 開頭）：監控開始與停止、監看框的位置。"""
        now = self.clock()
        self.sink.submit(_day(now), [f"# {format_time(now)} {clean_field(text)}"])

    def _close(self, event: _Event, ended: float, *, replaced: bool) -> None:
        ended = max(ended, event.started)
        brief = replaced and (ended - event.started) < max(0.0, float(self.min_seconds))
        self._write(event, ended, brief=brief)

    def _write(self, event: _Event, ended: float | None, *, brief: bool) -> None:
        lines = []
        for row in event.rows:
            status = row.status + (f",{STATUS_BRIEF}" if brief else "")
            lines.append(format_line(event.started, ended, row, status, event.model))
        self.sink.submit(_day(event.started), lines)


def _day(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))
