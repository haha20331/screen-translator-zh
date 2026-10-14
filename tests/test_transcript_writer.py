"""輸出檔的寫入執行緒：BOM 與表頭、附加、依日期分檔、佇列寫完才結束、被鎖住時改寫備援檔、不拖慢呼叫端。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from services.transcript_writer import BOM, HEADER, TranscriptWriter


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _body(path: Path) -> list[str]:
    """檔案內容去掉 BOM 與表頭註解後的資料行。"""
    lines = _read(path).removeprefix(BOM).split("\r\n")
    return [line for line in lines if line and not line.startswith("# ScreenTranslator") and not line.startswith("# 開始時間")]


@pytest.fixture
def writer(tmp_path):
    problems: list[str] = []
    instance = TranscriptWriter(tmp_path / "out", retry_delay=0.0, on_problem=problems.append)
    instance.problems = problems
    yield instance
    instance.close(timeout=2.0)


def test_a_new_file_starts_with_a_utf8_bom_and_the_column_header_and_uses_crlf(writer, tmp_path):
    writer.submit("2026-10-14", ["甲\t乙"])
    assert writer.flush()

    raw = (tmp_path / "out" / "2026-10-14.txt").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # 記事本與 Excel 才會認成 UTF-8
    text = raw.decode("utf-8")
    assert text.removeprefix(BOM).startswith(HEADER.replace("\n", "\r\n"))
    assert text.endswith("甲\t乙\r\n")
    assert "\n" not in text.replace("\r\n", "")  # 全部都是 CRLF


def test_appending_later_does_not_repeat_the_bom_or_header_even_after_the_file_was_closed(tmp_path):
    first = TranscriptWriter(tmp_path, retry_delay=0.0)
    first.submit("2026-10-14", ["第一句"])
    assert first.close()
    second = TranscriptWriter(tmp_path, retry_delay=0.0)
    second.submit("2026-10-14", ["第二句"])
    assert second.close()

    path = tmp_path / "2026-10-14.txt"
    assert _read(path).count(BOM) == 1
    assert _read(path).count("# 開始時間") == 1
    assert _body(path) == ["第一句", "第二句"]


def test_lines_keep_their_order_and_each_day_goes_to_its_own_file(writer, tmp_path):
    writer.submit("2026-10-14", ["a1", "a2"])
    writer.submit("2026-10-15", ["b1"])
    writer.submit("2026-10-14", ["a3"])
    assert writer.flush()

    assert _body(tmp_path / "out" / "2026-10-14.txt") == ["a1", "a2", "a3"]
    assert _body(tmp_path / "out" / "2026-10-15.txt") == ["b1"]


def test_everything_in_the_queue_is_written_when_closing(tmp_path):
    writer = TranscriptWriter(tmp_path, retry_delay=0.0)
    for index in range(500):
        writer.submit("2026-10-14", [f"句{index}"])

    assert writer.close(timeout=5.0)

    assert _body(tmp_path / "2026-10-14.txt") == [f"句{index}" for index in range(500)]


def test_submitting_after_close_is_ignored_and_closing_twice_is_fine(tmp_path):
    writer = TranscriptWriter(tmp_path)
    writer.submit("2026-10-14", ["x"])
    assert writer.close()

    assert writer.submit("2026-10-14", ["y"]) is False
    assert writer.close()
    assert _body(tmp_path / "2026-10-14.txt") == ["x"]


def test_nothing_is_started_or_created_until_the_first_line(tmp_path):
    writer = TranscriptWriter(tmp_path / "never")

    assert writer.flush() and writer.close()
    assert not (tmp_path / "never").exists()
    assert writer._thread is None  # 沒有輸出就不開執行緒


def test_submit_returns_immediately_even_when_the_disk_is_slow(tmp_path, monkeypatch):
    writer = TranscriptWriter(tmp_path, retry_delay=0.0)
    real_write = writer._write

    def slow_write(entries):
        time.sleep(0.2)  # 磁碟慢（防毒掃描）
        real_write(entries)

    monkeypatch.setattr(writer, "_write", slow_write)
    started = time.perf_counter()
    for index in range(200):
        writer.submit("2026-10-14", [f"句{index}"])
    elapsed = time.perf_counter() - started

    assert elapsed < 0.15  # 呼叫端（主執行緒）不會等磁碟
    assert writer.close(timeout=5.0)
    assert len(_body(tmp_path / "2026-10-14.txt")) == 200  # 慢，但一句都不少


def test_queued_lines_are_merged_into_few_writes_without_waiting_for_a_batch(tmp_path, monkeypatch):
    writer = TranscriptWriter(tmp_path, retry_delay=0.0)
    calls: list[int] = []
    real_append = writer._append
    monkeypatch.setattr(writer, "_append", lambda day, text: (calls.append(text.count("\n")), real_append(day, text))[1])
    writer.submit("2026-10-14", ["第一句"])
    assert writer.flush()  # 只有一句時不會為了湊批次而等待：馬上就寫了
    assert calls == [1]

    writer.close()


def test_the_file_is_closed_after_a_quiet_moment_so_the_user_can_delete_or_rename_it(tmp_path):
    writer = TranscriptWriter(tmp_path, idle_close_seconds=0.05)
    writer.submit("2026-10-14", ["x"])
    assert writer.flush()
    deadline = time.monotonic() + 3.0
    while writer._handle is not None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert writer._handle is None

    writer.submit("2026-10-14", ["y"])  # 之後還能再開檔附加
    assert writer.close()
    assert _body(tmp_path / "2026-10-14.txt") == ["x", "y"]


def test_changing_the_directory_sends_later_lines_to_the_new_folder(writer, tmp_path):
    writer.submit("2026-10-14", ["舊"])
    assert writer.flush()
    writer.set_directory(tmp_path / "new")
    writer.submit("2026-10-14", ["新"])
    assert writer.flush()

    assert _body(tmp_path / "out" / "2026-10-14.txt") == ["舊"]
    assert _body(tmp_path / "new" / "2026-10-14.txt") == ["新"]


# ----------------------------------------------------------------------------- 寫不進去
def test_a_failure_that_clears_on_retry_loses_nothing_and_reports_nothing(tmp_path, monkeypatch):
    problems: list[str] = []
    writer = TranscriptWriter(tmp_path, retries=3, retry_delay=0.0, on_problem=problems.append)
    real_open = writer._open
    attempts = {"n": 0}

    def flaky_open(path):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise PermissionError(13, "另一個程序正在使用此檔案")
        return real_open(path)

    monkeypatch.setattr(writer, "_open", flaky_open)
    writer.submit("2026-10-14", ["一句"])
    assert writer.flush(timeout=5.0)  # 一般運作中（不是離開程式）：照設定重試
    assert writer.close(timeout=5.0)

    assert attempts["n"] == 3
    assert _body(tmp_path / "2026-10-14.txt") == ["一句"]
    assert problems == []


def test_a_locked_file_falls_back_to_a_side_file_and_tells_the_user_once(tmp_path):
    problems: list[str] = []
    writer = TranscriptWriter(tmp_path, retries=2, retry_delay=0.0, on_problem=problems.append)
    (tmp_path / "2026-10-14.txt").mkdir()  # 這個名字被占住了，開檔一定失敗（和被鎖住一樣是 OSError）
    writer.submit("2026-10-14", ["第一句"])
    writer.submit("2026-10-14", ["第二句"])
    assert writer.flush(timeout=5.0)
    writer.submit("2026-10-14", ["第三句"])
    assert writer.close(timeout=5.0)

    side_files = sorted(tmp_path.glob("2026-10-14.part-*.txt"))
    assert len(side_files) == 1
    assert _body(side_files[0]) == ["第一句", "第二句", "第三句"]  # 這一天剩下的都寫在備援檔，不再每句重試被鎖的檔案
    assert len(problems) == 1  # 只通知一次，不是每句都跳
    assert "2026-10-14.txt" in problems[0] and side_files[0].name in problems[0]


def test_once_the_lock_is_released_writing_returns_to_the_main_file(tmp_path):
    problems: list[str] = []
    clock = {"now": 0.0}
    writer = TranscriptWriter(
        tmp_path, retries=0, retry_delay=0.0, fallback_recheck_seconds=60.0,
        on_problem=problems.append, clock=lambda: clock["now"],
    )
    primary = tmp_path / "2026-10-14.txt"
    primary.mkdir()  # 被占住
    writer.submit("2026-10-14", ["被鎖時"])
    assert writer.flush(5.0)
    primary.rmdir()  # 使用者關掉了 Excel
    clock["now"] = 30.0
    writer.submit("2026-10-14", ["還沒到檢查時間"])
    assert writer.flush(5.0)
    assert not primary.exists()  # 不是每句都去敲被鎖的檔案：還在備援檔

    clock["now"] = 61.0  # 隔了一分鐘：試一次原檔
    writer.submit("2026-10-14", ["回到原檔"])
    writer.submit("2026-10-14", ["之後也寫原檔"])
    assert writer.close(5.0)

    assert _body(primary) == ["回到原檔", "之後也寫原檔"]
    (side,) = tmp_path.glob("2026-10-14.part-*.txt")
    assert _body(side) == ["被鎖時", "還沒到檢查時間"]
    assert len(problems) == 1


def test_a_second_lock_later_is_reported_again(tmp_path):
    problems: list[str] = []
    clock = {"now": 0.0}
    writer = TranscriptWriter(tmp_path, retries=0, fallback_recheck_seconds=10.0, on_problem=problems.append, clock=lambda: clock["now"])
    primary = tmp_path / "2026-10-14.txt"
    primary.mkdir()
    writer.submit("2026-10-14", ["a"])
    assert writer.flush(5.0)
    primary.rmdir()
    clock["now"] = 20.0
    writer.submit("2026-10-14", ["b"])  # 回到原檔
    assert writer.flush(5.0)
    writer._close_handle()
    primary.unlink()
    primary.mkdir()  # 又被占住
    writer.submit("2026-10-14", ["c"])
    assert writer.close(5.0)

    assert len(problems) == 2


def test_when_even_the_side_file_fails_the_writer_survives_and_reports_once(tmp_path):
    problems: list[str] = []
    writer = TranscriptWriter(tmp_path / "blocked", retries=1, retry_delay=0.0, on_problem=problems.append)
    (tmp_path / "blocked").write_text("我是檔案，不是資料夾", encoding="utf-8")  # 資料夾建不起來
    writer.submit("2026-10-14", ["丟掉"])
    writer.submit("2026-10-14", ["也丟掉"])
    assert writer.flush(timeout=5.0)

    assert len(problems) == 1 and "無法寫入" in problems[0]
    assert writer._thread.is_alive()  # 執行緒不會因此死掉

    writer.set_directory(tmp_path / "ok")  # 使用者在設定改了資料夾：之後的字幕寫得進去
    writer.submit("2026-10-14", ["後來"])
    assert writer.close(timeout=5.0)
    assert _body(tmp_path / "ok" / "2026-10-14.txt") == ["後來"]


def test_a_bug_in_the_problem_callback_cannot_kill_the_writer(tmp_path):
    def broken(_message: str) -> None:
        raise RuntimeError("通知出錯")

    writer = TranscriptWriter(tmp_path, retries=0, on_problem=broken)
    (tmp_path / "2026-10-14.txt").mkdir()
    writer.submit("2026-10-14", ["a"])
    writer.submit("2026-10-15", ["b"])
    assert writer.close(timeout=5.0)

    assert _body(tmp_path / "2026-10-15.txt") == ["b"]


def test_closing_does_not_wait_out_the_retries_when_the_program_is_exiting(tmp_path):
    writer = TranscriptWriter(tmp_path, retries=50, retry_delay=1.0)
    (tmp_path / "2026-10-14.txt").mkdir()
    writer.submit("2026-10-14", ["a"])
    time.sleep(0.1)

    started = time.perf_counter()
    assert writer.close(timeout=3.0)

    assert time.perf_counter() - started < 2.0  # 離開程式時不會為了重試被鎖住的檔案而卡住


def test_unicode_text_survives_a_round_trip(writer, tmp_path):
    line = "2026-10-14 12:00:00.000\t\t君の名前を呼び続けた\t一直叫著你的名字\t正常\toffline/qwen2.5-3b\t(1,2)(3,2)(3,4)(1,4)"
    writer.submit("2026-10-14", [line])
    assert writer.flush()

    assert _body(tmp_path / "out" / "2026-10-14.txt") == [line]
