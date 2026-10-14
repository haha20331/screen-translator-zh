"""文字輸出的內容：一句字幕「顯示到結束」記一筆、欄位格式、結束時間、短暫標記、單次截圖。"""

from __future__ import annotations

import time
from pathlib import Path

from services.transcript import (
    Row,
    TranscriptRecorder,
    clean_field,
    format_corners,
    format_line,
    format_time,
    resolve_directory,
)


class _Sink:
    def __init__(self) -> None:
        self.batches: list[tuple[str, list[str]]] = []

    def submit(self, day, lines):
        self.batches.append((day, list(lines)))
        return True

    @property
    def lines(self) -> list[str]:
        return [line for _day, lines in self.batches for line in lines]


class _Clock:
    """從 2026-10-14 12:00:00（本機時間）開始的假時鐘。"""

    def __init__(self) -> None:
        self.now = time.mktime((2026, 10, 14, 12, 0, 0, 0, 0, -1))

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _row(text="こんにちは", translation="你好", x=10, y=20, w=100, h=30) -> Row:
    return Row.from_box(text, translation, x, y, w, h)


def _recorder(**kwargs):
    sink, clock = _Sink(), _Clock()
    return TranscriptRecorder(sink, clock=clock, **kwargs), sink, clock


def _fields(line: str) -> list[str]:
    return line.split("\t")


# ----------------------------------------------------------------------------- 欄位格式
def test_corners_are_top_left_top_right_bottom_right_bottom_left_in_screen_pixels():
    row = _row(x=120, y=45, w=90, h=23)

    assert format_corners(row.corners) == "(120,45)(210,45)(210,68)(120,68)"  # PLAN 6 節的範例格式


def test_a_line_has_seven_tab_separated_fields_in_the_agreed_order():
    clock = _Clock()
    line = format_line(clock.now, clock.now + 2.5, _row(), "正常", "offline/qwen2.5-1.5b")

    fields = _fields(line)
    assert len(fields) == 7
    assert fields[0] == "2026-10-14 12:00:00.000"
    assert fields[1] == "2026-10-14 12:00:02.500"
    assert fields[2:6] == ["こんにちは", "你好", "正常", "offline/qwen2.5-1.5b"]
    assert fields[6] == "(10,20)(110,20)(110,50)(10,50)"


def test_missing_end_time_is_left_blank_but_keeps_the_column():
    line = format_line(_Clock().now, None, _row(), "正常", "m")

    assert _fields(line)[1] == ""
    assert len(_fields(line)) == 7


def test_newlines_and_tabs_inside_text_never_break_the_one_line_per_block_rule():
    assert clean_field("第一行\n第二行") == "第一行 / 第二行"
    assert clean_field("a\r\n\r\nb") == "a / b"
    assert clean_field("a\tb") == "a / b"
    assert clean_field("a \n b") == "a / b"
    assert clean_field("a\u2028b") == "a / b"
    line = format_line(_Clock().now, None, _row("上\n下", "up\tdown"), "正常", "m")
    assert len(_fields(line)) == 7
    assert "\n" not in line


def test_control_characters_and_unencodable_characters_are_removed_not_fatal():
    assert clean_field("a\x00b\x07c") == "abc"
    cleaned = clean_field("壞\ud800字")
    cleaned.encode("utf-8")  # 不會丟例外：寫檔時不會因為單一怪字元整批失敗
    assert cleaned.startswith("壞") and cleaned.endswith("字")


def test_time_has_milliseconds():
    clock = _Clock()

    assert format_time(clock.now + 0.0049) == "2026-10-14 12:00:00.005"
    assert format_time(clock.now + 61.2) == "2026-10-14 12:01:01.200"


def test_row_status():
    assert _row("こんにちは", "你好").status == "正常"
    assert _row("12:30", "12:30").status == "原樣"  # 譯文與原文相同：數字、已是中文、翻譯失敗保留原文
    assert _row("ドン！", "").status == "略過"  # 擬聲詞：不顯示


def test_directory_setting_empty_means_default_and_supports_home_and_env(tmp_path, monkeypatch):
    default = tmp_path / "ScreenTranslator" / "output"
    monkeypatch.setenv("MY_OUT", str(tmp_path / "somewhere"))

    assert resolve_directory("", default) == default
    assert resolve_directory("   ", default) == default
    assert resolve_directory(str(tmp_path / "x"), default) == tmp_path / "x"
    assert resolve_directory("$MY_OUT", default) == tmp_path / "somewhere"
    assert resolve_directory("txt", default) == tmp_path / "ScreenTranslator" / "txt"  # 相對路徑：相對於程式資料夾
    assert resolve_directory("~", default) == Path.home()


# ----------------------------------------------------------------------------- 一句字幕的生命週期
def test_a_subtitle_is_written_once_when_it_is_replaced_with_its_start_and_end_time():
    recorder, sink, clock = _recorder()
    recorder.show([_row("A", "甲")], "m")
    clock.advance(2.0)
    assert sink.lines == []  # 還在顯示：先不寫

    recorder.show([_row("B", "乙")], "m")  # 下一句出現 = 上一句結束

    (line,) = sink.lines
    fields = _fields(line)
    assert fields[0] == "2026-10-14 12:00:00.000" and fields[1] == "2026-10-14 12:00:02.000"
    assert fields[2:5] == ["A", "甲", "正常"]
    assert sink.batches[0][0] == "2026-10-14"  # 寫進開始那一天的檔案


def test_several_blocks_of_one_subtitle_are_written_together_with_the_same_times():
    recorder, sink, clock = _recorder()
    recorder.show([_row("上", "up", y=0), _row("下", "down", y=40)], "m")
    clock.advance(1.0)
    recorder.end("stop")

    assert [_fields(line)[2] for line in sink.lines] == ["上", "下"]
    assert {tuple(_fields(line)[:2]) for line in sink.lines} == {("2026-10-14 12:00:00.000", "2026-10-14 12:00:01.000")}


def test_the_same_subtitle_seen_again_is_the_same_entry_not_a_new_line():
    recorder, sink, clock = _recorder()
    recorder.show([_row("こんにちは", "你好")], "m")
    clock.advance(1.0)
    recorder.show([_row("こんにちは", "你好", y=99)], "m")  # 例如「重新整理」重翻、清除快取後重翻、位置抖動
    clock.advance(1.0)
    recorder.end("stop")

    (line,) = sink.lines
    assert _fields(line)[0] == "2026-10-14 12:00:00.000" and _fields(line)[1] == "2026-10-14 12:00:02.000"


def test_same_text_translated_by_another_model_is_a_new_entry_so_models_can_be_compared():
    recorder, sink, clock = _recorder()
    recorder.show([_row("こんにちは", "你好")], "offline/qwen2.5-1.5b")
    clock.advance(1.0)
    recorder.show([_row("こんにちは", "您好")], "offline/qwen2.5-3b")
    clock.advance(1.0)
    recorder.end("stop")

    assert [(_fields(line)[3], _fields(line)[5]) for line in sink.lines] == [
        ("你好", "offline/qwen2.5-1.5b"),
        ("您好", "offline/qwen2.5-3b"),
    ]


def test_spacing_and_width_differences_from_ocr_do_not_split_the_entry():
    recorder, sink, _clock = _recorder()
    recorder.show([_row("ＡＢＣ　def", "x")], "m")
    recorder.show([_row("ABC def", "x")], "m")  # 全形半形、空白差異：OCR 常抖動，與穩定化的判斷一致

    assert sink.lines == []
    recorder.end("stop")
    assert len(sink.lines) == 1


def test_subtitle_that_flashes_by_is_marked_brief_not_dropped():
    recorder, sink, clock = _recorder(min_seconds=0.3)
    recorder.show([_row("紡ぐた", "紡")], "m")  # 淡入到一半被辨識到的殘句
    clock.advance(0.1)
    recorder.show([_row("紡ぐたったひとつ絆", "只有一個羈絆")], "m")
    clock.advance(3.0)
    recorder.show([_row("次の", "下一句")], "m")

    assert [_fields(line)[4] for line in sink.lines] == ["正常,短暫", "正常"]
    assert _fields(sink.lines[0])[2] == "紡ぐた"  # 還是寫下來了，由使用者事後過濾


def test_brief_threshold_is_configurable_and_only_applies_to_replaced_entries():
    recorder, sink, clock = _recorder(min_seconds=1.0)
    recorder.show([_row("A", "甲")], "m")
    clock.advance(0.5)
    recorder.show([_row("B", "乙")], "m")
    clock.advance(0.2)
    recorder.end("stop")  # 使用者按了停止：不是殘句，不標短暫

    assert [_fields(line)[4] for line in sink.lines] == ["正常,短暫", "正常"]


def test_the_end_time_is_when_the_text_disappeared_not_when_the_timer_finally_fired():
    recorder, sink, clock = _recorder()
    recorder.show([_row("A", "甲")], "m")
    clock.advance(2.0)
    recorder.note_text_gone()  # OCR 這一輪抓不到字
    clock.advance(1.0)  # 等「無字幕自動關閉」的 1 秒
    recorder.end("gone")

    assert _fields(sink.lines[0])[1] == "2026-10-14 12:00:02.000"


def test_text_that_comes_back_before_the_timer_cancels_the_gone_time():
    recorder, sink, clock = _recorder()
    recorder.show([_row("A", "甲")], "m")
    clock.advance(2.0)
    recorder.note_text_gone()
    clock.advance(0.5)
    recorder.cancel_text_gone()  # 字又出現了
    clock.advance(5.0)
    recorder.end("stop")

    assert _fields(sink.lines[0])[1] == "2026-10-14 12:00:07.500"


def test_stopping_while_the_text_was_already_gone_uses_the_gone_time():
    recorder, sink, clock = _recorder()
    recorder.show([_row("A", "甲")], "m")
    clock.advance(2.0)
    recorder.note_text_gone()
    clock.advance(0.4)
    recorder.end("stop")

    assert _fields(sink.lines[0])[1] == "2026-10-14 12:00:02.000"


def test_ending_with_nothing_on_screen_does_nothing():
    recorder, sink, _clock = _recorder()

    recorder.end("stop")
    recorder.end("gone")
    recorder.note_text_gone()
    recorder.finish_single_shot()

    assert sink.batches == []
    assert not recorder.is_open


def test_an_empty_result_does_not_start_an_entry():
    recorder, sink, _clock = _recorder()
    recorder.show([], "m")
    recorder.end("stop")

    assert sink.batches == []


def test_a_new_entry_after_the_text_vanished_is_written_separately():
    recorder, sink, clock = _recorder()
    recorder.show([_row("はい", "是")], "m")
    clock.advance(1.0)
    recorder.end("gone")
    clock.advance(5.0)
    recorder.show([_row("はい", "是")], "m")  # 字幕消失之後同一句再出現：是新的一次
    clock.advance(1.0)
    recorder.end("stop")

    assert len(sink.lines) == 2


def test_an_entry_belongs_to_the_day_it_started_even_if_it_ends_after_midnight():
    recorder, sink, clock = _recorder()
    clock.now = time.mktime((2026, 10, 14, 23, 59, 59, 0, 0, -1))
    recorder.show([_row("A", "甲")], "m")
    clock.advance(5.0)
    recorder.end("stop")

    assert sink.batches[0][0] == "2026-10-14"
    assert _fields(sink.lines[0])[1].startswith("2026-10-15 00:00:04")


# ----------------------------------------------------------------------------- 單次截圖
def test_single_shot_not_followed_by_monitoring_is_written_at_once_with_a_blank_end():
    recorder, sink, _clock = _recorder()
    recorder.show([_row("A", "甲")], "m", single_shot=True)

    recorder.finish_single_shot()

    (line,) = sink.lines
    assert _fields(line)[1] == ""
    assert not recorder.is_open


def test_monitoring_taking_over_a_single_shot_gives_one_line_with_a_real_end_time():
    recorder, sink, clock = _recorder()
    recorder.show([_row("A", "甲")], "m", single_shot=True)
    clock.advance(0.5)
    recorder.show([_row("A", "甲")], "m")  # 監控接著辨識到同一句
    recorder.finish_single_shot()  # 監控已經在跑：不能在這裡寫出去
    assert sink.lines == []

    clock.advance(3.0)
    recorder.end("stop")

    (line,) = sink.lines
    assert _fields(line)[0] == "2026-10-14 12:00:00.000" and _fields(line)[1] == "2026-10-14 12:00:03.500"


def test_finish_single_shot_leaves_a_running_monitor_entry_alone():
    recorder, sink, _clock = _recorder()
    recorder.show([_row("A", "甲")], "m")  # 監控中的字幕

    recorder.finish_single_shot()

    assert sink.lines == [] and recorder.is_open


# ----------------------------------------------------------------------------- 註解
def test_comment_lines_start_with_a_hash_and_a_timestamp():
    recorder, sink, _clock = _recorder()

    recorder.comment("監控開始 模式=region 框=(0,0,1920,1080)")

    assert sink.lines == ["# 2026-10-14 12:00:00.000 監控開始 模式=region 框=(0,0,1920,1080)"]
    assert sink.batches[0][0] == "2026-10-14"
