"""文字輸出接進監控與管線：字幕被取代／消失／停止／離開程式時寫出，設定、托盤與設定頁。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import DEFAULTS, AppConfig
from app.models import CaptureInfo, TextRegion
from services.transcript_writer import BOM
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from ui.tray_icon import TrayIcon
from utils.stability import TextStabilizer

DAY = "2026-10-14"


class _Clock:
    def __init__(self) -> None:
        self.now = time.mktime((2026, 10, 14, 12, 0, 0, 0, 0, -1))

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeOverlay:
    def __init__(self) -> None:
        self.cleared = 0

    def show_regions(self, capture, regions, animate_new_only=False):
        return True

    def clear_all(self):
        self.cleared += 1

    def hide_all(self, animate=False):
        pass

    def is_visible(self):
        return True


class _Translator:
    name = "offline"

    def model_label(self) -> str:
        return "offline/qwen2.5-1.5b-instruct-gguf-q4_k_m"

    def reset_context(self) -> None:
        pass


def _payload(texts=("こんにちは",), translated=None, x=5, y=5):
    capture = CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=(0, 0, 120, 40), mode="region")
    regions = [
        TextRegion(
            text=text,
            translated_text=(translated or {}).get(text, f"譯:{text}"),
            x=x,
            y=y + index * 20,
            width=60,
            height=15,
        )
        for index, text in enumerate(texts)
    ]
    return {"capture": capture, "regions": regions, "recognized_count": len(regions), "translated_count": len(regions), "failed_count": 0}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(app)
    instance.config.set("output.directory", str(tmp_path / "out"))
    instance._apply_transcript_settings()
    instance.clock = _Clock()
    instance._transcript.clock = instance.clock
    instance.overlay_manager = _FakeOverlay()
    instance.translator = _Translator()
    instance._monitor_active = True
    instance._monitor_request = {"mode": "region", "bbox": (10, 20, 330, 260), "hwnd": 0}
    instance._monitor_stabilizer = TextStabilizer(rounds=1)
    instance._capture_session = None
    instance._pipeline_succeeded = False
    yield instance
    instance._transcript_writer.close(timeout=2.0)
    instance.floating_status.deleteLater()


def _lines(controller, tmp_path):
    """寫完佇列之後，當天檔案的資料行與註解行（去掉 BOM 與表頭）。"""
    assert controller._transcript_writer.flush(timeout=3.0)
    path = tmp_path / "out" / f"{DAY}.txt"
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8").removeprefix(BOM)
    return [line for line in text.split("\r\n") if line and not line.startswith("# ScreenTranslator") and not line.startswith("# 開始時間")]


def _fields(line: str) -> list[str]:
    return line.split("\t")


# ----------------------------------------------------------------------------- 監控中
def test_a_subtitle_is_written_when_the_next_one_replaces_it_with_text_translation_corners_and_model(controller, tmp_path):
    controller.on_pipeline_result(_payload(("こんにちは",)))
    controller.clock.advance(2.0)
    assert _lines(controller, tmp_path) == []  # 還在顯示：先不寫

    controller.on_pipeline_result(_payload(("さようなら",), x=40))

    (line,) = _lines(controller, tmp_path)
    fields = _fields(line)
    assert fields[0] == "2026-10-14 12:00:00.000" and fields[1] == "2026-10-14 12:00:02.000"
    assert fields[2:6] == ["こんにちは", "譯:こんにちは", "正常", "offline/qwen2.5-1.5b-instruct-gguf-q4_k_m"]
    assert fields[6] == "(5,5)(65,5)(65,20)(5,20)"  # 螢幕絕對座標：x, y, x+寬, y+高


def test_a_result_that_changes_nothing_on_screen_adds_no_line(controller, tmp_path):
    payload = _payload()
    controller.on_pipeline_result(payload)
    controller.on_pipeline_result(_payload())  # 內容與位置都沒變：畫面不重畫，也不記一筆
    controller.stop_monitor()

    assert len([line for line in _lines(controller, tmp_path) if not line.startswith("#")]) == 1


def test_stopping_the_monitor_writes_the_open_subtitle_and_a_stop_comment(controller, tmp_path):
    controller.on_pipeline_result(_payload())
    controller.clock.advance(3.0)

    controller.stop_monitor()

    lines = _lines(controller, tmp_path)
    assert _fields(lines[0])[1] == "2026-10-14 12:00:03.000"
    assert lines[-1].endswith("監控停止") and lines[-1].startswith("# 2026-10-14 12:00:03.000")


def test_the_start_comment_records_the_mode_the_frame_and_the_model(controller, tmp_path):
    controller._transcript_note_monitor(started=True)

    (line,) = _lines(controller, tmp_path)
    assert line.startswith("# 2026-10-14 12:00:00.000 監控開始 ")
    assert "模式=region" in line and "框=(10,20,330,260)" in line and "offline/qwen2.5-1.5b-instruct-gguf-q4_k_m" in line


def test_end_time_is_when_the_text_disappeared_not_when_the_close_timer_fires(controller, tmp_path):
    controller.on_pipeline_result(_payload())
    controller.clock.advance(2.0)
    controller._arm_no_text_timer()  # OCR 這一輪抓不到字
    controller.clock.advance(1.0)

    controller._close_translation_after_no_text()  # 「無字幕自動關閉」的計時到了

    (line,) = _lines(controller, tmp_path)
    assert _fields(line)[1] == "2026-10-14 12:00:02.000"
    assert controller.overlay_manager.cleared == 1


def test_text_coming_back_before_the_timer_keeps_the_subtitle_going(controller, tmp_path):
    controller.on_pipeline_result(_payload())
    controller.clock.advance(2.0)
    controller._arm_no_text_timer()
    controller.clock.advance(0.5)
    controller.on_pipeline_result(_payload(("こんにちは",)))  # 又抓到字（相同字幕：不重畫）
    controller.clock.advance(4.0)

    controller.stop_monitor()

    (line, _stop_comment) = _lines(controller, tmp_path)
    assert _fields(line)[1] == "2026-10-14 12:00:06.500"


def test_the_same_subtitle_after_it_vanished_is_a_new_entry(controller, tmp_path):
    controller.on_pipeline_result(_payload(("はい",)))
    controller.clock.advance(1.0)
    controller._close_translation_after_no_text()
    controller.clock.advance(4.0)
    controller.on_pipeline_result(_payload(("はい",)))
    controller.clock.advance(1.0)
    controller.stop_monitor()

    data = [line for line in _lines(controller, tmp_path) if not line.startswith("#")]
    assert [_fields(line)[0] for line in data] == ["2026-10-14 12:00:00.000", "2026-10-14 12:00:05.000"]


def test_translation_kept_with_the_original_is_written_as_the_translation_only(controller, tmp_path):
    controller.on_pipeline_result(_payload(("こんにちは",), translated={"こんにちは": "こんにちは\n你好"}))
    controller.stop_monitor()

    (line, _stop) = _lines(controller, tmp_path)
    assert _fields(line)[3] == "你好"


def test_a_block_translated_to_nothing_is_recorded_as_skipped(controller, tmp_path):
    controller.on_pipeline_result(_payload(("ドン！",), translated={"ドン！": ""}))
    controller.stop_monitor()

    (line, _stop) = _lines(controller, tmp_path)
    assert _fields(line)[2:5] == ["ドン！", "", "略過"]


def test_two_blocks_of_one_result_share_the_times_and_keep_reading_order(controller, tmp_path):
    controller.on_pipeline_result(_payload(("上の行", "下の行")))
    controller.clock.advance(1.0)
    controller.stop_monitor()

    data = [line for line in _lines(controller, tmp_path) if not line.startswith("#")]
    assert [_fields(line)[2] for line in data] == ["上の行", "下の行"]
    assert {tuple(_fields(line)[:2]) for line in data} == {("2026-10-14 12:00:00.000", "2026-10-14 12:00:01.000")}


def test_changing_the_translation_model_while_the_text_stays_starts_a_new_entry(controller, tmp_path):
    controller.on_pipeline_result(_payload(("こんにちは",)))
    controller.clock.advance(1.0)
    controller.translator = type("T", (), {"name": "offline", "model_label": lambda self: "offline/qwen2.5-3b-instruct-gguf-q4_k_m", "reset_context": lambda self: None})()
    controller._forget_displayed_translations()  # 設定存檔換了模型：下一輪重新翻譯、重新顯示
    controller.on_pipeline_result(_payload(("こんにちは",), translated={"こんにちは": "你好啊"}))
    controller.clock.advance(1.0)
    controller.stop_monitor()

    data = [line for line in _lines(controller, tmp_path) if not line.startswith("#")]
    assert [(_fields(line)[3], _fields(line)[5]) for line in data] == [
        ("譯:こんにちは", "offline/qwen2.5-1.5b-instruct-gguf-q4_k_m"),
        ("你好啊", "offline/qwen2.5-3b-instruct-gguf-q4_k_m"),
    ]


def test_without_a_model_label_the_service_name_is_used(controller, tmp_path):
    controller.translator = None
    controller.on_pipeline_result(_payload())
    controller.stop_monitor()

    (line, _stop) = _lines(controller, tmp_path)
    assert _fields(line)[5] == controller.config.get("translation.service")


# ----------------------------------------------------------------------------- 單次截圖
def test_single_capture_without_monitoring_is_written_at_once_with_a_blank_end(controller, tmp_path):
    controller._monitor_active = False
    controller._capture_session = object()
    controller.on_pipeline_result(_payload())

    controller._transcript_finish_single_shot()

    (line,) = _lines(controller, tmp_path)
    assert _fields(line)[1] == ""


def test_single_capture_followed_by_monitoring_gives_one_line_not_two(controller, tmp_path):
    controller._monitor_active = False
    controller._capture_session = object()
    controller.on_pipeline_result(_payload())
    controller._capture_session = None
    controller._monitor_active = True  # 自動進入即時監控
    controller._transcript_finish_single_shot()  # 監控已經在跑：不能在這裡寫出去
    assert _lines(controller, tmp_path) == []
    controller._monitor_stabilizer.reset()
    controller._monitor_signature = ""
    controller.on_pipeline_result(_payload())  # 監控第一輪辨識到同一句
    controller.clock.advance(5.0)
    controller.stop_monitor()

    data = [line for line in _lines(controller, tmp_path) if not line.startswith("#")]
    assert len(data) == 1 and _fields(data[0])[1] == "2026-10-14 12:00:05.000"


# ----------------------------------------------------------------------------- 設定與離開
def test_turning_the_output_off_writes_nothing_new(controller, tmp_path):
    controller.config.set("output.enabled", False)

    controller.on_pipeline_result(_payload())
    controller._transcript_note_monitor(started=True)
    controller.stop_monitor()

    assert _lines(controller, tmp_path) == []


def test_turning_it_off_while_a_subtitle_is_showing_still_finishes_that_line(controller, tmp_path):
    controller.on_pipeline_result(_payload())
    controller.clock.advance(1.0)
    controller.config.set("output.enabled", False)

    controller._apply_transcript_settings()
    controller.on_pipeline_result(_payload(("次の",)))  # 關掉之後的字幕不再記錄

    (line,) = _lines(controller, tmp_path)
    assert _fields(line)[2] == "こんにちは" and _fields(line)[1] == "2026-10-14 12:00:01.000"


def test_changing_the_folder_or_threshold_in_settings_takes_effect(controller, tmp_path):
    controller.config.set("output.directory", str(tmp_path / "elsewhere"))
    controller.config.set("output.min_seconds", 1.5)

    controller._apply_transcript_settings()

    assert controller._transcript_writer.directory == tmp_path / "elsewhere"
    assert controller._transcript.min_seconds == 1.5


def test_default_folder_is_next_to_the_config_and_empty_means_default(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "ScreenTranslator" / "config.json"))
    instance = Application(app)
    try:
        assert instance.transcript_directory() == tmp_path / "ScreenTranslator" / "output"
    finally:
        instance.floating_status.deleteLater()


def test_leaving_the_program_writes_the_open_subtitle_and_closes_the_file(controller, tmp_path):
    controller.on_pipeline_result(_payload())
    controller.clock.advance(1.0)

    controller._transcript_shutdown()

    path = tmp_path / "out" / f"{DAY}.txt"
    assert "こんにちは" in path.read_bytes().decode("utf-8")
    assert controller._transcript_writer.submit(DAY, ["x"]) is False


def test_a_bug_while_recording_never_breaks_the_translation_display(controller, tmp_path, monkeypatch):
    monkeypatch.setattr(controller._transcript, "show", lambda *args, **kwargs: 1 / 0)

    controller.on_pipeline_result(_payload())

    assert controller._pipeline_succeeded is True  # 畫面照常；錯誤只記在紀錄檔


def test_a_write_problem_reaches_the_main_thread_as_one_status_message(controller, tmp_path):
    messages: list[str] = []
    controller.set_status = messages.append
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / f"{DAY}.txt").mkdir()  # 檔名被占住：開檔失敗，模擬被其他程式鎖住
    controller._transcript_writer._retries = 0
    controller.on_pipeline_result(_payload())
    controller.clock.advance(1.0)
    controller.stop_monitor()
    assert controller._transcript_writer.flush(timeout=3.0)

    deadline = time.monotonic() + 3.0
    while not any("輸出檔" in message for message in messages) and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)

    assert len([message for message in messages if "輸出檔" in message]) == 1
    assert list((tmp_path / "out").glob(f"{DAY}.part-*.txt"))


def test_open_output_folder_creates_it_writes_pending_lines_and_opens_it(controller, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("app.transcript_mixin.QDesktopServices.openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    controller.on_pipeline_result(_payload())
    controller.stop_monitor()

    controller.open_output_folder()

    assert opened == [str(tmp_path / "out")]
    assert (tmp_path / "out" / f"{DAY}.txt").exists()  # 打開時看得到最新的


def test_tray_menu_has_an_open_output_folder_entry(controller, monkeypatch):
    opened = []
    controller.open_output_folder = lambda: opened.append(True)
    tray = TrayIcon(controller)
    try:
        assert tray.act_open_output.text() == "開啟輸出資料夾"
        tray.act_open_output.trigger()
        assert opened == [True]
    finally:
        tray.deleteLater()


# ----------------------------------------------------------------------------- 設定檔與設定頁
def test_defaults_are_on_with_the_default_folder_and_a_short_threshold():
    assert DEFAULTS["output"] == {"enabled": True, "directory": "", "min_seconds": 0.3}


def test_broken_output_settings_fall_back_to_safe_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"output": {"enabled": "yes", "directory": 5, "min_seconds": "abc"}}', encoding="utf-8")

    output = AppConfig(path).section("output")

    assert output == {"enabled": True, "directory": "", "min_seconds": 0.3}


def test_threshold_is_clamped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"output": {"enabled": false, "directory": "  D:\\\\txt  ", "min_seconds": 99}}', encoding="utf-8")

    output = AppConfig(path).section("output")

    assert output == {"enabled": False, "directory": "D:\\txt", "min_seconds": 10.0}


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def test_settings_page_loads_and_saves_the_output_options(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    try:
        assert dialog.chk_output.isChecked()
        assert dialog.output_dir.text() == ""
        assert dialog.output_dir.placeholderText() == str(tmp_path / "output")  # 留空 = 預設資料夾，顯示給使用者看
        assert dialog.pages.widget(0).widget().isAncestorOf(dialog.chk_output)  # 一般頁

        dialog.chk_output.setChecked(False)
        dialog.output_dir.setText("  D:\\字幕  ")
        dialog._commit_accept(dialog._validated_hotkeys())

        assert config.get("output.enabled") is False
        assert config.get("output.directory") == "D:\\字幕"
    finally:
        dialog.deleteLater()


def test_settings_page_open_button_creates_and_opens_the_folder(qapp, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("ui.settings_dialog.QDesktopServices.openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    try:
        dialog.output_dir.setText(str(tmp_path / "mine"))
        dialog.btn_open_output.click()

        assert opened == [str(tmp_path / "mine")]
        assert (tmp_path / "mine").is_dir()
    finally:
        dialog.deleteLater()
