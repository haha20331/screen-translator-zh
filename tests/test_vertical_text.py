"""直向文字：版面（欄由右往左）、翻譯流程、主視窗的「文字方向」選擇（預設橫向，只在這次執行有效）。"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app import pipeline_mixin
from app.application import Application
from app.config import AppConfig
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCRLine
from ui.main_window import MainWindow
from ui.style import apply_style
from utils.layout_utils import ocr_lines_to_regions, translation_lines, vertical_regions
from workers.translation_worker import PipelineTask

IMAGE_WIDTH, IMAGE_HEIGHT = 400, 300


def _regions(lines, offset=(0, 0), **overrides):
    settings = dict(
        image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, offset_x=offset[0], offset_y=offset[1],
        min_confidence=0.6, y_tolerance_ratio=0.5, x_gap_ratio=0.8,
        target_language="zh-hant", max_width_ratio=1.0, merge_centered=False, han_is_source=True,
    )
    settings.update(overrides)
    return vertical_regions(lines, **settings)


def _column(text, x, y=20, width=50, height=240):
    return OCRLine(text, (x, y, width, height), 0.95)


# ----------------------------------------------------------------------------- 版面
def test_a_single_column_stays_where_it_is_with_the_column_width_as_the_glyph_size():
    (region,) = _regions([_column("夜空に浮かぶ星を数えて", 40, 10, 50, 200)])

    assert (region.x, region.y, region.width, region.height) == (40, 10, 50, 200)
    assert region.text == "夜空に浮かぶ星を数えて" and region.vertical
    assert region.source_line_height == 50  # 字的大小是欄的寬度，不是欄的長度


def test_two_neighbouring_columns_become_one_paragraph_read_from_the_right_column_first():
    right, left = _column("今日はいい天気ですね", 320), _column("散歩に行きましょう", 262)

    (region,) = _regions([left, right])  # OCR 給的順序不一定是閱讀順序

    assert region.text == "今日はいい天気ですね 散歩に行きましょう"
    assert (region.x, region.y, region.width, region.height) == (262, 20, 108, 240)  # 兩欄合起來的框
    assert region.source_line_count == 2


def test_columns_that_are_far_apart_stay_separate_and_are_ordered_right_to_left():
    regions = _regions([_column("左邊的欄", 20), _column("右邊的欄", 340)])

    assert [region.text for region in regions] == ["右邊的欄", "左邊的欄"]


def test_a_column_that_the_ocr_cut_into_two_pieces_is_joined_from_top_to_bottom():
    lower = OCRLine("星を数えて", (40, 130, 50, 100), 0.95)
    upper = OCRLine("夜空に浮かぶ", (40, 20, 50, 105), 0.95)

    (region,) = _regions([lower, upper])

    assert region.text == "夜空に浮かぶ 星を数えて"
    assert (region.x, region.y, region.width, region.height) == (40, 20, 50, 210)


def test_the_screen_offset_moves_the_regions_to_where_the_capture_is_on_screen():
    (region,) = _regions([_column("夜空", 40, 10, 50, 200)], offset=(1000, 500))

    assert (region.x, region.y, region.width, region.height) == (1040, 510, 50, 200)


def test_low_confidence_lines_are_dropped_like_in_horizontal_mode():
    weak = OCRLine("雑音", (200, 20, 50, 100), 0.2)

    regions = _regions([_column("夜空", 40), weak])

    assert [region.text for region in regions] == ["夜空"]


def test_horizontal_layout_is_untouched_and_never_marked_vertical():
    regions = ocr_lines_to_regions([OCRLine("夜空に浮かぶ星", (40, 10, 300, 40), 0.95)], 0, 0)

    assert [region.vertical for region in regions] == [False]


def test_the_panel_lists_vertical_translations_from_the_rightmost_column_first():
    def region(text, translated, x):
        return TextRegion(text=text, translated_text=translated, x=x, y=20, width=50, height=200, vertical=True)

    lines = translation_lines([region("左", "第二句", 100), region("右", "第一句", 300), region("中", "第一點五句", 200)])

    assert lines == ["第一句", "第一點五句", "第二句"]


def test_horizontal_panel_order_is_still_top_to_bottom():
    def region(translated, y):
        return TextRegion(text=translated + "原文", translated_text=translated, x=10, y=y, width=300, height=40)

    assert translation_lines([region("第二行", 200), region("第一行", 20)]) == ["第一行", "第二行"]


# ----------------------------------------------------------------------------- 翻譯流程
class _OCR:
    supports_vertical = True
    benefits_from_upscale = False  # 像 RapidOCR：管線不放大影像，OCR 座標就是影像座標

    def __init__(self, lines):
        self.lines = lines

    def recognize(self, _image, lang="auto"):
        return list(self.lines)


class _Translator:
    han_is_source = True
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self):
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        return [f"譯:{text}" for text in texts]


def _run(tmp_path, engine, *, vertical):
    config = AppConfig(tmp_path / "config.json")
    capture = CaptureInfo(
        image=np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 255, dtype=np.uint8), bbox=(100, 50, 100 + IMAGE_WIDTH, 50 + IMAGE_HEIGHT),
        monitor_indices=[0], mode="region",  # 擷取在螢幕上的位置（100, 50）
    )
    translator = _Translator()
    task = PipelineTask(capture, engine, translator, config, vertical=vertical)
    results, errors = [], []
    task.result.connect(results.append)
    task.error.connect(errors.append)
    task.run()
    return translator, results, errors


def test_the_pipeline_reads_vertical_columns_in_order_and_translates_them_as_one_paragraph(tmp_path):
    engine = _OCR([_column("散歩に行きましょう", 262), _column("今日はいい天気ですね", 320)])

    translator, results, errors = _run(tmp_path, engine, vertical=True)

    assert not errors
    assert translator.calls == [["今日はいい天気ですね 散歩に行きましょう"]]
    (region,) = results[0]["regions"]
    assert region.vertical and (region.x, region.y) == (100 + 262, 50 + 20)  # 擷取在螢幕上的位置也加進去了
    assert translation_lines(results[0]["regions"]) == ["譯:今日はいい天気ですね 散歩に行きましょう"]


def test_the_same_columns_in_horizontal_mode_are_not_read_as_a_paragraph(tmp_path):
    engine = _OCR([_column("散歩に行きましょう", 262), _column("今日はいい天気ですね", 320)])

    translator, results, _ = _run(tmp_path, engine, vertical=False)

    assert not any(region.vertical for region in results[0]["regions"])


def test_an_engine_that_cannot_read_vertical_text_says_so_instead_of_returning_rubbish(tmp_path):
    engine = _OCR([_column("夜空", 40)])
    engine.supports_vertical = False

    translator, results, errors = _run(tmp_path, engine, vertical=True)

    assert results == [] and translator.calls == []
    assert errors and "直向" in errors[0] and "rapidocr" in errors[0]


def test_horizontal_mode_does_not_care_whether_the_engine_supports_vertical(tmp_path):
    engine = _OCR([OCRLine("夜空に浮かぶ星", (40, 10, 300, 40), 0.95)])
    engine.supports_vertical = False

    translator, results, errors = _run(tmp_path, engine, vertical=False)

    assert not errors and translator.calls == [["夜空に浮かぶ星"]]


# ----------------------------------------------------------------------------- 控制器
@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


@pytest.fixture
def controller(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(qapp)
    yield instance
    instance.floating_status.deleteLater()


def test_the_default_direction_is_horizontal_and_is_not_saved(controller, tmp_path):
    assert controller._text_vertical is False

    controller.set_text_direction(True)
    controller.config.save()

    assert controller._text_vertical is True
    assert "text_direction" not in (tmp_path / "config.json").read_text(encoding="utf-8")  # 只在這次執行有效


def test_changing_the_direction_while_monitoring_forces_a_fresh_look_at_the_screen(controller):
    controller._monitor_active = True
    controller._monitor_known = {"夜空": "夜空的譯文"}
    controller._monitor_signature = "shown"
    controller._monitor_frame_hash = "frame"

    controller.set_text_direction(True)

    assert controller._monitor_known == {} and controller._monitor_signature == "" and controller._monitor_frame_hash == ""


def test_choosing_the_same_direction_again_does_nothing(controller):
    controller._monitor_active = True
    controller._monitor_frame_hash = "frame"

    controller.set_text_direction(False)

    assert controller._monitor_frame_hash == "frame"


def test_an_engine_that_cannot_do_vertical_is_replaced_by_rapidocr_only_for_vertical_text(controller, monkeypatch):
    plain = SimpleNamespace(supports_vertical=False)
    fallback = SimpleNamespace(supports_vertical=True)
    created = []
    monkeypatch.setattr(pipeline_mixin, "create_ocr_engine", lambda *args: created.append(args) or fallback)
    controller.ocr_engine = plain

    assert controller._ocr_engine_for_direction() is plain  # 橫向：照原本的引擎
    controller._text_vertical = True
    assert controller._ocr_engine_for_direction() is fallback
    assert controller._ocr_engine_for_direction() is fallback and len(created) == 1  # 備用引擎只建一次
    assert created[0][0] == "rapidocr"


def test_an_engine_that_can_do_vertical_is_used_as_it_is(controller, monkeypatch):
    capable = SimpleNamespace(supports_vertical=True)
    controller.ocr_engine = capable
    controller._text_vertical = True
    monkeypatch.setattr(pipeline_mixin, "create_ocr_engine", lambda *args: pytest.fail("不必建備用引擎"))

    assert controller._ocr_engine_for_direction() is capable


def test_when_no_vertical_engine_can_be_built_the_original_engine_is_returned_and_the_pipeline_explains(controller, monkeypatch):
    plain = SimpleNamespace(supports_vertical=False)

    def broken(*args):
        raise RuntimeError("RapidOCR 沒有安裝")

    monkeypatch.setattr(pipeline_mixin, "create_ocr_engine", broken)
    controller.ocr_engine = plain
    controller._text_vertical = True

    assert controller._ocr_engine_for_direction() is plain


def test_the_ocr_settings_change_drops_the_spare_vertical_engine(controller, monkeypatch):
    controller._vertical_ocr_engine = object()
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    monkeypatch.setattr(controller, "_rebuild_translator", lambda: 0)
    monkeypatch.setattr(controller, "_warmup", lambda: "")
    monkeypatch.setattr(controller, "_refresh_appearance", lambda: None)
    monkeypatch.setattr(controller, "refresh_monitor_map", lambda: None)
    monkeypatch.setattr(controller, "_apply_hotkeys", lambda: None)
    monkeypatch.setattr(controller, "_apply_autostart", lambda: None)
    monkeypatch.setattr(controller, "_apply_transcript_settings", lambda: None)

    controller._apply_settings()

    assert controller._vertical_ocr_engine is None


# ----------------------------------------------------------------------------- 主視窗
class _Controller:
    def __init__(self, config):
        self.config = config
        self._text_vertical = False
        self.directions = []

    def set_text_direction(self, vertical):
        self.directions.append(vertical)
        self._text_vertical = vertical

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_the_main_window_offers_horizontal_by_default_and_tells_the_controller_when_changed(qapp, tmp_path):
    controller = _Controller(AppConfig(tmp_path / "config.json"))
    window = MainWindow(controller)

    assert [window.combo_direction.itemText(i) for i in range(window.combo_direction.count())] == ["橫向", "直向"]
    assert window.combo_direction.currentData() is False and controller.directions == []  # 預設橫向，開窗時不會誤觸發

    window.combo_direction.setCurrentIndex(1)
    window.combo_direction.setCurrentIndex(0)

    assert controller.directions == [True, False]
    window.deleteLater()


def test_the_main_window_shows_the_direction_the_controller_is_using(qapp, tmp_path):
    controller = _Controller(AppConfig(tmp_path / "config.json"))
    controller._text_vertical = True
    window = MainWindow(controller)

    assert window.combo_direction.currentData() is True and controller.directions == []  # 同步畫面不能反過來改控制器
    window.deleteLater()
