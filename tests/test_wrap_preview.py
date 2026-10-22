"""多行合併範例：範例上的距離要與真正的合併規則（merge_wrapped_labels）一致，設定要存得進去、流程要吃得到。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCRLine
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from ui.wrap_preview import (
    SAMPLE_LINE_HEIGHT,
    SAMPLE_TEXTS,
    SEPARATE_EXTRA,
    WrapPreview,
    wrap_limits,
    wrap_summary,
)
from utils import layout_utils
from utils.layout_utils import merge_wrapped_labels, vertical_regions
from workers.translation_worker import PipelineTask

H = SAMPLE_LINE_HEIGHT
CAPTURE_WIDTH = 2000  # 夠寬，單行寬度上限不會擋到範例裡的行


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _region(text, x, y, width=120):
    return TextRegion(text=text, x=x, y=y, width=width, height=H, source_line_height=float(H))


def _merge(gap_ratio, align_ratio, second_x, second_y, *, centered=False, second_width=120):
    first = _region(SAMPLE_TEXTS[0], 0, 100, 100)
    second = _region(SAMPLE_TEXTS[1], second_x, second_y, second_width)
    return merge_wrapped_labels(
        [first, second], "zh-hant", CAPTURE_WIDTH, max_width_ratio=1.0, merge_centered=centered,
        gap_ratio=gap_ratio, align_ratio=align_ratio,
    )


# ----------------------------------------------------------------------------- 距離與真正的合併一致
def test_limits_use_the_same_functions_as_the_real_merge_including_the_minimums():
    assert wrap_limits(1.0, 0.35) == (24, 8)  # 預設：24 像素 × 1、× 0.35 再四捨五入
    assert wrap_limits(0.8, 0.35) == (19, 8)
    assert wrap_limits(0.0, 0.0) == (layout_utils.MIN_WRAP_GAP, layout_utils.MIN_WRAP_ALIGN) == (8, 5)  # 太小用最小像素數
    assert wrap_limits(0.8, 0.35, line_height=48) == (38, 17)  # 字大一倍，距離大一倍


@pytest.mark.parametrize("gap_ratio, align_ratio", [(0.8, 0.35), (0.0, 0.0), (1.5, 0.9), (2.0, 2.0)])
def test_the_farthest_position_in_the_example_really_merges_and_one_pixel_more_does_not(gap_ratio, align_ratio):
    gap, align = wrap_limits(gap_ratio, align_ratio)
    second_top = 100 + H + gap

    assert len(_merge(gap_ratio, align_ratio, align, second_top)) == 1  # 空隙與左緣誤差都在上限：併成一句
    assert len(_merge(gap_ratio, align_ratio, align, second_top + 1)) == 2  # 空隙多 1 像素：分開
    assert len(_merge(gap_ratio, align_ratio, align + 1, second_top)) == 2  # 左緣多 1 像素：分開
    assert len(_merge(gap_ratio, align_ratio, -align, second_top)) == 1  # 往左錯開一樣的距離也算


def test_centered_mode_measures_the_centres_instead_of_the_left_edges():
    gap, align = wrap_limits(0.8, 0.35)
    top = 100 + H + gap
    # 第二行比第一行（寬 100）窄：左緣差很多，但中心點只差 align 像素
    second_width = 40
    centre_x = 50 + align - second_width / 2  # 中心點在第一行中心（50）右邊 align 像素
    assert len(_merge(0.8, 0.35, int(centre_x), top, centered=False, second_width=second_width)) == 2
    assert len(_merge(0.8, 0.35, int(centre_x), top, centered=True, second_width=second_width)) == 1
    assert len(_merge(0.8, 0.35, int(centre_x) + 1, top, centered=True, second_width=second_width)) == 2


def test_the_default_gap_is_one_line_height_and_0_8_still_gives_the_old_hard_coded_behaviour():
    for dy, dx, expected in ((24, 8, 1), (25, 8, 2), (24, 9, 2)):  # 預設：行距 1 倍行高 = 24 像素、對齊 0.35 倍 = 8 像素
        first = _region("こんにちは、", 0, 100, 100)  # 合併會直接改動傳進去的第一個區塊，每次都要重建
        second = _region("お元気ですか？", dx, 100 + H + dy)
        assert len(merge_wrapped_labels([first, second], "zh-hant", CAPTURE_WIDTH, max_width_ratio=1.0)) == expected, (dy, dx)
    for dy, expected in ((19, 1), (20, 2)):  # 舊的寫死值 0.8 倍 = 19 像素
        first = _region("こんにちは、", 0, 100, 100)
        second = _region("お元気ですか？", 8, 100 + H + dy)
        assert len(merge_wrapped_labels([first, second], "zh-hant", CAPTURE_WIDTH, max_width_ratio=1.0, gap_ratio=0.8)) == expected, dy


@pytest.mark.parametrize("gap_ratio, align_ratio", [(0.8, 0.35), (0.0, 0.0), (2.0, 2.0)])
def test_the_picture_is_what_the_real_merge_does_with_those_three_lines(qapp, gap_ratio, align_ratio):
    preview = WrapPreview()
    preview.set_values(gap_ratio, align_ratio, True)
    info = preview.layout_info()
    gap, align = wrap_limits(gap_ratio, align_ratio)

    assert info.second.left() - info.first.left() == pytest.approx(align)  # 第二行在左緣誤差的邊緣
    assert info.second.top() - info.first.bottom() == pytest.approx(gap)  # 並且在行距的上限
    assert info.third.top() - info.second.bottom() == pytest.approx(gap + SEPARATE_EXTRA)  # 第三行在範圍外
    assert info.third.left() == pytest.approx(info.first.left())
    regions = [
        _region(text, round(rect.left()), round(rect.top()), round(rect.width()))
        for text, rect in zip(SAMPLE_TEXTS, (info.first, info.second, info.third))
    ]
    merged = merge_wrapped_labels(
        regions, "zh-hant", CAPTURE_WIDTH, max_width_ratio=1.0, gap_ratio=gap_ratio, align_ratio=align_ratio
    )
    assert [region.text for region in merged] == [f"{SAMPLE_TEXTS[0]} {SAMPLE_TEXTS[1]}", SAMPLE_TEXTS[2]]


def test_the_first_line_does_not_move_while_the_values_change_inside_the_settings_range(qapp):
    preview = WrapPreview()
    preview.set_values(0.0, 0.0, True)
    before = preview.layout_info().first

    preview.set_values(2.0, 2.0, False)

    assert preview.layout_info().first == before  # 只有第二、第三行移動，比較看得出變化


# ----------------------------------------------------------------------------- 畫面
def _count(preview: WrapPreview, color: str, tolerance: int = 40) -> int:
    image = preview.grab().toImage()
    target = QColor(color)
    return sum(
        1
        for y in range(image.height())
        for x in range(image.width())
        if abs(image.pixelColor(x, y).red() - target.red()) <= tolerance
        and abs(image.pixelColor(x, y).green() - target.green()) <= tolerance
        and abs(image.pixelColor(x, y).blue() - target.blue()) <= tolerance
    )


def test_painting_shows_the_merged_range_and_only_draws_the_centre_zone_when_centered_mode_is_on(qapp):
    preview = WrapPreview()
    pictures = []
    for gap_ratio, align_ratio in ((0.8, 0.35), (0.0, 0.0), (2.0, 2.0)):
        preview.set_values(gap_ratio, align_ratio, True)
        assert _count(preview, "#6EE7A8") > 30  # 「同一句」的虛線框
        assert _count(preview, "#FFC857") > 10  # 量距標示
        pictures.append(preview.grab().toImage())
    assert pictures[0] != pictures[1] != pictures[2]

    # 第一行正上方、只有「中心點範圍」會蓋到的那個點：置中模式開著才有顏色，關掉就是模擬畫面的底色
    preview.set_values(0.8, 0.35, True)
    info = preview.layout_info()
    probe = (round(info.first.center().x()), round(info.first.top()) - 3)
    on = preview.grab().toImage()
    preview.set_values(0.8, 0.35, False)
    off = preview.grab().toImage()
    assert on != off
    assert off.pixelColor(*probe) == QColor("#27456F") and on.pixelColor(*probe) != QColor("#27456F")


def test_summary_spells_out_the_pixels_for_the_current_values():
    text = wrap_summary(0.8, 0.35, True)

    assert f"{H} 像素" in text and "≤ 19 像素（0.8 倍行高）" in text and "≤ 8 像素（0.35 倍行高）" in text
    assert "左緣或中心點" in text and "擷取寬度" not in text  # 單行寬度上限不開放設定，後台固定 1.0
    assert "左緣或中心點" not in wrap_summary(0.8, 0.35, False)
    assert "最少 8" in wrap_summary(0.0, 0.5, True) and "最少 5" in wrap_summary(0.5, 0.0, True)
    assert "最少" not in text


# ----------------------------------------------------------------------------- 設定畫面與設定檔
def test_the_config_defaults(tmp_path):
    config = AppConfig(tmp_path / "config.json")

    assert config.get("ocr.merge_wrapped_gap_ratio") == 1.0
    assert config.get("ocr.merge_wrapped_align_ratio") == 0.35
    assert config.get("ocr.merge_wrapped_max_width_ratio") == 1.0
    assert config.get("ocr.merge_centered_lines") is True
    assert config.get("ocr.merge_y_tolerance_ratio") == 0.3


def test_the_general_page_has_the_card_and_starts_in_sync_with_the_saved_values(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("ocr.merge_wrapped_gap_ratio", 1.4)
    config.set("ocr.merge_wrapped_align_ratio", 0.6)
    config.set("ocr.merge_centered_lines", False)

    dialog = SettingsDialog(config)

    general_page = dialog.pages.widget(0).widget()
    assert general_page.isAncestorOf(dialog.wrap_preview) and general_page.isAncestorOf(dialog.spin_wrap_gap)
    assert dialog.wrap_preview.values() == pytest.approx((1.4, 0.6, False))
    assert (dialog.spin_wrap_gap.value(), dialog.spin_wrap_align.value()) == pytest.approx((1.4, 0.6))
    assert not hasattr(dialog, "spin_wrap_width")  # 單行寬度上限不在設定畫面
    assert not dialog.chk_wrap_centered.isChecked()
    assert "≤ 34 像素（1.4 倍行高）" in dialog.wrap_summary.text()
    dialog.deleteLater()


def test_the_example_follows_the_fields_before_saving(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    dialog.spin_wrap_gap.setValue(2.0)
    dialog.spin_wrap_align.setValue(1.0)
    dialog.chk_wrap_centered.setChecked(False)

    assert dialog.wrap_preview.values() == pytest.approx((2.0, 1.0, False))
    assert "≤ 48 像素（2 倍行高）" in dialog.wrap_summary.text() and "≤ 24 像素（1 倍行高）" in dialog.wrap_summary.text()
    dialog.deleteLater()


def test_the_three_values_are_saved_and_shown_again(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    dialog.spin_wrap_gap.setValue(1.2)
    dialog.spin_wrap_align.setValue(0.5)
    dialog.chk_wrap_centered.setChecked(False)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("ocr.merge_wrapped_gap_ratio") == pytest.approx(1.2)
    assert saved.get("ocr.merge_wrapped_align_ratio") == pytest.approx(0.5)
    assert saved.get("ocr.merge_wrapped_max_width_ratio") == 1.0  # 設定畫面不寫它，維持預設 1.0
    assert saved.get("ocr.merge_centered_lines") is False
    reopened = SettingsDialog(saved)
    assert reopened.wrap_preview.values() == pytest.approx((1.2, 0.5, False))
    dialog.deleteLater()
    reopened.deleteLater()


# ----------------------------------------------------------------------------- 翻譯流程吃得到設定
def test_vertical_text_passes_the_two_values_on(monkeypatch):
    seen = {}

    def fake_merge(regions, **kwargs):
        seen.update(kwargs)
        return regions

    monkeypatch.setattr(layout_utils, "merge_wrapped_labels", fake_merge)
    vertical_regions(
        [OCRLine("夜空", (40, 10, 50, 200), 0.95)],
        image_width=400, image_height=300, offset_x=0, offset_y=0, min_confidence=0.6,
        y_tolerance_ratio=0.3, x_gap_ratio=0.8, target_language="zh-hant", max_width_ratio=1.0,
        merge_centered=True, han_is_source=True, gap_ratio=1.7, align_ratio=0.9,
    )

    assert seen["gap_ratio"] == 1.7 and seen["align_ratio"] == 0.9


class _OCR:
    supports_vertical = True
    benefits_from_upscale = False

    def __init__(self, lines):
        self.lines = lines

    def recognize(self, _image, lang="auto"):
        return list(self.lines)


class _Translator:
    han_is_source = False
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self):
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        return [f"譯:{text}" for text in texts]


def _run(tmp_path, **ocr_settings):
    config = AppConfig(tmp_path / "config.json")
    for key, value in ocr_settings.items():
        config.set(f"ocr.{key}", value)
    capture = CaptureInfo(
        image=np.full((300, 400, 3), 255, dtype=np.uint8), bbox=(0, 0, 400, 300), monitor_indices=[0], mode="region",
    )
    # 兩行字幕：高 24 像素，第二行在第一行下面 30 像素（預設 1 倍行高 = 24 像素放不下，1.5 倍 = 36 像素放得下）
    ocr = _OCR([OCRLine("こんにちは、", (20, 20, 100, 24), 0.95), OCRLine("お元気ですか？", (20, 74, 120, 24), 0.95)])
    translator = _Translator()
    task = PipelineTask(capture, ocr, translator, config)
    results, errors = [], []
    task.result.connect(results.append)
    task.error.connect(errors.append)
    task.run()
    assert not errors
    return translator.calls


def test_the_pipeline_reads_the_row_gap_from_the_settings(tmp_path):
    assert _run(tmp_path) == [["こんにちは、", "お元気ですか？"]]  # 預設 1 倍行高 = 24 像素：隔 30 像素，分開翻譯
    assert _run(tmp_path, merge_wrapped_gap_ratio=1.5) == [["こんにちは、 お元気ですか？"]]  # 36 像素：併成一句
