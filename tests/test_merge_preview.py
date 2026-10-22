"""文本組合範例：範例上的距離要與真正的合併規則一致，設定畫面要隨欄位即時更新。"""

from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from services.ocr.base import OCRLine
from ui.merge_preview import (
    SAMPLE_LINE_HEIGHT,
    SEPARATE_EXTRA,
    MergePreview,
    merge_limits,
    merge_summary,
)
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.layout_utils import merge_lines

H = SAMPLE_LINE_HEIGHT


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _regions(y_ratio: float, x_ratio: float, gap: int, offset: int) -> int:
    """兩塊同高的字：第二塊在第一塊右邊 ``gap`` 像素、往上 ``offset`` 像素，真正的合併會剩幾個框。"""
    first = OCRLine(text="a", box=(0, 100, 60, H), confidence=0.9)
    second = OCRLine(text="b", box=(60 + gap, 100 - offset, 40, H), confidence=0.9)
    return len(merge_lines([first, second], min_confidence=0.0, y_tolerance_ratio=y_ratio, x_gap_ratio=x_ratio))


def test_limits_convert_the_ratios_to_pixels_of_the_sample_line():
    gap, offset = merge_limits(0.3, 0.8)

    assert gap == pytest.approx(0.8 * 0.6 * H)  # 橫向：倍字寬 × （行高 × 0.6）
    assert offset == pytest.approx(0.3 * H)  # 上下：倍行高 × 行高
    assert merge_limits(0.0, 0.0) == (0.0, 0.0)
    assert merge_limits(0.3, 0.8, line_height=48) == pytest.approx((2 * gap, 2 * offset))  # 字大一倍，距離大一倍
    assert merge_limits(0.3, 5.0, line_height=2)[0] == pytest.approx(5.0 * 4)  # 字很小時字寬至少 4 像素


@pytest.mark.parametrize("x_ratio", [0.3, 0.8, 1.7, 3.3])
def test_horizontal_limit_matches_the_real_merge(x_ratio):
    limit, _ = merge_limits(0.3, x_ratio)

    assert _regions(0.3, x_ratio, gap=math.ceil(limit) - 1, offset=0) == 1  # 還在範圍內：接成一個框
    assert _regions(0.3, x_ratio, gap=math.floor(limit) + 1, offset=0) == 2  # 超過一點：分開


@pytest.mark.parametrize("y_ratio", [0.0, 0.3, 0.55, 1.4])
def test_vertical_limit_matches_the_real_merge(y_ratio):
    _, limit = merge_limits(y_ratio, 0.8)

    assert _regions(y_ratio, 0.8, gap=0, offset=math.floor(limit)) == 1
    assert _regions(y_ratio, 0.8, gap=0, offset=math.floor(limit) + 1) == 2


def test_zero_horizontal_threshold_only_merges_overlapping_boxes():
    assert _regions(0.3, 0.0, gap=-3, offset=0) == 1
    assert _regions(0.3, 0.0, gap=0, offset=0) == 2


@pytest.mark.parametrize("y_ratio, x_ratio", [(0.3, 0.8), (0.0, 0.0), (1.0, 2.0), (2.0, 5.0)])
def test_layout_draws_the_limits_and_a_block_beyond_them(qapp, y_ratio, x_ratio):
    preview = MergePreview()
    preview.set_values(y_ratio, x_ratio)
    gap, offset = merge_limits(y_ratio, x_ratio)

    info = preview.layout_info()

    assert info.second.left() - info.first.right() == pytest.approx(gap)  # 第二塊剛好在最遠的空隙
    assert info.first.top() - info.second.top() == pytest.approx(offset)  # 並且往上錯開到最多
    assert info.third.left() - info.second.right() == pytest.approx(gap + SEPARATE_EXTRA)  # 第三塊在範圍外
    assert info.third.top() == pytest.approx(info.second.top())
    assert _regions(y_ratio, x_ratio, gap=math.floor(gap) + SEPARATE_EXTRA, offset=0) == 2
    assert info.second.top() >= 0 and info.height >= info.first.bottom()  # 沒有畫到畫面外


def test_first_block_does_not_move_when_the_values_change(qapp):
    preview = MergePreview()
    preview.set_values(0.0, 0.0)
    before = preview.layout_info().first

    preview.set_values(2.0, 5.0)

    assert preview.layout_info().first == before  # 只有第二、第三塊移動，比較看得出變化


def _count(preview: MergePreview, color: str, tolerance: int = 40) -> int:
    """範例裡接近這個顏色的像素數；細線有反鋸齒，所以容差放寬（底色與兩條線的顏色差都超過 70）。"""
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


def test_painting_shows_the_merged_range_at_every_extreme(qapp):
    preview = MergePreview()
    pictures = []
    for y_ratio, x_ratio in ((0.3, 0.8), (0.0, 0.0), (2.0, 5.0)):
        preview.set_values(y_ratio, x_ratio)
        assert _count(preview, "#6EE7A8") > 30  # 「同一句」的虛線框
        assert _count(preview, "#FFC857") > 10  # 兩條量距標示
        pictures.append(preview.grab().toImage())
    assert pictures[0] != pictures[1] != pictures[2]  # 數值不同，畫面就不同


def test_summary_spells_out_the_pixels_for_the_current_values():
    text = merge_summary(0.3, 0.8)

    assert f"{H} 像素" in text
    assert "11.5 像素（0.8 倍字寬）" in text
    assert "7.2 像素（0.3 倍行高）" in text
    assert "重疊" in merge_summary(0.3, 0.0)  # 0：只有框重疊才接，不寫成「小於 0 像素」
    assert "小於 0.0" not in merge_summary(0.3, 0.0)


def test_dialog_starts_in_sync_with_the_saved_values(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("ocr.merge_y_tolerance_ratio", 0.55)
    config.set("ocr.merge_x_gap_ratio", 1.7)

    dialog = SettingsDialog(config)

    assert dialog.merge_preview.values() == pytest.approx((0.55, 1.7))
    assert "24.5 像素（1.7 倍字寬）" in dialog.merge_summary.text()
    ocr_page = dialog.pages.widget(2).widget()
    assert ocr_page.isAncestorOf(dialog.merge_preview) and ocr_page.isAncestorOf(dialog.spin_x_gap)


def test_dialog_preview_follows_the_fields_before_saving(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    dialog.spin_x_gap.setValue(2.0)
    assert dialog.merge_preview.values()[1] == pytest.approx(2.0)
    assert "28.8 像素（2 倍字寬）" in dialog.merge_summary.text()

    dialog.spin_y_tol.setValue(1.0)
    assert dialog.merge_preview.values()[0] == pytest.approx(1.0)
    assert "24.0 像素（1 倍行高）" in dialog.merge_summary.text()
