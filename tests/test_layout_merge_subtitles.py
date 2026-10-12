"""字幕／對白的多行合併：放寬寬度限制、置中對齊、漢字來源。"""

from __future__ import annotations

from app.models import TextRegion
from utils.layout_utils import merge_wrapped_labels


def _region(text, x, y, w, h=30) -> TextRegion:
    return TextRegion(text=text, x=x, y=y, width=w, height=h, source_line_height=float(h))


def test_upstream_defaults_do_not_merge_wide_subtitle_lines():
    regions = [_region("心にあった抵抗感無く", 24, 26, 600), _region("なる本能的衝動", 25, 106, 432)]
    # 兩行相距 50px，在 0.8×行高內；但單行寬度 600 > 664 的 36%
    regions[1].y = 60

    assert len(merge_wrapped_labels(regions, "zh-hant", 664)) == 2


def test_wide_lines_merge_when_width_ratio_is_relaxed():
    regions = [_region("心にあった抵抗感無く", 24, 26, 600), _region("なる本能的衝動", 25, 60, 432)]

    (merged,) = merge_wrapped_labels(regions, "zh-hant", 664, max_width_ratio=1.0)

    assert merged.text == "心にあった抵抗感無く なる本能的衝動"
    assert merged.source_line_count == 2
    assert merged.width == 600 and merged.bottom == 90


def test_centered_second_line_merges_only_when_enabled():
    def lines():
        return [_region("そんなに早く走ったら", 100, 20, 400), _region("顔を上げるしかないだろ", 40, 55, 520)]

    assert len(merge_wrapped_labels(lines(), "zh-hant", 640, max_width_ratio=1.0)) == 2
    (merged,) = merge_wrapped_labels(lines(), "zh-hant", 640, max_width_ratio=1.0, merge_centered=True)
    assert merged.text == "そんなに早く走ったら 顔を上げるしかないだろ"


def test_distant_lines_are_never_merged():
    regions = [_region("上の文章です", 20, 10, 200), _region("下の文章です", 20, 200, 200)]

    assert len(merge_wrapped_labels(regions, "zh-hant", 640, max_width_ratio=1.0, merge_centered=True)) == 2


def test_han_only_lines_merge_only_when_declared_foreign():
    def lines():
        return [_region("進化", 20, 10, 100), _region("連続", 20, 45, 100)]

    assert len(merge_wrapped_labels(lines(), "zh-hant", 640, max_width_ratio=1.0)) == 2  # 當成已是中文
    assert len(merge_wrapped_labels(lines(), "zh-hant", 640, max_width_ratio=1.0, han_is_source=True)) == 1


def test_untranslatable_neighbours_are_not_absorbed():
    regions = [_region("設定を保存する", 20, 10, 200), _region("12345", 20, 45, 80)]

    assert len(merge_wrapped_labels(regions, "zh-hant", 640, max_width_ratio=1.0)) == 2


def test_overlapping_boxes_of_large_stylised_text_are_merged_in_reading_order():
    # val_data/2.png：三行大字的 OCR 框彼此重疊，不能跳過中間那一行
    regions = [
        _region("進化するうまさで", 87, 8, 648, 113),
        _region("20年連続", 76, 91, 677, 191),
        _region("5億本！", 184, 232, 459, 182),
    ]

    (merged,) = merge_wrapped_labels(
        regions, "zh-hant", 801, max_width_ratio=1.0, merge_centered=True, han_is_source=True
    )

    assert merged.text == "進化するうまさで 20年連続 5億本！"
    assert merged.source_line_count == 3
