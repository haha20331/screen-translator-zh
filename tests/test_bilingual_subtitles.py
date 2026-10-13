"""日文＋羅馬拼音＋中文疊成幾行的歌詞／雙語字幕：不能併成一段，羅馬拼音與附帶的中文不翻。

實際發生過的症狀：三行被併成一個文字塊，譯文變成「木塔塔赫托圖吉蘇納」這類音譯怪字串，
小模型還會把同一句重複好幾遍。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from app.config import AppConfig
from app.models import CaptureInfo, TextRegion
from app.monitor_utils import known_translations
from services.ocr.base import OCRLine
from utils import chinese
from utils.language_utils import is_romanized_japanese, needs_translation, same_script_class, script_class
from utils.layout_utils import companion_translation_texts, merge_wrapped_labels
from workers.translation_worker import PipelineTask

# ----------------------------------------------------------------------------- 羅馬拼音
ROMAJI = [
    "tsumugu tatta hitotsu kizuna",
    "aoi shoudou ni kararete hashiru",
    "watashitachi daremo shiranai",
    "inochi hakanai koiseyo shoujo yo",
    "Inochi hakanal koiseyo shoujo yo",  # OCR 認錯一個字母
    "te ni shita n da hitori ja nai",
    "ima mo kimi wo omou",
    "Arigatō gozaimasu",
]
ENGLISH = [
    "Take me home",
    "Same name",
    "No more",
    "See you",
    "Hello world",
    "Settings",
    "She is here",
    "Press any key to continue",
    "Make a wish",
    "Let me see",
    "Yo Mike",
    "tokyo",
    "kizuna",  # 單一單字不判定
]


@pytest.mark.parametrize("text", ROMAJI)
def test_romanized_japanese_is_recognised(text):
    assert is_romanized_japanese(text)
    assert not needs_translation(text, "zh-hant", han_is_source=True)


@pytest.mark.parametrize("text", ENGLISH)
def test_english_is_not_mistaken_for_romaji(text):
    assert not is_romanized_japanese(text)


def test_english_sentences_are_still_translated():
    assert needs_translation("Take me home", "zh-hant", han_is_source=True)


# ----------------------------------------------------------------------------- 書寫系統
@pytest.mark.parametrize(
    ("text", "expected"),
    [("紡ぐたったひとつ絆", "ja"), ("編織唯一的羈絆", "han"), ("tsumugu kizuna", "latin"), ("3", "other"), ("!?", "other")],
)
def test_script_class(text, expected):
    assert script_class(text) == expected


def test_different_scripts_are_not_the_same_class_but_symbols_fit_anywhere():
    assert not same_script_class("紡ぐ絆", "tsumugu")
    assert not same_script_class("tsumugu", "編織唯一")
    assert same_script_class("紡ぐ絆", "ありがとう")
    assert same_script_class("Settings", "File")
    assert same_script_class("紡ぐ絆", "3")


def test_a_japanese_line_may_continue_on_a_kana_less_kanji_line():
    # val_data/2.png 的標語：第二、三行沒有假名，仍是同一句日文
    assert same_script_class("進化するうまさで", "20年連続")
    assert same_script_class("20年連続", "5億本！")


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
@pytest.mark.parametrize(
    "text",
    ["被青澀的衝動驅使奔跑", "我們誰都不知道", "愛上你脆弱的生命吧，女孩", "我明白了，我並不孤單", "这个没有", "跑得那麼快的話，就只能抬起頭來了吧。"],
)
def test_chinese_lines_are_recognised_from_positive_evidence(text):
    assert chinese.looks_chinese(text)
    assert not same_script_class("紡ぐたったひとつ絆", text)


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
@pytest.mark.parametrize("text", ["20年連続", "5億本！", "主人大人!?", "設定", "新規作成", "恋愛", "国民的人気", "冒険者ギルド"])
def test_japanese_kanji_lines_are_not_mistaken_for_chinese(text):
    assert not chinese.looks_chinese(text)


# ----------------------------------------------------------------------------- 版面
def _region(text, x, y, width, height):
    return TextRegion(text=text, x=x, y=y, width=width, height=height, source_line_height=float(height))


def _stack():
    """置中疊放的三行字幕：日文、羅馬拼音、中文譯文。"""
    return [
        _region("紡ぐたったひとつ絆", 332, 29, 432, 62),
        _region("tsumugu tatta hitotsu kizuna", 352, 119, 395, 46),
        _region("編織唯一的羈絆", 425, 190, 250, 46),
    ]


def _merge(regions):
    return merge_wrapped_labels(regions, "zh-hant", 1100, max_width_ratio=1.0, merge_centered=True, han_is_source=True)


def test_japanese_romaji_and_chinese_lines_are_not_merged():
    merged = _merge(_stack())

    assert [region.text for region in merged] == [
        "紡ぐたったひとつ絆",
        "tsumugu tatta hitotsu kizuna",
        "編織唯一的羈絆",
    ]


def test_wrapped_japanese_lines_are_still_merged():
    lines = [_region("きっと星座も掴める", 300, 100, 500, 50), _region("あなたの存在を", 380, 156, 340, 50)]

    merged = _merge(lines)

    assert len(merged) == 1
    assert merged[0].source_line_count == 2


def test_the_chinese_line_stacked_with_a_japanese_subtitle_is_a_companion():
    assert companion_translation_texts(_stack()) == {"編織唯一的羈絆"}


def test_companion_is_found_even_without_the_romaji_line_between():
    regions = [_region("青い衝動に駆られて走る", 292, 34, 512, 53), _region("被青澀的衝動驅使奔跑", 376, 100, 347, 41)]

    assert companion_translation_texts(regions) == {"被青澀的衝動驅使奔跑"}


def test_chinese_line_above_the_japanese_one_is_found_too():
    regions = [_region("被青澀的衝動驅使奔跑", 376, 20, 347, 41), _region("青い衝動に駆られて走る", 292, 80, 512, 53)]

    assert companion_translation_texts(regions) == {"被青澀的衝動驅使奔跑"}


def test_a_japanese_kanji_slogan_is_not_a_companion():
    # val_data/2.png：三行大字標語，第二、三行是日文漢字，不是附帶的中文
    regions = [
        _region("進化するうまさで", 87, 8, 648, 113),
        _region("20年連続", 76, 91, 677, 191),
        _region("5億本！", 184, 232, 459, 182),
    ]

    assert companion_translation_texts(regions) == set()


def test_unmarked_chinese_without_a_romaji_line_cannot_be_told_apart_from_japanese():
    # 沒有羅馬拼音、字面上也沒有中文獨有的字：只能當日文翻（已知限制）
    regions = [_region("紡ぐたったひとつ絆", 332, 29, 432, 62), _region("編織唯一的羈絆", 425, 100, 250, 46)]

    assert companion_translation_texts(regions) == set()


def test_han_only_text_far_from_any_japanese_line_is_not_a_companion():
    regions = [_region("設定", 20, 20, 60, 24), _region("きっと星座も掴める", 300, 500, 500, 50)]

    assert companion_translation_texts(regions) == set()


def test_han_only_text_beside_japanese_on_the_same_row_is_not_a_companion():
    regions = [_region("設定", 20, 100, 60, 40), _region("きっと星座も掴める", 120, 100, 500, 40)]

    assert companion_translation_texts(regions) == set()


def test_no_japanese_means_no_companions():
    regions = [_region("設定", 20, 20, 60, 24), _region("保存", 20, 60, 60, 24)]

    assert companion_translation_texts(regions) == set()


# ----------------------------------------------------------------------------- 管線
class _OCR:
    def recognize(self, _image, lang="auto"):
        return [
            OCRLine("紡ぐたったひとつ絆", (332, 29, 432, 62), 0.95),
            OCRLine("tsumugu tatta hitotsu kizuna", (352, 119, 395, 46), 1.0),
            OCRLine("編織唯一的羈絆", (425, 190, 250, 46), 1.0),
        ]


class _Translator:
    han_is_source = True
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self) -> None:
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        return ["只有一個羈絆" for _ in texts]


def _run(tmp_path, **settings):
    config = AppConfig(tmp_path / "config.json")
    for key, value in settings.items():
        config.set(key, value)
    capture = CaptureInfo(
        image=np.full((260, 1100, 3), 255, dtype=np.uint8), bbox=(0, 0, 1100, 260), monitor_indices=[0], mode="region"
    )
    translator = _Translator()
    task = PipelineTask(capture, _OCR(), translator, config)
    results = []
    task.result.connect(results.append)
    task.run()
    return translator, results[0]["regions"]


def test_pipeline_only_translates_the_japanese_line(tmp_path):
    translator, regions = _run(tmp_path)

    assert translator.calls == [["紡ぐたったひとつ絆"]]
    by_text = {region.text: region.translated_text for region in regions}
    assert by_text["紡ぐたったひとつ絆"] == "只有一個羈絆"
    assert by_text["tsumugu tatta hitotsu kizuna"] == "tsumugu tatta hitotsu kizuna"
    assert by_text["編織唯一的羈絆"] == "編織唯一的羈絆"


def test_companion_skipping_can_be_turned_off(tmp_path):
    translator, _ = _run(tmp_path, **{"ocr.skip_companion_translation": False})

    assert translator.calls == [["紡ぐたったひとつ絆", "編織唯一的羈絆"]]


# ----------------------------------------------------------------------------- 小模型的重複迴圈
@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
@pytest.mark.parametrize(
    "output",
    ["我知道你是什麼意思，我知道你是什麼意思。", "的衝動驅動著青色衝動，衝動著青色衝動的衝動。", "我知道，我在，我知道，我在，我知道。"],
)
def test_runaway_repetition_is_detected(output):
    assert chinese.has_runaway_repetition(output, "inochi hakanal koiseyo shoujo yo")


@pytest.mark.parametrize("output", ["我愛你，我愛你", "哈哈哈哈", "我們都不知道。", "只有一個結合。"])
def test_normal_translations_are_not_flagged(output):
    assert not chinese.has_runaway_repetition(output, "きっと星座も掴める")


def test_repetition_is_fine_when_the_source_repeats_too():
    assert not chinese.has_runaway_repetition(
        "我愛你我愛你我愛你，永遠在一起永遠在一起", "愛してる愛してる愛してる、ずっと一緒ずっと一緒"
    )


def test_leading_punctuation_is_removed_from_translations():
    assert chinese.normalize_punctuation(",我知道你是什麼意思") == "我知道你是什麼意思"
    assert chinese.normalize_punctuation("，，你好.") == "你好。"


# ----------------------------------------------------------------------------- 即時監控的預熱
def test_known_translations_never_carry_the_kept_original():
    regions = [
        TextRegion(text="こんにちは", translated_text="こんにちは\nこんにちは\n你好"),
        TextRegion(text="さようなら", translated_text="再見"),
        TextRegion(text="空", translated_text="  "),
    ]

    assert known_translations(regions) == {"こんにちは": "你好", "さようなら": "再見"}
