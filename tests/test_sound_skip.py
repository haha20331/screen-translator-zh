"""只有擬聲詞（音效、笑聲、狀聲詞）的句子：不翻譯也不顯示。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.config import DEFAULTS, AppConfig
from app.models import TextRegion
from services.translation.cache import TranslationCache
from services.translation.offline_common import (
    SKIP_MARKER,
    is_onomatopoeia_output,
    is_skip_output,
    looks_like_sound_ja,
)
from services.translation.offline_translator import OfflineTranslator, static_prompt
from tests.fake_llama import offline_translator, source_of, static_part_of
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils import chinese
from utils.layout_utils import translation_lines

pytestmark = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")


# ----------------------------------------------------------------------------- 判斷函式
@pytest.mark.parametrize("output", ["[SKIP]", "[skip]", "【SKIP】", "SKIP", "[SKIP].", " [SKIP] "])
def test_skip_marker_is_recognised(output):
    assert is_skip_output(output)


@pytest.mark.parametrize("output", ["skip", "Skip", "skip it", "", "跳過", "[SKIP] 你好"])
def test_ordinary_text_is_not_a_skip_marker(output):
    # 小寫的 skip 可能是模型把「スキ」翻成英文的失敗結果，要算失敗而不是略過
    assert not is_skip_output(output)


@pytest.mark.parametrize("output", ["汪汪", "砰！", "哈哈", "啊啊啊", "。..", "滴答滴答", "嘻嘻嘻", "噗！"])
def test_chinese_sound_words_are_onomatopoeia(output):
    assert is_onomatopoeia_output(output)


@pytest.mark.parametrize("output", ["嗯", "啊！", "你好", "哈哈，你好", "再見", "心跳加速", "我回來了", ""])
def test_real_words_and_lone_interjections_are_not(output):
    assert not is_onomatopoeia_output(output)  # 單獨一個「嗯」是應答（うん），不能當音效濾掉


@pytest.mark.parametrize("text", ["ドン！", "ガシャーン", "ふふっ", "ははは", "ふぅ…", "ドキドキ", "ワンワン", "あああ", "ニャー"])
def test_japanese_sound_effects_look_like_sounds(text):
    assert looks_like_sound_ja(text)


@pytest.mark.parametrize("text", ["またね", "ただいま", "はい", "愛してる", "ありがとう", "おかえり", "大丈夫？", "行こう！"])
def test_ordinary_japanese_does_not(text):
    assert not looks_like_sound_ja(text)


# ----------------------------------------------------------------------------- 提示詞
def test_prompt_tells_the_model_to_skip_sound_effects_with_examples():
    prompt = static_prompt("ja")

    assert SKIP_MARKER in prompt and "sound effect" in prompt
    assert f"<|im_start|>assistant\n{SKIP_MARKER}<|im_end|>" in prompt  # 示範：音效 → [SKIP]
    assert "<|im_start|>assistant\n歡迎回來<|im_end|>" in prompt  # 對照：短短的真話照翻
    assert SKIP_MARKER in static_prompt("en", lyrics=True) and SKIP_MARKER in static_prompt("ja", lyrics=True)


@pytest.mark.parametrize("lyrics", [False, True])
@pytest.mark.parametrize("language", ["ja", "en"])
def test_prompt_without_the_rule_has_no_skip_instruction_or_examples(language, lyrics):
    prompt = static_prompt(language, lyrics, skip_sounds=False)

    assert SKIP_MARKER not in prompt and "sound effect" not in prompt
    assert prompt.endswith("<|im_end|>\n")


# ----------------------------------------------------------------------------- 翻譯（假模型）
def decode(prompt: str) -> str:
    source = source_of(prompt)
    has_rule = SKIP_MARKER in static_part_of(prompt)
    if source in {"ドン！", "Boom!"}:
        return SKIP_MARKER if has_rule else "砰！"
    if source == "ワンワン":
        return "汪汪"  # 模型沒照規則回標記，卻「翻」成了中文擬聲詞
    if source == "またね":
        return SKIP_MARKER if has_rule else "再見"  # 模型誤判：一般短句被回報成音效
    return {"こんにちは": "你好", "さようなら": "再見"}.get(source, "")


def _translator(**section):
    return offline_translator(decode, **section)


@pytest.mark.parametrize(("text", "language"), [("ドン！", "ja"), ("Boom!", "en"), ("ワンワン", "ja")])
def test_sound_effects_are_kept_as_is_and_are_not_failures(text, language):
    instance, _ = _translator(use_context=True)

    out = instance.translate([text, "こんにちは"], "auto", "zh-hant")

    assert out == [text, "你好"]  # 譯文等於原文：之後不會顯示
    assert instance.last_failed_count == 0
    assert [entry.source for entry in instance._history] == ["こんにちは"]  # 也不會被記成上下文


def test_a_skipped_sound_is_cached_so_it_is_not_asked_again(tmp_path):
    instance, generator = _translator()
    instance.cache = TranslationCache(tmp_path / "cache.json")

    instance.translate(["ドン！"], "auto", "zh-hant")
    instance.translate(["ドン！"], "auto", "zh-hant")

    assert generator.calls == 1


def test_an_ordinary_phrase_wrongly_reported_as_a_sound_is_translated_again_without_the_rule():
    instance, generator = _translator()

    out = instance.translate(["またね"], "auto", "zh-hant")

    assert out == ["再見"]
    assert generator.calls == 2
    assert SKIP_MARKER in generator.prefixes[0]
    assert SKIP_MARKER not in generator.prefixes[1]


def test_with_the_retry_off_an_ordinary_phrase_reported_as_a_sound_is_not_run_again():
    instance, generator = _translator(retry_non_chinese=False)

    out = instance.translate(["またね"], "auto", "zh-hant")

    assert generator.calls == 1  # 只進模型一次
    assert out == ["またね"] and instance.last_failed_count == 1  # 不顯示「[SKIP]」，原文保留、算失敗


def test_turning_the_option_off_translates_everything_and_separates_the_cache():
    instance, generator = _translator(skip_sounds=False)

    out = instance.translate(["ドン！", "ワンワン"], "auto", "zh-hant")

    assert out == ["砰！", "汪汪"]  # 關閉：連擬聲詞也翻
    assert all(SKIP_MARKER not in prefix for prefix in generator.prefixes)
    assert instance.cache_namespace("auto") == "auto|nosound|p2|1.5b-q4_k_m"
    instance.config["skip_sounds"] = True
    assert instance.cache_namespace("auto") == "auto|p2|1.5b-q4_k_m"


def test_the_marker_is_never_shown_when_skipping_is_not_allowed_and_the_retry_also_fails():
    instance, _ = _translator()
    # 英文沒有「像不像音效」的限制；直接測基底類別：不允許略過卻回了標記，要當失敗而不是把「[SKIP]」顯示出來
    instance.config["skip_sounds"] = False
    instance._generate = lambda texts, language: [SKIP_MARKER for _ in texts]

    out = instance.translate(["Hello"], "auto", "zh-hant")

    assert out == ["Hello"] and instance.last_failed_count == 1


# ----------------------------------------------------------------------------- 只過濾譯文（模型沒照規則回標記時）
def _answering(outputs):
    instance, _ = _translator()
    instance._generate = lambda texts, language: [outputs[text] for text in texts]
    return instance


def test_chinese_sound_words_are_dropped_for_sound_like_sources():
    instance = _answering({"ワンワン": "汪汪", "Woof woof": "汪汪", "ふふっ": "嘻嘻", "こんにちは": "你好"})

    out = instance.translate(["ワンワン", "Woof woof", "ふふっ", "こんにちは"], "auto", "zh-hant")

    assert out == ["ワンワン", "Woof woof", "ふふっ", "你好"]
    assert instance.last_failed_count == 0


def test_ordinary_japanese_is_still_translated_even_if_it_comes_out_sound_like():
    instance = _answering({"またね": "哈哈"})

    out = instance.translate(["またね"], "auto", "zh-hant")

    assert out == ["哈哈"]  # 原文不像音效：不採信「全是擬聲用字」的判斷


# ----------------------------------------------------------------------------- 顯示
def test_a_skipped_sentence_is_not_shown_in_the_panel():
    regions = [
        TextRegion(text="ドン！", translated_text="ドン！", x=0, y=0, width=50, height=20),
        TextRegion(text="こんにちは", translated_text="你好", x=0, y=40, width=80, height=20),
    ]

    assert translation_lines(regions) == ["你好"]


# ----------------------------------------------------------------------------- 設定
def test_default_and_settings_checkbox(tmp_path):
    assert DEFAULTS["translation"]["skip_sounds"] is True
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.chk_skip_sounds.isChecked()

    dialog.chk_skip_sounds.setChecked(False)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("translation.skip_sounds") is False
    dialog.deleteLater()

