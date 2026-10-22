"""上下文（記住前 N 句）與歌詞模式：提示詞、歷史管理、快取分流、設定介面。"""

from __future__ import annotations

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.config import DEFAULTS, AppConfig
from services.translation import offline_translator as module
from services.translation.cache import TranslationCache
from services.translation.offline_translator import (
    ContextEntry,
    OfflineTranslator,
    build_prompt,
    history_prompt,
    static_prompt,
    user_prompt,
)
from tests.fake_llama import FakeLlamaEngine, offline_translator, source_of
from ui.main_window import MainWindow
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils import chinese

pytestmark = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")

TRANSLATIONS = {
    "こんにちは": "你好",
    "元気ですか": "你好嗎",
    "ありがとう": "謝謝",
    "さようなら": "再見",
    "進む": "前進",
}


def decode(prompt: str) -> str:
    source = source_of(prompt)
    if "歌詞翻譯" in prompt and source == "進む":  # 歌詞提示讓模型把日文留在譯文裡
        return "進むと"
    return TRANSLATIONS.get(source, "")


def _translator(**section) -> tuple[OfflineTranslator, FakeLlamaEngine]:
    return offline_translator(decode, **section)


def _user_turns(prompt: str) -> list[str]:
    return re.findall(r"<\|im_start\|>user\n(?:Translate this \w+ text:|翻譯這句[日英]文：)\n(.*?)<\|im_end\|>", prompt)


def _last_prompt(generator: FakeLlamaEngine) -> str:
    """最後一次送進模型的提示詞，去掉固定前綴（系統提示 + 示範）：剩下前文與待翻的句子。"""
    return generator.prompts[-1][len(generator.prefixes[-1]) :]


# ----------------------------------------------------------------------------- 提示詞
@pytest.mark.parametrize(("style", "marker"), [("zh", "歌詞翻譯"), ("en", "song lyrics")])
def test_lyrics_prompt_changes_system_text_and_examples_but_keeps_the_shape(style, marker):
    plain, lyrics = static_prompt("ja", style=style), static_prompt("ja", lyrics=True, style=style)

    assert marker in lyrics and marker not in plain
    assert lyrics != plain
    assert lyrics.startswith("<|im_start|>system\n") and lyrics.endswith("<|im_end|>\n")
    assert lyrics.count("<|im_start|>assistant\n") == len(module._LYRICS_SHOTS["ja"])
    assert plain.count("<|im_start|>assistant\n") == len(module._SHOTS["ja"])
    assert static_prompt("en", lyrics=True, style=style) != static_prompt("en", style=style)


_NO_SUBJECT_MARKERS = {"zh": ("原文沒有主詞時，譯文也不要加主詞",), "en": ("Do not add a subject or pronoun", "leave it out of the translation too")}


@pytest.mark.parametrize("style", ["zh", "en"])
@pytest.mark.parametrize("lyrics", [False, True])
@pytest.mark.parametrize("skip_sounds", [False, True])
@pytest.mark.parametrize("language", ["ja", "en"])
def test_the_prompt_tells_the_model_not_to_add_a_subject_the_original_does_not_have(language, lyrics, skip_sounds, style):
    system = static_prompt(language, lyrics, skip_sounds, style).split("<|im_end|>", 1)[0]

    for marker in _NO_SUBJECT_MARKERS[style]:
        assert marker in system
    assert "我" in system and "你" in system  # 舉例說明是哪種字


@pytest.mark.parametrize("style", ["zh", "en"])
def test_the_examples_do_not_contradict_the_no_subject_rule(style):
    everything = "".join(static_prompt(language, lyrics, style=style) for language in ("ja", "en") for lyrics in (False, True))
    ja_header, en_header = ("翻譯這句日文：", "翻譯這句英文：") if style == "zh" else ("Translate this Japanese text:", "Translate this English text:")

    # 原文沒有主詞的示範，譯文也不能有（這兩句以前的示範自己加了「我」）
    assert "<|im_start|>assistant\n我喜歡你<|im_end|>" not in everything
    assert "<|im_start|>assistant\n我聽見了你的聲音<|im_end|>" not in everything
    assert "<|im_start|>assistant\n喜歡喔<|im_end|>" in everything and "<|im_start|>assistant\n聽得見你的聲音<|im_end|>" in everything
    # 原文有主詞的示範照翻，才不會被教成「一律拿掉主詞」
    assert f"{ja_header}\n私は学生です。<|im_end|>\n<|im_start|>assistant\n我是學生。" in static_prompt("ja", style=style)
    assert f"{en_header}\nI am a student.<|im_end|>\n<|im_start|>assistant\n我是學生。" in static_prompt("en", style=style)


def test_the_cache_key_carries_the_prompt_version_so_old_prompts_leave_no_stale_translations():
    instance, _ = _translator()

    assert module.PROMPT_VERSION >= 2
    assert f"|p{module.PROMPT_VERSION}|" in instance.cache_namespace("auto")


@pytest.mark.parametrize(
    ("style", "ja_header", "en_header"),
    [("zh", "翻譯這句日文：", "翻譯這句英文："), ("en", "Translate this Japanese text:", "Translate this English text:")],
)
def test_history_prompt_is_finished_chat_turns_in_time_order(style, ja_header, en_header):
    entries = [
        ContextEntry("こんにちは", "你好", "ja", 1.0),
        ContextEntry("Thank you", "謝謝", "en", 2.0),
    ]

    text = history_prompt(entries, style)

    assert text == (
        f"<|im_start|>user\n{ja_header}\nこんにちは<|im_end|>\n"
        "<|im_start|>assistant\n你好<|im_end|>\n"
        f"<|im_start|>user\n{en_header}\nThank you<|im_end|>\n"
        "<|im_start|>assistant\n謝謝<|im_end|>\n"
    )
    assert history_prompt([]) == ""


def test_build_prompt_puts_history_between_examples_and_current_text():
    history = [ContextEntry("こんにちは", "你好", "ja", 1.0)]

    prompt = build_prompt("元気ですか", "ja", history)

    assert prompt == static_prompt("ja") + history_prompt(history) + user_prompt("元気ですか", "ja")
    assert build_prompt("元気ですか", "ja") == static_prompt("ja") + user_prompt("元気ですか", "ja")


# ----------------------------------------------------------------------------- 上下文
def test_a_translator_without_the_setting_does_not_use_context():
    """設定檔裡沒有這個鍵時（只有直接建立翻譯器的情況）不用上下文；程式的預設設定是開，見下面的 DEFAULTS 測試。"""
    instance, generator = _translator()
    instance.translate(["こんにちは"], "auto", "zh-hant")
    instance.translate(["元気ですか"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["元気ですか"]
    assert not instance._history


def test_previous_sentences_become_context_for_the_next_one():
    instance, generator = _translator(use_context=True, context_size=2)

    instance.translate(["こんにちは"], "auto", "zh-hant")
    assert _user_turns(_last_prompt(generator)) == ["こんにちは"]  # 第一句沒有前文
    instance.translate(["元気ですか"], "auto", "zh-hant")

    prompt = _last_prompt(generator)
    assert _user_turns(prompt) == ["こんにちは", "元気ですか"]
    assert "<|im_start|>assistant\n你好<|im_end|>" in prompt  # 前文帶著當時的譯文
    assert prompt.endswith("元気ですか<|im_end|>\n<|im_start|>assistant\n")


def test_only_the_latest_n_sentences_are_used():
    instance, generator = _translator(use_context=True, context_size=2)
    for text in ("こんにちは", "元気ですか", "ありがとう"):
        instance.translate([text], "auto", "zh-hant")

    instance.translate(["さようなら"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["元気ですか", "ありがとう", "さようなら"]


def test_context_size_can_be_changed_while_running():
    instance, generator = _translator(use_context=True, context_size=1)
    for text in ("こんにちは", "元気ですか", "ありがとう"):
        instance.translate([text], "auto", "zh-hant")
    instance.config["context_size"] = 3

    instance.translate(["さようなら"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["こんにちは", "元気ですか", "ありがとう", "さようなら"]


def test_a_subtitle_seen_every_round_is_remembered_once():
    instance, generator = _translator(use_context=True, context_size=2)
    for _ in range(3):  # 監控每一輪都重新辨識同一句
        instance.translate(["こんにちは"], "auto", "zh-hant")

    instance.translate(["元気ですか"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["こんにちは", "元気ですか"]


def test_the_current_sentence_is_never_its_own_context():
    instance, generator = _translator(use_context=True, context_size=2)
    instance.translate(["こんにちは"], "auto", "zh-hant")
    instance.translate(["元気ですか"], "auto", "zh-hant")

    instance.translate(["こんにちは"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["元気ですか", "こんにちは"]


def test_failed_sentences_are_not_remembered():
    instance, _ = _translator(use_context=True)

    instance.translate(["未登録の文章です"], "auto", "zh-hant")  # 模型回空字串 = 失敗

    assert not instance._history


def test_old_context_expires(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])
    instance, generator = _translator(use_context=True, context_ttl_seconds=60)
    instance.translate(["こんにちは"], "auto", "zh-hant")

    clock["now"] += 61
    instance.translate(["元気ですか"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["元気ですか"]


def test_reset_context_forgets_everything():
    instance, generator = _translator(use_context=True)
    instance.translate(["こんにちは"], "auto", "zh-hant")

    instance.reset_context()
    instance.translate(["元気ですか"], "auto", "zh-hant")

    assert _user_turns(_last_prompt(generator)) == ["元気ですか"]


def test_placeholders_are_not_put_into_the_context():
    instance, _ = _translator(use_context=True)
    TRANSLATIONS["__ST_KEEP_0000__ ありがとう"] = "__ST_KEEP_0000__ 謝謝"
    try:
        instance.translate(["__ST_KEEP_0000__ ありがとう"], "auto", "zh-hant")
    finally:
        del TRANSLATIONS["__ST_KEEP_0000__ ありがとう"]

    assert [entry.source for entry in instance._history] == ["ありがとう"]
    assert [entry.translation for entry in instance._history] == ["謝謝"]


# ----------------------------------------------------------------------------- 歌詞模式
def test_lyrics_mode_switches_the_system_prompt_live():
    instance, generator = _translator()
    instance.translate(["こんにちは"], "auto", "zh-hant")
    assert "歌詞翻譯" not in generator.prefixes[-1]

    instance.config["lyrics_mode"] = True
    instance.translate(["元気ですか"], "auto", "zh-hant")
    assert "歌詞翻譯" in generator.prefixes[-1]

    instance.config["lyrics_mode"] = False
    instance.translate(["ありがとう"], "auto", "zh-hant")
    assert "歌詞翻譯" not in generator.prefixes[-1]


def test_japanese_left_in_the_output_is_retried_with_the_plain_prompt():
    instance, generator = _translator(lyrics_mode=True)

    out = instance.translate(["進む"], "auto", "zh-hant")

    assert out == ["前進"]  # 歌詞提示回「進むと」（殘留假名），改用一般提示重翻
    assert generator.calls == 2
    assert "歌詞翻譯" not in generator.prefixes[-1]


def test_retry_when_the_output_is_not_chinese_is_on_by_default():
    assert DEFAULTS["translation"]["retry_non_chinese"] is True
    assert _translator()[0].retry_enabled()


def test_with_the_retry_off_the_model_runs_once_and_the_output_is_used_as_is():
    instance, generator = _translator(lyrics_mode=True, retry_non_chinese=False)

    out = instance.translate(["進む"], "auto", "zh-hant")

    assert out == ["進むと"]  # 殘留假名也照輸出
    assert generator.calls == 1
    assert instance.last_failed_count == 0


def test_with_the_retry_off_a_non_chinese_answer_is_still_shown_but_a_pure_echo_is_not():
    off, _ = _translator(retry_non_chinese=False)
    on, _ = _translator()

    off._generate = on._generate = lambda texts, language: ["こんばんは" for _ in texts]  # 全是假名、但不是抄回原文
    assert off.translate(["こんにちは"], "auto", "zh-hant") == ["こんばんは"]
    assert on.translate(["こんにちは"], "auto", "zh-hant") == ["こんにちは"]  # 開著重翻時維持原本的把關：算失敗、保留原文

    off._generate = lambda texts, language: list(texts)  # 原樣抄回原文：顯示出來只是重複
    assert off.translate(["こんにちは"], "auto", "zh-hant") == ["こんにちは"] and off.last_failed_count == 1


def test_the_retry_switch_takes_effect_while_running_and_does_not_mix_up_the_cache(tmp_path):
    instance, generator = _translator(lyrics_mode=True)
    instance.cache = TranslationCache(tmp_path / "cache.json")

    instance.config["retry_non_chinese"] = False
    assert instance.translate(["進む"], "auto", "zh-hant") == ["進むと"]
    assert generator.calls == 1

    instance.config["retry_non_chinese"] = True  # 重新打開：不會拿到沒重翻過的舊結果
    assert instance.translate(["進む"], "auto", "zh-hant") == ["前進"]
    assert generator.calls == 3  # 第一次進模型 + 重翻（命中快取的沒有再算）


def test_no_retry_when_the_output_is_clean():
    instance, generator = _translator(lyrics_mode=True)

    instance.translate(["こんにちは"], "auto", "zh-hant")

    assert generator.calls == 1


# ----------------------------------------------------------------------------- 快取分流
@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({}, "auto|p5|1.5b-q8_0"),
        ({"lyrics_mode": True}, "auto|lyrics|p5|1.5b-q8_0"),
        ({"use_context": True}, "auto|ctx|p5|1.5b-q8_0"),
        ({"lyrics_mode": True, "use_context": True}, "auto|lyrics|ctx|p5|1.5b-q8_0"),
        ({"retry_non_chinese": True}, "auto|p5|1.5b-q8_0"),  # 預設就是開，不另外分流
        ({"retry_non_chinese": False}, "auto|noretry|p5|1.5b-q8_0"),
        ({"lyrics_mode": True, "retry_non_chinese": False}, "auto|lyrics|noretry|p5|1.5b-q8_0"),
    ],
)
def test_cache_namespace_separates_translation_styles(section, expected):
    instance, _ = _translator(offline={"model_name": "qwen2.5-1.5b-instruct-gguf-q8_0"}, **section)  # 模型固定，不受這台電腦下載了什麼影響

    assert instance.cache_namespace("auto") == expected


def test_toggling_a_mode_does_not_reuse_translations_made_without_it(tmp_path):
    instance, generator = _translator()
    instance.cache = TranslationCache(tmp_path / "cache.json")
    instance.translate(["こんにちは"], "auto", "zh-hant")
    assert generator.calls == 1
    instance.translate(["こんにちは"], "auto", "zh-hant")
    assert generator.calls == 1  # 同樣的設定：命中快取

    instance.config["lyrics_mode"] = True
    instance.translate(["こんにちは"], "auto", "zh-hant")

    assert generator.calls == 2  # 換成歌詞模式：重新翻譯


def test_a_poisoned_cache_entry_with_the_original_prepended_is_ignored(tmp_path):
    instance, generator = _translator()
    instance.cache = TranslationCache(tmp_path / "cache.json")
    # 舊版「保留原文」+ 即時監控把「原文\n譯文」寫回快取，之後每輪再多疊一份原文
    namespace = instance.cache_namespace("auto")
    instance.cache.set(namespace, "zh-hant", "こんにちは", "こんにちは\nこんにちは\n你好")

    out = instance.translate(["こんにちは"], "auto", "zh-hant")

    assert out == ["你好"]
    assert instance.cache.get(namespace, "zh-hant", "こんにちは") == "你好"


# ----------------------------------------------------------------------------- 設定與介面
def test_defaults_are_context_on_with_two_sentences_and_lyrics_off():
    translation = DEFAULTS["translation"]

    assert translation["use_context"] is True  # 盲測對照：關掉上下文譯文品質下降（PROMPT_TUNING.md）
    assert translation["context_size"] == 2
    assert translation["lyrics_mode"] is False


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


class _Controller:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_settings_dialog_round_trips_the_two_options(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    assert dialog.chk_use_context.isChecked()  # 預設開
    assert dialog.spin_context_size.value() == 2
    assert dialog.spin_context_size.isEnabled()
    assert not dialog.chk_lyrics_mode.isChecked()

    dialog.chk_use_context.setChecked(False)
    assert not dialog.spin_context_size.isEnabled()  # 沒勾上下文時，句數不能改
    dialog.chk_use_context.setChecked(True)
    assert dialog.spin_context_size.isEnabled()
    dialog.spin_context_size.setValue(4)
    dialog.chk_lyrics_mode.setChecked(True)
    hotkeys = dialog._validated_hotkeys()
    dialog._commit_accept(hotkeys)
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("translation.use_context") is True
    assert saved.get("translation.context_size") == 4
    assert saved.get("translation.lyrics_mode") is True
    dialog.deleteLater()


def test_settings_dialog_has_the_retry_switch_on_by_default_and_saves_it(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.chk_retry_non_chinese.isChecked()
    assert dialog.pages.widget(3).widget().isAncestorOf(dialog.chk_retry_non_chinese)  # 「翻譯」頁

    dialog.chk_retry_non_chinese.setChecked(False)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("translation.retry_non_chinese") is False
    reopened = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert not reopened.chk_retry_non_chinese.isChecked()
    dialog.deleteLater()
    reopened.deleteLater()


def test_main_window_lyrics_toggle_writes_the_config_immediately(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    window = MainWindow(_Controller(config))
    assert not window.check_lyrics.isChecked()

    window.check_lyrics.setChecked(True)

    assert AppConfig(tmp_path / "config.json").get("translation.lyrics_mode") is True
    window.check_lyrics.setChecked(False)
    assert AppConfig(tmp_path / "config.json").get("translation.lyrics_mode") is False
    window.deleteLater()


def test_context_is_only_switched_in_the_settings_dialog(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    assert not hasattr(window, "check_context")  # 主視窗下方不放上下文勾選
    window.deleteLater()


def test_main_window_lyrics_toggle_stays_usable_while_monitoring(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    window.set_busy(True)

    assert window.check_lyrics.isEnabled()
    window.deleteLater()


def test_main_window_reload_shows_saved_values_without_rewriting_them(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.lyrics_mode", True)
    window = MainWindow(_Controller(config))

    window.reload_values()

    assert window.check_lyrics.isChecked()
    assert AppConfig(tmp_path / "config.json").get("translation.lyrics_mode") is False  # 沒存檔就不會寫入
    window.deleteLater()


def test_the_english_lyrics_prompt_asks_for_plain_modern_chinese_not_classical_chinese():
    lyrics = module.static_prompt("ja", lyrics=True, style="en")
    plain = module.static_prompt("ja", style="en")

    assert "modern vernacular Chinese" in lyrics and "do not use classical or literary Chinese" in lyrics
    assert "白話文" in lyrics and "文言文" in lyrics
    assert "classical" not in plain and "vernacular" not in plain  # 一般模式的提示詞沒有動
    assert module.static_prompt("en", lyrics=True, style="en").count("vernacular") == 1  # 英文歌詞也一樣
    # 白話文的要求放在系統提示詞裡（固定前綴），不是每句都重複；快取版本跟著提高
    assert lyrics.index("vernacular") < lyrics.index("<|im_start|>user")
    assert module.PROMPT_VERSION >= 4


def test_the_chinese_lyrics_prompt_asks_for_plain_modern_chinese_not_classical_chinese():
    lyrics = module.static_prompt("ja", lyrics=True, style="zh")
    plain = module.static_prompt("ja", style="zh")

    assert "現代白話文" in lyrics and "不要用文言文" in lyrics
    for archaic in "之乃吾汝爾兮矣焉":
        assert archaic in lyrics.split("<|im_start|>user")[0]  # 古字列在系統提示詞裡，告訴模型避免
    assert "文言文" not in plain and "白話文" not in plain  # 一般模式的提示詞沒有動
    assert lyrics.count("白話文") == 1 and module.static_prompt("en", lyrics=True, style="zh").count("白話文") == 1
    assert module.PROMPT_VERSION >= 5
