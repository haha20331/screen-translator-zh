"""上下文（記住前 N 句）與歌詞模式：提示詞、歷史管理、快取分流、設定介面。"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

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


class FakeTokenizer:
    """把整段提示詞當成一個「詞」，方便測試檢查送進模型的文字。"""

    def encode(self, prompt, add_special_tokens):
        return SimpleNamespace(tokens=[prompt])

    def decode(self, ids, skip_special_tokens):
        context = ids[0]
        source = re.findall(r"Translate this \w+ text:\n(.*?)<\|im_end\|>", context)[-1]
        if "song lyrics" in context and source == "進む":  # 歌詞提示讓模型把日文留在譯文裡
            return "進むと"
        return TRANSLATIONS.get(source, "")


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_batch(self, batches, **kwargs):
        self.calls.append({"batches": batches, **kwargs})
        static = "".join(kwargs["static_prompt"])
        return [SimpleNamespace(sequences_ids=[[static + "".join(batch)]]) for batch in batches]


def _translator(**section) -> tuple[OfflineTranslator, FakeGenerator]:
    config = {"offline": {}, **section}
    instance = OfflineTranslator(config)
    generator = FakeGenerator()
    instance._generator, instance._tokenizer = generator, FakeTokenizer()
    return instance, generator


def _user_turns(prompt: str) -> list[str]:
    return re.findall(r"<\|im_start\|>user\nTranslate this \w+ text:\n(.*?)<\|im_end\|>", prompt)


def _last_prompt(generator: FakeGenerator, index: int = 0) -> str:
    return "".join(generator.calls[-1]["batches"][index])


# ----------------------------------------------------------------------------- 提示詞
def test_lyrics_prompt_changes_system_text_and_examples_but_keeps_the_shape():
    plain, lyrics = static_prompt("ja"), static_prompt("ja", lyrics=True)

    assert "song lyrics" in lyrics and "song lyrics" not in plain
    assert lyrics != plain
    assert lyrics.startswith("<|im_start|>system\n") and lyrics.endswith("<|im_end|>\n")
    assert lyrics.count("<|im_start|>assistant\n") == len(module._LYRICS_SHOTS["ja"])
    assert plain.count("<|im_start|>assistant\n") == len(module._SHOTS["ja"])
    assert static_prompt("en", lyrics=True) != static_prompt("en")


def test_history_prompt_is_finished_chat_turns_in_time_order():
    entries = [
        ContextEntry("こんにちは", "你好", "ja", 1.0),
        ContextEntry("Thank you", "謝謝", "en", 2.0),
    ]

    text = history_prompt(entries)

    assert text == (
        "<|im_start|>user\nTranslate this Japanese text:\nこんにちは<|im_end|>\n"
        "<|im_start|>assistant\n你好<|im_end|>\n"
        "<|im_start|>user\nTranslate this English text:\nThank you<|im_end|>\n"
        "<|im_start|>assistant\n謝謝<|im_end|>\n"
    )
    assert history_prompt([]) == ""


def test_build_prompt_puts_history_between_examples_and_current_text():
    history = [ContextEntry("こんにちは", "你好", "ja", 1.0)]

    prompt = build_prompt("元気ですか", "ja", history)

    assert prompt == static_prompt("ja") + history_prompt(history) + user_prompt("元気ですか", "ja")
    assert build_prompt("元気ですか", "ja") == static_prompt("ja") + user_prompt("元気ですか", "ja")


# ----------------------------------------------------------------------------- 上下文
def test_context_is_off_by_default():
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
    assert "song lyrics" not in "".join(generator.calls[-1]["static_prompt"])

    instance.config["lyrics_mode"] = True
    instance.translate(["元気ですか"], "auto", "zh-hant")
    assert "song lyrics" in "".join(generator.calls[-1]["static_prompt"])

    instance.config["lyrics_mode"] = False
    instance.translate(["ありがとう"], "auto", "zh-hant")
    assert "song lyrics" not in "".join(generator.calls[-1]["static_prompt"])


def test_japanese_left_in_the_output_is_retried_with_the_plain_prompt():
    instance, generator = _translator(lyrics_mode=True)

    out = instance.translate(["進む"], "auto", "zh-hant")

    assert out == ["前進"]  # 歌詞提示回「進むと」（殘留假名），改用一般提示重翻
    assert len(generator.calls) == 2
    assert "song lyrics" not in "".join(generator.calls[-1]["static_prompt"])


def test_no_retry_when_the_output_is_clean():
    instance, generator = _translator(lyrics_mode=True)

    instance.translate(["こんにちは"], "auto", "zh-hant")

    assert len(generator.calls) == 1


# ----------------------------------------------------------------------------- 快取分流
@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({}, "auto"),
        ({"lyrics_mode": True}, "auto|lyrics"),
        ({"use_context": True}, "auto|ctx"),
        ({"lyrics_mode": True, "use_context": True}, "auto|lyrics|ctx"),
    ],
)
def test_cache_namespace_separates_translation_styles(section, expected):
    instance, _ = _translator(**section)

    assert instance.cache_namespace("auto") == expected


def test_toggling_a_mode_does_not_reuse_translations_made_without_it(tmp_path):
    instance, generator = _translator()
    instance.cache = TranslationCache(tmp_path / "cache.json")
    instance.translate(["こんにちは"], "auto", "zh-hant")
    assert len(generator.calls) == 1
    instance.translate(["こんにちは"], "auto", "zh-hant")
    assert len(generator.calls) == 1  # 同樣的設定：命中快取

    instance.config["lyrics_mode"] = True
    instance.translate(["こんにちは"], "auto", "zh-hant")

    assert len(generator.calls) == 2  # 換成歌詞模式：重新翻譯


def test_a_poisoned_cache_entry_with_the_original_prepended_is_ignored(tmp_path):
    instance, generator = _translator()
    instance.cache = TranslationCache(tmp_path / "cache.json")
    # 舊版「保留原文」+ 即時監控把「原文\n譯文」寫回快取，之後每輪再多疊一份原文
    instance.cache.set("auto", "zh-hant", "こんにちは", "こんにちは\nこんにちは\n你好")

    out = instance.translate(["こんにちは"], "auto", "zh-hant")

    assert out == ["你好"]
    assert instance.cache.get("auto", "zh-hant", "こんにちは") == "你好"


# ----------------------------------------------------------------------------- 設定與介面
def test_defaults_are_off_with_two_sentences_of_context():
    translation = DEFAULTS["translation"]

    assert translation["use_context"] is False
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
    assert not dialog.chk_use_context.isChecked()
    assert dialog.spin_context_size.value() == 2
    assert not dialog.spin_context_size.isEnabled()  # 沒勾上下文時，句數不能改
    assert not dialog.chk_lyrics_mode.isChecked()

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
