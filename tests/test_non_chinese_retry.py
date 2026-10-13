"""譯文夾雜外文（假名、韓文、俄文、原文沒有的英文單字）時，改用取樣重翻。"""

from __future__ import annotations

import pytest

from services.translation import llama_backend
from services.translation.offline_common import SKIP_MARKER, looks_not_chinese
from tests.fake_llama import offline_translator, source_of
from utils import chinese
from utils.language_utils import has_foreign_script, has_spaced_out_chars, unexpected_latin_words

pytestmark = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")


# ----------------------------------------------------------------------------- 判斷
@pytest.mark.parametrize("text", ["계속你的名字", "привет", "你好 مرحبا", "สวัสดี", "ไ", "שלום", "γειά", "नमस्ते", "ㅋㅋ"])
def test_other_writing_systems_are_foreign_to_a_chinese_translation(text):
    assert has_foreign_script(text)


@pytest.mark.parametrize("text", ["你好", "恢復了 50 點 HP！", "明天見", "ありがとう", "Hello", "１２３", ""])
def test_chinese_japanese_and_latin_are_not_foreign_scripts(text):
    assert not has_foreign_script(text)


@pytest.mark.parametrize(
    ("source", "output", "words"),
    [
        ("夜空に浮かぶ星を数えて", "數 NIGHT SKY 中的星星", ["NIGHT", "SKY"]),
        ("さよならの代わりにありがとう", "謝謝代替見 Wrest", ["Wrest"]),
        ("電車が遅れているようです", "電車似乎 delayed。", ["delayed"]),
        ("HPが50回復した！", "恢復了 50 點 HP！", []),  # 原文有的、兩個字母的縮寫都不算
        ("Press Ctrl+S to save", "按 Ctrl+S 儲存", []),  # 原文就有的英文單字照留（大小寫不分）
        ("Where are you going?", "你要去 WHERE？", []),
        ("こんにちは", "你好", []),
    ],
)
def test_latin_words_the_original_does_not_have(source, output, words):
    assert unexpected_latin_words(source, output) == words


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        ("淚 的 數 越 總 勝 達", True),
        ("你 的 聲 在 聽", True),
        ("你 的 聲 音", True),  # 四個字、每個之間一個空格
        ("你 好", False),
        ("你 好 嗎", False),  # 三個字：不夠，避免誤傷
        ("恢復了 50 點 HP！", False),
        ("按 Ctrl+S 儲存", False),
        ("在 NIGHT 的 夜空", False),
        ("一直叫著你的名字", False),
        ("", False),
    ],
)
def test_text_with_a_space_between_every_character_is_runaway_output(text, bad):
    assert has_spaced_out_chars(text) is bad


@pytest.mark.parametrize(
    ("source", "output", "bad"),
    [
        ("見る", "看見ます", True),
        ("君の名前を呼び続けた", "계속你的名字", True),
        ("夜空に浮かぶ星を数えて", "數 NIGHT SKY 中的星星", True),
        ("涙の数だけ強くなれる", "淚 的 數 越 總 勝 達", True),
        ("今日はいい天気ですね。", "今天天氣真好呢。", False),
        ("HPが50回復した！", "恢復了 50 點 HP！", False),
        ("ドン！", "ドン！", True),  # 原文抄回來（有假名）：呼叫端自己會因為是擬聲詞略過，這裡只管「像不像中文」
    ],
)
def test_looks_not_chinese(source, output, bad):
    assert looks_not_chinese(source, output) is bad


# ----------------------------------------------------------------------------- 重翻
def _decoder(answers):
    """answers：種子（None = 貪婪）→ 模型寫的字。"""

    def decode(prompt, seed=None):
        source = source_of(prompt)
        table = answers.get(source, {})
        return table.get(seed, table.get(None, ""))

    return decode


BAD = "계속你的名字"
SOURCE = "君の名前を呼び続けた"


def test_a_clean_answer_is_never_retried():
    instance, engine = offline_translator(_decoder({SOURCE: {None: "一直叫著你的名字"}}))

    assert instance.translate([SOURCE], "auto", "zh-hant") == ["一直叫著你的名字"]
    assert engine.seeds == [None]


def test_a_foreign_answer_is_translated_again_with_sampling_until_it_looks_chinese():
    instance, engine = offline_translator(_decoder({SOURCE: {None: BAD, 1: "數 NIGHT 中", 2: "不停地呼叫著你的名字"}}))

    out = instance.translate([SOURCE], "auto", "zh-hant")

    assert out == ["不停地呼叫著你的名字"]
    assert engine.seeds == [None, 1, 2]  # 貪婪 → 取樣（種子 1）→ 取樣（種子 2）


def test_the_first_good_resample_wins_and_the_second_is_not_asked():
    instance, engine = offline_translator(_decoder({SOURCE: {None: BAD, 1: "不停地呼叫著你的名字", 2: "別的譯法"}}))

    assert instance.translate([SOURCE], "auto", "zh-hant") == ["不停地呼叫著你的名字"]
    assert engine.seeds == [None, 1]


def test_if_every_try_is_bad_the_original_answer_is_shown_and_it_is_not_a_failure():
    instance, engine = offline_translator(_decoder({SOURCE: {None: BAD, 1: "again 계속", 2: "Wrest"}}))

    out = instance.translate([SOURCE], "auto", "zh-hant")

    assert out == [BAD]  # 照原樣輸出（使用者說夾雜外文也沒關係），不是翻譯失敗
    assert instance.last_failed_count == 0
    assert engine.seeds == [None, 1, 2]


def test_only_the_bad_sentences_of_a_batch_are_translated_again():
    instance, engine = offline_translator(
        _decoder({SOURCE: {None: BAD, 1: "不停地呼叫著你的名字"}, "こんにちは": {None: "你好"}, "ありがとう": {None: "謝謝"}})
    )

    out = instance.translate(["こんにちは", SOURCE, "ありがとう"], "auto", "zh-hant")

    assert out == ["你好", "不停地呼叫著你的名字", "謝謝"]
    assert engine.seeds == [None, None, None, 1]  # 三句各翻一次，只有壞的那句再翻一次


def test_a_swapped_in_word_from_the_original_is_not_treated_as_foreign():
    instance, engine = offline_translator(_decoder({"HPが50回復した！": {None: "恢復了 50 點 HP！"}}))

    instance.translate(["HPが50回復した！"], "auto", "zh-hant")

    assert engine.seeds == [None]


def test_turning_the_retry_switch_off_asks_the_model_only_once():
    instance, engine = offline_translator(_decoder({SOURCE: {None: BAD, 1: "不停地呼叫著你的名字"}}), retry_non_chinese=False)

    assert instance.translate([SOURCE], "auto", "zh-hant") == [BAD]  # 夾雜韓文也照輸出
    assert engine.seeds == [None]


def test_a_skip_answer_is_not_a_reason_to_retry():
    instance, engine = offline_translator(_decoder({"ドン！": {None: SKIP_MARKER, 1: "砰"}}))

    assert instance.translate(["ドン！"], "auto", "zh-hant") == ["ドン！"]
    assert engine.seeds == [None]


def test_kana_left_in_a_plain_answer_is_also_resampled():
    # 「See you tomorrow → 明天見ます」：沒開上下文與歌詞時，一般提示重翻只會得到一模一樣的結果，要靠取樣才有機會翻對
    instance, engine = offline_translator(_decoder({"See you tomorrow": {None: "明天見ます", 1: "明天見"}}))

    assert instance.translate(["See you tomorrow"], "auto", "zh-hant") == ["明天見"]
    assert engine.seeds == [None, 1]


# ----------------------------------------------------------------------------- 引擎
class _Llama:
    def __init__(self) -> None:
        self.kwargs = {}

    def create_completion(self, prompt, **kwargs):
        self.kwargs = kwargs
        return {"choices": [{"text": "譯文"}]}


def _engine():
    engine = llama_backend.LlamaCppEngine.__new__(llama_backend.LlamaCppEngine)
    engine._llama = _Llama()
    engine._prefix_cache_ok = False
    return engine


def test_without_a_seed_the_engine_decodes_greedily():
    engine = _engine()
    engine.complete("提示詞", 50)

    assert engine._llama.kwargs["temperature"] == 0.0 and engine._llama.kwargs["top_k"] == 1
    assert "seed" not in engine._llama.kwargs


def test_with_a_seed_the_engine_uses_qwens_recommended_sampling_and_the_seed():
    engine = _engine()
    engine.complete("提示詞", 50, seed=2)

    kwargs = engine._llama.kwargs
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"], kwargs["repeat_penalty"], kwargs["seed"]) == (0.7, 0.8, 20, 1.05, 2)
    assert kwargs["stop"] == ["<|im_end|>"] and kwargs["max_tokens"] == 50
