"""Qwen3 系列（混合思考模型）：翻譯用非思考模式，提示詞在回答開頭先放一個空的思考區塊。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from services.translation.model_names import (
    QWEN3_NON_THINKING_PREFIX,
    answer_prefix,
    display_name,
    model_family,
    model_tag,
    prompt_style,
    strip_thinking,
)
from services.translation.offline_translator import (
    OFFLINE_MODELS,
    OfflineTranslator,
    clean_output,
    static_prompt,
    user_prompt,
)
from tests.fake_llama import offline_translator, source_of
from utils import chinese

# 1.7B 已不在清單內，但使用者放進模型資料夾的自訂 Qwen3 模型照樣要認得
QWEN3 = ("qwen3-1.7b-gguf-q8_0", "qwen3-4b-gguf-q4_k_m")


def test_the_non_thinking_opening_is_an_empty_think_block_exactly_like_the_chat_template():
    assert QWEN3_NON_THINKING_PREFIX == "<think>\n\n</think>\n\n"  # enable_thinking=False 時 Qwen3 的聊天範本放在回答前面的字


@pytest.mark.parametrize("name", QWEN3)
def test_qwen3_models_get_the_opening_and_qwen25_models_do_not(name):
    assert model_family(name) == "qwen3" and answer_prefix(name) == QWEN3_NON_THINKING_PREFIX
    for other in ("qwen2.5-3b-instruct-gguf-q4_k_m", "qwen2.5-0.5b-instruct-gguf-q8_0", "custom"):
        assert model_family(other) == "qwen2.5" and answer_prefix(other) == ""


def test_qwen3_models_use_the_english_prompts_and_every_other_model_the_chinese_ones():
    """盲測：中文提示詞對 Qwen2.5 3B 明顯較好，對 Qwen3 1.7B 的台詞反而較差（PROMPT_TUNING.md）。"""
    assert all(prompt_style(name) == "en" for name in QWEN3)
    for other in ("qwen2.5-3b-instruct-gguf-q4_k_m", "qwen2.5-1.5b-instruct-gguf-q4_k_m", "qwen2.5-7b-instruct-gguf-q4_k_m", "custom"):
        assert prompt_style(other) == "zh"


def test_the_translator_sends_the_prompt_style_that_belongs_to_the_model():
    for name, header in (("qwen3-1.7b-gguf-q8_0", "Translate this Japanese text:\n"), ("qwen2.5-3b-instruct-gguf-q4_k_m", "翻譯這句日文：\n")):
        instance, fake = offline_translator(lambda prompt: "你好", offline={"model_name": name})

        instance.translate(["こんにちは"], "auto", "zh-hant")

        assert header in fake.prompts[0]
        assert fake.prefixes[0] == static_prompt("ja", style=prompt_style(name))


def test_the_user_turn_ends_with_the_opening_only_when_one_is_given():
    plain = user_prompt("会いたい", "ja", style="en")
    opened = user_prompt("会いたい", "ja", QWEN3_NON_THINKING_PREFIX, style="en")

    assert plain.endswith("<|im_start|>assistant\n")
    assert opened == plain + "<think>\n\n</think>\n\n"
    assert opened.startswith("<|im_start|>user\nTranslate this Japanese text:\n会いたい<|im_end|>")


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_the_translator_puts_the_opening_after_the_last_assistant_turn_for_qwen3_only():
    qwen3, fake3 = offline_translator(lambda prompt: "想見", offline={"model_name": "qwen3-1.7b-gguf-q8_0"})
    qwen25, fake25 = offline_translator(lambda prompt: "想見", offline={"model_name": "qwen2.5-3b-instruct-gguf-q4_k_m"})

    assert qwen3.translate(["会いたい"], "auto", "zh-hant") == ["想見"]
    qwen25.translate(["会いたい"], "auto", "zh-hant")

    assert fake3.prompts[0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert fake25.prompts[0].endswith("<|im_start|>assistant\n")
    # 示範（前面的 assistant 回答）不放思考區塊：聊天範本只在最後一輪放
    assert fake3.prompts[0].count("<think>") == 1
    assert source_of(fake3.prompts[0]) == "会いたい"


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_retries_keep_the_opening_and_the_fixed_prefix_is_shared_between_families():
    def decode(prompt: str, seed) -> str:
        return "想見ます" if seed is None else "想見"  # 第一次殘留假名，取樣重翻

    instance, fake = offline_translator(decode, offline={"model_name": "qwen3-4b-gguf-q4_k_m"})

    assert instance.translate(["会いたい"], "auto", "zh-hant") == ["想見"]
    assert fake.seeds == [None, 1]
    assert all(prompt.endswith("<think>\n\n</think>\n\n") for prompt in fake.prompts)
    assert set(fake.prefixes) == {static_prompt("ja", style="en")}  # Qwen3 用英文提示詞；固定前綴整場翻譯都一樣，快取照用


def test_a_thinking_block_the_model_writes_anyway_is_removed_from_the_output():
    assert strip_thinking("<think>\n先想一下\n</think>\n\n你好") == "\n\n你好"
    assert strip_thinking("你好") == "你好"
    assert strip_thinking("</think>\n你好") == "\n你好"  # 只剩結尾標記也去掉
    assert clean_output("<think>\n先想一下\n</think>\n\n  你好  \n（註解）") == "你好"
    assert clean_output("<think>沒有結尾") == "<think>沒有結尾".replace("<think>", "")  # 標記本身不顯示


def test_the_names_and_cache_tags_keep_the_two_families_apart():
    assert model_tag("qwen3-1.7b-gguf-q8_0") == "qwen3-1.7b-q8_0"
    assert model_tag("qwen2.5-1.5b-instruct-gguf-q8_0") == "1.5b-q8_0"
    assert display_name("qwen3-1.7b-gguf-q8_0") == "Qwen3 1.7B Q8_0"
    assert display_name("qwen3-4b-gguf-q4_k_m") == "Qwen3 4B Q4_K_M"
    assert display_name("qwen2.5-3b-instruct-gguf-q4_k_m") == "3B Q4_K_M"
    assert display_name("my-own-model") == "my-own-model"
    tags = {OfflineTranslator({"offline": {"model_name": model.name}}).cache_namespace("auto") for model in OFFLINE_MODELS}
    assert len(tags) == len(OFFLINE_MODELS)  # 每個模型各有自己的快取鍵


def test_qwen3_4b_is_listed_between_the_qwen25_3b_and_7b():
    names = [model.name for model in OFFLINE_MODELS]

    assert names.index("qwen2.5-3b-instruct-gguf-q4_k_m") < names.index("qwen3-4b-gguf-q4_k_m") < names.index("qwen2.5-7b-instruct-gguf-q4_k_m")
