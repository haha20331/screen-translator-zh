"""生成時擋詞：原文決定擋哪幾類、詞彙表分類、遮罩（含放行 [SKIP]）、接進引擎與翻譯器。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import collections

import numpy as np
import pytest

from services.translation import llama_backend
from services.translation.offline_common import SKIP_MARKER
from services.translation.token_filter import FOREIGN, LATIN, SPACES, TokenFilter, bans_for, source_has_latin
from tests.fake_llama import offline_translator, source_of
from utils import chinese

# ----------------------------------------------------------------------------- 原文決定要擋什麼
ALL = frozenset({FOREIGN, LATIN, SPACES})


def test_a_source_without_latin_letters_blocks_everything_foreign_latin_and_spacing():
    assert bans_for("夜空に浮かぶ星を数えて") == ALL
    assert bans_for("会いたい") == ALL
    assert bans_for("12:30") == ALL  # 數字不是英文字母


def test_a_source_with_latin_letters_only_blocks_foreign_scripts():
    assert bans_for("HPが50回復した！") == {FOREIGN}  # 譯文保留 HP 是對的，也可能有「HP 50」這種空格
    assert bans_for("Press any key to continue") == {FOREIGN}
    assert bans_for("ＨＰが回復") == {FOREIGN}  # 全形英文字母
    assert bans_for("__ST_KEEP_0__ を開く") == {FOREIGN}  # 保護用的標記含英文字母，必須能原樣寫回去


def test_latin_detection_ignores_digits_symbols_and_kana():
    assert not source_has_latin("１２３、あ")
    assert source_has_latin("Ａ計画")
    assert not source_has_latin("")


# ----------------------------------------------------------------------------- 詞彙表分類與遮罩
VOCAB = {
    0: "你".encode(),
    1: "の".encode(),  # 假名
    2: "계속".encode(),  # 韓文
    3: "привет".encode(),  # 西里爾
    4: b"hello",
    5: b" the",
    6: " 的".encode(),  # 空白＋漢字
    7: b" ",
    8: b"  ",
    9: b"\n",  # 換行不擋
    10: b"<|im_end|>",  # 控制記號含英文字母，擋了模型就停不下來
    11: b"\xe3\x81",  # 只有半個字的位元組片段：解不出來，不擋
    12: "ー".encode(),
    13: b"SKIP",
    14: b" SKIP",
    15: b"[",
    16: b"]",
    17: b"123",
    18: " ，".encode(),  # 空白＋全形標點
    19: b"",
}


def _filter():
    texts = {"[SKIP]": [15, 13, 16], "SKIP": [13], " SKIP": [14]}
    return TokenFilter(lambda: (VOCAB[i] for i in range(len(VOCAB))), lambda text: texts.get(text, []))


def _blocked(mask) -> set[int]:
    return {int(i) for i in np.flatnonzero(np.isneginf(mask))}


def test_the_vocabulary_is_split_into_the_three_classes():
    assert _filter().counts() == {FOREIGN: 4, LATIN: 4, SPACES: 4}


def test_each_class_blocks_exactly_its_own_tokens_and_never_chinese_digits_newlines_or_control_tokens():
    f = _filter()

    assert _blocked(f.mask({FOREIGN})) == {1, 2, 3, 12}
    assert _blocked(f.mask({LATIN})) == {4, 5, 13, 14}
    assert _blocked(f.mask({SPACES})) == {6, 7, 8, 18}
    assert not (_blocked(f.mask(ALL)) & {0, 9, 10, 11, 15, 16, 17, 19})  # 你、換行、<|im_end|>、半個字、[ ]、數字、空詞


def test_allowed_texts_are_exempt_even_when_their_letters_are_blocked():
    f = _filter()

    blocked = _blocked(f.mask(ALL, ("[SKIP]", "SKIP")))

    assert 13 not in blocked and 14 not in blocked  # SKIP 與 空白SKIP 放行（[SKIP] 回報標記）
    assert {4, 5, 1, 2, 3}.issubset(blocked)  # 其他英文與外文照擋


def test_masks_are_cached_and_nothing_to_block_means_no_processor():
    f = _filter()

    assert f.mask(ALL) is f.mask(ALL)
    assert f.mask(frozenset()) is None
    assert f.processor(frozenset()) is None


def test_the_processor_adds_the_mask_to_the_scores_and_ignores_a_mismatched_size():
    f = _filter()
    apply = f.processor({FOREIGN})
    scores = np.zeros(len(VOCAB), dtype=np.float32)

    out = apply(None, scores)

    assert np.isneginf(out[1]) and out[0] == 0
    short = np.zeros(3, dtype=np.float32)
    assert apply(None, short) is short  # 詞彙表大小對不上：不亂動


def test_the_vocabulary_is_classified_once_and_only_when_needed():
    calls = []

    def pieces():
        calls.append(1)
        return iter([b"a"])

    f = TokenFilter(pieces, lambda text: [])
    assert calls == []
    f.mask({LATIN})
    f.mask({FOREIGN})
    f.counts()
    assert calls == [1]


# ----------------------------------------------------------------------------- 接進引擎
class _Llama:
    def __init__(self) -> None:
        self.kwargs = {}

    def n_vocab(self) -> int:
        return len(VOCAB)

    def detokenize(self, tokens):
        return VOCAB[tokens[0]]

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False):
        return {"[SKIP]": [15, 13, 16]}.get(data.decode(), [])

    def create_completion(self, prompt, **kwargs):
        self.kwargs = kwargs
        return {"choices": [{"text": "譯文"}]}


def _engine():
    engine = llama_backend.LlamaCppEngine.__new__(llama_backend.LlamaCppEngine)
    engine._llama = _Llama()
    engine._prefix_states = collections.OrderedDict()
    engine._prefix_cache_ok = True
    engine._init_token_filter()
    return engine


def test_the_engine_hands_a_logits_processor_to_llama_only_when_something_is_banned():
    engine = _engine()

    engine.complete("p", 10)
    assert "logits_processor" not in engine._llama.kwargs

    engine.complete("p", 10, bans=frozenset({FOREIGN}))
    (processor,) = engine._llama.kwargs["logits_processor"]
    assert np.isneginf(processor(None, np.zeros(len(VOCAB), dtype=np.float32))[1])


def test_banning_works_with_sampling_retries_too_and_keeps_the_sampling_settings():
    engine = _engine()

    engine.complete("p", 10, seed=1, bans=ALL, allow=("[SKIP]", "SKIP"))

    kwargs = engine._llama.kwargs
    assert kwargs["seed"] == 1 and kwargs["top_k"] == 20 and "logits_processor" in kwargs


def test_a_vocabulary_that_cannot_be_classified_turns_the_ban_off_instead_of_failing(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine._llama, "n_vocab", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert engine.complete("p", 10, bans=ALL) == "譯文"  # 照原本的行為翻譯，事後檢查還在
    assert "logits_processor" not in engine._llama.kwargs
    assert engine._token_filter_ok is False
    engine.complete("p", 10, bans=ALL)  # 之後不再嘗試


def test_prepare_token_filter_classifies_ahead_of_time():
    engine = _engine()

    engine.prepare_token_filter()

    assert engine._token_filter._classes is not None


# ----------------------------------------------------------------------------- 接進翻譯器
pytestmark_chinese = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")


def _decode(prompt: str) -> str:
    return {"夜空に浮かぶ星": "夜空中的星星", "Hello": "哈囉", "ドン！": SKIP_MARKER}.get(source_of(prompt), "")


@pytestmark_chinese
def test_japanese_sentences_block_foreign_latin_and_spacing_and_english_ones_only_foreign():
    instance, fake = offline_translator(_decode)

    instance.translate(["夜空に浮かぶ星"], "auto", "zh-hant")
    instance.translate(["Hello"], "auto", "zh-hant")

    assert fake.bans == [ALL, frozenset({FOREIGN})]


@pytestmark_chinese
def test_the_skip_marker_is_allowed_through_only_while_the_sound_skip_is_on():
    on, fake_on = offline_translator(_decode)
    on.translate(["ドン！"], "auto", "zh-hant")
    off, fake_off = offline_translator(_decode, skip_sounds=False)
    off.translate(["夜空に浮かぶ星"], "auto", "zh-hant")

    assert fake_on.allows[0] == ("[SKIP]", "SKIP")
    assert fake_off.allows[0] == ()


@pytestmark_chinese
def test_with_the_retry_option_off_the_model_is_left_alone():
    instance, fake = offline_translator(_decode, retry_non_chinese=False)

    instance.translate(["夜空に浮かぶ星"], "auto", "zh-hant")

    assert fake.bans == [frozenset()] and fake.allows == [()]  # 「翻出什麼就輸出什麼」


@pytestmark_chinese
def test_retries_with_sampling_keep_the_bans():
    def decode(prompt: str, seed) -> str:
        return "夜空 NIGHT" if seed is None else "夜空中的星星"

    instance, fake = offline_translator(decode)

    out = instance.translate(["夜空に浮かぶ星"], "auto", "zh-hant")

    assert out == ["夜空中的星星"] and fake.seeds == [None, 1]
    assert fake.bans == [ALL, ALL]


@pytestmark_chinese
def test_warmup_classifies_the_vocabulary_only_when_the_filter_is_in_use():
    on, fake_on = offline_translator(_decode)
    on.warmup()
    off, fake_off = offline_translator(_decode, retry_non_chinese=False)
    off.warmup()

    assert fake_on.filter_prepared == 1 and fake_off.filter_prepared == 0


# ----------------------------------------------------------------------------- 擋詞之後的亂碼與預熱
def test_a_run_of_symbols_is_junk_unless_the_source_has_it_too():
    from services.translation.offline_common import looks_not_chinese
    from utils.language_utils import has_symbol_run

    assert has_symbol_run("謝謝代替見######")  # 好詞被擋掉之後，模型有時會退而求其次寫出一排符號
    assert has_symbol_run("****")
    assert not has_symbol_run("謝謝再見")
    assert not has_symbol_run("等一下……！！！")  # 中文常見的省略號、驚嘆號不算
    assert not has_symbol_run("——")
    assert not has_symbol_run("=====", "=====")  # 原文就有的分隔線
    assert looks_not_chinese("さよならの代わりにありがとう", "謝謝代替見######")  # 會被取樣重翻救回


@pytestmark_chinese
def test_junk_after_blocking_is_retried_with_sampling():
    def decode(prompt: str, seed) -> str:
        return "謝謝代替見######" if seed is None else "謝謝你代替說再見"

    instance, fake = offline_translator(decode)

    assert instance.translate(["さよならの代わりにありがとう"], "auto", "zh-hant") == ["謝謝你代替說再見"]
    assert fake.seeds == [None, 1]


@pytestmark_chinese
def test_warmup_prepares_both_languages_so_the_first_subtitle_does_not_pay_for_the_prefix():
    instance, fake = offline_translator(_decode)

    instance.warmup()

    prefixes = set(fake.prefixes)
    assert len(prefixes) == 2  # 日文與英文各一種固定前綴（各要算一次，3B 約 3 秒、7B 約 7 秒）
    assert [source_of(prompt) for prompt in fake.prompts] == ["Hello", "こんにちは"]


def test_code_like_symbols_are_junk_too_and_can_be_stripped_as_a_last_resort():
    from utils.language_utils import has_symbol_junk, strip_symbol_junk

    junk = '感謝代替見面別！!");'  # 7B 的好詞被擋掉後寫出來的亂碼
    assert has_symbol_junk(junk, "さよならの代わりにありがとう")
    assert strip_symbol_junk(junk, "さよならの代わりにありがとう") == "感謝代替見面別！!"
    assert strip_symbol_junk("######", "x") == ""
    assert not has_symbol_junk("她說 \"你好\"", 'She said "hi"')  # 原文有引號：保留是對的
    assert not has_symbol_junk("(笑)", "（笑）")  # 全形括號正規化後就是半形括號
    assert not has_symbol_junk("謝謝；再見！")
    assert strip_symbol_junk("謝謝再見", "さよなら") == "謝謝再見"  # 沒有亂碼就原樣


@pytestmark_chinese
def test_junk_that_no_retry_can_fix_is_stripped_before_it_is_shown():
    instance, fake = offline_translator(lambda prompt, seed: '感謝代替見面別！!");')

    out = instance.translate(["さよならの代わりにありがとう"], "auto", "zh-hant")

    assert out == ["感謝代替見面別！！"]  # 標點之後還會整理成全形
    assert fake.seeds == [None, 1, 2]  # 取樣重翻最多兩次都救不回來


@pytestmark_chinese
def test_the_skip_marker_survives_the_final_junk_cleanup():
    instance, _ = offline_translator(lambda prompt: SKIP_MARKER)

    assert instance.translate(["ドン！"], "auto", "zh-hant") == ["ドン！"]  # 擬聲詞照樣略過（原文保留、不算失敗）
    assert instance.last_failed_count == 0
