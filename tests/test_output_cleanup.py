"""譯文的後處理：把小模型常見的、與翻譯好壞無關的瑕疵用程式整理掉（日文頓號、日文新字體、「觸摸」被換成「觸控」、
略過標記混進句子、半形括號、客套回覆、英文單字原樣留著）。這些都是真實歌詞譯文裡出現過的。"""

from __future__ import annotations

import pytest

from services.translation.offline_common import looks_not_chinese
from tests.fake_llama import offline_translator, source_of
from utils import chinese
from utils.language_utils import untranslated_english_words

pytestmark = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")


# ----------------------------------------------------------------------------- 日文頓號
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("那、還沒有回去啊？", "那，還沒有回去啊？"),
        ("謝謝、媽媽", "謝謝，媽媽"),
        ("嗯、剛被老師叫出去一下", "嗯，剛被老師叫出去一下"),
        ("下一站是新宿、新宿。", "下一站是新宿，新宿。"),  # 只有兩項：當成逗號
        ("蘋果、香蕉、橘子很好吃", "蘋果、香蕉、橘子很好吃"),  # 三項以上的短並列是中文的頓號
        ("他買了蘋果、香蕉、橘子，然後回家、睡覺", "他買了蘋果、香蕉、橘子，然後回家，睡覺"),  # 一段一段判斷
        ("沒有頓號的句子", "沒有頓號的句子"),
    ],
)
def test_the_japanese_comma_becomes_a_chinese_comma_unless_it_separates_a_list(text, expected):
    assert chinese.normalize_punctuation(text) == expected


# ----------------------------------------------------------------------------- 半形括號
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今天去哪裡玩呢？(嗯….)", "今天去哪裡玩呢？（嗯….）"),
        ("二！(二！)一！", "二！（二！）一！"),
        ("f(x) 和 (1) 與 (A)", "f(x) 和 (1) 與 (A)"),  # 裡面沒有中文字：不動
    ],
)
def test_half_width_parentheses_around_chinese_become_full_width(text, expected):
    assert chinese.normalize_punctuation(text) == expected


# ----------------------------------------------------------------------------- 日文新字體、觸摸
def test_japanese_only_kanji_become_traditional():
    assert chinese.to_traditional("硬闘著過去") == "硬鬥著過去"
    assert chinese.to_traditional("気持ち") == "氣持ち"
    assert chinese.to_traditional("観る楽しみ") == "觀る樂しみ"


def test_traditional_text_is_left_alone_and_conversion_is_still_idempotent():
    text = "為什麼這麼真實，並且即時"
    assert chinese.to_traditional(text) == text
    assert chinese.to_traditional(chinese.to_traditional("软件视频")) == "軟體影片"


def test_touching_stays_touching_but_a_touch_screen_is_still_a_touch_screen():
    assert chinese.to_traditional("觸摸衣袖，触摸你的手") == "觸摸衣袖，觸摸你的手"
    assert chinese.to_traditional("觸摸屏") == chinese.to_traditional("触摸屏") != "觸摸屏"
    assert "" not in chinese.to_traditional("觸摸 触摸")  # 暫時用的標記不會漏出去


# ----------------------------------------------------------------------------- 略過標記混進句子
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("二！SKIP一！就出發吧", "二！一！就出發吧"),
        ("二！[SKIP] 一！", "二！一！"),
        ("[SKIP]", "[SKIP]"),  # 整句就是標記：交給略過標記的判斷
        ("SKIP", "SKIP"),
        ("skip the intro", "skip the intro"),  # 小寫的 skip 是一般的字
        ("好", "好"),
    ],
)
def test_the_skip_marker_is_removed_from_the_middle_of_a_sentence(text, expected):
    assert chinese.strip_skip_leak(text) == expected


# ----------------------------------------------------------------------------- 客套回覆
@pytest.mark.parametrize(
    ("text", "meta"),
    [
        ("對不起，這句話我有點不太明白，可能需要更多上下文來正確翻譯。", True),
        ("抱歉，我無法翻譯這個。", True),
        ("請提供更多資訊", True),
        ("對不起，我來晚了", False),
        ("謝謝你一直陪著我", False),
        ("", False),
    ],
)
def test_apologies_and_requests_for_context_are_not_translations(text, meta):
    assert chinese.is_meta_reply(text) is meta


# ----------------------------------------------------------------------------- 英文單字原樣留著
@pytest.mark.parametrize(
    ("source", "output", "words"),
    [
        ("Deep inside the asteroid belt, sir.", "在小行星帶深處， sir。", ["sir"]),
        ("We were young and we were reckless", "我們年輕且 reckless", ["reckless"]),
        ("Probably not. Don't wait up for me.", "probably not, 不要等我。", ["probably", "not"]),
        ("Press Esc to continue", "按 Esc 繼續", []),  # 按鍵、縮寫、人名不算
        ("Get the HP potion", "拿 HP 藥水", []),
        ("Check your email", "查看你的 email", []),  # 中文常保留的拼法
        ("Yes sir", "Yes sir", []),  # 完全沒翻是另一回事
        ("こんにちは", "你好", []),  # 日文原文不看
        ("Hello", "", []),
    ],
)
def test_english_words_left_untranslated_in_a_chinese_translation(source, output, words):
    assert untranslated_english_words(source, output) == words


def test_untranslated_english_and_meta_replies_count_as_not_chinese_so_they_get_retried():
    assert looks_not_chinese("Deep inside the asteroid belt, sir.", "在小行星帶深處， sir。")
    assert looks_not_chinese("見る", "對不起，這句話我有點不太明白，可能需要更多上下文來正確翻譯。")
    assert not looks_not_chinese("Deep inside the asteroid belt, sir.", "在小行星帶深處，長官。")


# ----------------------------------------------------------------------------- 整個翻譯流程
def test_the_whole_pipeline_cleans_what_the_model_writes():
    answers = {
        "ツー！(ツー！) ワン！で駆け出そう": "二！SKIP一！就出發吧",
        "あれ、まだ帰ってなかったの？": "那、還沒有回去啊？",
        "ならば力ずくで通ってみろ": "那就硬闘著過去吧。",
        "袖触れて満ち引いて": "觸摸衣袖，漾起波紋",
        "今日はどこ遊びにいこ？ (うーん)": "今天去哪裡玩呢？(嗯….)",
    }
    instance, _engine = offline_translator(lambda prompt: answers[source_of(prompt)])

    out = instance.translate(list(answers), "auto", "zh-hant")

    assert out == ["二！一！就出發吧", "那，還沒有回去啊？", "那就硬鬥著過去吧。", "觸摸衣袖，漾起波紋", "今天去哪裡玩呢？（嗯….）"]


def test_a_meta_reply_is_a_failed_translation_that_shows_the_original_and_is_not_cached():
    reply = "對不起，這句話我有點不太明白，可能需要更多上下文來正確翻譯。"
    instance, _engine = offline_translator(lambda prompt, seed=None: reply, retry_non_chinese=False)

    out = instance.translate(["お大事に対談だランバダ"], "auto", "zh-hant")

    assert out == ["お大事に対談だランバダ"] and instance.last_failed_count == 1


def test_an_untranslated_english_word_is_translated_again_with_sampling():
    source = "Deep inside the asteroid belt, sir."

    def decode(prompt, seed=None):
        return "在小行星帶深處，長官。" if seed else "在小行星帶深處， sir。"

    instance, engine = offline_translator(decode)

    assert instance.translate([source], "auto", "zh-hant") == ["在小行星帶深處，長官。"]
    assert engine.seeds == [None, 1]
