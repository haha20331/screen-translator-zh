"""辨識與翻譯分開進行的排程（只保留最新一筆）、穩定器的指定顯示內容、可被放棄的離線翻譯。"""

from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from services.translation.base import TranslationInterrupted
from tests.fake_llama import offline_translator
from utils.stability import CONFIRMED, SAME, TextStabilizer
from utils.translation_scheduler import IGNORED, QUEUED, START, TranslationScheduler


# ----------------------------------------------------------------------------- 排程
def test_the_first_subtitle_starts_immediately():
    scheduler = TranslationScheduler()

    assert scheduler.submit("A", "jobA") == START
    assert scheduler.busy and scheduler.running
    assert scheduler.is_current("A")


def test_the_same_subtitle_seen_every_round_is_only_translated_once():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")

    assert scheduler.submit("A", "jobA") == IGNORED
    assert scheduler.submit("A", "jobA") == IGNORED
    assert scheduler.finished() is None


def test_a_newer_subtitle_waits_and_makes_the_running_one_stale():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")

    assert scheduler.submit("B", "jobB") == QUEUED
    assert scheduler.should_stop_running() is True  # 正在翻的 A 已經過時
    assert scheduler.should_stop_running() is False  # 只通知一次
    assert not scheduler.is_current("A")
    assert scheduler.is_current("B")
    assert scheduler.finished() == ("B", "jobB")  # A 結束後接著翻 B
    assert scheduler.running


def test_only_the_latest_of_several_waiting_subtitles_is_kept():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")
    scheduler.submit("B", "jobB")
    scheduler.submit("C", "jobC")

    assert scheduler.finished() == ("C", "jobC")  # B 被蓋掉，從來沒翻過
    assert scheduler.finished() is None
    assert not scheduler.busy


def test_the_waiting_subtitle_is_not_queued_twice():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")
    scheduler.submit("B", "jobB")

    assert scheduler.submit("B", "jobB") == IGNORED


def test_a_flicker_back_to_the_running_subtitle_keeps_translating_it():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")
    # 中途閃過 B，但 B 還沒被通知停止 A 就又變回 A：A 繼續翻，B 作廢
    scheduler.submit("B", "jobB")
    scheduler._inflight_stopped = False  # 假設呼叫端還沒來得及中斷 A

    assert scheduler.submit("A", "jobA") == IGNORED
    assert scheduler.is_current("A")
    assert scheduler.finished() is None


def test_returning_to_a_subtitle_that_was_already_stopped_translates_it_again():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")
    scheduler.submit("B", "jobB")
    assert scheduler.should_stop_running() is True

    assert scheduler.submit("A", "jobA2") == QUEUED  # A 已經被叫停：翻完（放棄）後重新翻
    assert scheduler.should_stop_running() is False
    assert scheduler.finished() == ("A", "jobA2")


def test_cancel_all_drops_the_queue_and_stops_the_running_one_once():
    scheduler = TranslationScheduler()
    scheduler.submit("A", "jobA")
    scheduler.submit("B", "jobB")
    scheduler.should_stop_running()

    assert scheduler.cancel_all() is False  # A 已經被叫停過
    assert scheduler.finished() is None  # B 被丟掉
    assert not scheduler.is_current("B")

    scheduler.submit("C", "jobC")
    assert scheduler.cancel_all() is True
    assert scheduler.cancel_all() is False
    assert not scheduler.is_current("C")


def test_idle_scheduler_has_nothing_to_stop():
    scheduler = TranslationScheduler()

    assert not scheduler.busy
    assert scheduler.should_stop_running() is False
    assert scheduler.cancel_all() is False
    assert not scheduler.is_current(None)


# ----------------------------------------------------------------------------- 穩定器：指定顯示了哪一組
def test_commit_texts_marks_exactly_that_text_as_displayed_even_after_later_observations():
    stabilizer = TextStabilizer(rounds=1)
    assert stabilizer.observe(["A"]) == CONFIRMED
    assert stabilizer.observe(["B"]) == CONFIRMED  # 辨識繼續往下看：_confirmed 已經是 B

    stabilizer.commit_texts(["A"])  # 翻好的是 A

    assert stabilizer.observe(["A"]) == SAME
    assert stabilizer.observe(["B"]) == CONFIRMED  # B 還沒顯示過


def test_commit_texts_keeps_a_different_candidate_waiting():
    stabilizer = TextStabilizer(rounds=2)
    stabilizer.observe(["B"])  # B 等第二次確認

    stabilizer.commit_texts(["A"])

    assert stabilizer.has_pending
    assert stabilizer.observe(["B"]) == CONFIRMED


def test_commit_texts_clears_the_candidate_that_was_just_shown():
    stabilizer = TextStabilizer(rounds=2)
    stabilizer.observe(["A"])

    stabilizer.commit_texts(["A"])

    assert not stabilizer.has_pending


# ----------------------------------------------------------------------------- 可被放棄的離線翻譯
def _translator():
    return offline_translator(lambda prompt: "譯文", use_context=False)


def test_an_interruptible_translation_stops_before_asking_the_model():
    translator, engine = _translator()

    with translator.interruptible(lambda: True), pytest.raises(TranslationInterrupted):
        translator.translate(["こんにちは"], "ja", "zh-hant")

    assert engine.calls == 0


class _Cache:
    """只記錄有沒有被寫入的最小快取。"""

    def __init__(self) -> None:
        self.stored: dict = {}

    def get(self, source, target, text):
        return self.stored.get((source, target, text))

    def set(self, source, target, text, value) -> None:
        self.stored[(source, target, text)] = value


def test_an_interrupted_translation_gives_no_partial_result_and_is_not_cached():
    translator, engine = _translator()
    translator.cache = cache = _Cache()
    stop = {"now": False}

    def decode(prompt):
        stop["now"] = True  # 生成途中被要求放棄
        return "半句"

    engine.decode = decode

    with translator.interruptible(lambda: stop["now"]), pytest.raises(TranslationInterrupted):
        translator.translate(["こんにちは"], "ja", "zh-hant")

    assert cache.stored == {}  # 半句不寫進快取
    engine.decode = lambda prompt: "譯文"
    assert translator.translate(["こんにちは"], "ja", "zh-hant") == ["譯文"]  # 之後正常翻，不受影響


def test_the_stop_condition_is_asked_between_sentences():
    translator, engine = _translator()
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 3  # 第一句翻完之後才要求放棄

    with translator.interruptible(should_stop), pytest.raises(TranslationInterrupted):
        translator.translate(["こんにちは", "さようなら", "ありがとう"], "ja", "zh-hant")

    assert 1 <= engine.calls < 3


def test_the_engine_is_given_the_stop_condition_only_when_interruptible():
    translator, engine = _translator()

    translator.translate(["こんにちは"], "ja", "zh-hant")
    assert engine.should_stops == [None]

    with translator.interruptible(lambda: False):
        translator.translate(["さようなら"], "ja", "zh-hant")
    assert engine.should_stops[-1] is not None and engine.should_stops[-1]() is False


def test_other_threads_are_not_interrupted():
    """背景預熱在另一條執行緒：主執行緒設的放棄條件不能影響它。"""
    translator, engine = _translator()
    outcome: list = []

    with translator.interruptible(lambda: True):
        thread = threading.Thread(target=lambda: outcome.append(translator._generate(["Hello"], "en")))
        thread.start()
        thread.join()

    assert outcome == [["譯文"]]


def test_translation_works_normally_after_leaving_the_interruptible_block():
    translator, _engine = _translator()

    with translator.interruptible(lambda: True):
        pass

    assert translator.translate(["こんにちは"], "ja", "zh-hant") == ["譯文"]
    assert not translator.stop_requested()
