"""設定 → 翻譯 →「測試速度」：用 3 句測資量每個已下載模型的熱機後翻譯時間，結果存檔、下次還看得到。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.application import Application
from app.config import AppConfig
from services.translation import speed_test as module
from services.translation.offline_translator import OFFLINE_MODELS, OfflineTranslator
from services.translation.speed_test import (
    SPEED_TEST_SENTENCES,
    STATUS_ERROR,
    STATUS_MEMORY,
    STATUS_MISSING,
    STATUS_OK,
    SpeedTestStore,
    memory_needed,
    model_bytes,
    run_speed_test,
    short_label,
)
from tests.fake_llama import offline_translator
from ui import settings_dialog as dialog_module
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils import chinese

MODELS = [("m-small", "小"), ("m-big", "大")]


# ----------------------------------------------------------------------------- 假的時鐘與翻譯器
class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeTranslator:
    """翻譯一句花多少時間由 ``costs`` 決定（秒）；預熱花 ``warmup_cost``。第 0 輪（不計時的那輪）故意很慢。"""

    def __init__(self, clock: _Clock, costs, warmup_cost=3.0, first_round_cost=9.0, fail_on=None, log=None) -> None:
        self.clock, self.costs, self.warmup_cost = clock, costs, warmup_cost
        self.first_round_cost, self.fail_on = first_round_cost, fail_on
        self.config = {"offline": {"threads": 4}}
        self.calls = 0
        self.closed = False
        self.log = log if log is not None else []

    def warmup(self) -> None:
        self.log.append("warmup")
        self.clock.now += self.warmup_cost

    def translate(self, texts, source, target):
        self.log.append("translate")
        index = self.calls % len(self.costs)
        in_first_round = self.calls < len(self.costs)
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise RuntimeError("記憶體不足")
        self.clock.now += self.first_round_cost if in_first_round else self.costs[index]
        return ["譯文"]

    def close(self) -> None:
        self.closed = True


def _run(clock, factory, **kwargs):
    events = {"start": [], "result": [], "progress": []}
    results = run_speed_test(
        {},
        kwargs.pop("models", MODELS),
        progress=events["progress"].append,
        on_start=events["start"].append,
        on_result=events["result"].append,
        make_translator=factory,
        memory_available=kwargs.pop("memory_available", lambda: None),
        sizes=kwargs.pop("sizes", lambda name: 1_000_000_000),
        clock=clock,
        **kwargs,
    )
    return results, events


# ----------------------------------------------------------------------------- 量測
def test_the_test_uses_exactly_three_sentences_covering_japanese_and_english():
    assert len(SPEED_TEST_SENTENCES) == 3
    assert [name for name, _ in SPEED_TEST_SENTENCES] == ["日文一般", "日文歌詞", "英文"]


def test_only_the_warmed_up_rounds_are_timed_and_the_load_time_is_reported_separately():
    clock = _Clock()
    made = []

    def factory(name):
        made.append(name)
        return _FakeTranslator(clock, costs=[0.1, 0.2, 0.3])

    results, events = _run(clock, factory, models=[("m-small", "小")])

    (result,) = results
    assert result["status"] == STATUS_OK
    assert result["ms"] == {"日文一般": 100, "日文歌詞": 200, "英文": 300}  # 不含不計時的第一輪（那輪故意花了 9 秒）
    assert result["avg_ms"] == 200
    assert result["load_s"] == pytest.approx(3.0, abs=0.01)  # 載入＋預熱單獨記，不混進每句的推論時間
    assert result["threads"] == 4


def test_the_median_of_the_timed_rounds_is_used():
    clock = _Clock()
    translator = _FakeTranslator(clock, costs=[0.1, 0.1, 0.1])
    results, _ = _run(clock, lambda name: translator, models=[("m", "m")], rounds=3)

    assert results[0]["ms"]["日文一般"] == 100


def test_each_sentence_is_translated_on_its_own_and_warmup_comes_first():
    clock = _Clock()
    translator = _FakeTranslator(clock, costs=[0.1, 0.1, 0.1])
    _run(clock, lambda name: translator, models=[("m", "m")])

    assert translator.log[0] == "warmup"
    assert translator.log.count("translate") == 3 * (1 + module.TIMED_ROUNDS)  # 一輪不計時 + 計時的輪數，每輪 3 句


def test_every_model_is_started_reported_and_released_in_order():
    clock = _Clock()
    translators = {}

    def factory(name):
        translators[name] = _FakeTranslator(clock, costs=[0.1, 0.1, 0.1])
        return translators[name]

    results, events = _run(clock, factory)

    assert [r["model"] for r in results] == ["m-small", "m-big"]
    assert events["start"] == ["m-small", "m-big"]
    assert [r["model"] for r in events["result"]] == ["m-small", "m-big"]
    assert all(t.closed for t in translators.values())  # 記憶體要還給系統，下一個模型才載得進來
    assert any("小" in text for text in events["progress"]) and any("測試中" in text for text in events["progress"])
    assert any("熱機中" in text and "不計時" in text for text in events["progress"])  # 不計時的那輪標明是熱機，不要和正式的輪數混在一起
    first_timed = f"小：測試中…（第 1／{module.TIMED_ROUNDS} 輪，第 1／3 句）"
    assert events["progress"].count(first_timed) == 1


# ----------------------------------------------------------------------------- 略過與失敗
def test_a_model_that_was_not_downloaded_is_marked_missing_and_never_loaded():
    clock = _Clock()
    made = []
    results, events = _run(clock, lambda name: made.append(name), sizes=lambda name: None)

    assert [r["status"] for r in results] == [STATUS_MISSING, STATUS_MISSING]
    assert made == [] and events["start"] == []


def test_a_model_that_does_not_fit_in_the_free_memory_is_skipped_with_the_numbers():
    clock = _Clock()
    made = []
    sizes = {"m-small": 1_000_000_000, "m-big": 7_000_000_000}
    results, events = _run(
        clock,
        lambda name: made.append(name) or _FakeTranslator(clock, costs=[0.1, 0.1, 0.1]),
        sizes=sizes.get,
        memory_available=lambda: 4_000_000_000,
    )

    assert [r["status"] for r in results] == [STATUS_OK, STATUS_MEMORY]
    assert made == ["m-small"]
    assert results[1]["message"] == f"可用 4.0 GB，需要約 {memory_needed(7_000_000_000) / 1e9:.1f} GB"


def test_unknown_free_memory_does_not_skip_anything():
    clock = _Clock()
    results, _ = _run(clock, lambda name: _FakeTranslator(clock, costs=[0.1, 0.1, 0.1]), sizes=lambda name: 99_000_000_000)

    assert [r["status"] for r in results] == [STATUS_OK, STATUS_OK]


def test_a_failing_model_is_recorded_and_the_next_one_still_runs():
    clock = _Clock()
    translators = []

    def factory(name):
        translator = _FakeTranslator(clock, costs=[0.1, 0.1, 0.1], fail_on=2 if name == "m-small" else None)
        translators.append(translator)
        return translator

    results, _ = _run(clock, factory)

    assert [r["status"] for r in results] == [STATUS_ERROR, STATUS_OK]
    assert "記憶體不足" in results[0]["message"]
    assert all(t.closed for t in translators)


def test_a_model_file_that_cannot_be_loaded_is_a_failure_not_a_crash():
    clock = _Clock()

    def factory(name):
        raise OSError("檔案損毀")

    results, _ = _run(clock, factory, models=[("m", "m")])

    assert results[0]["status"] == STATUS_ERROR and "檔案損毀" in results[0]["message"]


def test_stopping_halts_after_the_current_sentence_and_does_not_record_a_half_finished_model():
    clock = _Clock()
    translators = []
    stop = {"flag": False}

    def factory(name):
        translator = _FakeTranslator(clock, costs=[0.1, 0.1, 0.1])
        original = translator.translate

        def translate(*args):
            if name == "m-big" and translator.calls == 2:
                stop["flag"] = True
            return original(*args)

        translator.translate = translate
        translators.append(translator)
        return translator

    results, events = _run(clock, factory, cancelled=lambda: stop["flag"])

    assert [r["model"] for r in results] == ["m-small"]  # 第二個測到一半被停下：不存半成品
    assert [r["model"] for r in events["result"]] == ["m-small"]
    assert all(t.closed for t in translators)


def test_stopping_before_the_first_model_does_nothing():
    clock = _Clock()
    made = []
    results, _ = _run(clock, lambda name: made.append(name), cancelled=lambda: True)

    assert results == [] and made == []


# ----------------------------------------------------------------------------- 設定、檔案、名稱
def test_the_test_settings_keep_the_users_threads_but_switch_off_context_and_lyrics_for_every_model():
    original = {"use_context": True, "lyrics_mode": True, "offline": {"threads": 3, "model_dir": "D:/mine", "model_name": "auto"}}

    section = module._section_for(original, "m-big")

    assert section["use_context"] is False and section["lyrics_mode"] is False
    assert section["offline"] == {"threads": 3, "model_dir": "", "model_name": "m-big"}
    assert original["use_context"] is True and original["offline"]["model_name"] == "auto"  # 不改到使用者的設定


def test_short_names():
    assert short_label("qwen2.5-3b-instruct-gguf-q4_k_m") == "3B Q4_K_M"
    assert short_label("qwen2.5-0.5b-instruct-gguf-q8_0") == "0.5B Q8_0"


def test_the_needed_memory_is_a_bit_more_than_the_file():
    assert memory_needed(2_000_000_000) > 2_000_000_000 * 1.15


def test_model_size_adds_up_all_the_shards_of_a_model(tmp_path, monkeypatch):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "a-00001-of-00002.gguf").write_bytes(b"x" * 1000)
    (folder / "a-00002-of-00002.gguf").write_bytes(b"x" * 500)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "model_search_paths", lambda name, model_dir="": [tmp_path / "nothing", folder])

    assert model_bytes("m") == 1500
    monkeypatch.setattr(module, "model_search_paths", lambda name, model_dir="": [tmp_path / "nothing"])
    assert model_bytes("m") is None


def test_free_memory_is_a_positive_number_or_unknown():
    free = module.available_memory_bytes()

    assert free is None or free > 0


def test_the_result_file_keeps_each_model_and_survives_new_runs(tmp_path):
    store = SpeedTestStore(tmp_path / "speed_test.json")
    assert store.load()["models"] == {}

    store.save_result({"model": "a", "status": STATUS_OK, "avg_ms": 100}, {"cpu": "x"})
    store.save_result({"model": "b", "status": STATUS_OK, "avg_ms": 200})
    store.save_result({"model": "a", "status": STATUS_OK, "avg_ms": 90})  # 重測：只換掉 a

    data = SpeedTestStore(tmp_path / "speed_test.json").load()
    assert {name: entry["avg_ms"] for name, entry in data["models"].items()} == {"a": 90, "b": 200}
    assert data["machine"] == {"cpu": "x"}
    assert [s["name"] for s in data["sentences"]] == ["日文一般", "日文歌詞", "英文"]
    assert not (tmp_path / "speed_test.json.tmp").exists()


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"models": []}', '{"models": {"a": 5}}'])
def test_a_broken_result_file_is_treated_as_no_record(tmp_path, content):
    path = tmp_path / "speed_test.json"
    path.write_text(content, encoding="utf-8")

    assert SpeedTestStore(path).load()["models"] == {}


def test_a_result_file_that_cannot_be_written_does_not_stop_the_test(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    store = SpeedTestStore(blocked / "speed_test.json")  # 上層是檔案，建不起資料夾

    store.save_result({"model": "a", "status": STATUS_OK})  # 只記錄、不丟例外


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_the_translator_can_release_its_model():
    instance, engine = offline_translator(lambda prompt: "譯文")
    closed = []
    engine.close = lambda: closed.append(True)

    instance.close()

    assert instance._llama is None and closed == [True]
    instance.close()  # 再關一次也沒事


# ----------------------------------------------------------------------------- 設定畫面
@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _config(tmp_path, model="qwen2.5-3b-instruct-gguf-q4_k_m"):
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.offline.model_name", model)
    return config


def _result(name, avg=500, **extra):
    return {
        "model": name, "status": STATUS_OK, "ms": {"日文一般": avg - 50, "日文歌詞": avg + 100, "英文": avg - 50},
        "avg_ms": avg, "load_s": 8.2, "tested_at": "2026-10-14T15:00:00", "threads": 6, **extra,
    }


def _cells(dialog, row):
    return [dialog.table_speed.item(row, column).text() for column in range(dialog.table_speed.columnCount())]


def _row_of(name):
    return [model.name for model in OFFLINE_MODELS].index(name)


@pytest.fixture
def downloaded(monkeypatch):
    """假設這些模型都下載了（不要看這台電腦實際有什麼）。"""
    names = {model.name for model in OFFLINE_MODELS}
    monkeypatch.setattr(dialog_module, "llama_model_available", lambda name, model_dir="": name in names)
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))
    return names


def _wait_for_the_test_to_finish(dialog, seconds=5.0):
    deadline = time.monotonic() + seconds
    while dialog._speed_testing and time.monotonic() < deadline:
        dialog._poll_speed_test()
        time.sleep(0.01)
    dialog._poll_speed_test()
    assert not dialog._speed_testing, "測試沒有在時限內結束"


def test_the_table_sits_under_the_model_selector_with_a_button_and_lists_every_model(qapp, tmp_path, downloaded):
    dialog = SettingsDialog(_config(tmp_path))
    try:
        page = dialog.pages.widget(3).widget()  # 翻譯頁
        assert page.isAncestorOf(dialog.btn_speed_test) and page.isAncestorOf(dialog.table_speed)
        assert dialog.btn_speed_test.text() == "測試速度"
        assert dialog.table_speed.rowCount() == len(OFFLINE_MODELS)
        headers = [dialog.table_speed.horizontalHeaderItem(c).text() for c in range(dialog.table_speed.columnCount())]
        assert headers == ["模型", "日文一般", "日文歌詞", "英文", "平均", "載入（秒）"]
        assert [_cells(dialog, r)[0] for r in range(7)] == [
            "0.5B Q8_0", "1.5B Q4_K_M", "1.5B Q8_0", "Qwen3 1.7B Q8_0", "★ 3B Q4_K_M", "Qwen3 4B Q4_K_M", "7B Q4_K_M",
        ]  # ★ = 使用中
        assert dialog.lbl_speed_test.text() == "尚未測試過"
        assert _cells(dialog, 0)[1] == "尚未測試"
    finally:
        dialog.deleteLater()


def test_models_that_are_not_downloaded_say_so(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(dialog_module, "llama_model_available", lambda name, model_dir="": name.endswith("q4_k_m") and "1.5b" in name)
    dialog = SettingsDialog(_config(tmp_path))
    try:
        assert _cells(dialog, _row_of("qwen2.5-7b-instruct-gguf-q4_k_m"))[1] == "未下載"
        assert _cells(dialog, _row_of("qwen2.5-1.5b-instruct-gguf-q4_k_m"))[1] == "尚未測試"
    finally:
        dialog.deleteLater()


def test_saved_results_are_shown_as_soon_as_the_settings_open(qapp, tmp_path, downloaded):
    SpeedTestStore(tmp_path / "speed_test.json").save_result(_result("qwen2.5-3b-instruct-gguf-q4_k_m", avg=550))
    SpeedTestStore(tmp_path / "speed_test.json").save_result(
        {"model": "qwen2.5-7b-instruct-gguf-q4_k_m", "status": STATUS_MEMORY, "message": "可用 4.0 GB，需要約 5.9 GB", "tested_at": "2026-10-14T15:01:00"}
    )
    dialog = SettingsDialog(_config(tmp_path))
    try:
        assert _cells(dialog, _row_of("qwen2.5-3b-instruct-gguf-q4_k_m")) == ["★ 3B Q4_K_M", "500", "650", "500", "550", "8.2"]
        seven = _row_of("qwen2.5-7b-instruct-gguf-q4_k_m")
        assert _cells(dialog, seven)[1] == "記憶體不足，略過（可用 4.0 GB，需要約 5.9 GB）"  # 數字直接顯示，不必猜為什麼被略過
        assert "2026-10-14 15:01" in dialog.lbl_speed_test.text()  # 最近一次測試的時間
    finally:
        dialog.deleteLater()


def test_the_star_follows_the_model_in_use_and_auto_resolves_to_the_real_one(qapp, tmp_path, downloaded):
    dialog = SettingsDialog(_config(tmp_path, model="auto"))
    try:
        stars = [r for r in range(7) if _cells(dialog, r)[0].startswith("★")]
        assert stars == [_row_of("qwen2.5-3b-instruct-gguf-q4_k_m")]  # auto → 預設的 3B（這台有下載）
    finally:
        dialog.deleteLater()


def _fake_runner(record, *, wait_for_stop=False, fail=False):
    def runner(section, models, *, progress, on_start, on_result, cancelled):
        record["models"] = [name for name, _ in models]
        record["section"] = section
        for name, label in models:
            if cancelled():
                break
            on_start(name)
            progress(f"{label}：測試中…")
            if wait_for_stop:
                while not cancelled():
                    time.sleep(0.005)
                break
            if fail:
                raise RuntimeError("爆了")
            on_result(_result(name, avg=100 * (1 + len(record.setdefault("done", [])))))
            record["done"].append(name)
        return []

    return runner


def _yes(monkeypatch):
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Yes)
    return asked


def test_pressing_the_button_tests_every_downloaded_model_shows_and_saves_the_results(qapp, tmp_path, downloaded, monkeypatch):
    record = {}
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner(record))
    asked = _yes(monkeypatch)
    dialog = SettingsDialog(_config(tmp_path))
    try:
        dialog.btn_speed_test.click()
        assert dialog._speed_testing or record  # 一按就開始（在背景執行緒，不會卡住畫面）
        _wait_for_the_test_to_finish(dialog)

        assert "7 個模型" in asked[0] and "3 句" in asked[0]
        assert record["models"] == [model.name for model in OFFLINE_MODELS]
        assert [_cells(dialog, r)[4] for r in range(7)] == ["100", "200", "300", "400", "500", "600", "700"]
        assert dialog.btn_speed_test.text() == "測試速度" and dialog.btn_speed_test.isEnabled()
        saved = SpeedTestStore(tmp_path / "speed_test.json").load()
        assert set(saved["models"]) == {model.name for model in OFFLINE_MODELS}
        assert saved["machine"]["logical_cpus"] > 0  # 順便記下這是哪台電腦
    finally:
        dialog.deleteLater()


def test_results_are_there_again_after_closing_and_reopening_the_settings(qapp, tmp_path, downloaded, monkeypatch):
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner({}))
    _yes(monkeypatch)
    first = SettingsDialog(_config(tmp_path))
    first.btn_speed_test.click()
    _wait_for_the_test_to_finish(first)
    first.deleteLater()

    second = SettingsDialog(_config(tmp_path))
    try:
        assert [_cells(second, r)[4] for r in range(7)] == ["100", "200", "300", "400", "500", "600", "700"]
    finally:
        second.deleteLater()


def test_only_the_downloaded_models_are_tested_and_the_test_uses_the_saved_translation_settings(qapp, tmp_path, monkeypatch):
    only = {"qwen2.5-1.5b-instruct-gguf-q4_k_m", "qwen2.5-3b-instruct-gguf-q4_k_m"}
    monkeypatch.setattr(dialog_module, "llama_model_available", lambda name, model_dir="": name in only)
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))
    record = {}
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner(record))
    _yes(monkeypatch)
    config = _config(tmp_path)
    config.set("translation.offline.threads", 5)
    dialog = SettingsDialog(config)
    try:
        dialog.btn_speed_test.click()
        _wait_for_the_test_to_finish(dialog)

        assert record["models"] == sorted(only, key=lambda n: [m.name for m in OFFLINE_MODELS].index(n))
        assert record["section"]["offline"]["threads"] == 5
        assert _cells(dialog, _row_of("qwen2.5-7b-instruct-gguf-q4_k_m"))[1] == "未下載"
    finally:
        dialog.deleteLater()


def test_declining_the_confirmation_starts_nothing(qapp, tmp_path, downloaded, monkeypatch):
    record = {}
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner(record))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    dialog = SettingsDialog(_config(tmp_path))
    try:
        dialog.btn_speed_test.click()

        assert record == {} and not dialog._speed_testing
    finally:
        dialog.deleteLater()


def test_monitoring_blocks_the_test_with_the_reason(qapp, tmp_path, downloaded, monkeypatch):
    record = {}
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner(record))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    dialog = SettingsDialog(_config(tmp_path), speed_test_guard=lambda: "即時監控進行中，請先停止")
    try:
        dialog.btn_speed_test.click()

        assert shown == ["即時監控進行中，請先停止"] and record == {} and not dialog._speed_testing
    finally:
        dialog.deleteLater()


def test_a_missing_llama_library_is_explained_instead_of_failing(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: "請安裝 llama-cpp-python"))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    dialog = SettingsDialog(_config(tmp_path))
    try:
        dialog.btn_speed_test.click()

        assert shown == ["請安裝 llama-cpp-python"]
    finally:
        dialog.deleteLater()


def test_no_downloaded_model_tells_the_user_how_to_get_one(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(dialog_module, "llama_model_available", lambda name, model_dir="": False)
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    dialog = SettingsDialog(_config(tmp_path))
    try:
        dialog.btn_speed_test.click()

        assert len(shown) == 1 and "fetch_translation_models.py" in shown[0]
    finally:
        dialog.deleteLater()


def test_the_button_becomes_stop_while_testing_and_stopping_ends_the_test(qapp, tmp_path, downloaded, monkeypatch):
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner({}, wait_for_stop=True))
    _yes(monkeypatch)
    dialog = SettingsDialog(_config(tmp_path))
    try:
        dialog.btn_speed_test.click()
        deadline = time.monotonic() + 3
        while not dialog._speed_running and time.monotonic() < deadline:
            dialog._poll_speed_test()
            time.sleep(0.01)
        assert dialog.btn_speed_test.text() == "停止測試"
        assert "測試中…" in _cells(dialog, 0)[1] or any("測試中…" in _cells(dialog, r)[1] for r in range(7))

        dialog.btn_speed_test.click()  # 停止
        assert not dialog.btn_speed_test.isEnabled()  # 等目前這個模型測完
        _wait_for_the_test_to_finish(dialog)

        assert dialog.btn_speed_test.text() == "測試速度" and dialog.btn_speed_test.isEnabled()
        assert dialog.lbl_speed_test.text().startswith("已停止")
    finally:
        dialog.deleteLater()


def test_an_unexpected_error_is_reported_and_the_button_comes_back(qapp, tmp_path, downloaded, monkeypatch):
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner({}, fail=True))
    _yes(monkeypatch)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    dialog = SettingsDialog(_config(tmp_path))
    try:
        dialog.btn_speed_test.click()
        _wait_for_the_test_to_finish(dialog)

        assert len(warned) == 1 and "爆了" in warned[0]
        assert dialog.btn_speed_test.text() == "測試速度"
    finally:
        dialog.deleteLater()


def test_closing_the_settings_tells_the_background_test_to_stop(qapp, tmp_path, downloaded, monkeypatch):
    seen = {}

    def runner(section, models, *, progress, on_start, on_result, cancelled):
        seen["cancelled"] = cancelled
        started.set()
        while not cancelled():
            time.sleep(0.005)

    started = threading.Event()
    monkeypatch.setattr(dialog_module, "run_speed_test", runner)
    _yes(monkeypatch)
    dialog = SettingsDialog(_config(tmp_path))
    dialog.btn_speed_test.click()
    assert started.wait(3)

    dialog.deleteLater()
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    deadline = time.monotonic() + 3
    while not seen["cancelled"]() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert seen["cancelled"]()  # 測試執行緒會自己收尾，不會永遠佔著 CPU 與記憶體


def test_the_application_refuses_the_test_while_monitoring_or_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(QApplication.instance() or QApplication([]))
    try:
        assert instance._speed_test_blocker() == ""
        instance._monitor_active = True
        assert "即時監控" in instance._speed_test_blocker()
        instance._monitor_active = False
        instance._busy = True
        assert instance._speed_test_blocker() != ""
    finally:
        instance.floating_status.deleteLater()


def test_the_apps_own_model_is_released_before_the_test_and_loaded_back_after_it(qapp, tmp_path, downloaded, monkeypatch):
    order = []
    monkeypatch.setattr(dialog_module, "run_speed_test", lambda *a, **k: order.append("run"))
    _yes(monkeypatch)
    dialog = SettingsDialog(
        _config(tmp_path), speed_test_hooks=(lambda: order.append("release"), lambda: order.append("reload"))
    )
    try:
        dialog.btn_speed_test.click()
        _wait_for_the_test_to_finish(dialog)

        assert order == ["release", "run", "reload"]  # 先騰出記憶體，測完再載回來
    finally:
        dialog.deleteLater()


def test_the_model_is_loaded_back_even_when_the_test_is_stopped_or_fails(qapp, tmp_path, downloaded, monkeypatch):
    order = []
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner({}, fail=True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    _yes(monkeypatch)
    dialog = SettingsDialog(_config(tmp_path), speed_test_hooks=(lambda: order.append("release"), lambda: order.append("reload")))
    try:
        dialog.btn_speed_test.click()
        _wait_for_the_test_to_finish(dialog)

        assert order == ["release", "reload"]
    finally:
        dialog.deleteLater()


def test_a_failing_hook_never_blocks_the_test(qapp, tmp_path, downloaded, monkeypatch):
    record = {}
    monkeypatch.setattr(dialog_module, "run_speed_test", _fake_runner(record))
    _yes(monkeypatch)

    def broken():
        raise RuntimeError("放不掉")

    dialog = SettingsDialog(_config(tmp_path), speed_test_hooks=(broken, broken))
    try:
        dialog.btn_speed_test.click()
        _wait_for_the_test_to_finish(dialog)

        assert record["models"] and not dialog._speed_testing
    finally:
        dialog.deleteLater()


def test_the_application_frees_and_restores_its_translation_model_around_the_test(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(QApplication.instance() or QApplication([]))
    try:
        closed, warmed = [], []
        instance.translator = type("T", (), {"close": lambda self: closed.append(True)})()
        monkeypatch.setattr(instance, "_warmup", lambda: warmed.append(True))

        instance._release_translation_model()
        instance._reload_translation_model()
        instance._shutting_down = True
        instance._reload_translation_model()  # 離開程式中：不要再載模型

        assert closed == [True] and warmed == [True]
        instance.translator = None
        instance._release_translation_model()  # 沒有翻譯器也不會出錯
    finally:
        instance.floating_status.deleteLater()


# ----------------------------------------------------------------------------- GPU
class _DeviceTranslator(_FakeTranslator):
    def __init__(self, clock, device):
        super().__init__(clock, costs=[0.1, 0.1, 0.1])
        self.device = device


@pytest.mark.parametrize(
    ("supported", "setting", "expected"),
    [(False, -1, False), (False, 8, False), (True, -1, True), (True, 0, False), (True, 8, True), (True, "abc", True), (True, None, True)],
)
def test_the_gpu_is_requested_only_with_a_gpu_build_and_a_non_zero_layer_setting(monkeypatch, supported, setting, expected):
    monkeypatch.setattr(module.llama_backend, "gpu_offload_supported", lambda: supported)
    section = {"offline": {} if setting is None else {"gpu_layers": setting}}

    assert module.gpu_requested(section) is expected
    assert module.gpu_requested({}) is (supported)  # 沒有設定：預設 -1


def test_the_device_that_was_really_used_is_recorded(monkeypatch):
    clock = _Clock()
    on_gpu, _ = _run(clock, lambda name: _DeviceTranslator(clock, "gpu"), models=[("m", "m")], gpu=True)
    on_cpu, _ = _run(clock, lambda name: _DeviceTranslator(clock, "cpu"), models=[("m", "m")], gpu=False)
    fell_back, _ = _run(clock, lambda name: _DeviceTranslator(clock, "cpu"), models=[("m", "m")], gpu=True)
    mislabeled, _ = _run(clock, lambda name: _DeviceTranslator(clock, "gpu"), models=[("m", "m")], gpu=False)  # CPU 版：翻譯器自稱 gpu
    unknown, _ = _run(clock, lambda name: _FakeTranslator(clock, costs=[0.1, 0.1, 0.1]), models=[("m", "m")], gpu=False)

    assert (on_gpu[0]["device"], on_gpu[0]["gpu_fallback"]) == ("gpu", False)
    assert (on_cpu[0]["device"], on_cpu[0]["gpu_fallback"]) == ("cpu", False)
    assert (fell_back[0]["device"], fell_back[0]["gpu_fallback"]) == ("cpu", True)  # 要求用 GPU 卻退回 CPU：要讓人看得出來
    assert (mislabeled[0]["device"], mislabeled[0]["gpu_fallback"]) == ("cpu", False)
    assert "device" not in unknown[0]


def test_a_gpu_run_does_not_look_at_the_system_memory(monkeypatch):
    clock = _Clock()
    asked = []
    monkeypatch.setattr(module, "_default_translator", lambda section, name, cpu_fallback=True: asked.append(cpu_fallback) or _DeviceTranslator(clock, "gpu"))
    results = run_speed_test(
        {}, [("m", "m")], sizes=lambda name: 7_000_000_000, memory_available=lambda: 1_000_000_000, clock=clock, gpu=True, discrete=True,
    )

    assert results[0]["status"] == STATUS_OK  # 獨立顯示卡：模型放在顯示卡記憶體，系統記憶體只剩 1 GB 也不略過
    assert asked == [False]  # 但 GPU 載不進去的話，系統記憶體也載不進這個模型，就不要退回 CPU


def test_a_gpu_run_with_enough_system_memory_may_fall_back_to_the_cpu(monkeypatch):
    clock = _Clock()
    asked = []
    monkeypatch.setattr(module, "_default_translator", lambda section, name, cpu_fallback=True: asked.append(cpu_fallback) or _DeviceTranslator(clock, "gpu"))
    run_speed_test({}, [("m", "m")], sizes=lambda name: 2_000_000_000, memory_available=lambda: 16_000_000_000, clock=clock, gpu=True, discrete=True)

    assert asked == [True]


def test_an_integrated_gpu_shares_the_system_memory_so_the_memory_check_still_applies(monkeypatch):
    clock = _Clock()
    made = []
    monkeypatch.setattr(module, "_default_translator", lambda *a, **k: made.append(a))
    results = run_speed_test(
        {}, [("m", "m")], sizes=lambda name: 7_000_000_000, memory_available=lambda: 3_000_000_000, clock=clock, gpu=True, discrete=False,
    )

    assert results[0]["status"] == STATUS_MEMORY and made == []  # 內顯要載入的模型也是放在系統記憶體裡


def test_the_discrete_gpu_is_looked_up_when_not_told(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(module, "_default_translator", lambda *a, **k: _DeviceTranslator(clock, "gpu"))
    monkeypatch.setattr(module.llama_backend, "discrete_gpu_free_bytes", lambda: 8_000_000_000)

    results = run_speed_test({}, [("m", "m")], sizes=lambda name: 7_000_000_000, memory_available=lambda: 1_000_000_000, clock=clock, gpu=True)
    monkeypatch.setattr(module.llama_backend, "discrete_gpu_free_bytes", lambda: 0)
    skipped = run_speed_test({}, [("m", "m")], sizes=lambda name: 7_000_000_000, memory_available=lambda: 1_000_000_000, clock=clock, gpu=True)

    assert results[0]["status"] == STATUS_OK and skipped[0]["status"] == STATUS_MEMORY


def test_a_cpu_run_still_skips_models_that_do_not_fit(monkeypatch):
    clock = _Clock()
    made = []
    monkeypatch.setattr(module, "_default_translator", lambda *a, **k: made.append(a))
    results = run_speed_test(
        {}, [("m", "m")], sizes=lambda name: 7_000_000_000, memory_available=lambda: 1_000_000_000, clock=clock, gpu=False,
    )

    assert results[0]["status"] == STATUS_MEMORY and made == []


def test_the_default_translator_turns_the_cpu_fallback_off_only_when_asked(monkeypatch):
    captured = []
    monkeypatch.setattr("services.translation.factory.create_translator", lambda service, section, cache=None: captured.append(section))

    module._default_translator({}, "m")
    module._default_translator({}, "m", cpu_fallback=False)

    assert "cpu_fallback" not in captured[0]["offline"] and captured[1]["offline"]["cpu_fallback"] is False


def test_the_table_names_the_device_each_number_was_measured_on(qapp, tmp_path, downloaded):
    store = SpeedTestStore(tmp_path / "speed_test.json")
    store.save_result(_result("qwen2.5-3b-instruct-gguf-q4_k_m", device="gpu", gpu_fallback=False))
    store.save_result(_result("qwen2.5-1.5b-instruct-gguf-q4_k_m", device="cpu", gpu_fallback=False))
    store.save_result(_result("qwen2.5-0.5b-instruct-gguf-q8_0", device="cpu", gpu_fallback=True))
    store.save_result(_result("qwen2.5-1.5b-instruct-gguf-q8_0"))  # 舊版存的結果沒有裝置欄位
    dialog = SettingsDialog(_config(tmp_path))
    try:
        assert _cells(dialog, _row_of("qwen2.5-3b-instruct-gguf-q4_k_m"))[0] == "★ 3B Q4_K_M · GPU"
        assert _cells(dialog, _row_of("qwen2.5-1.5b-instruct-gguf-q4_k_m"))[0] == "1.5B Q4_K_M · CPU"
        assert _cells(dialog, _row_of("qwen2.5-0.5b-instruct-gguf-q8_0"))[0] == "0.5B Q8_0 · CPU（GPU 失敗）"
        assert "GPU" in dialog.table_speed.item(_row_of("qwen2.5-0.5b-instruct-gguf-q8_0"), 0).toolTip()
        assert _cells(dialog, _row_of("qwen2.5-1.5b-instruct-gguf-q8_0"))[0] == "1.5B Q8_0"  # 沒有紀錄就不亂標
    finally:
        dialog.deleteLater()
