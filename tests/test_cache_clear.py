"""設定裡的「清除翻譯快取」：方便換不同模型做比較時，不會看到上一個模型留下的譯文。"""

from __future__ import annotations

import copy
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.application import Application
from app.config import AppConfig
from services.translation.cache import TranslationCache
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.stability import SAME, TextStabilizer


# ----------------------------------------------------------------------------- 快取本身
def test_count_and_clear_report_how_many_were_removed_and_empty_the_file(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json")
    for index in range(3):
        cache.set("auto", "zh-hant", f"原文{index}", f"譯文{index}")
    cache.flush()
    assert cache.count() == 3

    assert cache.clear() == 3

    assert cache.count() == 0 and cache.get("auto", "zh-hant", "原文0") is None
    assert json.loads((tmp_path / "cache.json").read_text(encoding="utf-8")) == {}  # 檔案也清空了
    assert TranslationCache(tmp_path / "cache.json").count() == 0  # 重開程式不會又出現
    assert cache.clear() == 0


# ----------------------------------------------------------------------------- 設定畫面
@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


class _Actions:
    def __init__(self, count: int) -> None:
        self.entries = count
        self.cleared = 0

    def count(self) -> int:
        return self.entries

    def clear(self) -> int:
        removed, self.entries = self.entries, 0
        self.cleared += 1
        return removed


def _dialog(tmp_path, actions):
    return SettingsDialog(AppConfig(tmp_path / "config.json"), cache_actions=(actions.count, actions.clear) if actions else None)


def test_the_button_shows_how_many_translations_are_stored(qapp, tmp_path):
    dialog = _dialog(tmp_path, _Actions(42))

    assert dialog.btn_clear_cache.isEnabled()
    assert dialog.lbl_cache_count.text() == "目前存了 42 句"
    assert dialog.pages.widget(3).widget().isAncestorOf(dialog.btn_clear_cache)  # 「翻譯」頁，和快取筆數放在一起
    dialog.deleteLater()


def test_confirming_clears_the_cache_right_away(qapp, tmp_path, monkeypatch):
    actions = _Actions(5)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.StandardButton.Yes)
    dialog = _dialog(tmp_path, actions)

    dialog.btn_clear_cache.click()

    assert actions.cleared == 1 and actions.entries == 0
    assert "5 句" in asked[0]  # 問的時候就說明會刪幾句
    assert dialog.lbl_cache_count.text() == "已清除 5 句"
    assert not dialog.btn_clear_cache.isEnabled()  # 沒東西可清了
    dialog.deleteLater()


def test_declining_keeps_everything(qapp, tmp_path, monkeypatch):
    actions = _Actions(5)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    dialog = _dialog(tmp_path, actions)

    dialog.btn_clear_cache.click()

    assert actions.cleared == 0 and actions.entries == 5
    assert dialog.btn_clear_cache.isEnabled()
    dialog.deleteLater()


def test_an_empty_cache_has_nothing_to_clear_and_no_cache_object_disables_the_button(qapp, tmp_path):
    empty = _dialog(tmp_path, _Actions(0))
    assert not empty.btn_clear_cache.isEnabled()
    alone = _dialog(tmp_path, None)  # 單獨開設定視窗（沒有應用程式）：沒有快取可清
    assert not alone.btn_clear_cache.isEnabled() and alone.lbl_cache_count.text() == ""
    alone._on_clear_cache()  # 即使被呼叫也不會出錯
    empty.deleteLater()
    alone.deleteLater()


# ----------------------------------------------------------------------------- 應用程式
@pytest.fixture
def controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(app)
    yield instance
    instance.floating_status.deleteLater()


def test_clearing_also_makes_the_monitor_forget_what_is_on_screen(controller, tmp_path):
    controller.cache = TranslationCache(tmp_path / "cache.json")
    controller.cache.set("auto", "zh-hant", "こんにちは", "舊模型的譯文")
    controller._monitor_active = True
    controller._monitor_stabilizer = TextStabilizer(rounds=1)
    controller._monitor_stabilizer.observe(["こんにちは"])
    controller._monitor_stabilizer.commit()
    assert controller._monitor_stabilizer.observe(["こんにちは"]) == SAME
    controller._monitor_known = {"こんにちは": "舊模型的譯文"}
    controller._monitor_signature = "shown"
    controller._monitor_frame_hash = "frame"

    removed = controller.clear_translation_cache()

    assert removed >= 1 and controller.cache.count() == 0
    # 不忘掉的話，監控下一輪會把舊模型的譯文又寫回快取，換模型比較就白做了
    assert controller._monitor_known == {} and controller._monitor_signature == "" and controller._monitor_frame_hash == ""
    assert controller._monitor_stabilizer.observe(["こんにちは"]) != SAME  # 同一句字幕會重新翻譯
    controller._monitor_active = False


def test_the_count_and_clear_work_without_a_cache(controller):
    controller.cache = None

    assert controller.translation_cache_count() == 0
    assert controller.clear_translation_cache() == 0


def test_saving_the_settings_forgets_the_displayed_translations_too(controller, monkeypatch):
    """換了模型再儲存設定：畫面上舊模型的譯文不能被寫進新模型的快取。"""
    for name in ("_refresh_appearance", "refresh_monitor_map", "_warmup", "_apply_hotkeys", "_apply_autostart"):
        monkeypatch.setattr(controller, name, lambda *a, **k: None)
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    monkeypatch.setattr(controller, "_build_translator", lambda: None)
    controller._monitor_known = {"こんにちは": "舊模型的譯文"}
    controller._monitor_signature = "shown"

    controller._apply_settings()

    assert controller._monitor_known == {} and controller._monitor_signature == ""


# ----------------------------------------------------------------------------- 換模型時自動清空
MODEL_1_5B = "qwen2.5-1.5b-instruct-gguf-q8_0"
MODEL_3B = "qwen2.5-3b-instruct-gguf-q4_k_m"


def _edit_settings(controller, **values):
    """像設定視窗存檔那樣改設定：先把整份設定換成副本再改（翻譯器手上拿的還是舊的那一份，才比得出有沒有變）。"""
    controller.config.data = copy.deepcopy(controller.config.data)
    for key, value in values.items():
        controller.config.set(key.replace("__", "."), value)


def _with_offline_model(controller, tmp_path, model=MODEL_1_5B):
    controller.cache = TranslationCache(tmp_path / "cache.json")
    controller.config.set("translation.service", "offline")
    controller.config.set("translation.offline.model_name", model)
    controller.translator = controller._build_translator()
    controller.cache.set("auto", "zh-hant", "こんにちは", "舊模型的譯文")
    controller.cache.set("auto", "zh-hant", "ありがとう", "舊模型的譯文二")
    controller._monitor_known = {"こんにちは": "舊模型的譯文"}
    controller._monitor_signature = "shown"
    return controller


def test_changing_the_offline_model_clears_the_cache_and_forgets_the_screen(controller, tmp_path):
    _with_offline_model(controller, tmp_path)
    _edit_settings(controller, translation__offline__model_name=MODEL_3B)

    cleared = controller._rebuild_translator()

    assert cleared == 2 and controller.cache.count() == 0
    assert controller._monitor_known == {} and controller._monitor_signature == ""
    assert controller._translator_label() == f"offline/{MODEL_3B}"


def test_an_unchanged_model_keeps_the_cache(controller, tmp_path):
    _with_offline_model(controller, tmp_path)
    first = controller.translator

    cleared = controller._rebuild_translator()  # 設定沒變：重用同一個翻譯器

    assert cleared == 0 and controller.cache.count() == 2 and controller.translator is first
    assert controller._monitor_signature == "shown"


def test_changing_other_translation_options_keeps_the_cache(controller, tmp_path):
    _with_offline_model(controller, tmp_path)
    _edit_settings(controller, translation__offline__threads=3)  # 換了設定、重建翻譯器，但模型還是同一個
    first = controller.translator

    assert controller._rebuild_translator() == 0
    assert controller.translator is not first  # 真的重建了
    assert controller.cache.count() == 2


def test_the_first_translator_has_no_previous_model_so_nothing_is_cleared(controller, tmp_path):
    controller.cache = TranslationCache(tmp_path / "cache.json")
    controller.cache.set("auto", "zh-hant", "こんにちは", "譯文")
    controller.translator = None
    controller.config.set("translation.service", "offline")

    assert controller._rebuild_translator() == 0
    assert controller.cache.count() == 1


def test_the_model_that_made_the_cache_is_remembered_in_the_config_file(controller, tmp_path):
    _with_offline_model(controller, tmp_path)
    controller._rebuild_translator()

    assert AppConfig(tmp_path / "config.json").get("translation.cache_model") == f"offline/{MODEL_1_5B}"


def test_starting_with_a_different_model_than_last_time_clears_the_cache(controller, tmp_path):
    _with_offline_model(controller, tmp_path)
    controller.config.set("translation.cache_model", f"offline/{MODEL_3B}")  # 上次是 3B（例如現在 3B 被刪了、改用 1.5B）

    assert controller.reconcile_cache_with_model() == 2
    assert controller.cache.count() == 0
    assert controller.config.get("translation.cache_model") == f"offline/{MODEL_1_5B}"


def test_starting_with_the_same_model_or_no_record_keeps_the_cache(controller, tmp_path):
    _with_offline_model(controller, tmp_path)

    assert controller.reconcile_cache_with_model() == 0  # 沒有紀錄（第一次）：只記下來
    assert controller.cache.count() == 2
    assert controller.config.get("translation.cache_model") == f"offline/{MODEL_1_5B}"
    assert controller.reconcile_cache_with_model() == 0  # 和上次相同
    assert controller.cache.count() == 2


def test_saving_the_settings_with_a_new_model_says_so_in_the_status(controller, tmp_path, monkeypatch):
    for name in ("_refresh_appearance", "refresh_monitor_map", "_warmup", "_apply_hotkeys", "_apply_autostart"):
        monkeypatch.setattr(controller, name, lambda *a, **k: None)
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    _with_offline_model(controller, tmp_path)
    _edit_settings(controller, translation__offline__model_name=MODEL_3B)
    messages: list[str] = []
    controller.set_status = messages.append

    controller._apply_settings()

    assert controller.cache.count() == 0
    assert messages[-1] == "設定已儲存（換了翻譯模型，已清空翻譯快取 2 句）"


def test_saving_without_a_new_model_keeps_the_plain_status(controller, tmp_path, monkeypatch):
    for name in ("_refresh_appearance", "refresh_monitor_map", "_warmup", "_apply_hotkeys", "_apply_autostart"):
        monkeypatch.setattr(controller, name, lambda *a, **k: None)
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    _with_offline_model(controller, tmp_path)
    messages: list[str] = []
    controller.set_status = messages.append

    controller._apply_settings()

    assert controller.cache.count() == 2 and messages[-1] == "設定已儲存"
