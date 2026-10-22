"""模型資料夾：把資料夾裡的模型讀進選單、寫清楚模型放在哪（沒有任何模型時也寫）。"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from services.translation import offline_common
from services.translation.offline_common import (
    custom_model_folder,
    discover_models,
    find_model_file,
    model_folders,
    model_search_paths,
)
from services.translation.offline_translator import llama_model_available
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style

KNOWN = "qwen2.5-1.5b-instruct-gguf-q8_0"


def _gguf(folder: Path, name: str = "model.gguf", size: int = 1000) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"0" * size)
    return path


@pytest.fixture
def places(tmp_path, monkeypatch):
    """程式資料夾與使用者資料夾都指到暫存資料夾，不碰真正的模型。"""
    app_root = tmp_path / "app"
    user_data = tmp_path / "user"
    monkeypatch.setattr(offline_common, "_app_roots", lambda: [app_root])
    monkeypatch.setattr(offline_common, "app_data_dir", lambda: user_data)
    return app_root / "models" / "translation", user_data / "models" / "translation"


# ----------------------------------------------------------------------------- 資料夾與掃描
def test_the_model_folders_are_the_app_folder_the_user_folder_and_the_custom_one(places, tmp_path):
    app_folder, user_folder = places

    assert model_folders("") == [app_folder, user_folder]
    assert model_folders(str(tmp_path / "mine")) == [app_folder, user_folder, tmp_path / "mine"]
    assert model_folders(str(app_folder)) == [app_folder, user_folder]  # 自訂的和標準的一樣：不重複列
    assert custom_model_folder("") is None


def test_a_relative_custom_folder_is_counted_from_the_app_folder(places):
    app_folder, _ = places

    assert custom_model_folder("my-models") == app_folder.parents[1] / "my-models"


def test_no_models_anywhere_finds_nothing_even_if_the_folders_do_not_exist(places):
    assert discover_models("") == []


def test_every_subfolder_with_a_gguf_file_is_a_model_named_after_the_folder(places):
    app_folder, user_folder = places
    _gguf(app_folder / KNOWN, "a.gguf", 300)
    _gguf(user_folder / "my-own-model", "b.gguf", 700)
    (app_folder / "empty-folder").mkdir(parents=True)
    (app_folder / "notes").mkdir()
    (app_folder / "notes" / "readme.txt").write_text("not a model")

    found = discover_models("")

    assert [(model.name, model.size) for model in found] == [(KNOWN, 300), ("my-own-model", 700)]
    assert found[1].file == user_folder / "my-own-model" / "b.gguf" and found[1].folder == user_folder / "my-own-model"


def test_the_same_name_in_two_folders_is_listed_once_from_the_first_folder(places):
    app_folder, user_folder = places
    first = _gguf(app_folder / "same", "one.gguf")
    _gguf(user_folder / "same", "two.gguf")

    assert [model.file for model in discover_models("")] == [first]


def test_a_custom_folder_works_both_as_a_folder_of_models_and_as_a_single_model(places, tmp_path):
    root = tmp_path / "mine"
    _gguf(root / "alpha")
    _gguf(root / "beta")
    single = tmp_path / "single-model"
    _gguf(single, "only.gguf")

    assert sorted(model.name for model in discover_models(str(root))) == ["alpha", "beta"]
    assert [model.name for model in discover_models(str(single))] == ["single-model"]


def test_the_model_file_is_looked_up_in_a_custom_folder_before_the_standard_ones(places, tmp_path):
    app_folder, _ = places
    standard = _gguf(app_folder / KNOWN, "standard.gguf")
    root = tmp_path / "mine"
    custom = _gguf(root / KNOWN, "custom.gguf")

    assert find_model_file(KNOWN, "") == standard
    assert find_model_file(KNOWN, str(root)) == custom
    assert find_model_file("not-there", str(root)) is None
    assert model_search_paths(KNOWN, str(root))[:2] == [root / KNOWN, root]  # 先找底下以名稱命名的，再把它本身當模型資料夾
    assert llama_model_available("alpha", str(root)) is False
    _gguf(root / "alpha")
    assert llama_model_available("alpha", str(root)) is True


# ----------------------------------------------------------------------------- 設定頁
@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _items(dialog):
    combo = dialog.combo_offline_model
    return [(combo.itemText(i), combo.itemData(i), combo.model().item(i).isEnabled()) for i in range(combo.count())]


def test_with_no_model_at_all_the_dialog_still_says_where_to_put_one(qapp, places, tmp_path):
    app_folder, user_folder = places
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    folders_text = dialog.lbl_model_folders.text()
    assert str(app_folder) in folders_text and str(user_folder) in folders_text
    assert "目前還沒有這個資料夾" in folders_text and ".gguf" in folders_text
    assert "沒有找到任何翻譯模型" in dialog.lbl_model_status.text()
    assert "scripts/fetch_translation_models.py" in dialog.lbl_model_status.text() or "重新執行安裝程式" in dialog.lbl_model_status.text()
    assert [name for _label, name, enabled in _items(dialog) if enabled] == ["auto"]  # 沒有模型檔的選項都停用，只剩「自動」
    assert not dialog.btn_clear_models.isVisible()
    dialog.deleteLater()


def test_models_found_in_the_folders_are_listed_after_the_known_ones(qapp, places, tmp_path):
    app_folder, user_folder = places
    _gguf(app_folder / KNOWN)
    _gguf(user_folder / "my-own-model", size=3_000_000)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    items = _items(dialog)

    names = [name for _label, name, _enabled in items]
    assert names[-1] == "my-own-model" and names.count(KNOWN) == 1  # 已知的模型不會重複出現
    assert items[-1][0].startswith("自訂：my-own-model") and items[-1][2]
    assert dict((name, enabled) for _l, name, enabled in items)[KNOWN] is True
    assert "找到 2 個模型" in dialog.lbl_model_status.text()
    dialog.deleteLater()


def test_a_model_added_later_shows_up_after_rescanning_and_can_be_chosen_and_saved(qapp, places, tmp_path):
    _, user_folder = places
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    assert "沒有找到任何翻譯模型" in dialog.lbl_model_status.text()

    _gguf(user_folder / "fresh-model")
    dialog.btn_rescan_models.click()

    assert "fresh-model" in [name for _l, name, _e in _items(dialog)]
    assert "找到 1 個模型" in dialog.lbl_model_status.text()
    dialog.combo_offline_model.setCurrentIndex(dialog.combo_offline_model.findData("fresh-model"))
    assert str(user_folder / "fresh-model" / "model.gguf") in dialog.lbl_model_status.text()  # 寫出實際讀的檔案
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["translation"]["offline"]["model_name"] == "fresh-model"
    dialog.deleteLater()


def test_a_saved_custom_model_is_selected_again_and_left_alone_when_the_menu_is_not_touched(qapp, places, tmp_path):
    _, user_folder = places
    _gguf(user_folder / "my-own-model")
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.offline.model_name", "my-own-model")
    first = SettingsDialog(config)
    assert first.combo_offline_model.currentData() == "my-own-model"
    first.deleteLater()

    (user_folder / "my-own-model" / "model.gguf").unlink()  # 檔案被刪掉了：選單顯示「自動」，但沒動選單就不能把設定改掉
    second = SettingsDialog(config)
    assert second.combo_offline_model.currentData() == "auto"
    second._commit_accept(second._validated_hotkeys())
    second._save_close_timer.stop()
    assert AppConfig(tmp_path / "config.json").get("translation.offline.model_name") == "my-own-model"
    second.deleteLater()


def test_choosing_another_folder_lists_its_models_and_is_saved_and_can_be_cleared(qapp, places, tmp_path):
    other = tmp_path / "elsewhere"
    _gguf(other / "alpha")
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    assert not dialog.btn_clear_models.isVisible() or not dialog._model_dir

    dialog._set_model_dir(str(other))

    assert "alpha" in [name for _l, name, _e in _items(dialog)]
    assert str(other) in dialog.lbl_model_folders.text() and "自訂資料夾" in dialog.lbl_model_folders.text()
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()
    assert AppConfig(tmp_path / "config.json").get("translation.offline.model_dir") == str(other)

    reopened = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert "alpha" in [name for _l, name, _e in _items(reopened)]
    reopened._set_model_dir("")
    assert "alpha" not in [name for _l, name, _e in _items(reopened)]
    reopened._commit_accept(reopened._validated_hotkeys())
    reopened._save_close_timer.stop()
    assert AppConfig(tmp_path / "config.json").get("translation.offline.model_dir") == ""
    dialog.deleteLater()
    reopened.deleteLater()
