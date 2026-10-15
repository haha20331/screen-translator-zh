"""翻譯的目標語言只有繁體中文：介面不能選別的，設定檔裡的其他值會被改回來。"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import DEFAULTS, AppConfig
from ui.main_window import MainWindow
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.language_utils import TARGET_LANGUAGE, TARGET_LANGUAGES


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


def test_the_only_target_language_is_traditional_chinese():
    assert TARGET_LANGUAGES == [("zh-hant", "繁體中文")] and TARGET_LANGUAGE == "zh-hant"
    assert DEFAULTS["translation"]["target_language"] == "zh-hant"


@pytest.mark.parametrize("stored", ["zh", "en", "ja", "ko", "", None, 5])
def test_any_other_target_in_an_old_config_file_is_changed_back_on_load(tmp_path, stored):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"translation": {"target_language": stored}}), encoding="utf-8")

    assert AppConfig(path).get("translation.target_language") == "zh-hant"


def test_setting_another_target_does_not_stick_and_is_never_saved(tmp_path):
    config = AppConfig(tmp_path / "config.json")

    config.set("translation.target_language", "en")
    config.save()

    assert config.get("translation.target_language") == "zh-hant"
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["translation"]["target_language"] == "zh-hant"


def test_the_main_window_shows_traditional_chinese_as_the_fixed_target(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    window = MainWindow(_Controller(config))

    items = [(window.combo_target.itemText(i), window.combo_target.itemData(i)) for i in range(window.combo_target.count())]

    assert items == [("繁體中文", "zh-hant")]
    assert window.combo_target.currentData() == "zh-hant" and not window.combo_target.isEnabled()
    assert window.combo_source.count() > 1  # 輸入語言照樣可以選
    window.deleteLater()


def test_the_settings_dialog_shows_the_fixed_target_and_saves_only_that(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)

    assert [dialog.combo_target.itemData(i) for i in range(dialog.combo_target.count())] == ["zh-hant"]
    assert not dialog.combo_target.isEnabled()
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("translation.target_language") == "zh-hant"
    dialog.deleteLater()


def test_picking_a_target_at_runtime_is_ignored(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    controller = Application(app)
    monkeypatch.setattr(controller, "_create_configured_ocr_engine", lambda: None)
    monkeypatch.setattr(controller, "_rebuild_translator", lambda: 0)
    monkeypatch.setattr(controller, "_warmup", lambda: None)
    controller.set_status = lambda text: None

    controller.apply_runtime_selection(target="en")

    assert controller.config.get("translation.target_language") == "zh-hant"
    controller.floating_status.deleteLater()
