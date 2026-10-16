"""程式只有離線翻譯：沒有線上翻譯服務，也不儲存任何服務的憑據；舊設定檔裡的會在載入時刪掉。"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLineEdit

import app.application  # noqa: F401  註冊翻譯器
from app.application import Application
from app.config import DEFAULTS, AppConfig
from services.translation.base import list_translators
from services.translation.factory import create_translator
from ui.main_window import MainWindow
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style

ROOT = Path(__file__).resolve().parents[1]
REMOVED_MODULES = (
    "azure_translator",
    "baidu_translator",
    "deepl_translator",
    "google_free_translator",
    "google_translator",
    "mymemory_translator",
    "openai_translator",
    "youdao_translator",
    "http_errors",
)


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


# ----------------------------------------------------------------------------- 程式
def test_only_the_offline_translator_and_the_test_double_are_registered():
    assert sorted(list_translators()) == ["mock", "offline"]


def test_the_online_translator_modules_are_gone():
    for name in REMOVED_MODULES:
        assert importlib.util.find_spec(f"services.translation.{name}") is None, name
    with pytest.raises(ValueError, match="未知翻譯服務"):
        create_translator("openai", {})


def test_no_program_code_talks_to_the_network_through_requests():
    offenders = []
    for folder in ("app", "services", "ui", "utils", "workers"):
        for path in (ROOT / folder).rglob("*.py"):
            if re.search(r"^\s*(import requests|from requests)", path.read_text(encoding="utf-8"), re.M):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_the_config_has_no_place_for_a_credential():
    names: list[str] = []

    def walk(node, path=""):
        for key, value in node.items():
            names.append(f"{path}{key}")
            if isinstance(value, dict):
                walk(value, f"{path}{key}.")

    walk(DEFAULTS)
    shaped_like_credentials = [n for n in names if re.search(r"api_?key|secret|(?<![a-z])token$|access_token|password|appid|app_key|credential", n, re.I)]
    assert shaped_like_credentials == []
    assert not any(hasattr(AppConfig, name) for name in ("api_key",))
    assert not hasattr(Application, "_resolve_translator_service")  # 以前會因為環境變數裡有 API Key 自動換成線上服務


# ----------------------------------------------------------------------------- 舊設定檔
def _old_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "translation": {
                    "service": "openai",
                    "auto_select_service": False,
                    "openai": {"api_key": "sk-old-secret-1234567890", "base_url": "https://api.openai.com/v1"},
                    "deepl": {"api_key": "deepl-old-secret"},
                    "google": {"api_key": "google-old-secret"},
                    "baidu": {"appid": "12345", "api_key": "baidu-old-secret"},
                    "youdao": {"app_key": "k", "app_secret": "youdao-old-secret"},
                    "azure": {"api_key": "azure-old-secret", "region": "eastasia"},
                    "google_free_max_workers": 8,
                    "use_context": False,
                    "offline": {"threads": 3},
                },
                "ocr": {"min_confidence": 0.77},
            }
        ),
        encoding="utf-8",
    )


def test_old_online_settings_and_credentials_are_removed_on_load_and_the_file_is_cleaned(tmp_path):
    path = tmp_path / "config.json"
    _old_config(path)

    config = AppConfig(path)

    assert config.get("translation.service") == "offline"
    for key in ("openai", "deepl", "google", "baidu", "youdao", "azure", "auto_select_service", "google_free_max_workers"):
        assert config.get(f"translation.{key}") is None, key
    text = path.read_text(encoding="utf-8")
    assert not any(secret in text for secret in ("old-secret", "sk-old", "eastasia", "12345"))  # 憑據不再留在磁碟上
    saved = json.loads(text)["translation"]
    assert saved["service"] == "offline" and "openai" not in saved
    # 使用者其他的設定照舊
    assert config.get("translation.use_context") is False and config.get("translation.offline.threads") == 3
    assert config.get("ocr.min_confidence") == 0.77


def test_a_clean_config_is_not_rewritten_on_every_start(tmp_path):
    path = tmp_path / "config.json"
    _old_config(path)
    AppConfig(path)
    before = path.stat().st_mtime_ns

    AppConfig(path)

    assert path.stat().st_mtime_ns == before


def test_the_translation_service_cannot_be_set_to_anything_but_offline(tmp_path):
    config = AppConfig(tmp_path / "config.json")

    config.set("translation.service", "google_free")

    assert config.get("translation.service") == "offline"


def test_an_api_key_in_the_environment_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")
    monkeypatch.setenv("DEEPL_API_KEY", "deepl-from-the-environment")

    config = AppConfig(tmp_path / "config.json")

    assert config.get("translation.service") == "offline"
    assert "sk-from-the-environment" not in json.dumps(config.masked_snapshot())


# ----------------------------------------------------------------------------- 介面
def test_the_settings_dialog_has_no_service_choice_and_no_credential_fields(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    assert not hasattr(dialog, "combo_service")
    assert [edit for edit in dialog.findChildren(QLineEdit) if edit.echoMode() == QLineEdit.EchoMode.Password] == []
    assert not any(hasattr(dialog, name) for name in ("edit_openai_key", "edit_deepl_key", "edit_google_key", "edit_baidu_key", "edit_azure_key"))
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["translation"]
    assert saved["service"] == "offline" and not {"openai", "deepl", "google", "baidu", "youdao", "azure"} & set(saved)
    dialog.deleteLater()


def test_the_main_window_has_no_service_dropdown_and_the_locked_target_stays_locked(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    assert not hasattr(window, "combo_service")
    window.set_busy(True)
    window.set_busy(False)
    assert not window.combo_target.isEnabled()  # 忙完不能把「固定繁體中文」的選單重新打開
    assert window.combo_source.isEnabled() and window.combo_ocr.isEnabled()
    window.deleteLater()
