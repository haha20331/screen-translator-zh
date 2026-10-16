"""主視窗精簡之後：只剩框選翻譯、即時監控與設定；托盤的兩個選項；歡迎文字與模型介紹要符合現在的程式。"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from app.config import AppConfig
from services.translation.offline_translator import OFFLINE_MODELS
from ui import main_window as main_window_module
from ui.main_window import MainWindow
from ui.settings_dialog import OFFLINE_MODEL_CHOICES, SettingsDialog
from ui.style import apply_style
from ui.welcome_dialog import WelcomeDialog


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


class _Controller:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _wait(qapp, seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


# ----------------------------------------------------------------------------- 主視窗的內容
def test_the_main_window_only_keeps_region_capture_monitor_and_settings(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))

    for name in ("btn_fullscreen", "btn_window", "btn_edit", "btn_refresh", "combo_source", "combo_target", "combo_ocr"):
        assert not hasattr(window, name), name
    for name in ("btn_region", "btn_monitor", "btn_settings", "check_lyrics"):
        assert hasattr(window, name), name
    window.deleteLater()


def test_the_intro_texts_describe_what_the_program_does_now(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))
    texts = [label.text() for label in window.findChildren(QLabel)]

    hint = next(text for text in texts if "框選" in text and "離線" in text)
    assert "全屏" not in hint and "視窗" not in hint  # 全屏、視窗翻譯已經不在主視窗上
    assert "快捷鍵" in window.btn_region._detail  # 卡片上直接寫出快捷鍵，不必去翻設定
    assert "框選翻譯" in window.status_label.text()
    window.deleteLater()


def test_the_shortcut_on_the_card_follows_the_settings(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("hotkeys.capture_region", "ctrl+alt+q")
    window = MainWindow(_Controller(config))

    assert "Ctrl+Alt+Q" in window.btn_region._detail
    config.set("hotkeys.capture_region", "ctrl+alt+w")
    window.reload_values()
    assert "Ctrl+Alt+W" in window.btn_region._detail
    window.deleteLater()


def test_a_fullscreen_or_window_capture_from_a_hotkey_still_gets_through_the_main_window(qapp, tmp_path):
    window = MainWindow(_Controller(AppConfig(tmp_path / "config.json")))
    called = []

    for mode in ("fullscreen", "window", "region"):
        window.play_capture_departure(mode, lambda mode=mode: called.append(mode))
        window.restore_after_capture(mode)
        window.play_capture_failure(mode)
        window.finish_capture(mode)

    assert called == ["fullscreen", "window", "region"]  # 沒有對應的卡片時照樣放行，只是沒有卡片動畫
    window.deleteLater()


# ----------------------------------------------------------------------------- 托盤的兩個選項
def test_the_tray_options_default_to_close_to_tray_on_and_minimize_to_tray_off(tmp_path):
    config = AppConfig(tmp_path / "config.json")

    assert config.get("general.minimize_to_tray") is True
    assert config.get("general.minimize_hides_to_tray") is False


def test_closing_the_main_window_goes_to_the_tray_by_default(qapp, tmp_path):
    controller = _Controller(AppConfig(tmp_path / "config.json"))
    window = MainWindow(controller)
    window.show()

    window.close()

    assert not window.isVisible() and controller.shutdowns == 0  # 只是收起來，程式繼續跑
    window.deleteLater()


def test_with_the_tray_option_off_closing_the_main_window_ends_the_program(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("general.minimize_to_tray", False)
    controller = _Controller(config)
    window = MainWindow(controller)
    window.show()

    window.close()

    assert controller.shutdowns == 1  # main.py 設了「關掉最後一個視窗不結束」，所以要自己結束
    window.deleteLater()


def _window_with_tray(qapp, tmp_path, monkeypatch, *, hides: bool, tray: bool = True):
    config = AppConfig(tmp_path / "config.json")
    config.set("general.minimize_hides_to_tray", hides)
    monkeypatch.setattr(main_window_module.QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: tray))
    window = MainWindow(_Controller(config))
    window.show()
    _wait(qapp)
    return window


def test_minimizing_goes_to_the_taskbar_by_default(qapp, tmp_path, monkeypatch):
    window = _window_with_tray(qapp, tmp_path, monkeypatch, hides=False)

    window.showMinimized()
    _wait(qapp)

    assert window.isMinimized() and window.isVisible()  # 和一般視窗一樣縮到工作列
    window.deleteLater()


def test_minimizing_hides_the_window_when_the_option_is_on_and_the_tray_click_brings_it_back(qapp, tmp_path, monkeypatch):
    window = _window_with_tray(qapp, tmp_path, monkeypatch, hides=True)

    window.showMinimized()
    _wait(qapp)
    assert not window.isVisible()  # 收到系統匣，不留在工作列

    window.showNormal()  # 系統匣圖示的點擊做的事
    _wait(qapp)
    assert window.isVisible() and not window.isMinimized()
    window.deleteLater()


def test_the_window_is_never_hidden_when_there_is_no_tray_to_bring_it_back_from(qapp, tmp_path, monkeypatch):
    window = _window_with_tray(qapp, tmp_path, monkeypatch, hides=True, tray=False)

    window.showMinimized()
    _wait(qapp)

    assert window.isMinimized() and window.isVisible()
    window.deleteLater()


def test_the_settings_page_saves_both_tray_options(qapp, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    assert dialog.chk_tray.isChecked() and not dialog.chk_minimize_tray.isChecked()

    dialog.chk_tray.setChecked(False)
    dialog.chk_minimize_tray.setChecked(True)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("general.minimize_to_tray") is False and saved.get("general.minimize_hides_to_tray") is True
    dialog.deleteLater()


# ----------------------------------------------------------------------------- 介紹文字
def test_the_welcome_dialog_matches_the_program_as_it_is_now(qapp):
    hotkeys = {"capture_region": "ctrl+shift+a", "monitor_toggle": "ctrl+shift+m"}
    dialog = WelcomeDialog(None, hotkeys)
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))

    for stale in ("API Key", "內建詞典", "線上翻譯", "覆蓋在螢幕上原文", "主介面頂部"):
        assert stale not in text, stale  # 這些是上游的舊說明，現在的程式沒有這些東西
    for current in ("框選翻譯", "即時監控", "系統匣", "設定", "繁體中文", "Ctrl+Shift+M"):
        assert current in text, current
    dialog.deleteLater()


def test_model_introductions_are_plain_and_do_not_talk_about_licences(qapp, tmp_path):
    import json
    from pathlib import Path

    catalog = json.loads((Path(__file__).resolve().parent.parent / "installer" / "models.json").read_text(encoding="utf-8"))
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    texts = [model.label for model in OFFLINE_MODELS]
    texts += [label for label, _name in OFFLINE_MODEL_CHOICES]
    texts += [model["label"] for model in catalog["models"]]
    texts += [dialog.combo_offline_model.toolTip(), dialog.lbl_model_status.text()]
    texts += [dialog.combo_offline_model.itemData(i, 3) or "" for i in range(dialog.combo_offline_model.count())]  # 3 = ToolTipRole

    for text in texts:
        for word in ("授權", "Apache", "License", "非商用", "散布", "Q4_K_M", "Q8_0", "GGUF 檔以外"):
            assert word not in text, (word, text)
    assert all(model.label and "GB" in model.label for model in OFFLINE_MODELS)  # 每個都寫了檔案大小
    assert any("推薦" in model.label for model in OFFLINE_MODELS) and any("推薦" in model["label"] for model in catalog["models"])
    dialog.deleteLater()
