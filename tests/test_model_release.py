"""停止即時監控超過一段時間（預設 10 秒），就把翻譯模型從記憶體放掉；用到時再自動載回。"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.config import AppConfig
from app.model_release_mixin import ModelReleaseMixin


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(qapp, seconds):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)


class _Translator:
    """有 close／warmup／is_ready 的翻譯器（像離線翻譯器）。"""

    def __init__(self, loaded=True):
        self.loaded = loaded
        self.closed = 0
        self.warmups = 0

    def close(self):
        self.loaded = False
        self.closed += 1

    def warmup(self):
        self.loaded = True
        self.warmups += 1

    def is_ready(self):
        return self.loaded

    def reset_context(self):
        pass


class _Host(QObject, ModelReleaseMixin):
    """只裝這個 mixin 需要的東西，不啟動整個應用程式。"""

    def __init__(self, tmp_path, seconds, translator):
        super().__init__()
        self.config = AppConfig(tmp_path / "config.json")
        self.config.set("monitor.release_model_after_seconds", seconds)
        self.translator = translator
        self._shutting_down = False
        self._monitor_active = False
        self._busy = False
        self._capture_session = None
        self.worker = None
        self.warming = False
        self.statuses = []
        self.warmup_calls = 0
        self._init_model_release()

    def _translator_warming(self):
        return self.warming

    def _warmup(self):
        self.warmup_calls += 1
        return "正在載入翻譯模型"

    def set_status(self, text):
        self.statuses.append(text)


def test_the_model_is_released_after_the_wait_and_the_user_is_told(qapp, tmp_path):
    translator = _Translator()
    host = _Host(tmp_path, 0.05, translator)

    host._arm_model_release()
    assert translator.closed == 0  # 還沒到時間
    _wait(qapp, 0.25)

    assert translator.closed == 1 and not translator.loaded
    assert any("釋放" in text for text in host.statuses)
    _wait(qapp, 0.15)
    assert translator.closed == 1  # 只放一次


def test_starting_the_monitor_again_before_the_wait_is_over_keeps_the_model(qapp, tmp_path):
    translator = _Translator()
    host = _Host(tmp_path, 0.1, translator)

    host._arm_model_release()
    host._cancel_model_release()  # 重新開始監控
    _wait(qapp, 0.3)

    assert translator.closed == 0


def test_a_capture_in_progress_blocks_the_release_until_it_is_over(qapp, tmp_path):
    translator = _Translator()
    host = _Host(tmp_path, 0.05, translator)
    host._busy = True  # 使用者正在框選或翻譯

    host._arm_model_release()
    _wait(qapp, 0.3)
    assert translator.closed == 0  # 忙的時候不放，而且會一直重新檢查

    host._busy = False
    _wait(qapp, 0.2)
    assert translator.closed == 1


def test_a_model_that_is_still_loading_is_not_released_under_its_feet(qapp, tmp_path):
    translator = _Translator()
    host = _Host(tmp_path, 0.05, translator)
    host.warming = True

    host._arm_model_release()
    _wait(qapp, 0.25)
    assert translator.closed == 0

    host.warming = False
    _wait(qapp, 0.2)
    assert translator.closed == 1


def test_the_release_can_be_switched_off_with_zero_seconds(qapp, tmp_path):
    translator = _Translator()
    host = _Host(tmp_path, 0, translator)

    host._arm_model_release()
    _wait(qapp, 0.2)

    assert translator.closed == 0


def test_a_translator_without_a_model_in_memory_is_left_alone(qapp, tmp_path):
    host = _Host(tmp_path, 0.05, SimpleNamespace(name="mock"))  # 沒有 close 的翻譯器（例如測試用的假翻譯器）

    host._arm_model_release()
    _wait(qapp, 0.15)

    assert host.statuses == []


def test_the_program_shutting_down_never_releases(qapp, tmp_path):
    translator = _Translator()
    host = _Host(tmp_path, 0.05, translator)
    host._arm_model_release()
    host._shutting_down = True

    _wait(qapp, 0.2)

    assert translator.closed == 0


def test_a_released_model_is_loaded_again_in_the_background_when_it_is_needed(qapp, tmp_path):
    translator = _Translator(loaded=False)  # 剛被釋放
    host = _Host(tmp_path, 10, translator)

    note = host._ensure_model_loaded()

    assert note == "正在載入翻譯模型" and host.warmup_calls == 1


def test_a_model_that_is_already_loaded_or_loading_is_not_loaded_twice(qapp, tmp_path):
    host = _Host(tmp_path, 10, _Translator(loaded=True))
    assert host._ensure_model_loaded() == "" and host.warmup_calls == 0

    host = _Host(tmp_path, 10, _Translator(loaded=False))
    host.warming = True
    assert host._ensure_model_loaded() == "" and host.warmup_calls == 0

    host = _Host(tmp_path, 10, SimpleNamespace(name="mock"))
    assert host._ensure_model_loaded() == "" and host.warmup_calls == 0


# ----------------------------------------------------------------------------- 接在真正的應用程式上
@pytest.fixture
def controller(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    instance = Application(qapp)
    instance.config.set("monitor.release_model_after_seconds", 0.05)
    instance.translator = _Translator()
    monkeypatch.setattr(instance, "_warmup", lambda: "")
    monkeypatch.setattr(instance, "_monitor_cycle", lambda: None)
    yield instance
    instance.floating_status.deleteLater()


def test_stopping_the_real_monitor_starts_the_countdown_and_the_model_goes(controller, qapp):
    controller._monitor_active = True

    controller.stop_monitor()
    _wait(qapp, 0.3)

    assert controller.translator.closed == 1


def test_starting_the_real_monitor_again_cancels_the_release_and_reloads_a_released_model(controller, qapp, monkeypatch):
    controller._last_capture = SimpleNamespace(mode="fullscreen", bbox=(0, 0, 10, 10))
    controller._monitor_active = True
    controller.stop_monitor()
    controller.translator.loaded = False  # 假裝剛好在這之前被放掉
    loads = []
    monkeypatch.setattr(controller, "_warmup", lambda: loads.append(1) or "正在載入翻譯模型")

    controller.start_monitor()
    _wait(qapp, 0.3)

    assert controller._monitor_active and controller.translator.closed == 0  # 倒數取消了
    assert loads == [1]  # 載回模型是在背景做的，監控這一輪先不送翻譯


def test_a_cancelled_capture_after_the_monitor_stopped_still_ends_with_the_release(controller, qapp):
    controller._monitor_active = True
    controller.stop_monitor()
    controller._set_busy(True)  # 使用者開始新的擷取
    _wait(qapp, 0.2)
    assert controller.translator.closed == 0  # 擷取中不放

    controller._set_busy(False)  # 擷取取消了，沒有接著進入監控
    _wait(qapp, 0.3)

    assert controller.translator.closed == 1


def test_beginning_a_capture_brings_a_released_model_back_while_the_user_is_framing(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "_ensure_model_loaded", lambda: calls.append(1) or "正在載入翻譯模型")
    controller.window = SimpleNamespace(
        settle_settings_transition_for_capture=lambda: None,
        isVisible=lambda: True,
        isMinimized=lambda: False,
        set_busy=lambda busy: None,
        play_capture_departure=lambda mode, callback: None,
    )
    controller.overlay_manager = SimpleNamespace(is_visible=lambda: False, hide_all=lambda: None)
    monkeypatch.setattr(controller, "refresh_monitor_map", lambda: None)
    statuses = []
    monkeypatch.setattr(controller, "set_status", statuses.append)

    assert controller._begin_capture_session("region", lambda: None) is True

    assert calls == [1] and "正在載入翻譯模型" in statuses


def test_the_setting_is_clamped_and_defaults_to_ten_seconds(tmp_path):
    config = AppConfig(tmp_path / "config.json")
    assert config.get("monitor.release_model_after_seconds") == 10
    for stored, expected in ((-5, 0), (99999, 3600), ("abc", 10), (30, 30)):
        path = tmp_path / f"c{expected}{stored}.json"
        path.write_text('{"monitor": {"release_model_after_seconds": %s}}' % (f'"{stored}"' if isinstance(stored, str) else stored), encoding="utf-8")
        assert AppConfig(path).get("monitor.release_model_after_seconds") == expected


def test_the_settings_page_shows_and_saves_the_release_time(qapp, tmp_path):
    import json

    from ui.settings_dialog import SettingsDialog

    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    assert dialog.spin_release_model.value() == 10
    dialog.spin_release_model.setValue(0)
    assert dialog.spin_release_model.text() == "不釋放"
    dialog.spin_release_model.setValue(45)

    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["monitor"]["release_model_after_seconds"] == 45
    dialog.deleteLater()
