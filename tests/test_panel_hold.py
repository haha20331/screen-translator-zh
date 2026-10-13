"""字幕翻得很快時：舊的一句譯文至少再留一下，新的一句接在下面。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.config import DEFAULTS, AppConfig
from app.models import CaptureInfo, TextRegion
from ui.overlay_manager import OverlayManager
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from utils.dpi_utils import MonitorInfo


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def setup(app, tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("overlay.placement", "below")
    manager = OverlayManager(config)
    manager.set_monitor_map([MonitorInfo(0, (0, 0, 1000, 600), (0, 0), 1.0)])
    clock = _Clock()
    manager._clock = clock
    yield manager, clock, config
    manager.clear_all()


def _capture(mode="region") -> CaptureInfo:
    return CaptureInfo(image=np.zeros((100, 400, 3), dtype=np.uint8), bbox=(100, 100, 500, 200), monitor_indices=[0], mode=mode)


def _regions(*pairs):
    return [
        TextRegion(text=source, translated_text=translated, x=120, y=110 + index * 30, width=100, height=20)
        for index, (source, translated) in enumerate(pairs)
    ]


def _show(manager, *pairs):
    return manager.show_regions(_capture(), _regions(*pairs))


def _panel(manager):
    return manager._panels[0]


def test_default_hold_is_half_a_second():
    assert DEFAULTS["overlay"]["min_hold_seconds"] == 0.5


def test_a_sentence_replaced_within_the_hold_stays_above_the_new_one(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.2

    assert _show(manager, ("二", "第二句"))

    assert _panel(manager).lines() == ["第一句", "第二句"]  # 第 N 句留著，第 N+1 句接在下面
    assert manager._hold_timer.isActive()
    assert 250 <= manager._hold_timer.interval() <= 310  # 第 N 句還差 0.3 秒才留夠


def test_the_old_sentence_goes_away_once_it_has_been_shown_long_enough(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.2
    _show(manager, ("二", "第二句"))

    clock.now += 0.31  # 第一句已經顯示 0.51 秒
    manager._expire_panel_entries()

    assert _panel(manager).lines() == ["第二句"]
    assert not manager._hold_timer.isActive()  # 目前這一句不會被計時器拿掉


def test_a_new_sentence_after_the_hold_simply_replaces_the_old_one(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.6

    _show(manager, ("二", "第二句"))

    assert _panel(manager).lines() == ["第二句"]
    assert not manager._hold_timer.isActive()


def test_each_sentence_is_kept_for_its_own_minimum_time(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.2
    _show(manager, ("二", "第二句"))
    clock.now += 0.2
    _show(manager, ("三", "第三句"))

    assert _panel(manager).lines() == ["第一句", "第二句", "第三句"]

    clock.now += 0.15  # t = 0.55：第一句已留 0.55 秒、第二句才 0.35 秒
    manager._expire_panel_entries()
    assert _panel(manager).lines() == ["第二句", "第三句"]
    assert manager._hold_timer.isActive()

    clock.now += 0.2  # t = 0.75：第二句已留 0.55 秒
    manager._expire_panel_entries()
    assert _panel(manager).lines() == ["第三句"]


def test_a_result_with_nothing_to_show_still_lets_the_last_sentence_finish_its_time(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.2

    still_shown = _show(manager, ("OK", "OK"))  # 字幕沒了（或只剩不翻的字）

    assert still_shown and _panel(manager).lines() == ["第一句"]
    clock.now += 0.31
    manager._expire_panel_entries()
    assert not manager.is_visible() and not _panel(manager).isVisible()


def test_hold_of_zero_replaces_immediately(setup):
    manager, clock, config = setup
    config.set("overlay.min_hold_seconds", 0)
    _show(manager, ("一", "第一句"))
    clock.now += 0.01

    _show(manager, ("二", "第二句"))

    assert _panel(manager).lines() == ["第二句"]
    assert not manager._hold_timer.isActive()


def test_the_same_sentence_is_not_added_twice(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.1

    _show(manager, ("一", "第一句"))

    assert _panel(manager).lines() == ["第一句"]


def test_the_stack_is_capped(setup):
    manager, clock, config = setup
    config.set("overlay.min_hold_seconds", 5)
    for index in range(7):
        _show(manager, (str(index), f"第{index}句"))
        clock.now += 0.1

    assert _panel(manager).lines() == ["第3句", "第4句", "第5句", "第6句"]


def test_the_timer_does_not_bring_back_a_panel_the_user_hid(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.2
    _show(manager, ("二", "第二句"))
    manager.toggle()  # 使用者按了「隱藏譯文」
    assert not _panel(manager).isVisible()

    clock.now += 0.4
    manager._expire_panel_entries()

    assert not _panel(manager).isVisible()
    manager.toggle()
    assert _panel(manager).lines() == ["第二句"]  # 再顯示時是更新後的內容


def test_the_stacked_panel_follows_the_frame(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.1
    _show(manager, ("二", "第二句"))
    before = _panel(manager).geometry().topLeft()

    manager.follow_region((300, 250, 700, 350))

    assert _panel(manager).lines() == ["第一句", "第二句"]
    assert _panel(manager).geometry().topLeft() != before


def test_clearing_everything_forgets_the_held_sentences(setup):
    manager, clock, _ = setup
    _show(manager, ("一", "第一句"))
    clock.now += 0.1
    _show(manager, ("二", "第二句"))

    manager.clear_all()

    assert manager._panel_entries == [] and not manager._hold_timer.isActive()
    clock.now += 0.1
    _show(manager, ("三", "第三句"))
    assert _panel(manager).lines() == ["第三句"]  # 停止監控後再開始：不會把舊句子帶回來


def test_whole_screen_captures_are_unaffected(setup):
    manager, clock, _ = setup
    capture = CaptureInfo(image=np.zeros((600, 1000, 3), dtype=np.uint8), bbox=(0, 0, 1000, 600), monitor_indices=[0], mode="fullscreen")

    manager.show_regions(capture, _regions(("一", "第一句")))

    assert manager._panel_entries == [] and len(manager._windows) == 1


# ----------------------------------------------------------------------------- 設定
def test_the_hold_is_clamped_when_loading_a_config(tmp_path):
    path = tmp_path / "config.json"
    for saved, expected in (({}, 0.5), ({"min_hold_seconds": 3}, 3.0), ({"min_hold_seconds": -1}, 0.0), ({"min_hold_seconds": 99}, 10.0), ({"min_hold_seconds": "x"}, 0.5)):
        path.write_text(json.dumps({"overlay": saved}), encoding="utf-8")
        assert AppConfig(path).get("overlay.min_hold_seconds") == expected


def test_translation_settings_let_the_user_choose_the_hold(app, tmp_path):
    apply_style(app)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.spin_min_hold.value() == 0.5
    assert dialog.pages.widget(3).widget().isAncestorOf(dialog.spin_min_hold)  # 翻譯頁

    dialog.spin_min_hold.setValue(1.2)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("overlay.min_hold_seconds") == 1.2
    dialog.deleteLater()


def test_translation_settings_expose_the_cache_size(app, tmp_path):
    apply_style(app)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    assert dialog.spin_cache_entries.value() == 2000
    assert "建議 2000" in dialog.spin_cache_entries.toolTip()

    dialog.spin_cache_entries.setValue(500)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    assert AppConfig(tmp_path / "config.json").get("translation.cache_max_entries") == 500
    dialog.deleteLater()
