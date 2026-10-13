"""框選區域時，譯文顯示在框的下方，原文保持可見（不被蓋住）。"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QRect, QSize
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCRLine
from ui.overlay_manager import OverlayManager
from ui.translation_panel import place_panel
from utils.dpi_utils import MonitorInfo
from utils.layout_utils import translation_lines, uses_translation_panel
from workers.translation_worker import PipelineTask

SCREEN = QRect(0, 0, 1000, 600)


# ------------------------------------------------------------------------- place_panel
def test_panel_goes_below_the_frame_and_is_centered_on_it():
    anchor = QRect(200, 100, 400, 80)
    rect = place_panel(anchor, QSize(200, 40), SCREEN, gap=6)
    assert rect.top() == anchor.bottom() + 1 + 6
    assert abs(rect.center().x() - anchor.center().x()) <= 1  # QRect.center() 取整數
    assert not rect.intersects(anchor)


def test_panel_moves_above_the_frame_when_there_is_no_room_below():
    anchor = QRect(200, 500, 400, 80)  # 下緣 580，只剩 20px
    rect = place_panel(anchor, QSize(200, 60), SCREEN, gap=6)
    assert rect.bottom() < anchor.top()
    assert not rect.intersects(anchor)


def test_panel_falls_back_inside_the_frame_when_it_fills_the_screen():
    rect = place_panel(SCREEN, QSize(300, 50), SCREEN, gap=6)
    assert SCREEN.contains(rect)
    assert rect.bottom() == SCREEN.bottom() - 6


def test_panel_stays_inside_the_screen_horizontally():
    left = place_panel(QRect(0, 100, 40, 40), QSize(300, 30), SCREEN)
    right = place_panel(QRect(960, 100, 40, 40), QSize(300, 30), SCREEN)
    assert left.left() >= SCREEN.left()
    assert right.right() <= SCREEN.right()


# ------------------------------------------------------------------------- translation_lines
def _region(text, translated, x, y, width=100, height=20):
    return TextRegion(text=text, translated_text=translated, x=x, y=y, width=width, height=height)


def test_lines_follow_reading_order_and_join_a_row():
    regions = [
        _region("c", "第三", 10, 100),
        _region("b", "第二右", 150, 50),
        _region("a", "第一", 10, 10),
        _region("b2", "第二左", 10, 52),
    ]
    assert translation_lines(regions) == ["第一", "第二左  第二右", "第三"]


def test_lines_skip_empty_and_untranslated_blocks():
    regions = [
        _region("OK", "OK", 0, 0),  # 譯文與原文相同：翻譯失敗或不需翻譯
        _region("x", "   ", 0, 30),
        _region("Hello", "你好", 0, 60),
    ]
    assert translation_lines(regions) == ["你好"]


def test_lines_are_empty_when_nothing_was_translated():
    assert translation_lines([_region("OK", "OK", 0, 0)]) == []


@pytest.mark.parametrize(
    ("placement", "mode", "expected"),
    [
        ("below", "region", True),
        (None, "region", True),
        ("below", "fullscreen", False),  # 全螢幕、視窗擷取沒有「框」可以放下方
        ("below", "window", False),
        ("in_place", "region", False),
    ],
)
def test_panel_applies_only_to_region_captures(placement, mode, expected):
    assert uses_translation_panel(placement, mode) is expected


# ------------------------------------------------------------------------- OverlayManager
@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _manager(tmp_path, placement="below") -> OverlayManager:
    config = AppConfig(tmp_path / "config.json")
    config.set("overlay.placement", placement)
    config.set("overlay.min_hold_seconds", 0)  # 這個檔案測的是位置與跟隨；保留秒數另有測試
    manager = OverlayManager(config)
    manager.set_monitor_map([MonitorInfo(0, (0, 0, 1000, 600), (0, 0), 1.0)])
    return manager


def _capture(mode="region", bbox=(100, 100, 500, 200)) -> CaptureInfo:
    return CaptureInfo(
        image=np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), dtype=np.uint8),
        bbox=bbox,
        monitor_indices=[0],
        mode=mode,
    )


def _regions():
    return [_region("こんにちは", "你好", 120, 120), _region("さようなら", "再見", 120, 150)]


def test_region_capture_shows_one_panel_below_the_frame_and_no_cover(app, tmp_path):
    manager = _manager(tmp_path)

    assert manager.show_regions(_capture(), _regions())

    assert manager._windows == {}  # 沒有任何蓋在原文上的視窗
    panel = manager._panels[0]
    assert panel.isVisible()
    assert panel.lines() == ["你好", "再見"]
    assert panel.geometry().top() > 200  # 框的下緣是 y=200
    assert not panel.geometry().intersects(QRect(100, 100, 400, 100))
    manager.clear_all()


def test_fullscreen_capture_still_covers_the_source_text(app, tmp_path):
    manager = _manager(tmp_path)
    capture = _capture("fullscreen", (0, 0, 1000, 600))

    assert manager.show_regions(capture, _regions())

    assert manager._panels == {}
    assert len(manager._windows) == 1
    manager.clear_all()


def test_in_place_setting_covers_the_source_text_even_for_regions(app, tmp_path):
    manager = _manager(tmp_path, placement="in_place")

    assert manager.show_regions(_capture(), _regions())

    assert manager._panels == {}
    assert len(manager._windows) == 1
    manager.clear_all()


def test_panel_follows_the_frame_when_it_is_dragged(app, tmp_path):
    manager = _manager(tmp_path)
    manager.show_regions(_capture(), _regions())
    panel = manager._panels[0]
    before = panel.geometry().topLeft()

    manager.follow_region((300, 250, 700, 350))

    after = panel.geometry().topLeft()
    assert after.x() > before.x() and after.y() > before.y()
    assert panel.geometry().top() > 350
    manager.clear_all()


def test_hide_and_show_toggle_the_panel(app, tmp_path):
    manager = _manager(tmp_path)
    manager.show_regions(_capture(), _regions())
    panel = manager._panels[0]

    assert manager.toggle() is False
    assert not panel.isVisible()
    assert manager.toggle() is True
    assert panel.isVisible()
    manager.clear_all()


def test_clear_all_removes_the_panel_and_does_not_come_back_on_drag(app, tmp_path):
    manager = _manager(tmp_path)
    manager.show_regions(_capture(), _regions())

    manager.clear_all()
    manager.follow_region((300, 250, 700, 350))

    assert not manager._panels[0].isVisible()
    assert not manager.is_visible()


def test_untranslated_result_hides_a_previous_panel(app, tmp_path):
    manager = _manager(tmp_path)
    manager.show_regions(_capture(), _regions())

    assert not manager.show_regions(_capture(), [_region("OK", "OK", 120, 120)])

    assert not manager._panels[0].isVisible()


# ------------------------------------------------------------------------- worker: 不再把原文併進譯文
class _OCR:
    def recognize(self, _image, lang="auto"):
        return [OCRLine("Settings", (5, 5, 70, 16), 1.0)]


class _Translator:
    last_failed_count = 0
    unsupported_direction = False

    def translate(self, texts, source, target):
        return ["設定" for _ in texts]


def _run_worker(tmp_path, mode, placement):
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.keep_original", True)
    config.set("overlay.placement", placement)
    capture = CaptureInfo(
        image=np.full((40, 120, 3), 255, dtype=np.uint8), bbox=(0, 0, 120, 40), monitor_indices=[0], mode=mode
    )
    task = PipelineTask(capture, _OCR(), _Translator(), config)
    results = []
    task.result.connect(results.append)
    task.run()
    return results[0]["regions"][0].translated_text


def test_keep_original_is_not_prepended_when_the_panel_is_used(tmp_path):
    assert _run_worker(tmp_path, "region", "below") == "設定"


def test_keep_original_still_works_when_covering_the_source(tmp_path):
    assert _run_worker(tmp_path, "region", "in_place") == "Settings\n設定"
    assert _run_worker(tmp_path, "fullscreen", "below") == "Settings\n設定"
