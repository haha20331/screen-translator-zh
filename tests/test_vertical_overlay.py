"""譯文位置選「蓋在原文上方」、文字方向選直向：譯文也由上往下、欄由右往左排，不必改成「顯示在框下方」。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from app.models import CaptureInfo, TextRegion
from ui import overlay_manager as overlay_module
from ui.overlay_manager import OverlayManager
from ui.translation_overlay import ROTATE_WHEN_VERTICAL, Block, TranslationOverlayWindow, vertical_cells
from utils.dpi_utils import MonitorInfo


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------------- 排版（不畫圖）
def test_a_short_text_fills_one_column_from_the_top_and_hugs_the_right_edge():
    rect = QRectF(100, 20, 60, 200)

    cells, overflow = vertical_cells("今天天氣好", rect, cell=20, pitch=22)

    assert [char for char, _ in cells] == list("今天天氣好")
    assert [box.top() for _, box in cells] == [20, 40, 60, 80, 100]  # 由上往下，一個字一格
    assert all(box.right() <= rect.right() and box.right() > rect.right() - 22 for _, box in cells)  # 靠右邊的那一欄
    assert not overflow


def test_a_long_text_continues_in_the_column_to_the_left():
    rect = QRectF(100, 0, 60, 60)  # 一欄放 3 個字

    cells, overflow = vertical_cells("一二三四五六", rect, cell=20, pitch=22)

    columns = {round(box.left(), 1) for _, box in cells}
    assert len(columns) == 2 and not overflow
    first_column = [char for char, box in cells if box.left() == max(box.left() for _, box in cells)]
    assert first_column == list("一二三")  # 右邊那一欄先讀
    second_column = [char for char, box in cells if box.left() == min(box.left() for _, box in cells)]
    assert second_column == list("四五六")


def test_text_that_needs_more_columns_than_the_box_has_is_reported_as_overflow():
    _, overflow = vertical_cells("一二三四五六七八九", QRectF(100, 0, 40, 40), cell=20, pitch=22)

    assert overflow


def test_a_new_line_starts_a_new_column_and_spaces_take_no_room():
    cells, _ = vertical_cells("你 好\n世 界", QRectF(0, 0, 100, 200), cell=20, pitch=22)

    assert [char for char, _ in cells] == list("你好世界")
    right, left = cells[0][1], cells[2][1]
    assert left.left() < right.left() and left.top() == right.top()  # 換行之後是左邊一欄的最上面
    assert cells[1][1].top() - cells[0][1].top() == 20  # 空白沒有佔一格


def test_empty_text_has_no_cells():
    assert vertical_cells("", QRectF(0, 0, 50, 50), 20, 22) == ([], False)


def test_brackets_and_dashes_are_the_characters_that_get_turned():
    assert set("「」（）—ー…") <= ROTATE_WHEN_VERTICAL
    assert not set("，。！？你好A1") & ROTATE_WHEN_VERTICAL  # 逗號句號置中、其他字直立


# ----------------------------------------------------------------------------- 真的畫出來
def _ink_box(image, rect: QRectF, text_color: QColor):
    """block 範圍內文字顏色的像素範圍：(左, 上, 右, 下)。"""
    xs, ys = [], []
    for y in range(int(rect.top()), int(rect.bottom())):
        for x in range(int(rect.left()), int(rect.right())):
            color = image.pixelColor(x, y)
            if color.red() > 110 and color.green() > 110 and color.blue() > 110:  # 黑底上的白字（含邊緣的半透明像素）
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _render(tmp_path, block: Block):
    overlay = TranslationOverlayWindow(AppConfig(tmp_path / "config.json"))
    overlay.resize(360, 300)
    overlay.set_blocks_static([block])
    image = overlay.grab().toImage()
    overlay.deleteLater()
    return image


def _block(vertical: bool, text="今天的天氣真好我們一起去散步吧") -> Block:
    return Block(QRectF(250, 30, 60, 240), text, QColor("#FFFFFF"), "#000000", 40.0, 1, vertical)


def test_a_vertical_block_is_drawn_as_a_tall_narrow_stack_of_characters(qapp, tmp_path):
    block = _block(True)

    left, top, right, bottom = _ink_box(_render(tmp_path, block), block.rect, block.color)

    assert bottom - top > 2 * (right - left)  # 高遠大於寬：一欄一欄往下排
    assert top < block.rect.top() + 40  # 從框的上面開始
    assert right > block.rect.right() - 40  # 靠右邊那一欄開始


def test_the_same_text_in_a_horizontal_block_is_not_stacked(qapp, tmp_path):
    block = _block(False)

    box = _ink_box(_render(tmp_path, block), block.rect, block.color)

    assert box is not None
    left, top, right, bottom = box
    assert (bottom - top) < 2.5 * (right - left)  # 橫排：沒有像直排那樣疊成細長的一條


def test_a_long_vertical_translation_shrinks_to_fit_before_it_spills_out_to_the_left(qapp, tmp_path):
    long_text = "今天的天氣真好我們一起去散步吧" * 3
    block = Block(QRectF(250, 30, 60, 240), long_text, QColor("#FFFFFF"), "#000000", 40.0, 1, True)
    overlay = TranslationOverlayWindow(AppConfig(tmp_path / "config.json"))
    overlay.resize(360, 300)

    font = overlay._fit_vertical_font(long_text, block.rect, size_height=40.0)
    cell, pitch = overlay._vertical_metrics(font)
    short_font = overlay._fit_vertical_font("好", block.rect, size_height=40.0)

    assert cell < overlay._vertical_metrics(short_font)[0]  # 字多就縮小
    _, overflow = vertical_cells(long_text, block.rect, cell, pitch)
    assert not overflow or font.pointSize() <= int(overlay._config.get("overlay.min_font_size", 8))
    overlay.deleteLater()


def test_punctuation_and_brackets_do_not_break_the_vertical_drawing(qapp, tmp_path):
    block = _block(True, "「你好，」他說。（真的！）Hello")

    assert _ink_box(_render(tmp_path, block), block.rect, block.color) is not None


# ----------------------------------------------------------------------------- 區域 → 方塊
class _Signal:
    def connect(self, _callback) -> None:
        pass


class _Window:
    def __init__(self, _config) -> None:
        self.close_requested = _Signal()
        self.blocks = []

    def setGeometry(self, *_geometry) -> None:
        pass

    def set_blocks(self, blocks, animate_new_only: bool = False) -> None:
        self.blocks = blocks

    def set_edit_mode(self, _enabled: bool) -> None:
        pass

    def show_fade(self) -> None:
        pass

    def raise_(self) -> None:
        pass

    def hide(self) -> None:
        pass


def _blocks_for(tmp_path, monkeypatch, regions, placement="in_place"):
    monkeypatch.setattr(overlay_module, "TranslationOverlayWindow", _Window)
    config = AppConfig(tmp_path / "config.json")
    config.set("overlay.placement", placement)
    manager = OverlayManager(config)
    manager.set_monitor_map([MonitorInfo(0, (0, 0, 400, 300), (0, 0), 1.0)])
    capture = CaptureInfo(image=np.zeros((300, 400, 3), dtype=np.uint8), bbox=(0, 0, 400, 300), monitor_indices=[0], mode="fullscreen")
    assert manager.show_regions(capture, regions)
    return manager._windows[0].blocks


def _region(vertical, width, height):
    return TextRegion(text="夜空に浮かぶ", translated_text="漂浮在夜空中", x=300, y=20, width=width, height=height, vertical=vertical)


def test_a_tall_vertical_region_becomes_a_vertical_block(tmp_path, monkeypatch):
    (block,) = _blocks_for(tmp_path, monkeypatch, [_region(True, 50, 200)])

    assert block.vertical is True


def test_a_horizontal_region_stays_a_horizontal_block(tmp_path, monkeypatch):
    (block,) = _blocks_for(tmp_path, monkeypatch, [_region(False, 50, 200)])

    assert block.vertical is False


def test_a_wide_region_in_vertical_mode_is_still_drawn_horizontally(tmp_path, monkeypatch):
    """直向模式下偵測到的橫條（浮水印、標題）不能硬排成直的。"""
    (block,) = _blocks_for(tmp_path, monkeypatch, [_region(True, 200, 40)])

    assert block.vertical is False
