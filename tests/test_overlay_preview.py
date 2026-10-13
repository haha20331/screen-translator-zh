"""譯文外觀範例：設定畫面隨調整即時顯示實際會看到的譯文。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QColorDialog

from app.config import AppConfig
from ui.overlay_preview import SAMPLE_TRANSLATION, OverlayOptions, OverlayPreview
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _count(preview: OverlayPreview, color: str, tolerance: int = 3) -> int:
    """範例裡有多少像素是這個顏色（用來確認設定真的畫出來了）。"""
    image = preview.grab().toImage()
    target = QColor(color)
    matches = 0
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if (
                abs(pixel.red() - target.red()) <= tolerance
                and abs(pixel.green() - target.green()) <= tolerance
                and abs(pixel.blue() - target.blue()) <= tolerance
            ):
                matches += 1
    return matches


def _options(**overrides) -> dict:
    base = {
        "overlay.placement": "below",
        "overlay.font_size": 18,
        "overlay.background_color": "#7A1F1F",
        "overlay.text_color": "#FFE9A8",
        "overlay.panel_alpha": 255,
        "overlay.border_radius": 4,
        "overlay.use_auto_text_color": True,
        "overlay.auto_background": True,
        "overlay.background_alpha": 255,
        "overlay.padding": 4,
        "overlay.show_border": False,
        "overlay.min_font_size": 8,
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------------------- 覆寫層
def test_options_use_the_overrides_and_fall_back_to_the_saved_config(tmp_path):
    config = AppConfig(tmp_path / "config.json")
    config.set("overlay.font_size", 30)

    options = OverlayOptions(config, {"overlay.text_color": "#112233"})

    assert options.get("overlay.text_color") == "#112233"  # 預覽中尚未儲存的值
    assert options.get("overlay.font_size") == 30  # 沒覆寫：沿用實際設定
    assert options.get("overlay.no_such_key", "x") == "x"
    assert OverlayOptions(None, {}).get("overlay.font_size", 7) == 7


# ----------------------------------------------------------------------------- 畫面內容
def test_below_mode_draws_the_panel_with_the_chosen_colors(qapp, tmp_path):
    preview = OverlayPreview(AppConfig(tmp_path / "config.json"))

    preview.set_options(_options())

    assert _count(preview, "#7A1F1F") > 2000  # 面板底色
    assert _count(preview, "#FFE9A8", tolerance=0) > 20  # 譯文文字色（筆畫中心的像素）
    assert SAMPLE_TRANSLATION  # 範例文字是一句中文譯文


def test_below_mode_panel_opacity_changes_the_picture(qapp, tmp_path):
    preview = OverlayPreview(AppConfig(tmp_path / "config.json"))
    preview.set_options(_options(**{"overlay.panel_alpha": 255}))
    opaque = _count(preview, "#7A1F1F")

    preview.set_options(_options(**{"overlay.panel_alpha": 0}))

    assert opaque > 2000
    assert _count(preview, "#7A1F1F") == 0  # 完全透明：只看得到模擬畫面


def test_in_place_mode_with_auto_background_covers_the_source_with_the_sampled_color(qapp, tmp_path):
    preview = OverlayPreview(AppConfig(tmp_path / "config.json"))

    preview.set_options(_options(**{"overlay.placement": "in_place", "overlay.auto_background": True}))

    assert _count(preview, "#3A5C8E") > 2000  # 自動背景：取畫面原本的底色


def test_in_place_mode_with_manual_colors_uses_them(qapp, tmp_path):
    preview = OverlayPreview(AppConfig(tmp_path / "config.json"))

    preview.set_options(
        _options(**{"overlay.placement": "in_place", "overlay.auto_background": False, "overlay.background_alpha": 255})
    )

    assert _count(preview, "#7A1F1F") > 2000


def test_the_preview_follows_the_placement(qapp, tmp_path):
    preview = OverlayPreview(AppConfig(tmp_path / "config.json"))
    preview.set_options(_options())
    below_panel = _count(preview, "#7A1F1F")

    preview.set_options(_options(**{"overlay.placement": "in_place", "overlay.auto_background": True}))

    assert below_panel > 2000 and _count(preview, "#7A1F1F") == 0


# ----------------------------------------------------------------------------- 設定畫面
def test_dialog_shows_the_preview_next_to_the_colors_and_starts_in_sync(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    colors_page = dialog.pages.widget(4).widget()

    assert colors_page.isAncestorOf(dialog.overlay_preview)
    assert colors_page.isAncestorOf(dialog.btn_bg_color)
    options = dialog.overlay_preview._overrides
    assert options["overlay.background_color"] == dialog._bg_color.name()
    assert options["overlay.panel_alpha"] == dialog.slider_panel_alpha.value() == 215
    assert options["overlay.placement"] == "below"
    dialog.deleteLater()


def test_every_adjustment_updates_the_preview_before_saving(qapp, tmp_path, monkeypatch):
    config = AppConfig(tmp_path / "config.json")
    dialog = SettingsDialog(config)
    preview = dialog.overlay_preview

    dialog.slider_panel_alpha.setValue(90)
    assert preview._overrides["overlay.panel_alpha"] == 90
    dialog.spin_font_size.setValue(26)
    assert preview._overrides["overlay.font_size"] == 26
    dialog.spin_radius.setValue(12)
    assert preview._overrides["overlay.border_radius"] == 12
    dialog.chk_auto_bg.setChecked(False)
    assert preview._overrides["overlay.auto_background"] is False
    dialog.combo_placement.setCurrentIndex(dialog.combo_placement.findData("in_place"))
    assert preview._overrides["overlay.placement"] == "in_place"

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *args, **kwargs: QColor("#336699")))
    dialog._pick_color("bg")
    assert preview._overrides["overlay.background_color"] == "#336699"
    dialog._pick_color("text")
    assert preview._overrides["overlay.text_color"] == "#336699"

    assert config.get("overlay.background_color") != "#336699"  # 範例不會動到實際設定
    assert config.get("overlay.panel_alpha") == 215
    dialog.deleteLater()


def test_only_the_controls_that_apply_to_the_placement_are_enabled(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    # 框的下方：固定的顏色與面板不透明度；自動模式、內邊距、邊框用不到
    assert dialog.btn_text_color.isEnabled() and dialog.btn_bg_color.isEnabled()
    assert dialog.slider_panel_alpha.isEnabled()
    for unused in (dialog.chk_auto_color, dialog.chk_auto_bg, dialog.slider_bg_alpha, dialog.spin_padding, dialog.chk_show_border):
        assert not unused.isEnabled() and unused.toolTip()

    dialog.combo_placement.setCurrentIndex(dialog.combo_placement.findData("in_place"))
    assert not dialog.slider_panel_alpha.isEnabled()
    assert dialog.chk_auto_color.isEnabled() and dialog.chk_auto_bg.isEnabled() and dialog.spin_padding.isEnabled()
    # 自動模式開著時，手動的顏色與不透明度用不到
    assert dialog.chk_auto_color.isChecked() and dialog.chk_auto_bg.isChecked()
    assert not dialog.btn_text_color.isEnabled() and not dialog.btn_bg_color.isEnabled() and not dialog.slider_bg_alpha.isEnabled()

    dialog.chk_auto_color.setChecked(False)
    dialog.chk_auto_bg.setChecked(False)
    assert dialog.btn_text_color.isEnabled() and dialog.btn_bg_color.isEnabled() and dialog.slider_bg_alpha.isEnabled()
    dialog.deleteLater()


def test_panel_opacity_is_saved_and_restored(qapp, tmp_path):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    dialog.slider_panel_alpha.setValue(130)
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("overlay.panel_alpha") == 130
    reopened = SettingsDialog(saved)
    assert reopened.slider_panel_alpha.value() == 130
    assert reopened.overlay_preview._overrides["overlay.panel_alpha"] == 130
    dialog.deleteLater()
    reopened.deleteLater()

