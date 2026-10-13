"""覆盖层管理器：按显示器分发覆盖窗口，负责 DPI 坐标换算。"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRect, QRectF
from PySide6.QtGui import QColor

from app.config import AppConfig
from app.logger import get_logger
from app.models import CaptureInfo, TextRegion
from ui.translation_overlay import Block, TranslationOverlayWindow
from ui.translation_panel import TranslationPanelWindow
from utils import dpi_utils
from utils.layout_utils import translation_lines, uses_translation_panel

log = get_logger("overlay.manager")


class OverlayManager(QObject):
    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._monitors: list[dpi_utils.MonitorInfo] = []
        self._windows: dict[int, TranslationOverlayWindow] = {}
        # 框選區域時，譯文放在框的下方（原文保持可見）：每個顯示器至多一個面板
        self._panels: dict[int, TranslationPanelWindow] = {}
        self._panel_lines: list[str] = []
        self._panel_bbox: tuple[int, int, int, int] | None = None
        self._visible = True
        self._edit_mode = False

    def set_monitor_map(self, monitors: list[dpi_utils.MonitorInfo]) -> None:
        self._monitors = monitors

    # ------------------------------------------------------------- capture result
    def show_regions(
        self, capture: CaptureInfo, regions: list[TextRegion], animate_new_only: bool = False
    ) -> bool:
        if not self._monitors:
            return False
        if uses_translation_panel(self._config.get("overlay.placement", "below"), capture.mode):
            return self._show_panel(capture, regions)
        self._clear_panels()
        groups: dict[int, list[TextRegion]] = {}
        for region in regions:
            monitor = dpi_utils.monitor_for_physical_point(region.center[0], region.center[1], self._monitors)
            groups.setdefault(monitor.index, []).append(region)

        # 监控模式下只更新有变化的显示器，避免整屏清屏再重绘带来的闪烁；
        # 普通截图仍走 hide_all 的完整重建。
        if not animate_new_only:
            self.hide_all()
        shown = False
        for monitor in self._monitors:
            regions_on_monitor = groups.get(monitor.index)
            if not regions_on_monitor:
                window = self._windows.get(monitor.index)
                if window is not None:
                    window.clear()
                    window.hide()
                continue
            mon_bbox = dpi_utils.clamp_bbox_to_monitor(capture.bbox, monitor)
            geo = dpi_utils.physical_rect_to_overlay_geometry(mon_bbox, monitor)
            blocks: list[Block] = []
            for region in regions_on_monitor:
                display_text = (region.translated_text or region.text).strip()
                # 翻译失败时服务会原样返回原文。不要再拿覆盖层重绘一遍，
                # 否则干净的原字会被换成不同字号和背景，整页看起来发脏。
                if not display_text or display_text == region.text.strip():
                    continue
                local = dpi_utils.physical_rect_to_local(
                    (region.x, region.y, region.right, region.bottom), mon_bbox, monitor
                )
                blocks.append(
                    Block(
                        QRectF(*local),
                        display_text,
                        QColor(region.text_color),
                        region.background_color,
                        (region.source_line_height or region.height) / monitor.dpr,
                        max(1, region.source_line_count),
                    )
                )
            if not blocks:
                continue

            window = self._windows.get(monitor.index)
            if window is None:
                window = TranslationOverlayWindow(self._config)
                window.close_requested.connect(self._on_overlay_close_requested)
                self._windows[monitor.index] = window

            window.setGeometry(geo[0], geo[1], max(1, geo[2]), max(1, geo[3]))
            window.set_blocks(blocks, animate_new_only=animate_new_only)
            window.set_edit_mode(self._edit_mode)
            window.show_fade()
            window.raise_()
            shown = True
        self._visible = shown
        return shown

    # ------------------------------------------------------------- panel below the selection
    def _show_panel(self, capture: CaptureInfo, regions: list[TextRegion]) -> bool:
        for window in self._windows.values():  # 先前以「蓋在原文上」顯示的內容不應殘留
            window.clear()
            window.hide()
        lines = translation_lines(regions)
        if not lines:
            self._clear_panels()
            self._visible = False
            return False
        self._panel_lines = lines
        self._panel_bbox = tuple(capture.bbox)
        self._visible = self._place_panel()
        return self._visible

    def _place_panel(self) -> bool:
        """把面板（重新）放到目前的框下方；框換了螢幕就換那個螢幕的面板。"""
        if self._panel_bbox is None or not self._panel_lines or not self._monitors:
            return False
        bbox = self._panel_bbox
        monitor = dpi_utils.monitor_for_physical_point(
            (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2, self._monitors
        )
        origin_x, origin_y = monitor.logical_origin
        physical = monitor.physical
        bounds = QRect(
            origin_x,
            origin_y,
            round((physical[2] - physical[0]) / monitor.dpr),
            round((physical[3] - physical[1]) / monitor.dpr),
        )
        geometry = dpi_utils.physical_rect_to_overlay_geometry(dpi_utils.clamp_bbox_to_monitor(bbox, monitor), monitor)
        anchor = QRect(geometry[0], geometry[1], max(1, geometry[2]), max(1, geometry[3]))
        for index, panel in self._panels.items():
            if index != monitor.index:
                panel.hide()
        panel = self._panels.get(monitor.index)
        if panel is None:
            panel = TranslationPanelWindow(self._config)
            self._panels[monitor.index] = panel
        panel.set_content(self._panel_lines, anchor, bounds)
        panel.show()
        panel.raise_()
        return True

    def follow_region(self, bbox: tuple[int, int, int, int]) -> None:
        """使用者拖動／縮放監控框時，讓譯文面板跟著框走。"""
        if self._panel_bbox is None or not self._visible:
            return
        self._panel_bbox = tuple(bbox)
        self._place_panel()

    def _clear_panels(self) -> None:
        for panel in self._panels.values():
            panel.clear()
        self._panel_lines = []
        self._panel_bbox = None

    def _on_overlay_close_requested(self) -> None:
        self.hide_all()
        self._visible = False
        self.set_edit_mode(False)

    # ------------------------------------------------------------- visibility
    def hide_all(self, animate: bool = False) -> None:
        for window in self._windows.values():
            if animate:
                window.hide_fade()
            else:
                window.hide()
        for panel in self._panels.values():
            panel.hide()
        self._visible = False

    def show_all(self) -> None:
        for window in self._windows.values():
            if window._blocks:
                window.show_fade()
                window.raise_()
        if self._panel_lines:
            self._place_panel()
        self._visible = True

    def clear_all(self) -> None:
        """立即清除所有覆盖窗口内容并隐藏，保证下一次翻译不叠加。"""
        for window in self._windows.values():
            window.clear()
            window.hide()
        self._clear_panels()
        self._visible = False

    def toggle(self) -> bool:
        if self._visible:
            self.hide_all(animate=True)
        else:
            self.show_all()
        return self._visible

    def is_visible(self) -> bool:
        return self._visible

    # ------------------------------------------------------------- edit/style
    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        for window in self._windows.values():
            window.set_edit_mode(enabled)
        if self._visible:
            for window in self._windows.values():
                if window._blocks:
                    window.show()
                    window.raise_()

    def apply_style(self) -> None:
        for window in self._windows.values():
            window.apply_style()
        for panel in self._panels.values():
            panel.apply_style()
