"""监控区域选择器：监控进行中可拖拽/缩放调整抓取范围（实时更新）。"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRect, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.motion import motion_duration
from utils import dpi_utils

# 框沒被動到（滑鼠不在框內）超過這麼久，就變得更透明，免得一直擋在字幕旁邊；滑鼠移進框內就恢復
IDLE_DIM_MS = 2000
DIM_OPACITY = 0.3  # 變透明之後的整體不透明度（框線還看得到，找得到框在哪）


class MonitorRegionSelector(QWidget):
    # physical (mss) bbox, 实时跟随拖拽
    region_changed = Signal(object)
    # physical (mss) bbox, 拖拽/缩放结束时触发一次
    region_committed = Signal(object)
    # 双击停止监控
    stop_requested = Signal()

    _HANDLE = 9

    def __init__(
        self,
        monitor_map,
        physical_bbox,
        accent: str = "#2878E8",
        parent=None,
        idle_ms: int = IDLE_DIM_MS,
        dim_opacity: float = DIM_OPACITY,
    ) -> None:
        super().__init__(parent)
        self._idle_ms = max(0, int(idle_ms))  # 0 = 不變透明
        self._dim_opacity = max(0.15, min(1.0, float(dim_opacity)))  # 太低的話框內填色的透明度四捨五入成 0，滑鼠會穿過框、移進來也收不到
        self._dimmed = False
        self._monitor_map = monitor_map
        self._accent = QColor(accent)
        self._highlight_color = QColor(accent)
        self._dragging = False
        self._resize_dir = 0
        self._drag_offset = QPoint()
        self._base_geometry = self._physical_to_geometry(physical_bbox)
        self._geometry = QRect(self._base_geometry)
        self._physical = tuple(physical_bbox)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        # 操作說明用滑鼠停留時的提示，不畫在框內：框內是被擷取、被辨識的區域，常駐的字會擋住字幕
        self.setToolTip("拖曳調整位置與大小 · 雙擊停止即時監控")
        self.setGeometry(self._geometry)

        self._pulse_t = 0.0
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.0)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(1100)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._set_pulse)
        self._pulse.start()

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(self._idle_ms)
        self._idle_timer.timeout.connect(self._dim_if_idle)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

    # --------------------------------------------------------- 閒置變透明
    def is_dimmed(self) -> bool:
        return self._dimmed

    def _cursor_inside(self) -> bool:
        # 不靠 enter／leave 事件：框剛顯示在靜止的游標底下、或被別的視窗蓋過時，事件可能不會來
        return self.geometry().contains(QCursor.pos())

    def _fade_to(self, opacity: float, duration: int) -> None:
        self._fade.stop()
        if duration <= 0:  # 動態效果關閉（減少動態）：直接切換
            self.setWindowOpacity(opacity)
            return
        self._fade.setDuration(duration)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(opacity)
        self._fade.start()

    def _arm_idle(self) -> None:
        self._idle_timer.stop()
        if self._idle_ms > 0 and not self._dragging:
            self._idle_timer.start()

    def _touch(self) -> None:
        """框被動到（滑鼠移進來、按下、拖曳）：不再倒數，已經變透明就恢復。"""
        self._idle_timer.stop()
        if self._dimmed:
            self._dimmed = False
            self._fade_to(1.0, motion_duration(140))

    def _dim_if_idle(self) -> None:
        if self._dragging or self._cursor_inside():
            return
        self._dimmed = True
        self._fade_to(self._dim_opacity, motion_duration(260))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._arm_idle()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._touch()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._arm_idle()

    # --------------------------------------------------------- coordinate mapping
    def _physical_to_geometry(self, physical) -> QRect:
        cx = (physical[0] + physical[2]) // 2
        cy = (physical[1] + physical[3]) // 2
        monitor = dpi_utils.monitor_for_physical_point(cx, cy, self._monitor_map)
        geo = dpi_utils.physical_rect_to_overlay_geometry(physical, monitor)
        return QRect(geo[0], geo[1], geo[2], geo[3])

    def _geometry_to_physical(self, geo: QRect):
        logical = (geo.x(), geo.y(), geo.x() + geo.width(), geo.y() + geo.height())
        parts = dpi_utils.logical_rect_to_physical_parts(logical, self._monitor_map)
        if not parts:
            return None
        return dpi_utils.union_rects(parts)

    def set_physical_bbox(self, physical) -> None:
        self._physical = tuple(physical)
        self._geometry = self._physical_to_geometry(physical)
        self.setGeometry(self._geometry)

    # --------------------------------------------------------- hit testing
    def _hit_test(self, pos: QPoint) -> int:
        h = self._HANDLE
        x, y = pos.x(), pos.y()
        w, ht = self.width(), self.height()
        left = x <= h
        right = x >= w - h
        top = y <= h
        bottom = y >= ht - h
        if left and top:
            return 1
        if right and top:
            return 2
        if left and bottom:
            return 3
        if right and bottom:
            return 4
        if left:
            return 5
        if right:
            return 6
        if top:
            return 7
        if bottom:
            return 8
        return 0

    def _apply_cursor(self, pos: QPoint) -> None:
        d = self._hit_test(pos)
        cursors = {
            0: Qt.CursorShape.SizeAllCursor,
            1: Qt.CursorShape.SizeFDiagCursor,
            2: Qt.CursorShape.SizeBDiagCursor,
            3: Qt.CursorShape.SizeFDiagCursor,
            4: Qt.CursorShape.SizeBDiagCursor,
            5: Qt.CursorShape.SizeHorCursor,
            6: Qt.CursorShape.SizeHorCursor,
            7: Qt.CursorShape.SizeVerCursor,
            8: Qt.CursorShape.SizeVerCursor,
        }
        self.setCursor(cursors[d])

    # --------------------------------------------------------- mouse
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        self._touch()
        self._resize_dir = self._hit_test(event.position().toPoint())
        self._drag_offset = event.position().toPoint()
        self._base_geometry = QRect(self._geometry)
        self._pulse.stop()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._dimmed:
            self._touch()
        if not self._dragging:
            self._apply_cursor(pos)
            return
        delta = pos - self._drag_offset
        g = QRect(self._base_geometry)
        d = self._resize_dir
        min_w, min_h = 32, 32
        if d == 0:
            g.translate(delta)
        else:
            if d in (1, 3, 5):
                g.setLeft(min(g.left() + delta.x(), g.right() - min_w))
            if d in (2, 4, 6):
                g.setRight(max(g.right() + delta.x(), g.left() + min_w))
            if d in (1, 2, 7):
                g.setTop(min(g.top() + delta.y(), g.bottom() - min_h))
            if d in (3, 4, 8):
                g.setBottom(max(g.bottom() + delta.y(), g.top() + min_h))
        self._geometry = g
        self.setGeometry(g)
        physical = self._geometry_to_physical(g)
        if physical is not None:
            self._physical = physical
            self.region_changed.emit(physical)

    def mouseReleaseEvent(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._resize_dir = 0
        physical = self._geometry_to_physical(self._geometry)
        if physical is not None:
            self._physical = physical
            self.region_committed.emit(physical)
        self._apply_cursor(event.position().toPoint())
        if not self._cursor_inside():
            self._arm_idle()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.stop_requested.emit()

    # --------------------------------------------------------- paint
    def _set_pulse(self, value) -> None:
        self._pulse_t = float(value)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        fill = QColor(self._accent)
        fill.setAlpha(14)
        painter.fillRect(self.rect(), fill)

        pulse = 0.5 + 0.5 * (1.0 - abs(1.0 - 2.0 * self._pulse_t))
        border = QColor(self._accent)
        border.setAlpha(int(200 + 45 * pulse))
        painter.setPen(QPen(border, 1.6 + 0.6 * pulse))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))

        handle = QColor(self._accent)
        handle.setAlpha(235)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(handle)
        hh = self._HANDLE
        for cx, cy in (
            (hh // 2, hh // 2),
            (w - hh // 2, hh // 2),
            (hh // 2, h - hh // 2),
            (w - hh // 2, h - hh // 2),
        ):
            painter.drawRoundedRect(cx - 3, cy - 3, 6, 6, 2, 2)
