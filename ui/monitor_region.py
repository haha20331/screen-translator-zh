"""监控区域选择器：监控进行中可拖拽/缩放调整抓取范围（实时更新）。"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QRegion
from PySide6.QtWidgets import QToolTip, QWidget

from app.logger import get_logger
from ui.motion import motion_duration
from utils import dpi_utils, native_window

log = get_logger("monitor_region")

# 框沒被動到（滑鼠不在框內）超過這麼久，就變得更透明，免得一直擋在字幕旁邊；滑鼠移進框內就恢復
IDLE_DIM_MS = 2000
DIM_OPACITY = 0.3  # 變透明之後的整體不透明度（框線還看得到，找得到框在哪）
# 框裡面幾乎全透明、變淡之後更是收不到滑鼠事件，所以每隔這麼久直接看游標在不在框裡
CURSOR_POLL_MS = 50  # 同時決定移動符號在游標進出框時多快出現、消失
# 拖曳、縮放時，通知譯文面板跟著框走的最短間隔：滑鼠事件可以一秒來好幾百個，面板不必每個都重排一次
REGION_UPDATE_MS = 16
# 框的最小尺寸（邏輯像素）：再小的話兩個移動符號就疊在一起了。拉出來的框比這個小，也照這個大小先建出來
MIN_FRAME_SIZE = (64, 40)
# 框是「中空」的：只有邊框那一圈（縮放用）與左上、右下兩個移動符號收得到滑鼠，中間整片讓滑鼠穿過去，
# 點得到下面真正的畫面。中間靠視窗的形狀（setMask）排除在視窗之外。
# 另外：Windows 的分層視窗（透明背景）對「逐點透明度是 0」的像素也讓滑鼠穿過去，所以邊框那一圈（9 px 寬，實際畫出來的線只有
# 2 px）要塗上透明度 1/255 的底色（肉眼看不出來），整圈才收得到滑鼠；實測沒有它，邊框中間的縮放把手會穿透
# 移動方式（設定 → 即時監控）：GRIPS = 拖左上、右下兩個移動符號（預設，中間穿透）；CENTER = 整個框的中央都能拖，但中間不再穿透
DRAG_GRIPS = "grips"
DRAG_CENTER = "center"
DRAG_MODES = (DRAG_GRIPS, DRAG_CENTER)
# 游標停在可以拖曳移動的地方（移動符號；「拖框的中央」模式則是框的中央）時另外顯示的提示，其餘地方顯示整體的操作說明
MOVE_TIP = "拖曳移動，雙擊關閉"
BAND = 9  # 邊框那一圈的寬度，也是縮放的把手範圍
RING_ALPHA = 1
GRIP = 18  # 移動符號的邊長
GRIP_INSET = BAND + 2  # 移動符號離框外緣的距離（在邊框那一圈的裡面）
# 剛顯示出來的前一秒左右，每隔這些毫秒檢查一次視窗實際的位置，和要求的不同就用系統座標校正（見 _settle_position）
SETTLE_AT_MS = (0, 40, 120, 300, 700, 1400)


class MonitorRegionSelector(QWidget):
    # physical (mss) bbox, 实时跟随拖拽
    region_changed = Signal(object)
    # physical (mss) bbox, 拖拽/缩放结束时触发一次
    region_committed = Signal(object)
    # 双击停止监控
    stop_requested = Signal()

    _HANDLE = BAND

    def __init__(
        self,
        monitor_map,
        physical_bbox,
        accent: str = "#2878E8",
        parent=None,
        idle_ms: int = IDLE_DIM_MS,
        dim_opacity: float = DIM_OPACITY,
        poll_ms: int = CURSOR_POLL_MS,
        drag_mode: str = DRAG_GRIPS,
    ) -> None:
        super().__init__(parent)
        self._drag_mode = drag_mode if drag_mode in DRAG_MODES else DRAG_GRIPS
        self._idle_ms = max(0, int(idle_ms))  # 0 = 不變透明
        self._dim_opacity = max(0.15, min(1.0, float(dim_opacity)))  # 太低的話框內填色的透明度四捨五入成 0，滑鼠會穿過框、移進來也收不到
        self._dimmed = False
        self._grips_visible = True  # 兩個移動符號只在游標在框的範圍內才畫出來（游標一離開就隱形）；剛建好的框游標就在上面
        self._monitor_map = monitor_map
        self._accent = QColor(accent)
        self._highlight_color = QColor(accent)
        self._dragging = False
        self._resize_dir = 0
        self._settled = False  # 使用者已經動過框（或位置已經確認過）：不再自動校正位置
        self._press_global = QPoint()  # 按下去那一刻游標的螢幕座標：拖曳量都用螢幕座標算（見 mouseMoveEvent）
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
        # 這個視窗從來不是作用中的視窗（不搶焦點）：Qt 預設只替作用中的視窗顯示提示，不加這個，滑鼠停在框上什麼提示都不會出現
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips)
        self.setMouseTracking(True)
        # 操作說明用滑鼠停留時的提示，不畫在框內：框內是被擷取、被辨識的區域，常駐的字會擋住字幕
        self._update_tooltip()
        self.setGeometry(self._geometry)
        self._update_shape()

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
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(max(10, int(poll_ms)))
        self._poll_timer.timeout.connect(self._poll_cursor)
        # 拖曳時送給譯文面板的更新：合併成每 REGION_UPDATE_MS 一次，最後一次一定會送
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(REGION_UPDATE_MS)
        self._update_timer.timeout.connect(self._flush_region_changed)

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

    def grips_visible(self) -> bool:
        return self._grips_visible

    def _set_grips_visible(self, visible: bool) -> None:
        if visible != self._grips_visible:
            self._grips_visible = visible
            self.update()

    def _touch(self) -> None:
        """框被動到（滑鼠移進來、按下、拖曳）：不再倒數，已經變透明就恢復，移動符號也顯示出來。"""
        self._set_grips_visible(True)
        self._idle_timer.stop()
        if self._dimmed:
            self._dimmed = False
            self._fade_to(1.0, motion_duration(140))

    def _dim_if_idle(self) -> None:
        if self._dragging or self._cursor_inside():
            return
        self._dimmed = True
        self._fade_to(self._dim_opacity, motion_duration(260))

    def _poll_cursor(self) -> None:
        """定時直接看游標在不在框裡：框裡面幾乎全透明、變淡之後更是收不到 enter／mouseMove，不能只靠事件。"""
        if self._dragging:
            self._set_grips_visible(True)  # 拖曳中游標可能暫時落後在框外，符號不能在手指底下消失
            return
        if self._cursor_inside():
            self._set_grips_visible(True)
            if self._dimmed or self._idle_timer.isActive():
                self._touch()
        else:
            self._set_grips_visible(False)
            if not self._dimmed and not self._idle_timer.isActive():
                self._arm_idle()

    # --------------------------------------------------------- 位置：用系統座標對準
    def _wanted_native(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self._physical)  # type: ignore[return-value]

    def _settle_position(self) -> None:
        """框剛顯示的前一秒：視窗實際的位置（系統座標，物理像素）和要求的不同，就直接用系統座標放過去。

        Qt 把「邏輯座標」換成螢幕上的位置時，用的是視窗「當時所在」螢幕的縮放倍率；第一次顯示的視窗還沒有確定在哪個螢幕，
        在多螢幕、縮放比例不同的電腦上會偏掉（之後用滑鼠拖才正常，因為那時視窗的螢幕已經確定了）。
        截圖用的是物理像素，所以框的位置直接以物理像素為準；位置本來就對（單螢幕、100% 縮放）時什麼都不做。"""
        if self._settled or self._dragging or not self.isVisible() or not native_window.available():
            return
        hwnd = int(self.winId())
        wanted = self._wanted_native()
        actual = native_window.window_rect(hwnd)
        if actual is None:
            return
        if max(abs(a - b) for a, b in zip(actual, wanted)) <= 1:
            return
        log.warning("監視框的位置與要求的不同（要求 %s，實際 %s），用系統座標校正", wanted, actual)
        native_window.set_window_rect(hwnd, wanted)

    def _schedule_settling(self) -> None:
        self._settled = False
        for delay in SETTLE_AT_MS:
            QTimer.singleShot(delay, self._settle_position)

    def _current_physical(self):
        """框現在在螢幕上的範圍（物理像素）：能問系統就問系統（就是使用者看到的位置），不行才用 Qt 的座標換算。"""
        if self.isVisible() and native_window.available():
            actual = native_window.window_rect(int(self.winId()))
            if actual is not None:
                return actual
        return self._geometry_to_physical(self._geometry)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._arm_idle()
        self._poll_timer.start()
        log.info(
            "監視框：要求的位置 %s，Qt 邏輯座標 %s，螢幕 %s",
            self._wanted_native(), self.geometry().getRect(),
            [(m.physical, m.logical_origin, m.dpr) for m in self._monitor_map],
        )
        self._schedule_settling()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._poll_timer.stop()
        self._update_timer.stop()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._touch()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        # 游標從邊框走進中間（中間是穿透的）也會收到 leave，但還在框的範圍內：符號留著，等游標真的離開框再隱形
        if not self._dragging and not self._cursor_inside():
            self._set_grips_visible(False)
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
        if self.isVisible():
            self._schedule_settling()

    # --------------------------------------------------------- 形狀與命中測試
    def _grip_rects(self) -> tuple[QRect, QRect]:
        """左上、右下兩個移動符號的位置（視窗內座標）。"""
        top_left = QRect(GRIP_INSET, GRIP_INSET, GRIP, GRIP)
        bottom_right = QRect(self.width() - GRIP_INSET - GRIP, self.height() - GRIP_INSET - GRIP, GRIP, GRIP)
        return top_left, bottom_right

    def _shape(self) -> QRegion:
        """收得到滑鼠的範圍：邊框那一圈加上兩個移動符號；其餘（框的中間）不屬於這個視窗，滑鼠直接穿到下面。
        「拖框的中央」模式整個框都收滑鼠。"""
        w, h = self.width(), self.height()
        if self._drag_mode == DRAG_CENTER:
            return QRegion(0, 0, w, h)
        ring = QRegion(0, 0, w, h).subtracted(QRegion(BAND, BAND, max(0, w - 2 * BAND), max(0, h - 2 * BAND)))
        for grip in self._grip_rects():
            ring = ring.united(QRegion(grip))
        return ring

    def _update_shape(self) -> None:
        if self._drag_mode == DRAG_CENTER:
            self.clearMask()  # 整個框都收滑鼠
        else:
            self.setMask(self._shape())

    def _update_tooltip(self) -> None:
        if self._drag_mode == DRAG_CENTER:
            self.setToolTip("按住框的中央拖曳來移動 · 拖邊框與四個角來縮放 · 雙擊停止即時監控")
        else:
            self.setToolTip("按住左上或右下角的移動符號拖曳來移動 · 拖邊框與四個角來縮放 · 雙擊邊框或符號停止即時監控")

    def _tooltip_at(self, pos: QPoint) -> tuple[str, QRect]:
        """游標在 pos（視窗內座標）時要顯示的提示，以及這段提示管到哪個範圍（離開那個範圍就換一則）。"""
        if self._hit_test(pos) == 0:
            for grip in self._grip_rects():
                if grip.contains(pos):
                    return MOVE_TIP, grip
            return MOVE_TIP, self.rect().adjusted(BAND, BAND, -BAND, -BAND)  # 「拖框的中央」模式：框的中央
        return self.toolTip(), QRect()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            text, area = self._tooltip_at(event.pos())
            if text:
                QToolTip.showText(event.globalPos(), text, self, area)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def drag_mode(self) -> str:
        return self._drag_mode

    def is_adjusting(self) -> bool:
        """使用者現在按著滑鼠在移動或調整大小（放開之前都算）。"""
        return self._dragging

    def set_drag_mode(self, mode: str) -> None:
        """換移動方式（設定改了，框還在畫面上時馬上生效）。"""
        mode = mode if mode in DRAG_MODES else DRAG_GRIPS
        if mode == self._drag_mode:
            return
        self._drag_mode = mode
        self._update_shape()
        self._update_tooltip()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_shape()

    def _hit_test(self, pos: QPoint) -> int:
        """游標在哪：1～8 = 邊框的八個縮放位置，0 = 移動符號，-1 = 都不是（框的中間，實際上收不到滑鼠）。"""
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
        if self._drag_mode == DRAG_CENTER:
            return 0  # 框的中央整片都是移動
        if any(grip.contains(pos) for grip in self._grip_rects()):
            return 0
        return -1

    def _apply_cursor(self, pos: QPoint) -> None:
        d = self._hit_test(pos)
        cursors = {
            -1: Qt.CursorShape.ArrowCursor,
            0: Qt.CursorShape.SizeAllCursor,
            1: Qt.CursorShape.SizeFDiagCursor,
            2: Qt.CursorShape.SizeBDiagCursor,
            3: Qt.CursorShape.SizeBDiagCursor,
            4: Qt.CursorShape.SizeFDiagCursor,
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
        direction = self._hit_test(event.position().toPoint())
        if direction < 0:
            return  # 框的中間不處理（它本來就收不到滑鼠，這裡只是保險）
        self._dragging = True
        self._settled = True  # 使用者開始動框了，不要再自動校正位置
        self._touch()
        self._resize_dir = direction
        self._press_global = event.globalPosition().toPoint()
        self._geometry = self.geometry()  # 以視窗現在實際的位置為準（校正位置之後 Qt 的記錄可能還沒更新）
        self._base_geometry = QRect(self._geometry)
        self._pulse.stop()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._dimmed:
            self._touch()
        if not self._dragging:
            self._apply_cursor(pos)
            return
        # 拖曳量用「螢幕座標」算：event.position() 是相對於框「現在的位置」，框一動它就跟著變，
        # 用它算移動量會讓框在兩個位置之間來回跳（抽搐）；螢幕座標不受框移動的影響
        delta = event.globalPosition().toPoint() - self._press_global
        g = QRect(self._base_geometry)
        d = self._resize_dir
        min_w, min_h = MIN_FRAME_SIZE
        if d == 0:
            g.translate(delta)
        else:
            # QRect 的 right()／bottom() 是最後一個像素（寬度 = right - left + 1），所以最小寬度要多算一個
            if d in (1, 3, 5):
                g.setLeft(min(g.left() + delta.x(), g.right() - min_w + 1))
            if d in (2, 4, 6):
                g.setRight(max(g.right() + delta.x(), g.left() + min_w - 1))
            if d in (1, 2, 7):
                g.setTop(min(g.top() + delta.y(), g.bottom() - min_h + 1))
            if d in (3, 4, 8):
                g.setBottom(max(g.bottom() + delta.y(), g.top() + min_h - 1))
        if g == self._geometry:
            return
        self._geometry = g
        self.setGeometry(g)
        if not self._update_timer.isActive():
            self._update_timer.start()

    def _flush_region_changed(self) -> None:
        physical = self._current_physical()
        if physical is not None:
            self._physical = physical
            self.region_changed.emit(physical)

    def mouseReleaseEvent(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._resize_dir = 0
        self._update_timer.stop()
        physical = self._current_physical()
        if physical is not None:
            self._physical = physical
            self.region_changed.emit(physical)  # 排隊中還沒送的最後一次更新，先送出去再說「定案」
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
        # 框裡面不畫任何東西（全透明）；只畫框線、四個角的把手，以及（預設的移動方式）兩個移動符號。
        # 底色只塗在 _shape 圈出來的範圍（邊框那一圈與移動符號；「拖框的中央」模式是整個框）：
        # 透明度 RING_ALPHA，肉眼看不出來，只是讓這些地方收得到滑鼠
        painter.setClipRegion(self._shape())
        ring = QColor(self._accent)
        ring.setAlpha(RING_ALPHA)
        painter.fillRect(self.rect(), ring)
        painter.setClipping(False)

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

        if self._grips_visible and self._drag_mode == DRAG_GRIPS:
            for grip in self._grip_rects():
                self._paint_grip(painter, grip)

    def _paint_grip(self, painter: QPainter, grip: QRect) -> None:
        """移動符號：主題色的小方塊，中間一個四個方向的箭頭（按住它拖曳來移動整個框）。"""
        chip = QColor(self._accent)
        chip.setAlpha(225)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(chip)
        painter.drawRoundedRect(grip, 5, 5)
        pen = QPen(QColor(255, 255, 255, 245), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = grip.center().x() + 0.5, grip.center().y() + 0.5
        reach, tip = GRIP / 2 - 3.5, 2.6
        painter.drawLine(cx - reach, cy, cx + reach, cy)
        painter.drawLine(cx, cy - reach, cx, cy + reach)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = cx + sx * reach, cy + sy * reach
            painter.drawLine(tx, ty, tx - sx * tip + sy * tip, ty - sy * tip + sx * tip)
            painter.drawLine(tx, ty, tx - sx * tip - sy * tip, ty - sy * tip - sx * tip)
