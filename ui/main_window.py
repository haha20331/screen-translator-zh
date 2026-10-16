"""主窗口：按产品视觉稿组织的屏幕翻译工作台。"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QAbstractAnimation, QEvent, QPropertyAnimation, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ui.appearance import current_tokens
from ui.motion import (
    BASE,
    ENTER_EASING,
    EXIT_EASING,
    FAST,
    MICRO,
    MOVE_EASING,
    RELEASE_EASING,
    motion_duration,
)
from ui.settings_transition import SettingsTransitionGuard
from ui.welcome_dialog import _hkfmt


class NativePageDeck(QWidget):
    """A clipped deck that moves the real pages without a layout fighting them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("NativePageDeck")
        self._widgets: list[QWidget] = []
        self._current: QWidget | None = None
        self._transition_home: QWidget | None = None
        self._transition_settings: QWidget | None = None
        self._transition_progress = 0.0
        self._transitioning = False

    def addWidget(self, widget: QWidget) -> int:
        index = self.indexOf(widget)
        if index >= 0:
            return index
        widget.setParent(self)
        self._widgets.append(widget)
        widget.setGeometry(self.rect())
        if self._current is None:
            self._current = widget
            widget.show()
        else:
            widget.hide()
        return len(self._widgets) - 1

    def removeWidget(self, widget: QWidget) -> None:
        index = self.indexOf(widget)
        if index < 0:
            return
        widget.hide()
        self._widgets.pop(index)
        if self._current is widget:
            self._current = self._widgets[0] if self._widgets else None
        if self._transition_home is widget or self._transition_settings is widget:
            self._transition_home = None
            self._transition_settings = None
            self._transitioning = False
        if self._current is not None:
            self.setCurrentWidget(self._current)

    def indexOf(self, widget: QWidget) -> int:
        # Qt wrappers may outlive their deleted C++ object. Equality can then
        # dereference the dead object, while identity remains safe.
        for index, candidate in enumerate(self._widgets):
            if candidate is widget:
                return index
        return -1

    def count(self) -> int:
        return len(self._widgets)

    def widget(self, index: int) -> QWidget | None:
        return self._widgets[index] if 0 <= index < len(self._widgets) else None

    def currentWidget(self) -> QWidget | None:
        return self._current

    def isTransitioning(self) -> bool:
        return self._transitioning

    def setCurrentWidget(self, widget: QWidget) -> None:
        if self.indexOf(widget) < 0:
            return
        self._transitioning = False
        self._transition_home = None
        self._transition_settings = None
        self._current = widget
        bounds = self.rect()
        for candidate in self._widgets:
            candidate.setGeometry(bounds)
            candidate.setVisible(candidate is widget)
        widget.raise_()

    def setTransitionProgress(
        self,
        home: QWidget,
        settings: QWidget,
        progress: float,
    ) -> None:
        if self.indexOf(home) < 0 or self.indexOf(settings) < 0:
            return
        self._transition_home = home
        self._transition_settings = settings
        self._transition_progress = max(0.0, min(1.0, float(progress)))
        self._transitioning = True
        self._apply_transition_frame()

    def _apply_transition_frame(self) -> None:
        home = self._transition_home
        settings = self._transition_settings
        if not self._transitioning or home is None or settings is None:
            return
        width = self.width()
        height = self.height()
        seam = round(width * (1.0 - self._transition_progress))
        home.setGeometry(seam - width, 0, width, height)
        settings.setGeometry(seam, 0, width, height)
        home.show()
        settings.show()
        settings.raise_()
        self._current = settings if self._transition_progress >= 0.5 else home

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._transitioning:
            self._apply_transition_frame()
        elif self._current is not None:
            self._current.setGeometry(self.rect())


class CaptureActionCard(QPushButton):
    """带线性图标和双行说明的捕获入口卡片。"""

    def __init__(self, title: str, detail: str, icon_kind: str, active: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._detail = detail
        self._icon_kind = icon_kind
        self._active = active
        self._hover = 0.0
        self._press = 0.0
        self._launch = 0.0
        self._shake = 0.0
        self._hover_animation = QVariantAnimation(self)
        self._hover_animation.setDuration(BASE)
        self._hover_animation.setEasingCurve(ENTER_EASING)
        self._hover_animation.valueChanged.connect(self._set_hover)
        self._press_animation = QVariantAnimation(self)
        self._press_animation.valueChanged.connect(self._set_press)
        self._launch_animation = QVariantAnimation(self)
        self._launch_animation.valueChanged.connect(self._set_launch)
        self._shake_animation = QVariantAnimation(self)
        self._shake_animation.valueChanged.connect(self._set_shake)
        self.pressed.connect(lambda: self._animate_press(1.0))
        self.released.connect(lambda: self._animate_press(0.0))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(102)

    def _set_hover(self, value) -> None:
        self._hover = float(value)
        self.update()

    def _animate_hover(self, target: float) -> None:
        self._hover_animation.stop()
        duration = motion_duration(BASE)
        if duration == 0:
            self._set_hover(target)
            return
        self._hover_animation.setDuration(duration)
        self._hover_animation.setStartValue(self._hover)
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

    def _set_press(self, value) -> None:
        self._press = float(value)
        self.update()

    def _animate_press(self, target: float) -> None:
        self._press_animation.stop()
        duration = motion_duration(MICRO if target else FAST)
        if duration == 0:
            self._set_press(target)
            return
        self._press_animation.setDuration(duration)
        self._press_animation.setEasingCurve(
            ENTER_EASING if target else RELEASE_EASING
        )
        self._press_animation.setStartValue(self._press)
        self._press_animation.setEndValue(target)
        self._press_animation.start()

    def _set_launch(self, value) -> None:
        self._launch = float(value)
        self.update()

    def _set_shake(self, value) -> None:
        self._shake = float(value)
        self.update()

    def play_launch(self) -> None:
        self._shake_animation.stop()
        self._launch_animation.stop()
        duration = motion_duration(BASE)
        if duration == 0:
            self._launch = 1.0
            self.update()
            return
        self._launch_animation.setDuration(duration)
        self._launch_animation.setEasingCurve(ENTER_EASING)
        self._launch_animation.setStartValue(self._launch)
        self._launch_animation.setEndValue(1.0)
        self._launch_animation.start()

    def reset_launch(self) -> None:
        self._launch_animation.stop()
        self._launch = 0.0
        self.update()

    def play_failure(self) -> None:
        self.reset_launch()
        self._shake_animation.stop()
        duration = motion_duration(BASE)
        if duration == 0:
            self._shake = 0.0
            self.update()
            return
        self._shake_animation.setDuration(duration)
        self._shake_animation.setEasingCurve(ENTER_EASING)
        self._shake_animation.setKeyValues(
            [(0.0, 0.0), (0.24, -1.0), (0.48, 1.0), (0.72, -0.45), (1.0, 0.0)]
        )
        self._shake_animation.start()

    def set_detail(self, detail: str) -> None:
        self._detail = detail
        self.update()

    @staticmethod
    def _mix(start: QColor, end: QColor, progress: float) -> QColor:
        progress = max(0.0, min(1.0, progress))
        return QColor.fromRgbF(
            start.redF() + (end.redF() - start.redF()) * progress,
            start.greenF() + (end.greenF() - start.greenF()) * progress,
            start.blueF() + (end.blueF() - start.blueF()) * progress,
            start.alphaF() + (end.alphaF() - start.alphaF()) * progress,
        )

    def enterEvent(self, event) -> None:
        if self.isEnabled():
            self._animate_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._animate_hover(0.0)
        self._animate_press(0.0)
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(round(self._shake * 2.2), 0)
        painter.translate(0, round(self._press * 2.0))
        rect = self.rect().adjusted(1, 1, -1, -3)
        visual_enabled = self.isEnabled() or self._launch > 0.0
        active = self._active and visual_enabled
        if not visual_enabled:
            fill = QColor(tokens.surface_alt)
            border = QColor(tokens.border)
        else:
            emphasis = max(self._hover, 0.45 if active else 0.0)
            fill = self._mix(QColor(tokens.surface), QColor(tokens.accent_soft), emphasis)
            border = self._mix(QColor(tokens.border), QColor(tokens.accent_border), emphasis)
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.5))
        painter.drawRoundedRect(rect, 14, 14)

        emphasis = max(self._hover, 0.45 if active else 0.0)
        icon_color = QColor(tokens.disabled) if not visual_enabled else self._mix(
            QColor(tokens.ink_soft), QColor(tokens.accent), emphasis
        )
        painter.setPen(QPen(icon_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        x, y = 29, max(38, self.height() // 2 - 11)
        if self._icon_kind == "region":
            for px, py, dx, dy in ((x, y + 8, 0, -7), (x, y + 1, 7, 0), (x + 31, y + 8, 0, -7), (x + 31, y + 1, -7, 0),
                                   (x, y + 23, 0, 7), (x, y + 30, 7, 0), (x + 31, y + 23, 0, 7), (x + 31, y + 30, -7, 0)):
                painter.drawLine(px, py, px + dx, py + dy)
            painter.drawLine(x + 35, y + 25, x + 49, y + 25)
            painter.drawLine(x + 42, y + 18, x + 42, y + 32)
        elif self._icon_kind == "screen":
            painter.drawRoundedRect(x, y + 1, 38, 27, 4, 4)
            painter.drawLine(x + 19, y + 28, x + 19, y + 35)
            painter.drawLine(x + 9, y + 35, x + 29, y + 35)
        else:
            painter.drawRoundedRect(x, y, 39, 31, 5, 5)
            painter.drawLine(x, y + 8, x + 39, y + 8)

        title_font = QFont()
        title_font.setFamilies(["Microsoft JhengHei UI", "Microsoft JhengHei", "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI"])
        title_font.setPixelSize(17)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(
            QColor(tokens.disabled)
            if not visual_enabled
            else self._mix(QColor(tokens.ink), QColor(tokens.accent_hover), emphasis)
        )
        text_x = 95
        title_y = max(53, self.height() // 2 - 4)
        painter.drawText(text_x, title_y, self._title)

        detail_font = QFont(title_font)
        detail_font.setPixelSize(12)
        detail_font.setWeight(QFont.Weight.Normal)
        painter.setFont(detail_font)
        painter.setPen(QColor(tokens.disabled) if not visual_enabled else QColor(tokens.muted))
        painter.drawText(text_x, title_y + 26, self._detail)

        if visual_enabled:
            line_width = 12 + round(18 * max(emphasis, self._launch))
            line_x = x + 19 - line_width // 2
            painter.setPen(
                QPen(
                    QColor(tokens.accent), 2.4,
                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(line_x, rect.bottom() - 10, line_x + line_width, rect.bottom() - 10)

# 主視窗的內容只剩標題、一張「框選翻譯」卡和底下的狀態列，約 404 px 高；設定頁也是嵌在這個視窗裡，
# 440 px 時左邊六個分頁鈕還排得下、右邊的內容靠捲動
MIN_WINDOW_SIZE = (900, 440)
# 預設尺寸：980×460。（拿掉全屏／視窗翻譯卡、語言與 OCR 那一欄之後，原本的 580 太空）
DEFAULT_WINDOW_WIDTH = 980
DEFAULT_WINDOW_HEIGHT = 460
_SCREEN_MARGIN = 80  # 留給工作列與視窗標題列


def default_window_size(available_height: int | None = None) -> tuple[int, int]:
    """主視窗開啟時的預設尺寸；螢幕可用高度放不下就縮小，但不會小於最小尺寸。"""
    height = DEFAULT_WINDOW_HEIGHT
    if available_height:
        height = min(height, max(MIN_WINDOW_SIZE[1], int(available_height) - _SCREEN_MARGIN))
    return DEFAULT_WINDOW_WIDTH, height


class MainWindow(QWidget):
    def __init__(self, controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("Root")
        self.setWindowTitle("螢幕翻譯")
        self.setMinimumSize(*MIN_WINDOW_SIZE)
        screen = QGuiApplication.primaryScreen()
        self.resize(*default_window_size(screen.availableGeometry().height() if screen is not None else None))
        self._faded_in = False
        self._capture_departure_callback: Callable[[], None] | None = None
        self._visibility_action = ""
        self._settings_page_widget: QWidget | None = None
        self._settings_progress = 0.0
        self._settings_transition_target = 0.0
        self._settings_transition_state = "home"
        self._settings_exit_callback: Callable[[], None] | None = None
        self._settings_previous_focus: QWidget | None = None
        self._visibility_animation = QPropertyAnimation(self, b"windowOpacity", self)
        self._visibility_animation.finished.connect(self._finish_visibility_animation)
        self._departure_timer = QTimer(self)
        self._departure_timer.setSingleShot(True)
        self._departure_timer.timeout.connect(self._start_capture_departure)
        self._build_ui()
        self._settings_guard = SettingsTransitionGuard(self)
        self._settings_transition_animation = QVariantAnimation(self)
        self._settings_transition_animation.setStartValue(0.0)
        self._settings_transition_animation.setEndValue(1.0)
        self._settings_transition_animation.valueChanged.connect(self._set_settings_progress)
        self._settings_transition_animation.finished.connect(self._finish_settings_transition)
        self._load_values()
        self.refresh_appearance()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self._pages = NativePageDeck(self)
        root_layout.addWidget(self._pages)
        self._home_page = QWidget()
        self._home_page.setObjectName("HomePage")
        self._pages.addWidget(self._home_page)
        layout = QVBoxLayout(self._home_page)
        self._home_layout = layout
        layout.setContentsMargins(40, 30, 40, 28)
        layout.setSpacing(22)

        header_frame = QWidget()
        header_frame.setMinimumHeight(94)
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(0, 0, 0, 0)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(7)
        title = QLabel("螢幕翻譯")
        title.setObjectName("TitleLabel")
        hint = QLabel("框選螢幕上的字幕或文字，離線翻成繁體中文，譯文就顯示在框的下方")
        hint.setObjectName("HintLabel")
        title_stack.addStretch(1)
        title_stack.addWidget(title)
        title_stack.addWidget(hint)
        title_stack.addStretch(1)
        header.addLayout(title_stack, 1)
        self.btn_settings = QPushButton("設定")
        self.btn_settings.setObjectName("SettingsButton")
        self.btn_settings.setFixedSize(102, 42)
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self.btn_settings, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(header_frame)

        action_panel = QFrame()
        action_panel.setObjectName("ActionPanel")
        action_row = QHBoxLayout(action_panel)
        self._action_layout = action_row
        action_row.setContentsMargins(18, 16, 18, 16)
        action_row.setSpacing(16)
        self.btn_region = CaptureActionCard("框選翻譯", "拖曳選擇螢幕區域", "region", active=False)
        action_row.addWidget(self.btn_region, 1)
        layout.addWidget(action_panel)

        footer_panel = QFrame()
        footer_panel.setObjectName("FooterPanel")
        bottom_row = QHBoxLayout(footer_panel)
        self._footer_layout = bottom_row
        bottom_row.setContentsMargins(22, 14, 18, 14)
        bottom_row.setSpacing(10)
        self.status_label = QLabel("就緒：按「框選翻譯」開始")
        self.status_label.setObjectName("StatusLabel")
        bottom_row.addWidget(self.status_label, 1)
        self.check_keep_original = QCheckBox("保留原文")
        self.check_lyrics = QCheckBox("歌詞")
        self.check_lyrics.setToolTip("告訴模型這是歌詞，譯得較通順有詩意。只對「離線翻譯」有效。")
        self.btn_monitor = QPushButton("即時監控")
        self.btn_monitor.setObjectName("MonitorButton")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.setMinimumHeight(42)
        self.btn_monitor.setCursor(Qt.CursorShape.PointingHandCursor)
        bottom_row.addWidget(self.check_lyrics)
        bottom_row.addWidget(self.check_keep_original)
        bottom_row.addWidget(self.btn_monitor)
        layout.addWidget(footer_panel)

        self.btn_region.clicked.connect(self.controller.start_region_capture)
        self.btn_settings.clicked.connect(self.controller.open_settings)
        self.btn_monitor.clicked.connect(self.controller.toggle_monitor)
        self.check_keep_original.toggled.connect(self._on_keep_original)
        self.check_lyrics.toggled.connect(lambda checked: self._on_translation_flag("translation.lyrics_mode", checked))
        if hasattr(self.controller, "show_about"):
            QShortcut(QKeySequence("F1"), self).activated.connect(self.controller.show_about)

    def _load_values(self) -> None:
        self.reload_values()

    def reload_values(self) -> None:
        """Refresh duplicated quick controls after settings are saved."""
        self._block_signals = True
        self.check_keep_original.setChecked(bool(self.controller.config.get("translation.keep_original", False)))
        self.check_lyrics.setChecked(bool(self.controller.config.get("translation.lyrics_mode", False)))
        hotkey = ""
        try:
            hotkey = _hkfmt(self.controller.config.hotkeys().get("capture_region", ""))
        except Exception:  # 測試用的假設定沒有快捷鍵
            pass
        self.btn_region.set_detail("拖曳選擇螢幕區域" + (f"　快捷鍵 {hotkey}" if hotkey else ""))
        # 譯文放在框下方時原文本來就留在畫面上，「保留原文」只對蓋在原文上的模式有意義
        self.check_keep_original.setVisible(self.controller.config.get("overlay.placement", "below") == "in_place")
        self._block_signals = False

    def refresh_appearance(self) -> None:
        """Apply density metrics and repaint custom line-art widgets."""
        tokens = current_tokens()
        density = tokens.density
        margins = {
            "spacious": (42, 32, 42, 30),
            "balanced": (40, 30, 40, 28),
            "compact": (32, 22, 32, 22),
        }[density]
        button_height = {"spacious": 46, "balanced": 42, "compact": 38}[density]
        self._home_layout.setContentsMargins(*margins)
        self._home_layout.setSpacing(tokens.main_spacing)
        self._action_layout.setSpacing(max(12, tokens.main_spacing - 6))
        self._footer_layout.setSpacing(max(8, tokens.main_spacing - 12))
        self.btn_settings.setFixedHeight(button_height)
        self.btn_region.update()
        self.btn_settings.update()
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._faded_in:
            self._faded_in = True
            self.setWindowOpacity(0.0)
            duration = motion_duration(BASE, large_surface=True)
            if duration == 0:
                self.setWindowOpacity(1.0)
            else:
                self._visibility_action = "return"
                self._visibility_animation.stop()
                self._visibility_animation.setDuration(duration)
                self._visibility_animation.setStartValue(0.0)
                self._visibility_animation.setEndValue(1.0)
                self._visibility_animation.setEasingCurve(ENTER_EASING)
                self._visibility_animation.start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_settings_guard"):
            self._settings_guard.setGeometry(self.contentsRect())

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def show_settings_page(self, page: QWidget) -> None:
        # An embedded page must not retain the 900x610 standalone-dialog minimum.
        # The main window deliberately supports 900x540 and never changes size.
        page.setMinimumSize(0, 0)
        if self._pages.indexOf(page) < 0:
            self._pages.addWidget(page)
        if self._settings_progress <= 0.001:
            focused = self.focusWidget()
            self._settings_previous_focus = (
                focused
                if focused is not None
                and (focused is self._home_page or self._home_page.isAncestorOf(focused))
                else self.btn_settings
            )
        self._settings_page_widget = page
        self._settings_exit_callback = None
        self._animate_settings_to(1.0)

    def begin_settings_exit(self, page: QWidget, callback: Callable[[], None]) -> None:
        if page is not self._settings_page_widget or self._pages.indexOf(page) < 0:
            callback()
            return
        if self._settings_transition_state == "exiting":
            return
        self._settings_exit_callback = callback
        self._animate_settings_to(0.0)

    def remove_settings_page(self, page: QWidget) -> None:
        self._pages.setCurrentWidget(self._home_page)
        if self._pages.indexOf(page) >= 0:
            self._pages.removeWidget(page)
        if self._settings_page_widget is page:
            self._settings_page_widget = None
        page.deleteLater()

    def close_settings_page(self, page: QWidget) -> None:
        """Compatibility wrapper for callers that do not need exit coordination."""
        self.begin_settings_exit(page, lambda: self.remove_settings_page(page))

    def _apply_settings_frame(self, progress: float) -> None:
        """Push the two real pages across one exact, gap-free seam."""
        progress = max(0.0, min(1.0, float(progress)))
        page = self._settings_page_widget
        if page is None or self._pages.indexOf(page) < 0:
            return
        self._pages.setTransitionProgress(self._home_page, page, progress)
        if self._settings_guard.isVisible():
            self._settings_guard.raise_()

    def _animate_settings_to(self, target: float) -> None:
        target = 1.0 if target >= 0.5 else 0.0
        live_progress = self._settings_progress
        self._settings_transition_target = target
        self._settings_transition_state = "entering" if target else "exiting"
        popup = QApplication.activePopupWidget()
        if popup is not None:
            popup.close()
        self._settings_guard.setGeometry(self.contentsRect())
        self._apply_settings_frame(live_progress)
        duration = motion_duration(BASE + FAST, large_surface=True)
        distance = abs(target - live_progress)
        if duration == 0 or distance < 0.001:
            self._settings_transition_animation.stop()
            self._set_settings_progress(target)
            self._finish_settings_transition()
            return
        self._settings_guard.show()
        self._settings_guard.raise_()
        self._settings_guard.setFocus(Qt.FocusReason.OtherFocusReason)
        direction = (
            QAbstractAnimation.Direction.Forward
            if target > live_progress
            else QAbstractAnimation.Direction.Backward
        )
        if self._settings_transition_animation.state() == QAbstractAnimation.State.Running:
            # Reverse the live clock; recreating its keyframes causes a visible jump.
            self._settings_transition_animation.setDirection(direction)
        else:
            self._settings_transition_animation.setDuration(max(1, duration))
            self._settings_transition_animation.setEasingCurve(MOVE_EASING)
            curve = self._settings_transition_animation.easingCurve()
            low, high = 0.0, 1.0
            for _ in range(18):
                phase = (low + high) / 2.0
                if curve.valueForProgress(phase) < live_progress:
                    low = phase
                else:
                    high = phase
            live_time = round(duration * (low + high) / 2.0)
            was_blocked = self._settings_transition_animation.blockSignals(True)
            try:
                self._settings_transition_animation.setDirection(direction)
                self._settings_transition_animation.start()
                # start() resets a stopped clock to its directional endpoint.
                # Restore the live clock before any valueChanged frame can run.
                self._settings_transition_animation.setCurrentTime(live_time)
            finally:
                self._settings_transition_animation.blockSignals(was_blocked)
            self._set_settings_progress(live_progress)
        # The moving settings page is raised by the deck. Reassert the guard so
        # there is no one-frame input gap at either edge of the handoff.
        self._settings_guard.raise_()
        self._settings_guard.setFocus(Qt.FocusReason.OtherFocusReason)

    def settle_settings_transition_for_capture(self) -> None:
        """Finish the page handoff before capture visibility takes ownership."""
        if self._settings_transition_state not in {"entering", "exiting"}:
            return
        self._settings_transition_animation.stop()
        self._set_settings_progress(self._settings_transition_target)
        self._finish_settings_transition()

    def _set_settings_progress(self, value) -> None:
        progress = max(0.0, min(1.0, float(value)))
        self._settings_progress = progress
        self._apply_settings_frame(progress)
        page = self._settings_page_widget
        if page is not None and hasattr(page, "set_lifecycle_progress"):
            page.set_lifecycle_progress(progress)
        if 0.0001 < progress < 0.9999:
            self._settings_guard.show()
            self._settings_guard.raise_()

    def _finish_settings_transition(self) -> None:
        target = self._settings_transition_target
        self._set_settings_progress(target)
        self._settings_guard.hide()
        page = self._settings_page_widget
        if target >= 0.5:
            self._settings_transition_state = "active"
            if page is not None and self._pages.indexOf(page) >= 0:
                self._pages.setCurrentWidget(page)
                if hasattr(page, "nav_buttons") and page.nav_buttons:
                    page.nav_buttons[page.pages.currentIndex()].setFocus()
            return
        self._settings_transition_state = "home"
        self._pages.setCurrentWidget(self._home_page)
        previous_focus = self._settings_previous_focus
        self._settings_previous_focus = None
        if (
            previous_focus is not None
            and previous_focus.isEnabled()
            and previous_focus.isVisibleTo(self)
        ):
            previous_focus.setFocus()
        else:
            self.btn_settings.setFocus()
        callback = self._settings_exit_callback
        self._settings_exit_callback = None
        if callback is not None:
            callback()

    def set_busy(self, busy: bool) -> None:
        for control in (self.btn_region, self.btn_settings, self.check_keep_original):
            control.setEnabled(not busy)

    def play_capture_departure(self, mode: str, callback: Callable[[], None]) -> None:
        """Confirm the chosen card, then remove the window before any screenshot."""
        self._capture_departure_callback = callback
        card = self._capture_card(mode)
        if card is not None:
            card.play_launch()
        if not self.isVisible():
            self._finish_capture_departure_immediately()
            return
        delay = motion_duration(MICRO, large_surface=True)
        if delay == 0:
            self._start_capture_departure()
        else:
            self._departure_timer.start(delay)

    def restore_after_capture(
        self,
        mode: str,
        *,
        was_minimized: bool = False,
        failed: bool = False,
    ) -> None:
        self._departure_timer.stop()
        self._visibility_animation.stop()
        card = self._capture_card(mode)
        if card is not None:
            card.reset_launch()
        if was_minimized:
            self.setWindowOpacity(1.0)
            self.showMinimized()
            return
        duration = motion_duration(BASE, large_surface=True)
        if duration > 0:
            self.setWindowOpacity(0.0)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if failed and card is not None:
            card.play_failure()
        if duration == 0:
            self.setWindowOpacity(1.0)
            return
        self._visibility_action = "return"
        self._visibility_animation.setDuration(duration)
        self._visibility_animation.setStartValue(0.0)
        self._visibility_animation.setEndValue(1.0)
        self._visibility_animation.setEasingCurve(ENTER_EASING)
        self._visibility_animation.start()

    def play_capture_failure(self, mode: str) -> None:
        card = self._capture_card(mode)
        if card is not None:
            card.play_failure()

    def finish_capture(self, mode: str) -> None:
        """Clear the hidden launch state before the window is shown again."""
        card = self._capture_card(mode)
        if card is not None:
            card.reset_launch()

    def _capture_card(self, mode: str) -> CaptureActionCard | None:
        # 主視窗只有框選翻譯這一張卡；全屏、視窗翻譯只剩快捷鍵與系統匣選單
        return {"region": self.btn_region}.get(mode)

    def _start_capture_departure(self) -> None:
        duration = motion_duration(FAST, large_surface=True)
        if duration == 0:
            self._finish_capture_departure_immediately()
            return
        self._visibility_animation.stop()
        self._visibility_action = "depart"
        self._visibility_animation.setDuration(duration)
        self._visibility_animation.setStartValue(self.windowOpacity())
        self._visibility_animation.setEndValue(0.0)
        self._visibility_animation.setEasingCurve(EXIT_EASING)
        self._visibility_animation.start()

    def _finish_capture_departure_immediately(self) -> None:
        self.hide()
        self.setWindowOpacity(1.0)
        callback = self._capture_departure_callback
        self._capture_departure_callback = None
        if callback is not None:
            callback()

    def _finish_visibility_animation(self) -> None:
        if self._visibility_action == "depart":
            self._finish_capture_departure_immediately()
        else:
            self.setWindowOpacity(1.0)
        self._visibility_action = ""

    def set_monitor_checked(self, active: bool) -> None:
        self._block_signals = True
        self.btn_monitor.setChecked(active)
        self.btn_monitor.setText("停止監控" if active else "即時監控")
        self._block_signals = False

    def _on_translation_flag(self, key: str, checked: bool) -> None:
        """上下文、歌詞模式：勾選立即生效（翻譯器每次翻譯都讀設定），不必重新載入模型。"""
        if getattr(self, "_block_signals", False):
            return
        self.controller.config.set(key, checked)
        self.controller.config.save()

    def _on_keep_original(self, checked: bool) -> None:
        if getattr(self, "_block_signals", False):
            return
        self.controller.config.set("translation.keep_original", checked)
        self.controller.config.save()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized() and self._minimize_hides_to_tray():
            # 在事件處理完之後再藏起來：這個事件還在處理視窗狀態時直接 hide 會和系統的最小化動作打架
            QTimer.singleShot(0, self._hide_to_tray)

    def _minimize_hides_to_tray(self) -> bool:
        """設定「最小化主視窗時收到托盤」（預設關）；沒有系統匣可用時不能藏，不然視窗就找不回來了。"""
        return bool(self.controller.config.get("general.minimize_hides_to_tray", False)) and QSystemTrayIcon.isSystemTrayAvailable()

    def _hide_to_tray(self) -> None:
        if self.isMinimized() and self._minimize_hides_to_tray():
            self.hide()  # 系統匣圖示的點擊會 showNormal() 把它帶回來

    def closeEvent(self, event) -> None:
        if self.controller.config.get("general.minimize_to_tray", True):
            event.ignore()
            self.hide()
        else:
            # 沒有選「關閉主視窗時最小化到托盤」：關掉主視窗就結束程式。
            # main.py 設了「關掉最後一個視窗不結束」（給最小化到托盤用），所以這裡要自己呼叫 shutdown
            super().closeEvent(event)
            shutdown = getattr(self.controller, "shutdown", None)
            if callable(shutdown):
                shutdown()
