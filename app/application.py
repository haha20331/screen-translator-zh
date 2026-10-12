"""应用控制器：把截图、OCR、翻译、覆盖层、托盘、快捷键串起来。

原本这是一个 1400+ 行的 god-object。现已按职责拆分为若干 Mixin
（``app/*_mixin.py``），``Application`` 只保留生命周期与更新逻辑并继承它们。
所有公开方法名、信号连接方式、以及测试直接访问的私有属性名都**完全不变**，
因此这是一次零行为变化的纯结构重构。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMessageBox, QWidget

import services.ocr.null_ocr  # noqa: F401
import services.ocr.rapidocr_ocr  # noqa: F401

# 确保各 OCR 引擎 / 翻译适配器已注册（顺序影响 UI 下拉框默认项）
import services.ocr.windows_ocr  # noqa: F401
import services.translation.deepl_translator  # noqa: F401
import services.translation.google_free_translator  # noqa: F401
import services.translation.google_translator  # noqa: F401
import services.translation.mock_translator  # noqa: F401
import services.translation.mymemory_translator  # noqa: F401
import services.translation.openai_translator  # noqa: F401

# 各职责拆分为 Mixin，Application 只保留生命周期/更新并继承它们。
from app.capture_mixin import CaptureMixin
from app.capture_session import _CaptureSession
from app.clipboard_mixin import ClipboardMixin
from app.config import AppConfig
from app.history_mixin import HistoryMixin
from app.hotkeys import HotkeyError, HotkeyManager
from app.logger import app_data_dir, get_logger
from app.models import CaptureInfo, TextRegion
from app.monitor_mixin import MonitorMixin
from app.pipeline_mixin import PipelineMixin
from app.runtime import main_script, sys_executable
from app.settings_mixin import SettingsMixin
from app.status_mixin import StatusMixin
from app.version import __version__
from services.ocr.base import OCRUnavailableError, create_ocr_engine, list_ocr_engines
from services.screenshot_service import ScreenshotService
from services.translation.base import Translator
from services.translation.cache import TranslationCache
from services.translation.factory import create_translator, list_translators
from services.window_capture_service import (
    WindowCaptureError,
    get_foreground_window,
    get_window_rect_physical,
    get_window_title,
    is_current_process_window,
    is_window_capturable,
    window_capture_available,
)
from ui.appearance import resolve_tokens
from ui.floating_status import FloatingStatus
from ui.main_window import MainWindow
from ui.motion import CAPTURE_SETTLE, SELECTION_SETTLE, STATUS_HOLD
from ui.overlay_manager import OverlayManager
from ui.selection_overlay import SelectionOverlay
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from ui.tray_icon import TrayIcon, build_icon
from ui.window_capture_highlight import WindowCaptureHighlight
from utils import dpi_utils
from utils.language_utils import LANGUAGE_CODES, LANGUAGES
from workers.translation_worker import PipelineTask

log = get_logger("application")


class Application(
    QObject,
    StatusMixin,
    ClipboardMixin,
    HistoryMixin,
    MonitorMixin,
    CaptureMixin,
    SettingsMixin,
    PipelineMixin,
):
    """顶层控制器（生命周期 + 更新；其余职责见各 Mixin）。"""

    def __init__(self, qt_app: QGuiApplication) -> None:
        super().__init__()
        self.qt_app = qt_app
        self.config = AppConfig()
        self.monitor_map: list[dpi_utils.MonitorInfo] = []

        self.screenshot_service = ScreenshotService()
        self.overlay_manager: OverlayManager | None = None
        self.floating_status = FloatingStatus()
        self.hotkey_manager = HotkeyManager(self)
        self.translator: Translator | None = None
        self.ocr_engine = None
        self._active_ocr_engine_name = "none"
        self.cache: TranslationCache | None = None

        self.window: MainWindow | None = None
        self._settings_page: SettingsDialog | None = None
        self._reopen_settings_after_close = False
        self.tray: TrayIcon | None = None
        self.selection: SelectionOverlay | None = None
        self.window_highlight: WindowCaptureHighlight | None = None
        self.worker: PipelineTask | None = None

        self._last_capture: CaptureInfo | None = None
        self._last_window_hwnd = 0
        self._overlay_visible = True
        self._busy = False
        self._last_external_hwnd = 0
        self._capture_session: _CaptureSession | None = None
        self._pending_capture_mode = ""
        self._pending_capture_bbox: tuple[int, int, int, int] | None = None
        self._pending_window_hwnd = 0
        self._highlight_window_rect: tuple[int, int, int, int] | None = None
        self._window_highlight_retries = 0
        self._pipeline_succeeded = False
        self._pipeline_error = ""
        self._shutting_down = False
        self._shutdown_finalized = False

        self._capture_timer = QTimer(self)
        self._capture_timer.setSingleShot(True)
        self._capture_timer.timeout.connect(self._execute_pending_capture)
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.timeout.connect(self._show_selection)
        self._foreground_tracker = QTimer(self)
        self._foreground_tracker.setInterval(200)
        self._foreground_tracker.timeout.connect(self._remember_external_foreground)

        self._monitor_active = False
        self._monitor_request: dict | None = None
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setSingleShot(True)
        self._monitor_timer.timeout.connect(self._monitor_cycle)
        self._monitor_known: dict[str, str] = {}
        self._monitor_signature = ""
        self._monitor_frame_hash = ""
        self._monitor_errors = 0
        self._monitor_no_text = False
        self._last_original_text = ""
        self._last_translated_text = ""
        self._monitor_region_selector = None

        try:
            self.qt_app.styleHints().colorSchemeChanged.connect(
                self._on_system_color_scheme_changed
            )
        except (AttributeError, RuntimeError):
            pass

        self.hotkey_manager.triggered.connect(self.on_hotkey)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        try:
            self._refresh_appearance()
            self._build_monitor_map()
            self.cache = TranslationCache(
                path=Path(self.config.path.parent) / "translation_cache.json",
                ttl_days=self.config.get("translation.cache_ttl_days", 30),
                max_entries=self.config.get("translation.cache_max_entries", 2000),
            )
            self.ocr_engine = self._create_configured_ocr_engine()
            resolved_service = self._resolve_translator_service()
            if resolved_service != self.config.get("translation.service", "mock"):
                self.config.set("translation.service", resolved_service)
                self.config.save()
                self.set_status(f"检测到 API Key，已自动切换到 {resolved_service} 翻译服务")
            self.translator = create_translator(
                self.config.get("translation.service", "mock"),
                self.config.section("translation"),
                cache=self.cache,
                api_key_resolver=self.config.api_key,
            )
            self.overlay_manager = OverlayManager(self.config)
            self.overlay_manager.set_monitor_map(self.monitor_map)

            self.window = MainWindow(self)
            self.tray = TrayIcon(self)
            capture_available = window_capture_available()
            capture_reason = "当前平台暂不支持原生窗口捕获"
            self.window.set_window_capture_available(capture_available, capture_reason)
            self.tray.set_window_capture_available(capture_available, capture_reason)
            self.window.show()
            self.tray.show()
            self._remember_external_foreground()
            self._foreground_tracker.start()
            self._apply_hotkeys()
            self._apply_autostart()
            self._warmup()
            if (not self.config.get("general.first_run_done", False)
                    and os.environ.get("SCREEN_TRANSLATOR_SELFTEST") != "1"):
                QTimer.singleShot(500, self._show_welcome)
        except Exception as exc:
            log.exception("应用启动失败")
            QMessageBox.critical(None, "启动失败", f"{exc}")

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            self.screenshot_service.close()
        except Exception:
            pass
        self._foreground_tracker.stop()
        self._capture_timer.stop()
        self._selection_timer.stop()
        self._monitor_timer.stop()
        self._monitor_active = False
        self._dispose_monitor_region()
        self._dispose_selection()
        self._dispose_window_highlight()
        self.floating_status.hide_immediate()
        self.hotkey_manager.stop()
        if self.overlay_manager is not None:
            self.overlay_manager.hide_all()
        if self.cache is not None:
            self.cache.flush()
        if self.window is not None:
            self.window.hide()
        if self.tray is not None:
            self.tray.hide()
        if self.worker is not None and self.worker.isRunning():
            self.worker.finished.connect(
                self._maybe_finalize_shutdown, Qt.ConnectionType.UniqueConnection
            )
            self.worker.cancel()
            self.set_status("正在安全结束当前任务…")
        self._maybe_finalize_shutdown()

    def _maybe_finalize_shutdown(self) -> None:
        if not self._shutting_down or self._shutdown_finalized:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._finalize_shutdown()

    def _finalize_shutdown(self) -> None:
        if self._shutdown_finalized:
            return
        self._shutdown_finalized = True
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self.qt_app.quit()

    def _warmup(self) -> None:
        """后台线程预热 OCR 模型，避免第一次截图时卡很久。"""
        if os.environ.get("SCREEN_TRANSLATOR_NO_WARMUP") == "1" or os.environ.get(
            "SCREEN_TRANSLATOR_SELFTEST"
        ) == "1":
            return

        def job() -> None:
            try:
                if self.ocr_engine is not None:
                    self.ocr_engine.warmup()
            except Exception as exc:
                log.warning("OCR 模型预热失败：%s", exc)

        import threading

        threading.Thread(target=job, daemon=True, name="ocr-warmup").start()

    def _remember_external_foreground(self) -> None:
        """保留最近一次非本程序前台窗口，供点击“当前窗口”后稳定捕获。"""
        try:
            hwnd = get_foreground_window()
            if hwnd and not is_current_process_window(hwnd) and is_window_capturable(hwnd):
                self._last_external_hwnd = hwnd
        except Exception as exc:
            log.debug("读取前台窗口失败：%s", exc)

    def _resolve_window_target(self) -> int:
        """Freeze the best live external window before the capture transition."""
        candidates = (get_foreground_window(), self._last_external_hwnd)
        for hwnd in candidates:
            try:
                if hwnd and not is_current_process_window(hwnd) and is_window_capturable(hwnd):
                    self._last_external_hwnd = hwnd
                    return hwnd
            except Exception:
                continue
        return 0

    # ------------------------------------------------------------------ dpi/monitor
    def _build_monitor_map(self) -> None:
        self.monitor_map = dpi_utils.build_monitor_map(
            self.qt_app.screens(), dpi_utils.enum_display_monitors_physical()
        )

    def refresh_monitor_map(self) -> None:
        self._build_monitor_map()
        if self.overlay_manager is not None:
            self.overlay_manager.set_monitor_map(self.monitor_map)

    # ------------------------------------------------------------------ actions
    def on_hotkey(self, action: str) -> None:
        log.debug("快捷键触发：%s", action)
        if action == "capture_region":
            self.start_region_capture()
        elif action == "capture_fullscreen":
            self.start_capture("fullscreen")
        elif action == "capture_window":
            self.start_capture("window")
        elif action == "toggle_overlay":
            self.toggle_overlay()
        elif action == "refresh":
            self.refresh()
        elif action == "monitor_toggle":
            self.toggle_monitor()

    # ------------------------------------------------------------------ onboarding / help
    def _show_welcome(self) -> None:
        """首次启动时弹出的三步引导；用户点“开始使用”后写入完成标志。"""
        if self.window is None:
            return
        from ui.welcome_dialog import WelcomeDialog

        def finish() -> None:
            self.config.set("general.first_run_done", True)
            self.config.save()

        WelcomeDialog(self.window, self.config.hotkeys(), on_finish=finish).exec()

    def show_about(self) -> None:
        """打开“关于”对话框（含版本、仓库与快捷键速查）。"""
        from ui.about_dialog import AboutDialog

        AboutDialog(self.window, self.config).exec()

