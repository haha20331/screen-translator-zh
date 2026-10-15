"""设置对话框：通用 / 个性化 / OCR / 翻译 / 覆盖层 / 快捷键。"""

from __future__ import annotations

import copy
import queue
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QPropertyAnimation, QRect, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.config import DEFAULTS, AppConfig
from app.hotkeys import HotkeyError, normalize_hotkey
from app.logger import app_data_dir, get_logger
from app.version import __version__
from services.diagnostics import DiagnosticsExportError, export_diagnostics
from services.ocr.base import list_ocr_engines
from services.transcript import resolve_directory
from services.translation.base import list_translators
from services.translation.factory import service_display_name
from services.translation.model_names import how_to_get
from services.translation.offline_translator import OFFLINE_MODELS, OfflineTranslator, llama_model_available
from services.translation.speed_test import (
    SPEED_TEST_SENTENCES,
    STATUS_ERROR,
    STATUS_MEMORY,
    STATUS_MISSING,
    STATUS_OK,
    SpeedTestStore,
    describe_machine,
    run_speed_test,
    short_label,
)
from ui.appearance import (
    ACCENT_PRESETS,
    DENSITY_PRESETS,
    MOTION_PRESETS,
    PALETTE_PRESETS,
    SURFACE_PRESETS,
    current_tokens,
)
from ui.motion import BASE, ENTER_EASING, EXIT_EASING, MOVE_EASING, SLOW, motion_duration
from ui.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider, NoWheelSpinBox
from ui.overlay_preview import OverlayPreview
from ui.personalization import AppearancePreview, PersonalizationChoiceRow
from utils import stability
from utils.language_utils import LANGUAGES

log = get_logger("settings")

_HOTKEY_ACTIONS = [
    ("capture_region", "框選翻譯"),
    ("capture_fullscreen", "全屏翻譯"),
    ("capture_window", "當前視窗翻譯"),
    ("toggle_overlay", "隱藏/顯示譯文"),
    ("refresh", "重新識別翻譯"),
    ("monitor_toggle", "即時監控"),
]


class SettingsGlyph(QLabel):
    """The canonical application icon, reused without redrawing it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(58, 58)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
        self._icon_source = root / "assets" / "app_launch_v4.png"
        source = QPixmap(str(self._icon_source))
        if not source.isNull():
            self.setPixmap(
                source.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.setAccessibleName("螢幕翻譯")

    def set_lifecycle_progress(self, _progress: float) -> None:
        """Keep the canonical mark still while the surrounding page moves."""


# 離線翻譯模型的選項：(顯示名稱, model_name)；auto 由翻譯器依硬體挑模型
OFFLINE_MODEL_CHOICES = (
    ("自動（預設 3B Q4_K_M，沒有下載就用 1.5B；有獨立顯示卡且放了 7B 時用 7B）", "auto"),
    *((model.label, model.name) for model in OFFLINE_MODELS),
)



def _speed_test_thread(section: dict, models: list, cancel: threading.Event, messages: queue.Queue, store: SpeedTestStore) -> None:
    """背景執行緒：跑速度測試，進度與結果丟進佇列給畫面，每個模型測完就存檔。不碰任何介面物件，設定頁關掉也能安全跑完。"""
    machine = describe_machine()

    def saved(result: dict) -> None:
        store.save_result(result, machine)
        messages.put(("result", result))

    try:
        run_speed_test(
            section,
            models,
            progress=lambda text: messages.put(("progress", text)),
            on_start=lambda name: messages.put(("start", name)),
            on_result=saved,
            cancelled=cancel.is_set,
        )
    except Exception as exc:  # 預期外的錯誤：告訴使用者，而不是讓畫面永遠停在「測試中」
        log.exception("速度測試中斷")
        messages.put(("error", str(exc)))
    finally:
        messages.put(("done", cancel.is_set()))

class SettingsDialog(QWidget):
    accepted = Signal()
    exit_requested = Signal(int)
    finished = Signal(int)
    def __init__(
        self, config: AppConfig, parent=None, cache_actions=None, speed_test_guard=None, speed_test_hooks=None
    ) -> None:
        super().__init__(parent)
        self.config = config
        # (數量, 清除) 兩個函式，由應用程式提供；沒有時（例如單獨開設定視窗）「清除翻譯快取」不能用
        self._cache_actions = cache_actions
        # 速度測試前先問應用程式現在能不能測：回傳不是空字串就是「不能測」的原因（例如即時監控中，會搶 CPU 讓數字不準）
        self._speed_guard = speed_test_guard
        # (開始前, 結束後) 兩個函式：測試要載入好幾個模型，開始前先請應用程式把自己載入的翻譯模型放掉（騰出記憶體），結束後再載回來
        self._speed_hooks = speed_test_hooks
        self._speed_store = SpeedTestStore(Path(config.path.parent) / "speed_test.json")
        self._speed_results: dict[str, dict] = self._speed_store.load()["models"]  # 上次（以及更早）測的結果，開設定就看得到
        self._speed_thread: threading.Thread | None = None
        self._speed_cancel = threading.Event()
        self._speed_messages: queue.Queue = queue.Queue()
        self._speed_running: str = ""  # 正在測的模型名稱
        self._speed_testing = False
        self.destroyed.connect(self._speed_cancel.set)  # 設定頁被關掉：測試做完手上這個模型就停（結果照樣存檔）
        self._speed_timer = QTimer(self)
        self._speed_timer.setInterval(150)
        self._speed_timer.timeout.connect(self._poll_speed_test)
        self.setObjectName("SettingsDialog")
        self.setWindowTitle("設定")
        self.setMinimumSize(900, 610)
        self.resize(1020, 680)
        self._page_sweep_in: QPropertyAnimation | None = None
        self._page_sweep_out: QPropertyAnimation | None = None
        self._active_page = 0
        self._embedded = False
        self._nav_initialized = False
        self._pending_page = 0
        self._queued_page: int | None = None
        self._page_transition_active = False
        self._sweep_moving_down = True
        self._exit_pending = False
        self._pending_exit_result = -1
        self._close_intent = -1
        self._lifecycle_progress = 0.0
        self._build_ui()
        self._load()
        self.refresh_appearance()

    def refresh_appearance(self) -> None:
        tokens = current_tokens()
        for button in self.nav_buttons:
            button.setMinimumHeight(tokens.nav_height)
            button.setMaximumHeight(tokens.nav_height)
        for card in self.findChildren(QFrame):
            if card.objectName() != "SettingsCard" or card.layout() is None:
                continue
            padding = tokens.card_padding
            card.layout().setContentsMargins(padding, max(12, padding - 4), padding, padding)
        for page_index in range(self.pages.count()):
            scroll = self.pages.widget(page_index)
            page = scroll.widget() if isinstance(scroll, QScrollArea) else None
            if page is not None and page.layout() is not None:
                page.layout().setSpacing(tokens.page_spacing)
        self._page_sweep.update()
        self._settings_glyph.update()
        self.update()

    def show_inside(self, parent: QWidget) -> None:
        self._embedded = True
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setMinimumSize(0, 0)
        parent.installEventFilter(self)
        self.setGeometry(parent.contentsRect())
        self.show()
        self.raise_()

    def eventFilter(self, watched, event) -> bool:
        if self._embedded and watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.contentsRect())
        return super().eventFilter(watched, event)

    def accept(self) -> None:
        if self._exit_pending:
            return
        self._commit_timer.stop()
        self.accepted.emit()
        self._request_exit(1)

    def reject(self) -> None:
        if self._exit_pending:
            return
        self._commit_timer.stop()
        self._save_close_timer.stop()
        self._pending_hotkeys = None
        self._request_exit(0)

    @property
    def exit_pending(self) -> bool:
        return self._exit_pending

    @property
    def pending_exit_result(self) -> int:
        return self._pending_exit_result

    @property
    def close_intent(self) -> int:
        """Requested result, including the short saved-confirmation hold."""
        return self._close_intent

    def set_lifecycle_progress(self, progress: float) -> None:
        self._lifecycle_progress = max(0.0, min(1.0, float(progress)))
        self._settings_glyph.set_lifecycle_progress(self._lifecycle_progress)

    def _request_exit(self, result: int) -> None:
        self._settle_page_transition_for_exit()
        self._exit_pending = True
        self._pending_exit_result = int(result)
        self._close_intent = int(result)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.exit_requested.emit(int(result))

    def complete_exit(self, result: int) -> None:
        if not self._exit_pending or int(result) != self._pending_exit_result:
            return
        self._exit_pending = False
        self._pending_exit_result = -1
        self._close_intent = -1
        self.finished.emit(int(result))
        self.hide()

    def cancel_exit(self) -> bool:
        if not self._exit_pending or self._pending_exit_result != 0:
            return False
        self._exit_pending = False
        self._pending_exit_result = -1
        self._close_intent = -1
        self._settings_shell.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_cancel.setText("取消")
        self.btn_save.setProperty("saved", False)
        self.btn_save.setText("儲存設定")
        self.btn_save.style().unpolish(self.btn_save)
        self.btn_save.style().polish(self.btn_save)
        return True

    def _settle_page_transition_for_exit(self) -> None:
        self._page_sweep_in.stop()
        self._page_sweep_out.stop()
        self._page_sweep.hide()
        self._page_transition_active = False
        self._queued_page = None
        current = self.pages.currentIndex()
        self._active_page = current
        self._pending_page = current
        self._pending_nav_index = current
        self._nav_move_timer.stop()
        for position, button in enumerate(self.nav_buttons):
            button.blockSignals(True)
            button.setChecked(position == current)
            button.blockSignals(False)
        self._move_nav_indicator(current)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(14)
        self._settings_glyph = SettingsGlyph()
        self._settings_glyph.set_lifecycle_progress(self._lifecycle_progress)
        header.addWidget(self._settings_glyph)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        title = QLabel("設定")
        title.setObjectName("SettingsTitle")
        subtitle = QLabel("按你的使用方式，安靜地完成每一次翻譯")
        subtitle.setObjectName("SettingsSubtitle")
        heading.addStretch(1)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        heading.addStretch(1)
        header.addLayout(heading)
        header.addStretch(1)
        layout.addLayout(header)

        shell = QFrame()
        self._settings_shell = shell
        shell.setObjectName("SettingsShell")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(10, 12, 14, 12)
        shell_layout.setSpacing(14)
        nav_host = QWidget()
        nav = QVBoxLayout(nav_host)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(6)
        self.nav_buttons: list[QPushButton] = []
        for label in ("通用", "個性化", "OCR", "翻譯", "譯文顯示", "快捷鍵"):
            button = QPushButton(label)
            button.setObjectName("SettingsNavButton")
            button.setCheckable(True)
            button.setMinimumSize(144, 42)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, b=button: self._select_page(self.nav_buttons.index(b)))
            self.nav_buttons.append(button)
            nav.addWidget(button)
        nav.addStretch(1)
        self._nav_indicator = QFrame(nav_host)
        self._nav_indicator.setObjectName("SettingsNavIndicator")
        self._nav_indicator.raise_()
        self._nav_animation = QPropertyAnimation(self._nav_indicator, b"geometry", self)
        self._nav_animation.setDuration(SLOW)
        self._nav_animation.setEasingCurve(MOVE_EASING)
        self._pending_nav_index = 0
        self._nav_move_timer = QTimer(self)
        self._nav_move_timer.setSingleShot(True)
        self._nav_move_timer.timeout.connect(self._move_pending_nav_indicator)
        shell_layout.addWidget(nav_host)
        divider = QFrame()
        divider.setObjectName("SettingsDivider")
        divider.setFrameShape(QFrame.Shape.VLine)
        shell_layout.addWidget(divider)

        page_host = QFrame()
        page_host.setObjectName("SettingsPageHost")
        page_host_layout = QVBoxLayout(page_host)
        page_host_layout.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        self.pages.setObjectName("SettingsPages")
        for page in (
            self._build_general_tab(), self._build_personalization_tab(),
            self._build_ocr_tab(), self._build_translation_tab(),
            self._build_overlay_tab(), self._build_hotkey_tab(),
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(page)
            self.pages.addWidget(scroll)
        page_host_layout.addWidget(self.pages)
        # Preserve the original full-page curtain: it covers the old page,
        # hands over at center, then exits in the navigation direction.
        self._page_sweep = QFrame(page_host)
        self._page_sweep.setObjectName("SettingsPageCurtain")
        self._page_sweep.hide()
        self._page_sweep.raise_()
        self._page_sweep_in = QPropertyAnimation(self._page_sweep, b"geometry", self)
        self._page_sweep_in.setDuration(BASE)
        self._page_sweep_in.setEasingCurve(ENTER_EASING)
        self._page_sweep_in.finished.connect(self._reveal_sweep_page)
        self._page_sweep_out = QPropertyAnimation(self._page_sweep, b"geometry", self)
        self._page_sweep_out.setDuration(BASE)
        self._page_sweep_out.setEasingCurve(EXIT_EASING)
        self._page_sweep_out.finished.connect(self._finish_page_switch)
        shell_layout.addWidget(page_host, 1)
        layout.addWidget(shell, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(10)
        footer.addStretch(1)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("DialogCancelButton")
        self.btn_save = QPushButton("儲存設定")
        self.btn_save.setObjectName("DialogSaveButton")
        for button in (self.btn_cancel, self.btn_save):
            button.setMinimumSize(116, 42)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_save.clicked.connect(self._on_accept)
        self._save_animation = QPropertyAnimation(self.btn_save, b"minimumWidth", self)
        self._save_animation.setDuration(SLOW)
        self._save_animation.setEasingCurve(ENTER_EASING)
        self._save_close_timer = QTimer(self)
        self._save_close_timer.setSingleShot(True)
        self._save_close_timer.timeout.connect(self.accept)
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.timeout.connect(self._commit_pending)
        self._pending_hotkeys: dict[str, str] | None = None
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_save)
        layout.addLayout(footer)
        self.combo_placement.currentIndexChanged.connect(self._sync_placement)
        self.combo_display_mode.currentIndexChanged.connect(self._sync_display_mode)
        self.chk_keep_original.toggled.connect(self._sync_keep_original)
        self._select_page(0)

    def _select_page(self, index: int) -> None:
        for position, button in enumerate(self.nav_buttons):
            button.blockSignals(True)
            button.setChecked(position == index)
            button.blockSignals(False)
        self._pending_nav_index = index
        self._nav_move_timer.start(0)

        if self._page_transition_active:
            # Keep one full curtain in flight. Re-clicking its target must not
            # expose the new page early; a different target becomes the next
            # transition and is measured from the page actually revealed.
            self._queued_page = None if index == self._pending_page else index
            return

        self._active_page = self.pages.currentIndex()
        if index == self._active_page:
            return
        self._run_page_switch(index, index > self._active_page)

    def _run_page_switch(self, index: int, moving_down: bool) -> None:
        """Switch behind the original full-page sliding curtain."""
        enter_duration = motion_duration(BASE, large_surface=True)
        if enter_duration == 0:
            self._page_sweep_in.stop()
            self._page_sweep_out.stop()
            self.pages.setCurrentIndex(index)
            self._active_page = index
            self._pending_page = index
            self._queued_page = None
            self._page_transition_active = False
            self._page_sweep.hide()
            return
        host = self._page_sweep.parentWidget()
        width = max(1, host.width())
        height = max(1, host.height())
        start_x = width if moving_down else -width
        self._page_sweep_in.stop()
        self._page_sweep_out.stop()
        self._pending_page = index
        self._page_transition_active = True
        self._sweep_moving_down = moving_down
        start = QRect(start_x, 0, width, height)
        covered = QRect(0, 0, width, height)
        self._page_sweep.setGeometry(start)
        self._page_sweep.show()
        self._page_sweep.raise_()
        self._page_sweep_in.setDuration(enter_duration)
        self._page_sweep_in.setStartValue(start)
        self._page_sweep_in.setEndValue(covered)
        self._page_sweep_in.start()

    def _reveal_sweep_page(self) -> None:
        self.pages.setCurrentIndex(self._pending_page)
        self._active_page = self._pending_page
        host = self._page_sweep.parentWidget()
        width = max(1, host.width())
        height = max(1, host.height())
        end_x = -width if self._sweep_moving_down else width
        self._page_sweep_out.setDuration(motion_duration(BASE, large_surface=True))
        self._page_sweep_out.setStartValue(QRect(0, 0, width, height))
        self._page_sweep_out.setEndValue(QRect(end_x, 0, width, height))
        self._page_sweep_out.start()

    def _finish_page_switch(self) -> None:
        self._page_sweep.hide()
        self._page_transition_active = False
        self._active_page = self.pages.currentIndex()
        queued_page = self._queued_page
        self._queued_page = None
        if queued_page is None or queued_page == self._active_page:
            return
        self._run_page_switch(queued_page, queued_page > self._active_page)

    def _move_nav_indicator(self, index: int) -> None:
        button = self.nav_buttons[index]
        target = button.geometry().adjusted(0, 7, -button.width() + 4, -7)
        if not self._nav_initialized:
            self._nav_initialized = True
            self._nav_indicator.setGeometry(target)
            self._nav_indicator.show()
            return
        self._nav_animation.stop()
        duration = motion_duration(SLOW)
        if duration == 0:
            self._nav_indicator.setGeometry(target)
            self._nav_indicator.show()
            return
        self._nav_animation.setDuration(duration)
        start = self._nav_indicator.geometry()
        top = min(start.top(), target.top())
        bottom = max(start.bottom(), target.bottom())
        stretched = QRect(target.x(), top, target.width(), bottom - top + 1)
        self._nav_animation.setKeyValues(
            [(0.0, start), (0.45, stretched), (0.62, stretched), (1.0, target)]
        )
        self._nav_animation.start()

    def _move_pending_nav_indicator(self) -> None:
        self._move_nav_indicator(self._pending_nav_index)

    def _on_cancel(self) -> None:
        self._commit_timer.stop()
        self._save_close_timer.stop()
        self._pending_hotkeys = None
        self.btn_cancel.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_cancel.setText("返回中")
        self.reject()

    # ------------------------------------------------------------- tabs
    @staticmethod
    def _new_page() -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 6, 10, 10)
        layout.setSpacing(12)
        return page, layout

    @staticmethod
    def _new_form() -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(22)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        return form

    @staticmethod
    def _card(title: str, description: str, body: QLayout) -> QFrame:
        card = QFrame()
        card.setObjectName("SettingsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("SettingsCardTitle")
        description_label = QLabel(description)
        description_label.setObjectName("SettingsCardDescription")
        description_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addSpacing(2)
        layout.addLayout(body)
        return card

    def _build_general_tab(self) -> QWidget:
        tab, layout = self._new_page()
        self.chk_autostart = QCheckBox("開機自動啟動")
        self.chk_tray = QCheckBox("關閉主視窗時最小化到托盤")
        self.chk_history = QCheckBox("儲存截圖與識別歷史")
        startup_form = self._new_form()
        startup_form.addRow(self.chk_autostart)
        startup_form.addRow(self.chk_tray)
        layout.addWidget(self._card("啟動與後臺", "決定應用如何進入工作狀態與停留在系統托盤。", startup_form))

        self.spin_scan_interval = NoWheelDoubleSpinBox()
        self.spin_scan_interval.setRange(stability.MIN_INTERVAL_MS / 1000, stability.MAX_INTERVAL_MS / 1000)
        self.spin_scan_interval.setSingleStep(0.05)
        self.spin_scan_interval.setDecimals(2)  # 可輸入到小數兩位（0.01 秒）
        self.spin_scan_interval.setSuffix(" 秒")
        self.spin_scan_interval.setToolTip(
            "每一輪檢查結束後，等這麼久再檢查下一次畫面（最短 0.1 秒，可輸入到小數兩位）。"
            "間隔越短，字幕一換就越快反應，但電腦負擔越重"
            "（影片畫面一直在動時，每次檢查都要重新辨識文字，0.1 秒幾乎是不停地辨識）；"
            "間隔越長越省資源，但換字幕要多等一下。畫面完全沒變時不會辨識，也不會翻譯。"
        )
        self.lbl_scan_rate = QLabel()
        self.lbl_scan_rate.setObjectName("SettingsCardDescription")
        self.spin_scan_interval.valueChanged.connect(self._refresh_scan_rate)
        scan_row = QHBoxLayout()
        scan_row.setSpacing(10)
        scan_row.addWidget(self.spin_scan_interval)
        scan_row.addWidget(self.lbl_scan_rate)
        scan_row.addStretch(1)
        self.spin_hide_no_text = NoWheelDoubleSpinBox()
        self.spin_hide_no_text.setRange(0.0, stability.MAX_NO_TEXT_SECONDS)
        self.spin_hide_no_text.setSingleStep(0.5)
        self.spin_hide_no_text.setDecimals(1)
        self.spin_hide_no_text.setSuffix(" 秒")
        self.spin_hide_no_text.setSpecialValueText("不自動關閉")
        self.spin_hide_no_text.setToolTip(
            "監控中，OCR 連續這麼久都抓不到任何文字（字幕消失了、換成沒有字的畫面），就自動把譯文關掉，"
            "不會一直留著上一句。字幕再出現時會重新翻譯顯示。0 = 不自動關閉。"
        )
        self.spin_confirm_rounds = NoWheelSpinBox()
        self.spin_confirm_rounds.setRange(stability.MIN_ROUNDS, stability.MAX_ROUNDS)
        self.spin_confirm_rounds.setSuffix(" 次")
        self.spin_confirm_rounds.setToolTip(
            "新字幕要連續辨識到幾次相同的結果，才翻譯並顯示。次數越少反應越快，但字幕淡入淡出、"
            "換字幕的過渡畫面可能先閃出殘缺的句子；次數越多越不會閃怪句子，但換字幕越慢"
            "（每多一次約慢 0.5～1 秒）。辨識結果與目前顯示的相同時，不論次數都不會重複翻譯。"
        )
        self.spin_max_wait = NoWheelDoubleSpinBox()
        self.spin_max_wait.setRange(stability.MIN_MAX_WAIT, stability.MAX_MAX_WAIT)
        self.spin_max_wait.setSingleStep(0.5)
        self.spin_max_wait.setDecimals(1)
        self.spin_max_wait.setSuffix(" 秒")
        self.spin_max_wait.setToolTip("新字幕一直在變（辨識結果抖動）時，最久等這麼久就直接顯示最新的一句，避免畫面卡住不更新。")
        self.btn_preset_fast = QPushButton("套用快速推薦")
        self.btn_preset_stable = QPushButton("套用穩定推薦")
        for button in (self.btn_preset_fast, self.btn_preset_stable):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preset_fast.clicked.connect(lambda: self._apply_stability_preset(stability.RECOMMENDED_FAST_ROUNDS))
        self.btn_preset_stable.clicked.connect(lambda: self._apply_stability_preset(stability.RECOMMENDED_STABLE_ROUNDS))
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_row.addWidget(self.btn_preset_fast)
        preset_row.addWidget(self.btn_preset_stable)
        preset_row.addStretch(1)
        self.chk_auto_monitor = QCheckBox("框選（或全螢幕、視窗）翻譯完成後，自動進入即時監控（推薦）")
        self.chk_auto_monitor.setToolTip(
            "開啟時，翻譯完成後會接著持續監控，畫面上的字幕一變就自動重譯；關閉即時監控或開始新的擷取即可停止。\n"
            "關閉這個選項則只翻譯一次，要監控請手動按「即時監控」或用快捷鍵。"
        )
        monitor_form = self._new_form()
        monitor_form.addRow(self.chk_auto_monitor)
        monitor_form.addRow("辨識間隔", scan_row)
        monitor_form.addRow("無字幕時關閉譯文", self.spin_hide_no_text)
        monitor_form.addRow("確認次數", self.spin_confirm_rounds)
        monitor_form.addRow("最久等待", self.spin_max_wait)
        monitor_form.addRow(preset_row)
        explanation = QLabel(
            f"系統推薦：\n"
            f"．辨識間隔：{stability.RECOMMENDED_INTERVAL_MS / 1000:g} 秒（約每秒檢查 10 次）。"
            "想省電腦資源就拉長，想讓換字幕更快反應就縮短（最短 0.1 秒，可輸入到小數兩位）。\n"
            f"．無字幕時關閉譯文：{stability.RECOMMENDED_NO_TEXT_SECONDS:g} 秒。OCR 連續抓不到字幕這麼久，就自動關掉譯文。\n"
            f"．快速：確認 {stability.RECOMMENDED_FAST_ROUNDS} 次，最久等待 {stability.RECOMMENDED_MAX_WAIT:g} 秒。"
            "辨識到就立刻顯示，反應最快，但字幕淡入淡出時可能先閃出殘缺的句子。\n"
            f"．穩定：確認 {stability.RECOMMENDED_STABLE_ROUNDS} 次，最久等待 {stability.RECOMMENDED_MAX_WAIT:g} 秒。"
            "同一句連續辨識到兩次才顯示，不會閃出辨識到一半的怪句子，換字幕約慢 0.5～1 秒。"
        )
        explanation.setObjectName("SettingsCardDescription")
        explanation.setWordWrap(True)
        monitor_form.addRow(explanation)
        layout.addWidget(self._card("即時監控", "翻譯完成後是否自動持續監控、多久檢查一次畫面（辨識間隔），以及新字幕要連續辨識到相同結果幾次才翻譯並顯示（確認次數）。", monitor_form))

        self.history_dir = QLineEdit()
        pick = QPushButton("選擇目錄")
        pick.clicked.connect(self._pick_history_dir)
        row = QHBoxLayout()
        row.addWidget(self.history_dir, 1)
        row.addWidget(pick)
        history_form = self._new_form()
        history_form.addRow(self.chk_history)
        history_form.addRow("歷史目錄", row)
        layout.addWidget(self._card("本地記錄", "歷史記錄只儲存在你選擇的本機目錄。", history_form))

        self.chk_output = QCheckBox("把字幕與譯文寫進 txt 檔（一句字幕一行）")
        self.chk_output.setToolTip(
            "每句字幕結束（被下一句取代、消失、停止監控、離開程式）時，寫一行：開始與結束時間、原文、譯文、狀態、"
            "翻譯用的模型、四角點座標（螢幕絕對座標，物理像素）。一天一個檔案，附加寫入，寫在背景，不會拖慢翻譯。"
        )
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText(str(self._default_output_dir()))
        pick_output = QPushButton("選擇目錄")
        pick_output.clicked.connect(self._pick_output_dir)
        self.btn_open_output = QPushButton("開啟資料夾")
        self.btn_open_output.clicked.connect(self._open_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(pick_output)
        output_row.addWidget(self.btn_open_output)
        output_note = QLabel(
            "檔名是「日期.txt」，UTF-8、欄位以 TAB 分隔，可以用記事本或 Excel 開啟。"
            "留空 = 預設資料夾。檔案會記下框內出現的所有文字，包括聊天或密碼畫面，請自行留意存放位置。"
        )
        output_note.setObjectName("SettingsCardDescription")
        output_note.setWordWrap(True)
        output_form = self._new_form()
        output_form.addRow(self.chk_output)
        output_form.addRow("輸出資料夾", output_row)
        output_form.addRow(output_note)
        layout.addWidget(self._card("文字輸出", "辨識到的字幕、譯文與座標，附加寫入 txt 檔。", output_form))

        self.btn_export_diagnostics = QPushButton("匯出診斷日誌")
        self.btn_export_diagnostics.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_diagnostics.clicked.connect(self._export_diagnostics)
        diagnostics_body = QHBoxLayout()
        diagnostics_body.setContentsMargins(0, 0, 0, 0)
        diagnostics_body.addWidget(QLabel("僅包含脫敏配置、執行環境和最近日誌。"), 1)
        diagnostics_body.addWidget(self.btn_export_diagnostics)
        layout.addWidget(
            self._card(
                "診斷與支援",
                "不會打包截圖、翻譯歷史、模型檔案或 API Key。",
                diagnostics_body,
            )
        )
        layout.addStretch(1)
        return tab

    def _export_diagnostics(self) -> None:
        suggested = app_data_dir() / f"ScreenTranslator-diagnostics-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        target, _ = QFileDialog.getSaveFileName(
            self, "匯出診斷日誌", str(suggested), "ZIP 壓縮包 (*.zip)"
        )
        if not target:
            return
        try:
            exported = export_diagnostics(target, self.config, app_version=__version__)
        except DiagnosticsExportError as exc:
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        QMessageBox.information(self, "匯出完成", f"診斷日誌已儲存到：\n{exported}")

    def _build_personalization_tab(self) -> QWidget:
        tab, layout = self._new_page()

        self.palette_choices = PersonalizationChoiceRow(
            list(PALETTE_PRESETS), mode="palette"
        )
        palette_body = QVBoxLayout()
        palette_body.setContentsMargins(0, 0, 0, 0)
        palette_body.addWidget(self.palette_choices)
        layout.addWidget(
            self._card(
                "主題氛圍",
                "每套配色都圍繞暖白、深灰線稿與短下劃線重新校準。",
                palette_body,
            )
        )

        accent_specs = [(value, label, "") for value, label in ACCENT_PRESETS]
        self.accent_choices = PersonalizationChoiceRow(accent_specs, mode="accent")
        accent_body = QVBoxLayout()
        accent_body.setContentsMargins(0, 0, 0, 0)
        accent_body.addWidget(self.accent_choices)
        layout.addWidget(
            self._card(
                "介面強調色",
                "同步改變按鈕、狀態、短線反饋與框選邊界，保持整套視覺一致。",
                accent_body,
            )
        )

        self.motion_choices = PersonalizationChoiceRow(list(MOTION_PRESETS))
        self.chk_reduce_motion = QCheckBox("減少動態效果（關閉位移與連續動畫）")
        motion_body = QVBoxLayout()
        motion_body.setContentsMargins(0, 0, 0, 0)
        motion_body.setSpacing(10)
        motion_body.addWidget(self.motion_choices)
        motion_body.addWidget(self.chk_reduce_motion)
        layout.addWidget(
            self._card(
                "動效節奏",
                "所有動畫都會停止在明確狀態，不使用掃描線或無意義迴圈。",
                motion_body,
            )
        )

        self.density_choices = PersonalizationChoiceRow(
            list(DENSITY_PRESETS), compact=True
        )
        self.surface_choices = PersonalizationChoiceRow(
            list(SURFACE_PRESETS), compact=True
        )
        layout_body = QVBoxLayout()
        layout_body.setContentsMargins(0, 0, 0, 0)
        layout_body.setSpacing(10)
        layout_body.addWidget(self.density_choices)
        layout_body.addWidget(self.surface_choices)
        layout.addWidget(
            self._card(
                "密度與表面",
                "改變控制元件呼吸感與層級，不使用廉價玻璃擬態。",
                layout_body,
            )
        )

        self.personalization_preview = AppearancePreview()
        self.personalization_hint = QLabel("即時預覽 · 儲存後應用到主介面")
        self.personalization_hint.setObjectName("PersonalizationHint")
        self.btn_replay_personalization = QPushButton("重播動效")
        self.btn_replay_personalization.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_personalization = QPushButton("恢復預設")
        self.btn_reset_personalization.setObjectName("PersonalizationResetButton")
        self.btn_reset_personalization.setCursor(Qt.CursorShape.PointingHandCursor)
        preview_actions = QHBoxLayout()
        preview_actions.setContentsMargins(0, 0, 0, 0)
        preview_actions.addWidget(self.personalization_hint, 1)
        preview_actions.addWidget(self.btn_replay_personalization)
        preview_actions.addWidget(self.btn_reset_personalization)
        preview_body = QVBoxLayout()
        preview_body.setContentsMargins(0, 0, 0, 0)
        preview_body.setSpacing(10)
        preview_body.addWidget(self.personalization_preview)
        preview_body.addLayout(preview_actions)
        layout.addWidget(
            self._card(
                "即時體驗",
                "每次選擇都會重演一次“框選—識別—譯文出現”，不會常駐迴圈。",
                preview_body,
            )
        )

        for choices in (
            self.palette_choices,
            self.accent_choices,
            self.motion_choices,
            self.density_choices,
            self.surface_choices,
        ):
            choices.valueChanged.connect(self._refresh_personalization_preview)
        self.chk_reduce_motion.toggled.connect(self._refresh_personalization_preview)
        self.btn_replay_personalization.clicked.connect(self.personalization_preview.replay)
        self.btn_reset_personalization.clicked.connect(self._reset_personalization)
        layout.addStretch(1)
        return tab

    def _build_ocr_tab(self) -> QWidget:
        tab, layout = self._new_page()
        self.combo_engine = NoWheelComboBox()
        for engine in list_ocr_engines():
            self.combo_engine.addItem(engine, engine)
        self.combo_lang = NoWheelComboBox()
        for code, name in LANGUAGES:
            self.combo_lang.addItem(name, code)
        self.spin_confidence = NoWheelDoubleSpinBox()
        self.spin_confidence.setRange(0.0, 1.0)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_y_tol = NoWheelDoubleSpinBox()
        self.spin_y_tol.setRange(0.0, 2.0)
        self.spin_y_tol.setSingleStep(0.1)
        self.spin_x_gap = NoWheelDoubleSpinBox()
        self.spin_x_gap.setRange(0.0, 5.0)
        self.spin_x_gap.setSingleStep(0.1)
        self.spin_confidence.setToolTip("低於該置信度的識別結果會被丟棄；0.6 左右較穩妥，越低越寬容但可能混入噪點。")
        self.spin_y_tol.setToolTip("同一行文字在垂直方向的容差；數值越大，間距較遠的文字塊也越容易合併到同一行。")
        self.spin_x_gap.setToolTip("同一行內相鄰文字塊的橫向最大間隔；超過該間隔則視為不同行。")

        engine_form = self._new_form()
        engine_form.addRow("OCR 引擎", self.combo_engine)
        engine_form.addRow("識別語言", self.combo_lang)
        layout.addWidget(self._card("識別引擎", "選擇文字識別方式與執行能力。", engine_form))

        merge_form = self._new_form()
        merge_form.addRow("最低置信度", self.spin_confidence)
        merge_form.addRow("行合併容差", self.spin_y_tol)
        merge_form.addRow("橫向合併閾值", self.spin_x_gap)
        layout.addWidget(self._card("文本組合", "控制零散文字塊如何合併成自然語句。", merge_form))

        layout.addStretch(1)
        return tab

    def _build_translation_tab(self) -> QWidget:
        tab, layout = self._new_page()
        form = self._new_form()
        self.combo_service = NoWheelComboBox()
        for service in list_translators():
            self.combo_service.addItem(service_display_name(service), service)
        self.combo_service.setToolTip(
            "選擇翻譯服務商；預設 mymemory 為免費詞典翻譯，開箱即用，"
            "填入對應 Key 可獲得更準的線上翻譯。"
        )
        self.combo_source = NoWheelComboBox()
        self.combo_target = NoWheelComboBox()
        for code, name in LANGUAGES:
            self.combo_source.addItem(name, code)
            self.combo_target.addItem(name, code)
        self.combo_offline_model = NoWheelComboBox()
        for label, name in OFFLINE_MODEL_CHOICES:
            self.combo_offline_model.addItem(label, name)
        self.combo_offline_model.setToolTip(
            "「離線翻譯」用哪個模型（同一個 Qwen2.5 系列，越大越準也越慢、越吃記憶體）。\n"
            "0.5B 最快但日文品質差；1.5B 是速度與品質的平衡；3B、7B 較準，在 CPU 上明顯變慢。\n"
            f"灰色的選項代表缺少模型檔，{how_to_get('<名稱>')}下載。\n"
            "改了之後下次翻譯會重新載入模型；換了模型，翻譯快取會自動清空。\n"
            "不確定這台電腦跑得動哪個，按下面的「測試速度」，會用 3 句測資量每個已下載模型的翻譯時間。"
        )
        self.chk_keep_original = QCheckBox("譯文上方同時保留原文")
        self.chk_use_context = QCheckBox("使用上下文：記住前面的句子，協助翻譯這一句")
        self.spin_context_size = NoWheelSpinBox()
        self.spin_context_size.setRange(1, 10)
        self.spin_context_size.setSuffix(" 句")
        self.spin_min_hold = NoWheelDoubleSpinBox()
        self.spin_min_hold.setRange(0.0, 10.0)
        self.spin_min_hold.setSingleStep(0.1)
        self.spin_min_hold.setDecimals(1)
        self.spin_min_hold.setSuffix(" 秒")
        self.spin_min_hold.setToolTip(
            "字幕換得很快、第 N 句剛翻好畫面就跳到第 N+1 句時，第 N 句的譯文至少還要留這麼久；"
            "第 N+1 句也翻好後，接在第 N 句的下方一起顯示，同樣至少留這麼久。預設 0.5 秒，0 = 立刻換掉。\n"
            "只適用於「顯示在框下方」的譯文面板（框選區域）。"
        )
        self.spin_cache_entries = NoWheelSpinBox()
        self.spin_cache_entries.setRange(50, 20000)
        self.spin_cache_entries.setSingleStep(50)
        self.spin_cache_entries.setSuffix(" 句")
        self.spin_cache_entries.setToolTip(
            "記住最近翻過的原文與譯文；一模一樣的原文再出現時直接用，不再翻譯（也不必重新呼叫模型）。\n"
            "每句約 0.2 KB，2000 句不到 1 MB，存在本機。超過上限時，先淘汰最久沒用到的。\n"
            "建議 2000：只留 50 句太少，歌詞的副歌常在 50 句以外才再出現；影片重看一次也能直接用。"
        )
        self.chk_skip_sounds = QCheckBox("略過只有擬聲詞（音效、笑聲、狀聲詞）的句子，不翻譯也不顯示")
        self.chk_skip_sounds.setToolTip(
            "原文只是「ドン！」「ガシャーン」「Boom!」「ははは」這類聲音時，不翻成「砰」「哈哈」，也不顯示。\n"
            "由模型判斷（實測約 85～90% 的音效會被略過，一般句子沒有被誤略過），"
            "再把「譯文全是擬聲用字」的結果（汪汪、砰、哈哈）也濾掉。\n"
            "日文原文要像音效（片假名、長音、促音、重複）才會略過，「またね」「はい」這類一般短句不受影響。"
        )
        self.chk_retry_non_chinese = QCheckBox("避免譯文夾雜外文、不是中文時重翻（推薦）")
        self.chk_retry_non_chinese.setToolTip(
            "模型偶爾會在中文裡夾日文假名、韓文、俄文、原文沒有的英文單字，或把中文字一個一個隔開（「數 星 星」）。\n"
            "開啟：生成的當下就讓模型選不到這些字（不多花時間，大多數句子因此不用重翻）；"
            "仍然翻壞的（例如殘留假名）再改用取樣重翻，最多兩次，這幾句會多花模型運算的時間。\n"
            "關閉：完全不干涉模型，每句只進模型一次，翻出什麼就輸出什麼，最快，但偶爾會看到夾雜外文的譯文。\n"
            "只對「離線翻譯」有效。"
        )
        self.chk_lyrics_mode = QCheckBox("歌詞模式：告訴模型這是歌詞，譯得較通順有詩意")
        _llm_only = "只對「離線翻譯」有效；線上服務不支援。"
        self.chk_use_context.setToolTip("把最近翻好的句子（原文與譯文）當成前文交給模型，代名詞與語氣會前後一致。\n" + _llm_only)
        self.spin_context_size.setToolTip("最多記住前幾句，預設 2。句子越多每次翻譯越慢。")
        self.chk_lyrics_mode.setToolTip("歌詞常是片段、有意象，用歌詞的提示與示範讓模型意譯而不是逐字直譯。\n" + _llm_only)
        self.chk_use_context.toggled.connect(self.spin_context_size.setEnabled)
        self.spin_timeout = NoWheelSpinBox()
        self.spin_timeout.setRange(5, 300)
        self.spin_retries = NoWheelSpinBox()
        self.spin_retries.setRange(0, 10)
        self.spin_interval = NoWheelDoubleSpinBox()
        self.spin_interval.setRange(0.0, 30.0)
        self.spin_interval.setSingleStep(0.1)

        form.addRow("翻譯服務", self.combo_service)
        form.addRow("輸入語言", self.combo_source)
        form.addRow("目標語言", self.combo_target)
        form.addRow("離線翻譯模型", self.combo_offline_model)
        form.addRow("　速度測試", self._build_speed_test_row())
        form.addRow(self.chk_keep_original)
        form.addRow(self.chk_use_context)
        form.addRow("　記住前幾句", self.spin_context_size)
        form.addRow(self.chk_lyrics_mode)
        form.addRow(self.chk_skip_sounds)
        form.addRow(self.chk_retry_non_chinese)
        form.addRow("譯文最短保留", self.spin_min_hold)
        self.lbl_cache_count = QLabel()
        self.lbl_cache_count.setObjectName("SettingsCardDescription")
        self.btn_clear_cache = QPushButton("清除翻譯快取")
        self.btn_clear_cache.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_cache.setToolTip(
            "刪除所有已儲存的翻譯（立刻生效，不必按儲存）。之後遇到同樣的原文會重新翻譯。\n"
            "換翻譯模型（或翻譯服務）並儲存設定時，程式會自動清空快取，不必手動清除；"
            "想用同一個模型重新翻譯時，才需要按這個鈕。"
        )
        self.btn_clear_cache.clicked.connect(self._on_clear_cache)
        cache_row = QHBoxLayout()
        cache_row.setSpacing(10)
        cache_row.addWidget(self.spin_cache_entries)
        cache_row.addWidget(self.btn_clear_cache)
        cache_row.addWidget(self.lbl_cache_count)
        cache_row.addStretch(1)
        form.addRow("翻譯快取", cache_row)
        self._refresh_cache_count()
        form.addRow("超時（秒）", self.spin_timeout)
        form.addRow("失敗重試次數", self.spin_retries)
        form.addRow("請求間隔（秒）", self.spin_interval)
        layout.addWidget(self._card("翻譯路徑", "選擇語言、服務與失敗重試策略。", form))

        key_form = self._new_form()
        self.edit_openai_key = QLineEdit()
        self.edit_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_openai_model = QLineEdit()
        self.edit_openai_url = QLineEdit()
        self.edit_deepl_key = QLineEdit()
        self.edit_deepl_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_google_key = QLineEdit()
        self.edit_google_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_baidu_appid = QLineEdit()
        self.edit_baidu_key = QLineEdit()
        self.edit_baidu_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_youdao_key = QLineEdit()
        self.edit_youdao_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_youdao_secret = QLineEdit()
        self.edit_youdao_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_azure_key = QLineEdit()
        self.edit_azure_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_azure_region = QLineEdit()
        key_form.addRow("OpenAI API Key", self.edit_openai_key)
        key_form.addRow("OpenAI 模型", self.edit_openai_model)
        key_form.addRow("OpenAI Base URL", self.edit_openai_url)
        key_form.addRow("DeepL API Key", self.edit_deepl_key)
        key_form.addRow("Google API Key", self.edit_google_key)
        key_form.addRow("百度 APP ID", self.edit_baidu_appid)
        key_form.addRow("百度 API Key", self.edit_baidu_key)
        key_form.addRow("有道 App Key", self.edit_youdao_key)
        key_form.addRow("有道 App Secret", self.edit_youdao_secret)
        key_form.addRow("Azure 翻譯 Key", self.edit_azure_key)
        key_form.addRow("Azure 區域(Region)", self.edit_azure_region)
        self.edit_openai_key.setToolTip("OpenAI 相容服務的 API Key（如 gpt-4o-mini 翻譯）。")
        self.edit_openai_model.setToolTip("模型名，例如 gpt-4o-mini；留空使用程式預設值。")
        self.edit_openai_url.setToolTip("自定義 Base URL，用於代理或第三方相容端點；留空使用官方地址。")
        self.edit_deepl_key.setToolTip("DeepL 的 API Key（Pro 或 Free 均可）。")
        self.edit_google_key.setToolTip("Google Cloud Translation 的 API Key。")
        self.edit_baidu_appid.setToolTip("百度翻譯開放平臺的 APP ID。")
        self.edit_baidu_key.setToolTip("百度翻譯開放平臺的金鑰（Secret）。")
        self.edit_youdao_key.setToolTip("有道智雲的 App Key。")
        self.edit_youdao_secret.setToolTip("有道智雲的 App Secret。")
        self.edit_azure_key.setToolTip("Azure AI 翻譯服務的金鑰。")
        self.edit_azure_region.setToolTip("Azure 翻譯資源所屬區域，例如 eastasia。")
        layout.addWidget(
            self._card(
                "線上服務憑據",
                "優先讀取環境變數；此處僅用於當前裝置的本地配置。",
                key_form,
            )
        )

        privacy = QLabel(
            "隱私提示：使用線上翻譯服務時，截圖識別出的文字會發送到第三方伺服器。"
            "本應用預設不儲存截圖、不上傳圖片，請自行確認服務商隱私政策。"
        )
        privacy.setObjectName("NoticeLabel")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        layout.addStretch(1)
        return tab

    def _build_overlay_tab(self) -> QWidget:
        tab, layout = self._new_page()

        self.combo_placement = NoWheelComboBox()
        self.combo_placement.addItem("框選區域時，顯示在框的下方（原文保持可見，推薦）", "below")
        self.combo_placement.addItem("蓋在原文上方（取代原文）", "in_place")
        self.combo_display_mode = NoWheelComboBox()
        self.combo_display_mode.addItem("僅顯示譯文", False)
        self.combo_display_mode.addItem("原文 + 譯文（兩行對照）", True)
        self.combo_font = NoWheelComboBox()
        self.combo_font.addItem("預設字型（自動選擇，推薦）", "")
        self.combo_font.addItems(QFontDatabase.families())
        self.spin_font_size = NoWheelSpinBox()
        self.spin_font_size.setRange(6, 72)
        self.spin_min_font = NoWheelSpinBox()
        self.spin_min_font.setRange(4, 36)
        self.btn_text_color = QPushButton("選擇顏色")
        self.btn_text_color.setFixedWidth(120)
        self.chk_auto_color = QCheckBox("自動識別文字顏色（推薦）")
        self.btn_bg_color = QPushButton("選擇顏色")
        self.btn_bg_color.setFixedWidth(120)
        self.chk_auto_bg = QCheckBox("背景顏色隨文字自動適配（推薦）")
        self.slider_bg_alpha = NoWheelSlider(Qt.Orientation.Horizontal)
        self.slider_bg_alpha.setRange(0, 255)
        self.slider_panel_alpha = NoWheelSlider(Qt.Orientation.Horizontal)
        self.slider_panel_alpha.setRange(0, 255)
        self.slider_panel_alpha.setToolTip("數值越大底色越不透明（255 = 完全不透明）。")
        self.slider_bg_alpha.setToolTip("數值越大底色越不透明（255 = 完全不透明）。只有關閉「背景顏色隨文字自動適配」時才使用。")
        self.spin_padding = NoWheelSpinBox()
        self.spin_padding.setRange(0, 32)
        self.spin_radius = NoWheelSpinBox()
        self.spin_radius.setRange(0, 32)
        self.chk_show_border = QCheckBox("顯示文本框邊框")

        text_form = self._new_form()
        text_form.addRow("字型", self.combo_font)
        text_form.addRow("譯文位置", self.combo_placement)
        text_form.addRow("蓋在原文上時", self.combo_display_mode)
        text_form.addRow("字號", self.spin_font_size)
        text_form.addRow("最小字號", self.spin_min_font)
        layout.addWidget(self._card("文字排版", "讓譯文保持清晰，並匹配原始文字層級。", text_form))

        color_form = self._new_form()
        color_form.addRow("文字顏色", self.btn_text_color)
        color_form.addRow(self.chk_auto_color)
        color_form.addRow("背景顏色", self.btn_bg_color)
        color_form.addRow(self.chk_auto_bg)
        color_form.addRow("面板不透明度", self.slider_panel_alpha)
        color_form.addRow("背景不透明度", self.slider_bg_alpha)
        self.overlay_preview = OverlayPreview(self.config)
        preview_title = QLabel("目前設定的範例")
        preview_title.setObjectName("SettingsCardDescription")
        preview_column = QVBoxLayout()
        preview_column.setContentsMargins(0, 0, 0, 0)
        preview_column.setSpacing(6)
        preview_column.addWidget(preview_title)
        preview_column.addWidget(self.overlay_preview)
        preview_column.addStretch(1)
        color_body = QHBoxLayout()
        color_body.setContentsMargins(0, 0, 0, 0)
        color_body.setSpacing(18)
        color_body.addLayout(color_form, 1)
        color_body.addLayout(preview_column)
        layout.addWidget(
            self._card(
                "顏色與背景",
                "右邊的範例會隨設定即時改變。框下方的面板用固定的文字與背景顏色；蓋在原文上時，自動模式會按螢幕內容選擇對比度。",
                color_body,
            )
        )

        shape_form = self._new_form()
        shape_form.addRow("內邊距", self.spin_padding)
        shape_form.addRow("圓角", self.spin_radius)
        shape_form.addRow(self.chk_show_border)
        layout.addWidget(self._card("邊界與留白", "調整譯文塊的呼吸感與邊界提示。", shape_form))
        layout.addStretch(1)
        return tab

    def _build_hotkey_tab(self) -> QWidget:
        tab, layout = self._new_page()
        form = self._new_form()
        self.hotkey_edits: dict[str, QKeySequenceEdit] = {}
        for action, label in _HOTKEY_ACTIONS:
            editor = QKeySequenceEdit()
            editor.setMaximumSequenceLength(1)
            self.hotkey_edits[action] = editor
            form.addRow(label, editor)
        layout.addWidget(self._card("全域快捷鍵", "即使主視窗隱藏，也能直接啟動常用動作。", form))
        layout.addStretch(1)
        return tab

    def _personalization_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "palette": self.palette_choices.value(),
            "accent": self.accent_choices.value(),
            "motion_profile": self.motion_choices.value(),
            "density": self.density_choices.value(),
            "surface": self.surface_choices.value(),
            "reduce_motion": self.chk_reduce_motion.isChecked(),
        }

    def _refresh_personalization_preview(
        self, *_args, animate: bool = True
    ) -> None:
        state = self._personalization_state()
        self.palette_choices.set_accent(str(state["accent"]))
        self.personalization_preview.set_options(state, animate=animate)
        dirty = state != getattr(self, "_loaded_appearance", state)
        self.personalization_hint.setProperty("dirty", dirty)
        self.personalization_hint.setText(
            "預覽中 · 儲存後應用到主介面"
            if dirty
            else "即時預覽 · 當前設定已儲存"
        )
        self.personalization_hint.style().unpolish(self.personalization_hint)
        self.personalization_hint.style().polish(self.personalization_hint)

    def _reset_personalization(self) -> None:
        state = copy.deepcopy(DEFAULTS["appearance"])
        self.palette_choices.set_value(str(state["palette"]), emit=False)
        self.accent_choices.set_value(str(state["accent"]), emit=False)
        self.motion_choices.set_value(str(state["motion_profile"]), emit=False)
        self.density_choices.set_value(str(state["density"]), emit=False)
        self.surface_choices.set_value(str(state["surface"]), emit=False)
        self.chk_reduce_motion.blockSignals(True)
        self.chk_reduce_motion.setChecked(bool(state["reduce_motion"]))
        self.chk_reduce_motion.blockSignals(False)
        self._refresh_personalization_preview()

    # ------------------------------------------------------------- load/save
    def _load(self) -> None:
        cfg = self.config
        self._set_combo(self.combo_display_mode, bool(cfg.get("translation.keep_original", False)))
        self.chk_autostart.setChecked(bool(cfg.get("general.startup_with_system", False)))
        self.chk_tray.setChecked(bool(cfg.get("general.minimize_to_tray", True)))
        self.chk_history.setChecked(bool(cfg.get("general.save_history", False)))
        self.history_dir.setText(str(cfg.get("general.history_dir", "")))
        self.chk_output.setChecked(bool(cfg.get("output.enabled", True)))
        self.output_dir.setText(str(cfg.get("output.directory", "")))

        appearance = copy.deepcopy(DEFAULTS["appearance"])
        appearance.update(cfg.section("appearance"))
        self.palette_choices.set_value(str(appearance["palette"]), emit=False, animate=False)
        self.accent_choices.set_value(str(appearance["accent"]), emit=False, animate=False)
        self.motion_choices.set_value(
            str(appearance["motion_profile"]), emit=False, animate=False
        )
        self.density_choices.set_value(
            str(appearance["density"]), emit=False, animate=False
        )
        self.surface_choices.set_value(
            str(appearance["surface"]), emit=False, animate=False
        )
        self.chk_reduce_motion.setChecked(bool(appearance["reduce_motion"]))
        self._loaded_appearance = self._personalization_state()
        self._refresh_personalization_preview(animate=False)

        self._set_combo(self.combo_engine, cfg.get("ocr.engine", "windows"))
        self._set_combo(self.combo_lang, cfg.get("ocr.lang", "auto"))
        self.chk_auto_monitor.setChecked(bool(cfg.get("monitor.auto_start", True)))
        self.spin_scan_interval.setValue(int(cfg.get("monitor.interval_ms", stability.RECOMMENDED_INTERVAL_MS)) / 1000)
        self._refresh_scan_rate()
        self.spin_hide_no_text.setValue(
            float(cfg.get("monitor.hide_after_no_text_seconds", stability.RECOMMENDED_NO_TEXT_SECONDS))
        )
        self.spin_confirm_rounds.setValue(int(cfg.get("monitor.confirm_rounds", stability.RECOMMENDED_FAST_ROUNDS)))
        self.spin_max_wait.setValue(float(cfg.get("monitor.max_wait_seconds", stability.RECOMMENDED_MAX_WAIT)))
        self.spin_confidence.setValue(float(cfg.get("ocr.min_confidence", 0.6)))
        self.spin_y_tol.setValue(float(cfg.get("ocr.merge_y_tolerance_ratio", 0.5)))
        self.spin_x_gap.setValue(float(cfg.get("ocr.merge_x_gap_ratio", 0.8)))

        self._set_combo(self.combo_service, cfg.get("translation.service", "mock"))
        self._set_combo(self.combo_source, cfg.get("translation.source_language", "auto"))
        self._set_combo(self.combo_target, cfg.get("translation.target_language", "zh"))
        self.chk_keep_original.setChecked(bool(cfg.get("translation.keep_original", False)))
        self.chk_use_context.setChecked(bool(cfg.get("translation.use_context", True)))
        self.spin_context_size.setValue(int(cfg.get("translation.context_size", 2)))
        self.spin_context_size.setEnabled(self.chk_use_context.isChecked())
        self.chk_lyrics_mode.setChecked(bool(cfg.get("translation.lyrics_mode", False)))
        self._load_offline_model(cfg)
        self.chk_skip_sounds.setChecked(bool(cfg.get("translation.skip_sounds", True)))
        self.chk_retry_non_chinese.setChecked(bool(cfg.get("translation.retry_non_chinese", True)))
        self.spin_min_hold.setValue(float(cfg.get("overlay.min_hold_seconds", 0.5)))
        self.spin_cache_entries.setValue(int(cfg.get("translation.cache_max_entries", 2000)))
        self.spin_timeout.setValue(int(cfg.get("translation.timeout_seconds", 30)))
        self.spin_retries.setValue(int(cfg.get("translation.max_retries", 3)))
        self.spin_interval.setValue(float(cfg.get("translation.request_interval_seconds", 0.0)))
        self.edit_openai_key.setText(str(cfg.get("translation.openai.api_key", "")))
        self.edit_openai_model.setText(str(cfg.get("translation.openai.model", "gpt-4o-mini")))
        self.edit_openai_url.setText(str(cfg.get("translation.openai.base_url", "https://api.openai.com/v1")))
        self.edit_deepl_key.setText(str(cfg.get("translation.deepl.api_key", "")))
        self.edit_google_key.setText(str(cfg.get("translation.google.api_key", "")))
        self.edit_baidu_appid.setText(str(cfg.get("translation.baidu.appid", "")))
        self.edit_baidu_key.setText(str(cfg.get("translation.baidu.api_key", "")))
        self.edit_youdao_key.setText(str(cfg.get("translation.youdao.app_key", "")))
        self.edit_youdao_secret.setText(str(cfg.get("translation.youdao.app_secret", "")))
        self.edit_azure_key.setText(str(cfg.get("translation.azure.api_key", "")))
        self.edit_azure_region.setText(str(cfg.get("translation.azure.region", "")))

        self._set_combo(self.combo_placement, cfg.get("overlay.placement", "below"))
        self._sync_placement()
        self._set_combo(self.combo_font, cfg.get("overlay.font_family", ""))
        self.spin_font_size.setValue(int(cfg.get("overlay.font_size", 18)))
        self.spin_min_font.setValue(int(cfg.get("overlay.min_font_size", 8)))
        self._text_color = QColor(str(cfg.get("overlay.text_color", "#FFFFFF")))
        self._bg_color = QColor(str(cfg.get("overlay.background_color", "#000000")))
        self._update_color_buttons()
        self.chk_auto_color.setChecked(bool(cfg.get("overlay.use_auto_text_color", True)))
        self.chk_auto_bg.setChecked(bool(cfg.get("overlay.auto_background", True)))
        self.slider_bg_alpha.setValue(int(cfg.get("overlay.background_alpha", 160)))
        self.slider_panel_alpha.setValue(int(cfg.get("overlay.panel_alpha", 215)))
        self.spin_padding.setValue(int(cfg.get("overlay.padding", 4)))
        self.spin_radius.setValue(int(cfg.get("overlay.border_radius", 4)))
        self.chk_show_border.setChecked(bool(cfg.get("overlay.show_border", True)))

        for action, _ in _HOTKEY_ACTIONS:
            self.hotkey_edits[action].setKeySequence(
                self._hotkey_sequence(str(cfg.get(f"hotkeys.{action}", "")))
            )

        for signal in (
            self.combo_font.currentIndexChanged,
            self.combo_placement.currentIndexChanged,
            self.spin_font_size.valueChanged,
            self.spin_min_font.valueChanged,
            self.chk_auto_color.toggled,
            self.chk_auto_bg.toggled,
            self.slider_bg_alpha.valueChanged,
            self.slider_panel_alpha.valueChanged,
            self.spin_padding.valueChanged,
            self.spin_radius.valueChanged,
            self.chk_show_border.toggled,
        ):
            signal.connect(self._sync_overlay_controls)
        self._sync_overlay_controls()
        self.btn_text_color.clicked.connect(lambda: self._pick_color("text"))
        self.btn_bg_color.clicked.connect(lambda: self._pick_color("bg"))

    def _on_accept(self) -> None:
        """A short, deterministic save acknowledgement; no page compositing involved."""
        hotkeys = self._validated_hotkeys()
        if hotkeys is None:
            return
        self._save_close_timer.stop()
        self.btn_save.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_save.setProperty("saved", False)
        self.btn_save.setText("儲存中…")
        self.btn_save.style().unpolish(self.btn_save)
        self.btn_save.style().polish(self.btn_save)
        self._pending_hotkeys = hotkeys
        self._commit_timer.start(80)

    def _commit_pending(self) -> None:
        hotkeys = self._pending_hotkeys
        self._pending_hotkeys = None
        if hotkeys is not None:
            self._commit_accept(hotkeys)

    def _validated_hotkeys(self) -> dict[str, str] | None:
        labels = dict(_HOTKEY_ACTIONS)
        hotkeys: dict[str, str] = {}
        normalized: dict[str, str] = {}
        try:
            for action, _ in _HOTKEY_ACTIONS:
                value = self.hotkey_edits[action].keySequence().toString(
                    QKeySequence.SequenceFormat.PortableText
                )
                hotkeys[action] = value
                normalized[action] = normalize_hotkey(value)
        except (HotkeyError, ValueError) as exc:
            QMessageBox.warning(self, "快捷鍵無效", str(exc))
            return None

        seen: dict[str, str] = {}
        for action, combo in normalized.items():
            if not combo:
                continue
            if combo in seen:
                first = labels.get(seen[combo], seen[combo])
                second = labels.get(action, action)
                QMessageBox.warning(
                    self,
                    "快捷鍵衝突",
                    f"{combo} 被重複繫結：{first} / {second}",
                )
                return None
            seen[combo] = action
        return hotkeys

    def _commit_accept(self, hotkeys: dict[str, str]) -> None:
        cfg = self.config
        original_data = cfg.data
        cfg.data = copy.deepcopy(original_data)
        try:
            cfg.set("general.startup_with_system", self.chk_autostart.isChecked())
            cfg.set("general.minimize_to_tray", self.chk_tray.isChecked())
            cfg.set("general.save_history", self.chk_history.isChecked())
            cfg.set("general.history_dir", self.history_dir.text().strip())
            cfg.set("output.enabled", self.chk_output.isChecked())
            cfg.set("output.directory", self.output_dir.text().strip())

            appearance = self._personalization_state()
            for key, value in appearance.items():
                cfg.set(f"appearance.{key}", value)

            cfg.set("ocr.engine", self.combo_engine.currentData() or "windows")
            cfg.set("ocr.lang", self.combo_lang.currentData() or "auto")
            cfg.set("monitor.auto_start", self.chk_auto_monitor.isChecked())
            cfg.set("monitor.interval_ms", int(round(self.spin_scan_interval.value() * 1000)))
            cfg.set("monitor.hide_after_no_text_seconds", self.spin_hide_no_text.value())
            cfg.set("monitor.confirm_rounds", self.spin_confirm_rounds.value())
            cfg.set("monitor.max_wait_seconds", self.spin_max_wait.value())
            cfg.set("ocr.min_confidence", self.spin_confidence.value())
            cfg.set("ocr.merge_y_tolerance_ratio", self.spin_y_tol.value())
            cfg.set("ocr.merge_x_gap_ratio", self.spin_x_gap.value())

            cfg.set("translation.service", self.combo_service.currentData() or "mock")
            cfg.set("translation.source_language", self.combo_source.currentData() or "auto")
            cfg.set("translation.target_language", self.combo_target.currentData() or "zh")
            cfg.set("translation.keep_original", self.chk_keep_original.isChecked())
            cfg.set("translation.use_context", self.chk_use_context.isChecked())
            cfg.set("translation.context_size", self.spin_context_size.value())
            cfg.set("translation.lyrics_mode", self.chk_lyrics_mode.isChecked())
            self._save_offline_model(cfg)
            cfg.set("translation.skip_sounds", self.chk_skip_sounds.isChecked())
            cfg.set("translation.retry_non_chinese", self.chk_retry_non_chinese.isChecked())
            cfg.set("overlay.min_hold_seconds", self.spin_min_hold.value())
            cfg.set("translation.cache_max_entries", self.spin_cache_entries.value())
            cfg.set("translation.timeout_seconds", self.spin_timeout.value())
            cfg.set("translation.max_retries", self.spin_retries.value())
            cfg.set("translation.request_interval_seconds", self.spin_interval.value())
            cfg.set("translation.openai.api_key", self.edit_openai_key.text().strip())
            cfg.set("translation.openai.model", self.edit_openai_model.text().strip())
            cfg.set("translation.openai.base_url", self.edit_openai_url.text().strip())
            cfg.set("translation.deepl.api_key", self.edit_deepl_key.text().strip())
            cfg.set("translation.google.api_key", self.edit_google_key.text().strip())
            cfg.set("translation.baidu.appid", self.edit_baidu_appid.text().strip())
            cfg.set("translation.baidu.api_key", self.edit_baidu_key.text().strip())
            cfg.set("translation.youdao.app_key", self.edit_youdao_key.text().strip())
            cfg.set("translation.youdao.app_secret", self.edit_youdao_secret.text().strip())
            cfg.set("translation.azure.api_key", self.edit_azure_key.text().strip())
            cfg.set("translation.azure.region", self.edit_azure_region.text().strip())

            cfg.set("overlay.placement", self.combo_placement.currentData() or "below")
            cfg.set("overlay.font_family", self.combo_font.currentData() or "")
            cfg.set("overlay.font_size", self.spin_font_size.value())
            cfg.set("overlay.min_font_size", self.spin_min_font.value())
            cfg.set("overlay.text_color", self._text_color.name())
            cfg.set("overlay.use_auto_text_color", self.chk_auto_color.isChecked())
            cfg.set("overlay.background_color", self._bg_color.name())
            cfg.set("overlay.auto_background", self.chk_auto_bg.isChecked())
            cfg.set("overlay.background_alpha", self.slider_bg_alpha.value())
            cfg.set("overlay.panel_alpha", self.slider_panel_alpha.value())
            cfg.set("overlay.padding", self.spin_padding.value())
            cfg.set("overlay.border_radius", self.spin_radius.value())
            cfg.set("overlay.show_border", self.chk_show_border.isChecked())

            for action, combo in hotkeys.items():
                cfg.set(f"hotkeys.{action}", combo)
            cfg.save()
        except (OSError, ValueError, TypeError) as exc:
            cfg.data = original_data
            self.btn_save.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            self.btn_save.setText("儲存設定")
            QMessageBox.warning(self, "儲存失敗", f"配置沒有被修改。\n{exc}")
            return

        self._loaded_appearance = self._personalization_state()
        self._refresh_personalization_preview(animate=False)

        self.btn_save.setProperty("saved", True)
        self.btn_save.setText("✓ 已儲存")
        self.btn_save.style().unpolish(self.btn_save)
        self.btn_save.style().polish(self.btn_save)
        self._close_intent = 1
        self._settings_shell.setEnabled(False)
        self._save_animation.stop()
        base_width = max(116, self.btn_save.width())
        duration = motion_duration(SLOW)
        if duration == 0:
            self.btn_save.setMinimumWidth(116)
        else:
            self._save_animation.setDuration(duration)
            self._save_animation.setKeyValues(
                [(0.0, base_width), (0.42, base_width + 14), (1.0, 116)]
            )
            self._save_animation.start()
        hold = motion_duration(SLOW)
        if hold == 0:
            self.accept()
        else:
            self._save_close_timer.start(hold + 80)

    # ------------------------------------------------------------- helpers
    def _set_combo(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    @staticmethod
    def _hotkey_sequence(value: str) -> QKeySequence:
        """Load both Qt portable shortcuts and older pynput-style values."""
        portable = value.strip()
        if "<" in portable:
            aliases = {
                "<ctrl>": "Ctrl",
                "<shift>": "Shift",
                "<alt>": "Alt",
                "<cmd>": "Meta",
                "<esc>": "Esc",
                "<space>": "Space",
            }
            lowered = portable.lower()
            for source, target in aliases.items():
                lowered = lowered.replace(source, target)
            portable = lowered.replace("<", "").replace(">", "")
        return QKeySequence.fromString(portable, QKeySequence.SequenceFormat.PortableText)

    def _default_output_dir(self) -> Path:
        return Path(self.config.path.parent) / "output"

    def _pick_output_dir(self) -> None:
        start = str(resolve_directory(self.output_dir.text(), self._default_output_dir()))
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", start)
        if directory:
            self.output_dir.setText(directory)

    def _open_output_dir(self) -> None:
        """打開（目前輸入框裡的）輸出資料夾；還沒有就先建立。"""
        directory = resolve_directory(self.output_dir.text(), self._default_output_dir())
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "輸出資料夾", f"無法建立資料夾：{exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _pick_history_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "選擇歷史記錄目錄", self.history_dir.text())
        if directory:
            self.history_dir.setText(directory)

    def _pick_color(self, target: str) -> None:
        if target == "text":
            color = QColorDialog.getColor(self._text_color, self, "選擇文字顏色")
            if color.isValid():
                self._text_color = color
        else:
            color = QColorDialog.getColor(self._bg_color, self, "選擇背景顏色")
            if color.isValid():
                self._bg_color = color
        self._update_color_buttons()
        self._refresh_overlay_preview()

    def _update_color_buttons(self) -> None:
        for button, color in (
            (self.btn_text_color, self._text_color),
            (self.btn_bg_color, self._bg_color),
        ):
            value = color.name().upper()
            foreground = "#292D2A" if color.lightnessF() > 0.58 else "#FFFFFF"
            button.setText(f"●  {value}")
            button.setStyleSheet(
                f"QPushButton {{ color: {foreground}; background: {value}; border: 1px solid {value};"
                "border-radius: 8px; padding: 6px 10px; }"
                # 用不到時（自動模式）把色塊淡掉，才看得出這個顏色目前沒有作用
                "QPushButton:disabled { color: #B8B8B1; background: #F5F4F1; border: 1px solid #E8E6E0; }"
            )

    def _overlay_preview_options(self) -> dict:
        """目前控制項上的數值（還沒儲存），範例用它來畫。"""
        return {
            "overlay.placement": self.combo_placement.currentData() or "below",
            "overlay.font_family": self.combo_font.currentData() or "",
            "overlay.font_size": self.spin_font_size.value(),
            "overlay.min_font_size": self.spin_min_font.value(),
            "overlay.text_color": self._text_color.name(),
            "overlay.use_auto_text_color": self.chk_auto_color.isChecked(),
            "overlay.background_color": self._bg_color.name(),
            "overlay.auto_background": self.chk_auto_bg.isChecked(),
            "overlay.background_alpha": self.slider_bg_alpha.value(),
            "overlay.panel_alpha": self.slider_panel_alpha.value(),
            "overlay.padding": self.spin_padding.value(),
            "overlay.border_radius": self.spin_radius.value(),
            "overlay.show_border": self.chk_show_border.isChecked(),
        }

    def _refresh_overlay_preview(self, *_args) -> None:
        if hasattr(self, "overlay_preview") and hasattr(self, "_text_color"):
            self.overlay_preview.set_options(self._overlay_preview_options())

    def _sync_overlay_controls(self, *_args) -> None:
        """只啟用目前譯文位置真的會用到的控制項，其餘停用並說明原因。"""
        below = self.combo_placement.currentData() != "in_place"
        covering_only = "只在「蓋在原文上方」時使用；框下方的面板用固定的顏色。"
        panel_only = "只在「框的下方」顯示時使用。"
        rows = (
            (self.chk_auto_color, not below, covering_only),
            (self.chk_auto_bg, not below, covering_only),
            (self.spin_padding, not below, covering_only),
            (self.chk_show_border, not below, covering_only),
            (self.spin_min_font, not below, covering_only),
            (self.slider_panel_alpha, below, panel_only),
            # 顏色按鈕、背景不透明度：蓋在原文上且開了自動模式時用不到
            (self.btn_text_color, below or not self.chk_auto_color.isChecked(), "已開啟「自動識別文字顏色」，這裡的顏色用不到。"),
            (self.btn_bg_color, below or not self.chk_auto_bg.isChecked(), "已開啟「背景顏色隨文字自動適配」，這裡的顏色用不到。"),
            (self.slider_bg_alpha, not below and not self.chk_auto_bg.isChecked(), "只在「蓋在原文上方」且關閉自動背景時使用。"),
        )
        for control, enabled, reason in rows:
            control.setEnabled(enabled)
            control.setToolTip("" if enabled else reason)
        self._refresh_overlay_preview()

    def _load_offline_model(self, cfg) -> None:
        """依設定選好模型；缺少模型檔的選項停用（並說明怎麼補）。"""
        model_dir = str(cfg.get("translation.offline.model_dir", "") or "")
        for index, (_label, name) in enumerate(OFFLINE_MODEL_CHOICES):
            item = self.combo_offline_model.model().item(index)
            usable = name == "auto" or llama_model_available(name, model_dir)
            item.setEnabled(usable)
            self.combo_offline_model.setItemData(
                index,
                "" if usable else f"找不到 llama-cpp-python 或模型檔：{how_to_get(name)}",
                Qt.ItemDataRole.ToolTipRole,
            )
        saved = str(cfg.get("translation.offline.model_name", "auto") or "auto")
        chosen = 0  # 自訂的模型名稱等其他情況一律顯示成「自動」
        for index, (_label, name) in enumerate(OFFLINE_MODEL_CHOICES):
            if name == saved:
                chosen = index
        self.combo_offline_model.setCurrentIndex(chosen)
        self._loaded_offline_model = chosen

    def _save_offline_model(self, cfg) -> None:
        """使用者沒動過這個選單就不寫入，避免把自訂的模型名稱蓋掉。"""
        if self.combo_offline_model.currentIndex() == self._loaded_offline_model:
            return
        cfg.set("translation.offline.model_name", self.combo_offline_model.currentData())

    # ------------------------------------------------------------- 速度測試
    def _build_speed_test_row(self) -> QHBoxLayout:
        """離線翻譯模型下面的「測試速度」：左邊按鈕與狀態，右邊是每個模型的結果表（存檔，下次開設定還在）。"""
        self.btn_speed_test = QPushButton("測試速度")
        self.btn_speed_test.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_speed_test.setToolTip(
            "用 3 句測資（日文一般、日文歌詞、英文），依序載入每個已下載的模型，量這台電腦翻一句要多久。\n"
            "只算「熱機之後」的推論時間：先預熱（載入、算好固定前綴），再翻一輪不計時，之後計時三輪取中位數。\n"
            "測試時會載入各模型（0.5B 到 7B），電腦會比較忙，請先停止即時監控；結果會存起來。"
        )
        self.btn_speed_test.clicked.connect(self._on_speed_test)
        self.lbl_speed_test = QLabel()
        self.lbl_speed_test.setObjectName("SettingsCardDescription")
        self.lbl_speed_test.setWordWrap(True)
        left = QWidget()
        left.setFixedWidth(138)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.btn_speed_test)
        left_layout.addWidget(self.lbl_speed_test)
        left_layout.addStretch(1)

        self.lbl_speed_caption = QLabel("每句翻譯時間（毫秒，熱機後、不含載入與固定前綴）。★ 是目前使用的模型")
        self.lbl_speed_caption.setObjectName("SettingsCardDescription")
        self.lbl_speed_caption.setWordWrap(True)
        columns = ["模型", *(name for name, _text in SPEED_TEST_SENTENCES), "平均", "載入（秒）"]
        self._speed_columns = columns
        self.table_speed = QTableWidget(len(OFFLINE_MODELS), len(columns))
        self.table_speed.setObjectName("SpeedTable")
        self.table_speed.setHorizontalHeaderLabels(columns)
        self.table_speed.verticalHeader().setVisible(False)
        self.table_speed.verticalHeader().setDefaultSectionSize(28)
        self.table_speed.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_speed.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table_speed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table_speed.setShowGrid(False)
        self.table_speed.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table_speed.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.table_speed.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(columns)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)
        self.table_speed.setFixedHeight(header.height() + 28 * len(OFFLINE_MODELS) + 8)
        self.table_speed.setMinimumWidth(420)
        for index, (name, text) in enumerate(SPEED_TEST_SENTENCES, start=1):
            self.table_speed.horizontalHeaderItem(index).setToolTip(text)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.lbl_speed_caption)
        right_layout.addWidget(self.table_speed)
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(left, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(right_layout, 1)
        self._refresh_speed_table()
        return row

    def _current_offline_model(self) -> str:
        """設定裡實際使用的離線模型（「自動」會解析成實際的那個）。"""
        saved = str(self.config.get("translation.offline.model_name", "auto") or "auto")
        try:
            return OfflineTranslator({"offline": {"model_name": saved}}).resolved_model_name()
        except Exception:
            return saved

    def _speed_cell(self, text: str, *, bold: bool = False, muted: bool = False, tip: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        if muted:
            item.setForeground(self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text))
        if tip:
            item.setToolTip(tip)
        return item

    def _refresh_speed_table(self) -> None:
        """依存檔的結果重畫表格；沒下載的模型標「未下載」，正在測的標「測試中」。"""
        current = self._current_offline_model()
        span_width = len(self._speed_columns) - 1
        for row, model in enumerate(OFFLINE_MODELS):
            name = model.name
            result = self._speed_results.get(name)
            available = llama_model_available(name)
            title = ("★ " if name == current else "") + short_label(name)
            tip = model.label
            if result is not None and result.get("status") == STATUS_OK and result.get("device"):
                # 這筆數字是用 CPU 還是 GPU 量的：換了裝置重測會覆蓋舊的，所以要標清楚
                if result["device"] == "gpu":
                    title += " · GPU"
                elif result.get("gpu_fallback"):
                    title += " · CPU（GPU 失敗）"
                    tip += "\n要求用 GPU，但載入失敗，改用 CPU 測，這是 CPU 的時間。"
                else:
                    title += " · CPU"
            label = self._speed_cell(title, bold=name == current, tip=tip)
            label.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.table_speed.setItem(row, 0, label)
            for column in range(1, len(self._speed_columns)):
                if self.table_speed.columnSpan(row, column) > 1:  # 上一次是整列的訊息（未下載、測試中…）：還原成一格一格
                    self.table_speed.setSpan(row, column, 1, 1)
                self.table_speed.setItem(row, column, QTableWidgetItem(""))
            message, muted = "", True
            if self._speed_testing and name == self._speed_running:
                message, muted = "測試中…", False
            elif result is None or result.get("status") == STATUS_MISSING:
                message = ("等待測試" if self._speed_testing else "尚未測試") if available else "未下載"
            elif result.get("status") == STATUS_MEMORY:
                message = "記憶體不足，略過" + (f"（{result['message']}）" if result.get("message") else "")
            elif result.get("status") == STATUS_ERROR:
                message = "測試失敗"
            if message:
                tip = (result or {}).get("message", "") if result else ""
                self.table_speed.setSpan(row, 1, 1, span_width)
                self.table_speed.setItem(row, 1, self._speed_cell(message, muted=muted, tip=tip))
                continue
            if result.get("status") == STATUS_OK:
                times = result.get("ms", {})
                when = str(result.get("tested_at", "")).replace("T", " ")
                tip = f"{when}，{result.get('threads', '?')} 執行緒"
                for index, (group, _text) in enumerate(SPEED_TEST_SENTENCES, start=1):
                    value = times.get(group)
                    self.table_speed.setItem(row, index, self._speed_cell("" if value is None else f"{value:,}", tip=tip))
                self.table_speed.setItem(row, len(SPEED_TEST_SENTENCES) + 1, self._speed_cell(f"{result.get('avg_ms', 0):,}", bold=True, tip=tip))
                self.table_speed.setItem(row, len(SPEED_TEST_SENTENCES) + 2, self._speed_cell(f"{result.get('load_s', 0):.1f}", tip=tip))
        self._refresh_speed_label()

    def _refresh_speed_label(self) -> None:
        if self._speed_testing:
            return
        stamps = [str(entry.get("tested_at", "")) for entry in self._speed_results.values() if entry.get("tested_at")]
        self.lbl_speed_test.setText(f"上次測試：\n{max(stamps).replace('T', ' ')}" if stamps else "尚未測試過")

    def _call_speed_hook(self, index: int) -> None:
        if self._speed_hooks is None:
            return
        try:
            self._speed_hooks[index]()
        except Exception:
            log.exception("速度測試的前置／善後動作失敗")

    def _set_speed_testing(self, testing: bool) -> None:
        self._speed_testing = testing
        self.btn_speed_test.setText("停止測試" if testing else "測試速度")
        self.btn_speed_test.setEnabled(True)
        if testing:
            self._speed_timer.start()
        else:
            self._speed_timer.stop()
            self._speed_running = ""

    def _on_speed_test(self) -> None:
        if self._speed_testing:
            self._speed_cancel.set()
            self.btn_speed_test.setEnabled(False)
            self.lbl_speed_test.setText("停止中…\n（等目前這個模型測完）")
            return
        reason = OfflineTranslator.availability_reason()
        if reason is None and self._speed_guard is not None:
            reason = self._speed_guard() or None
        if reason is not None:
            QMessageBox.information(self, "測試速度", reason)
            return
        models = [(model.name, short_label(model.name)) for model in OFFLINE_MODELS if llama_model_available(model.name)]
        if not models:
            QMessageBox.information(
                self,
                "測試速度",
                f"找不到任何已下載的模型。{how_to_get()}下載。",
            )
            return
        answer = QMessageBox.question(
            self,
            "測試速度",
            f"要依序載入已下載的 {len(models)} 個模型，各翻 3 句來量這台電腦的翻譯時間嗎？\n"
            "每個模型要載入並預熱數秒到半分鐘（7B 最久），整個測試約需幾分鐘，期間電腦會比較忙。\n"
            "結果會存起來，下次開設定還看得到。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._speed_cancel = threading.Event()
        self.destroyed.connect(self._speed_cancel.set)
        self._speed_messages = queue.Queue()
        self._call_speed_hook(0)
        section = copy.deepcopy(self.config.section("translation"))
        self._speed_thread = threading.Thread(
            target=_speed_test_thread,
            args=(section, models, self._speed_cancel, self._speed_messages, self._speed_store),
            name="speed-test",
            daemon=True,
        )
        self._set_speed_testing(True)
        self.lbl_speed_test.setText("準備中…")
        self._speed_thread.start()
        self._refresh_speed_table()

    def _poll_speed_test(self) -> None:
        """把測試執行緒送來的進度與結果搬到畫面上（執行緒不直接碰介面）。"""
        changed = False
        finished = False
        while True:
            try:
                kind, payload = self._speed_messages.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.lbl_speed_test.setText(str(payload))
            elif kind == "start":
                self._speed_running = str(payload)
                changed = True
            elif kind == "result":
                self._speed_results[payload["model"]] = payload
                if self._speed_running == payload["model"]:
                    self._speed_running = ""
                changed = True
            elif kind == "error":
                QMessageBox.warning(self, "測試速度", f"速度測試中斷：{payload}")
            elif kind == "done":
                finished = True
                stopped = bool(payload)
        if finished:
            self._set_speed_testing(False)
            self._call_speed_hook(1)
            self._refresh_speed_table()
            if stopped:
                self.lbl_speed_test.setText("已停止\n" + self.lbl_speed_test.text())
            return
        if changed:
            self._refresh_speed_table()

    def _refresh_cache_count(self) -> None:
        if self._cache_actions is None:
            self.btn_clear_cache.setEnabled(False)
            self.lbl_cache_count.setText("")
            return
        count = int(self._cache_actions[0]())
        self.lbl_cache_count.setText(f"目前存了 {count} 句")
        self.btn_clear_cache.setEnabled(count > 0)

    def _on_clear_cache(self) -> None:
        """刪除所有已儲存的翻譯：先確認，因為刪掉就回不來（只是之後要重新翻）。"""
        if self._cache_actions is None:
            return
        count = int(self._cache_actions[0]())
        answer = QMessageBox.question(
            self,
            "清除翻譯快取",
            f"要刪除目前儲存的 {count} 句翻譯嗎？\n之後遇到同樣的原文會重新翻譯（換模型比較時很有用）。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = int(self._cache_actions[1]())
        self._refresh_cache_count()
        self.lbl_cache_count.setText(f"已清除 {removed} 句")

    def _refresh_scan_rate(self, *_args) -> None:
        interval = max(self.spin_scan_interval.value(), stability.MIN_INTERVAL_MS / 1000)
        self.lbl_scan_rate.setText(f"約每秒檢查 {1 / interval:.1f} 次畫面")

    def _apply_stability_preset(self, rounds: int) -> None:
        """套用系統推薦：確認次數依快速／穩定，最久等待用推薦值。"""
        self.spin_confirm_rounds.setValue(rounds)
        self.spin_max_wait.setValue(stability.RECOMMENDED_MAX_WAIT)

    def _sync_placement(self, *_args) -> None:
        """譯文放在框下方時，原文本來就留在畫面上：「蓋在原文上時」的選項用不到，先停用。"""
        covering = self.combo_placement.currentData() == "in_place"
        reason = "" if covering else "譯文顯示在框的下方時，原文會保持原樣可見，這項設定用不到。"
        for control in (self.combo_display_mode, self.chk_keep_original):
            control.setEnabled(covering)
            control.setToolTip(reason)

    def _sync_display_mode(self, *_args) -> None:
        """覆盖层"译文显示形式"与翻译页"保留原文"保持同步。"""
        self.chk_keep_original.blockSignals(True)
        self.chk_keep_original.setChecked(bool(self.combo_display_mode.currentData()))
        self.chk_keep_original.blockSignals(False)

    def _sync_keep_original(self, checked: bool) -> None:
        self.combo_display_mode.blockSignals(True)
        self._set_combo(self.combo_display_mode, bool(checked))
        self.combo_display_mode.blockSignals(False)
