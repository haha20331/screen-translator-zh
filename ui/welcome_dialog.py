"""首次启动的欢迎引导：三步上手，降低新用户的使用门槛。

只在 ``general.first_run_done`` 为 False 时由控制器弹出一次；用户点“开始使用”
后由调用方把该标志写入配置。对话框完全复用项目的主题调色板（见 ``ui/style``）。
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


def _hkfmt(raw: str) -> str:
    """把 ``ctrl+shift+a`` 之类的配置值美化为 ``Ctrl+Shift+A`` 用于展示。"""
    if not raw:
        return ""
    return "+".join(part.capitalize() for part in raw.split("+"))


class WelcomeDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        hotkeys: dict,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_finish = on_finish
        self.setWindowTitle("歡迎使用 螢幕翻譯")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._hotkeys = hotkeys
        self._build_ui()
        self._page = 0
        self._show_page(0)

    # ------------------------------------------------------------------ ui
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        header.setObjectName("WelcomeHeader")
        hb = QVBoxLayout(header)
        hb.setContentsMargins(36, 28, 36, 18)
        hb.setSpacing(8)
        title = QLabel("歡迎使用 螢幕翻譯")
        title.setObjectName("WelcomeTitle")
        sub = QLabel("三步上手，把螢幕上任何看不懂的文字，一鍵變成你能讀懂的語言。")
        sub.setObjectName("WelcomeSubtitle")
        hb.addWidget(title)
        hb.addWidget(sub)
        root.addWidget(header)

        self._stack = QStackedWidget()
        self._stack.setObjectName("WelcomeStack")
        self._stack.addWidget(self._page_capture())
        self._stack.addWidget(self._page_language())
        self._stack.addWidget(self._page_tray())
        root.addWidget(self._stack, 1)

        footer = QWidget()
        footer.setObjectName("WelcomeFooter")
        fb = QHBoxLayout(footer)
        fb.setContentsMargins(36, 16, 36, 26)
        fb.setSpacing(10)
        dots = QWidget()
        dl = QHBoxLayout(dots)
        dl.setSpacing(8)
        self._dots = []
        for _ in range(3):
            d = QWidget()
            d.setFixedSize(8, 8)
            d.setObjectName("WelcomeDot")
            dl.addWidget(d)
            self._dots.append(d)
        fb.addWidget(dots)
        fb.addStretch(1)
        self.btn_back = QPushButton("上一步")
        self.btn_back.setObjectName("DialogCancelButton")
        self.btn_back.clicked.connect(self._back)
        self.btn_next = QPushButton("下一步")
        self.btn_next.setObjectName("PrimaryButton")
        self.btn_next.clicked.connect(self._next)
        fb.addWidget(self.btn_back)
        fb.addWidget(self.btn_next)
        root.addWidget(footer)

    def _mode_row(self, glyph: str, title: str, hotkey: str, desc: str) -> QWidget:
        row = QWidget()
        rb = QHBoxLayout(row)
        rb.setContentsMargins(2, 10, 2, 10)
        rb.setSpacing(14)
        badge = QLabel(glyph)
        badge.setObjectName("WelcomeBadge")
        badge.setFixedSize(42, 42)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rb.addWidget(badge)
        text = QVBoxLayout()
        text.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(10)
        t = QLabel(title)
        t.setObjectName("WelcomeRowTitle")
        chip = QLabel(_hkfmt(hotkey))
        chip.setObjectName("HotkeyChip")
        head.addWidget(t)
        head.addWidget(chip)
        head.addStretch(1)
        d = QLabel(desc)
        d.setObjectName("WelcomeRowDesc")
        d.setWordWrap(True)
        text.addLayout(head)
        text.addWidget(d)
        rb.addLayout(text, 1)
        return row

    def _page_capture(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(36, 12, 36, 8)
        v.setSpacing(10)
        intro = QLabel("先選一種截圖方式，框好範圍後即開始識別與翻譯：")
        intro.setObjectName("WelcomeBody")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(
            self._mode_row(
                "▦", "框選翻譯", self._hotkeys.get("capture_region", ""),
                "拖曳滑鼠圈出任意螢幕區域，適合零散段落。",
            )
        )
        v.addWidget(
            self._mode_row(
                "▣", "全屏翻譯", self._hotkeys.get("capture_fullscreen", ""),
                "一次翻譯整個螢幕，適合網頁或長文件。",
            )
        )
        v.addWidget(
            self._mode_row(
                "▢", "當前視窗", self._hotkeys.get("capture_window", ""),
                "自動鎖定最前面的應用視窗，無需手動框選。",
            )
        )
        v.addStretch(1)
        return page

    def _page_language(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(36, 12, 36, 8)
        v.setSpacing(12)
        body = QLabel(
            "在主介面頂部選好<strong>源語言</strong>與<strong>目標語言</strong>，"
            "點任意截圖方式即可翻譯。\n\n"
            "預設使用內建詞典翻譯，<strong>開箱即用、無需任何配置</strong>；"
            "想要更地道的線上翻譯，可在「設定 → 翻譯」填入對應服務的 API Key，"
            "程式會自動切換到該服務。"
        )
        body.setObjectName("WelcomeBody")
        body.setWordWrap(True)
        v.addWidget(body, 1)
        return page

    def _page_tray(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(36, 12, 36, 8)
        v.setSpacing(12)
        body = QLabel(
            "翻譯時主視窗會最小化到系統托盤，譯文直接覆蓋在螢幕上原文的位置。\n\n"
            "關閉主視窗不會退出程式——單擊托盤圖示即可隨時喚回；"
            "按 <strong>F1</strong> 可隨時檢視本幫助與全部快捷鍵。"
        )
        body.setObjectName("WelcomeBody")
        body.setWordWrap(True)
        v.addWidget(body, 1)
        return page

    # ------------------------------------------------------------------ nav
    def _show_page(self, index: int) -> None:
        self._page = index
        self._stack.setCurrentIndex(index)
        for i, dot in enumerate(self._dots):
            dot.setProperty("active", i == index)
            dot.style().unpolish(dot)
            dot.style().polish(dot)
        self.btn_back.setEnabled(index > 0)
        self.btn_next.setText("開始使用" if index == self._stack.count() - 1 else "下一步")

    def _back(self) -> None:
        if self._page > 0:
            self._show_page(self._page - 1)

    def _next(self) -> None:
        if self._page < self._stack.count() - 1:
            self._show_page(self._page + 1)
        else:
            self.accept()

    def accept(self) -> None:
        if self._on_finish is not None:
            self._on_finish()
        super().accept()
