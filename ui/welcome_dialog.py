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
        sub = QLabel("框選畫面上的文字，譯文馬上顯示在框的下方。翻譯全部在這台電腦上完成，不用網路帳號。")
        sub.setObjectName("WelcomeSubtitle")
        hb.addWidget(title)
        hb.addWidget(sub)
        root.addWidget(header)

        self._stack = QStackedWidget()
        self._stack.setObjectName("WelcomeStack")
        self._stack.addWidget(self._page_capture())
        self._stack.addWidget(self._page_display())
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
        intro = QLabel("兩種用法，都是先用滑鼠圈出想翻譯的範圍：")
        intro.setObjectName("WelcomeBody")
        intro.setWordWrap(True)
        v.addWidget(intro)
        v.addWidget(
            self._mode_row(
                "▦", "框選翻譯", self._hotkeys.get("capture_region", ""),
                "圈出一塊區域，翻一次。適合看一段文字、一張圖。",
            )
        )
        v.addWidget(
            self._mode_row(
                "▶", "即時監控", self._hotkeys.get("monitor_toggle", ""),
                "框會一直留在畫面上，字幕一換就自動翻下一句。適合看影片、看歌詞、玩遊戲。",
            )
        )
        note = QLabel("文字是直排的（例如日文直書）？先把主視窗的「文字方向」改成「直向」，再框選。")
        note.setObjectName("WelcomeBody")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return page

    def _page_display(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(36, 12, 36, 8)
        v.setSpacing(12)
        body = QLabel(
            "<strong>譯文怎麼看</strong><br>"
            "譯文顯示在框的下方，原文照常留在畫面上。框的中間是透明的，可以直接點到後面的畫面；按住框左上或右下角的移動符號（滑鼠移到框上才會出現）拖曳來移動、拉邊框改大小；"
            "滑鼠離開框超過 2 秒，框會變淡，免得擋住畫面，移回來就恢復。<br><br>"
            "<strong>第一次使用</strong><br>"
            "程式會先把翻譯模型載入記憶體，狀態列會顯示進度，載好就能翻。"
            "辨識與翻譯都在這台電腦上完成，畫面和文字不會傳到網路上。<br><br>"
            "翻過的字幕會依日期存成文字檔，位置可以在「設定 → 一般 → 文字輸出」查看與修改。"
        )
        body.setObjectName("WelcomeBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        v.addWidget(body, 1)
        return page

    def _page_tray(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(36, 12, 36, 8)
        v.setSpacing(12)
        body = QLabel(
            "<strong>設定</strong><br>"
            "主視窗右上角的「設定」可以換翻譯模型（越大越準，也越慢）、選用 CPU 或顯示卡來翻譯、"
            "改輸入語言和譯文的外觀。譯文固定翻成繁體中文。<br><br>"
            "<strong>系統匣</strong><br>"
            "程式也會有一個系統匣圖示（螢幕右下角、工作列右邊的小圖示），按右鍵可以快速操作或退出。"
            "預設按主視窗右上角的關閉鈕就會結束程式；想讓它關掉視窗後繼續在背景執行，"
            "到「設定 → 一般」勾選「關閉主視窗時最小化到托盤」。"
            "全螢幕翻譯、當前視窗翻譯用快捷鍵或系統匣選單。<br><br>"
            "按 <strong>F1</strong> 隨時查看全部快捷鍵。"
        )
        body.setObjectName("WelcomeBody")
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
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
