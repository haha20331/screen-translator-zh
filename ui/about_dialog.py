"""“关于”对话框：版本、仓库入口，以及一份随时可查的快捷键速查表。

主窗口按 F1、托盘菜单“关于”都会打开它。快捷键表同时覆盖全局热键与界面内
常用按键，让新用户一眼看懂如何操作。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.version import __version__
from ui.welcome_dialog import _hkfmt


def _shortcut_rows(hotkeys: dict) -> list[tuple[str, str]]:
    rows = [
        ("框選翻譯", hotkeys.get("capture_region", "")),
        ("全屏翻譯", hotkeys.get("capture_fullscreen", "")),
        ("當前視窗翻譯", hotkeys.get("capture_window", "")),
        ("隱藏 / 顯示譯文", hotkeys.get("toggle_overlay", "")),
        ("重新識別翻譯", hotkeys.get("refresh", "")),
        ("即時監控", hotkeys.get("monitor_toggle", "")),
        ("檢視幫助（本頁）", "F1"),
        ("取消當前選區", "Esc"),
    ]
    return [(label, _hkfmt(key)) for label, key in rows]


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None, config) -> None:
        super().__init__(parent)
        self.setWindowTitle("關於 螢幕翻譯")
        self.setMinimumSize(560, 520)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._build_ui(config)

    def _build_ui(self, config) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 22)
        root.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(14)
        mark = QLabel("譯")
        mark.setObjectName("AboutMark")
        mark.setFixedSize(56, 56)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info = QVBoxLayout()
        info.setSpacing(4)
        name = QLabel("螢幕翻譯")
        name.setObjectName("AboutName")
        version = QLabel(f"版本 {__version__}")
        version.setObjectName("AboutVersion")
        tagline = QLabel("把螢幕上任何文字，一鍵翻譯成你能讀懂的語言。")
        tagline.setObjectName("AboutTagline")
        info.addWidget(name)
        info.addWidget(version)
        info.addWidget(tagline)
        header.addWidget(mark)
        header.addLayout(info, 1)
        root.addLayout(header)

        shortcuts = self._card("快捷鍵", "即使主視窗隱藏，全域熱鍵也能直接啟動動作。", self._shortcut_table(config.hotkeys()))
        root.addWidget(shortcuts, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        ok = QPushButton("知道了")
        ok.setObjectName("PrimaryButton")
        ok.setMinimumWidth(120)
        ok.clicked.connect(self.accept)
        footer.addWidget(ok)
        root.addLayout(footer)

    def _card(self, title: str, description: str, body: QWidget) -> QFrame:
        card = QFrame()
        card.setObjectName("SettingsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        t = QLabel(title)
        t.setObjectName("SettingsCardTitle")
        d = QLabel(description)
        d.setObjectName("SettingsCardDescription")
        layout.addWidget(t)
        layout.addWidget(d)
        layout.addWidget(body)
        return card

    def _shortcut_table(self, hotkeys: dict) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for label, key in _shortcut_rows(hotkeys):
            row = QHBoxLayout()
            row.setSpacing(12)
            name = QLabel(label)
            name.setObjectName("ShortcutName")
            chip = QLabel(key)
            chip.setObjectName("HotkeyChip")
            row.addWidget(name)
            row.addStretch(1)
            row.addWidget(chip)
            layout.addLayout(row)
        return widget
