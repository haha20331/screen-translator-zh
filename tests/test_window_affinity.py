"""程式自己的視窗必須被排除在螢幕擷取之外，否則監控框的提示文字與譯文標籤會被 OCR 讀回去。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from utils import window_affinity
from utils.window_affinity import WDA_EXCLUDEFROMCAPTURE, CaptureExclusionFilter, exclude_from_capture


class _FakeUser32:
    def __init__(self, result=1, error=None) -> None:
        self.calls = []
        self._result, self._error = result, error

    def SetWindowDisplayAffinity(self, hwnd, affinity):
        self.calls.append((hwnd, affinity))
        if self._error:
            raise self._error
        return self._result


def test_exclude_from_capture_sets_the_exclude_affinity():
    user32 = _FakeUser32()

    assert exclude_from_capture(1234, user32) is True
    assert user32.calls == [(1234, WDA_EXCLUDEFROMCAPTURE)]
    assert WDA_EXCLUDEFROMCAPTURE == 0x11


def test_exclude_from_capture_reports_failure():
    assert exclude_from_capture(1234, _FakeUser32(result=0)) is False
    assert exclude_from_capture(1234, _FakeUser32(error=OSError("denied"))) is False
    assert exclude_from_capture(0, _FakeUser32()) is False  # 沒有視窗代號


def test_unsupported_platform_does_nothing(monkeypatch):
    monkeypatch.setattr(window_affinity, "supported", lambda: False)

    assert exclude_from_capture(1234) is False


def test_supported_is_false_off_windows(monkeypatch):
    monkeypatch.setattr(window_affinity.sys, "platform", "linux")

    assert window_affinity.supported() is False


def test_supported_requires_windows_10_2004(monkeypatch):
    monkeypatch.setattr(window_affinity.sys, "platform", "win32")
    for build, expected in ((19040, False), (19041, True), (22631, True)):
        monkeypatch.setattr(window_affinity.sys, "getwindowsversion", lambda build=build: type("V", (), {"build": build})(), raising=False)
        assert window_affinity.supported() is expected


def test_filter_excludes_every_top_level_window_when_shown():
    app = QApplication.instance() or QApplication([])
    seen = []
    capture_filter = CaptureExclusionFilter(app, apply=lambda hwnd: seen.append(hwnd) or True)
    app.installEventFilter(capture_filter)
    try:
        window, child = QWidget(), None
        child = QWidget(window)
        window.show()
        child.show()
        app.processEvents()
        window_id = int(window.winId())
    finally:
        app.removeEventFilter(capture_filter)
        window.close()

    assert seen == [window_id]  # 只處理頂層視窗，不處理子元件


def test_filter_never_breaks_window_display_when_the_api_fails():
    app = QApplication.instance() or QApplication([])

    def boom(_hwnd):
        raise RuntimeError("api failure")

    capture_filter = CaptureExclusionFilter(app, apply=boom)
    app.installEventFilter(capture_filter)
    try:
        window = QWidget()
        window.show()
        app.processEvents()
        assert window.isVisible()
    finally:
        app.removeEventFilter(capture_filter)
        window.close()
