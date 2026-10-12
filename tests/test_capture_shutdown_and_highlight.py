"""Shutdown, OCR-engine fallback and moving-target confirmation regressions."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

import app.application as application_module
from app.application import Application, _CaptureSession
from app.config import AppConfig
from app.models import CaptureInfo
from services.ocr.base import OCRUnavailableError
from workers.translation_worker import PipelineTask


class _SlowOCR:
    def recognize(self, _image, lang="ch"):
        loop = QEventLoop()
        QTimer.singleShot(60, loop.quit)
        loop.exec()
        return []


def test_shutdown_waits_for_real_worker_finished_before_quitting(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    controller = Application(app)
    capture = CaptureInfo(
        image=np.zeros((40, 80, 3), dtype=np.uint8),
        bbox=(0, 0, 80, 40),
        mode="region",
    )
    controller.worker = PipelineTask(
        capture=capture,
        ocr_engine=_SlowOCR(),
        translator=None,
        config=AppConfig(tmp_path / "config.json"),
    )
    quits = []
    monkeypatch.setattr(app, "quit", lambda: quits.append(True))
    controller.worker.start()

    controller.shutdown()

    assert controller._shutting_down
    assert controller.worker is not None
    assert quits == []
    assert controller.worker.wait(3000)
    app.processEvents()
    assert controller.worker is None
    assert quits == [True]
    controller.floating_status.deleteLater()


def test_default_ocr_fallback_uses_none_only_after_rapidocr_and_windows_fail(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    controller = Application(app)
    controller.config = AppConfig(tmp_path / "config.json")
    attempts = []
    sentinel = object()

    def create(name, *_args):
        attempts.append(name)
        if name != "none":
            raise OCRUnavailableError(f"{name} unavailable")
        return sentinel

    monkeypatch.setattr(application_module, "create_ocr_engine", create)

    assert controller._create_configured_ocr_engine() is sentinel
    assert attempts == ["rapidocr", "windows", "none"]
    assert controller._active_ocr_engine_name == "none"
    controller.floating_status.deleteLater()
    app.processEvents()


def _controller_with_engine(tmp_path, monkeypatch, usable: str, requested_error=OCRUnavailableError):
    """Application whose create_ocr_engine only succeeds for `usable`.

    The configured engine ("missing_engine") fails with `requested_error`; every other engine
    except `usable` raises OCRUnavailableError. Returns (controller, attempts, sentinel).
    """
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    controller = Application(app)
    controller.config = AppConfig(tmp_path / "config.json")
    controller.config.set("ocr.engine", "missing_engine")
    attempts = []
    sentinel = object()

    def create(name, *_args):
        attempts.append(name)
        if name == usable:
            return sentinel
        if name == "none":
            raise AssertionError("none must not be used while a real engine works")
        if name == "missing_engine":
            raise requested_error("bad engine configuration")
        raise OCRUnavailableError(f"{name} unavailable")

    monkeypatch.setattr(application_module, "create_ocr_engine", create)
    return controller, attempts, sentinel


def test_requested_engine_falls_back_to_rapidocr_when_unavailable(tmp_path, monkeypatch):
    controller, attempts, sentinel = _controller_with_engine(tmp_path, monkeypatch, usable="rapidocr")

    assert controller._create_configured_ocr_engine() is sentinel
    assert attempts == ["missing_engine", "rapidocr"]
    assert controller._active_ocr_engine_name == "rapidocr"
    assert controller.config.get("ocr.engine") == "rapidocr"
    controller.floating_status.deleteLater()
    QApplication.instance().processEvents()


def test_windows_ocr_is_the_fallback_when_rapidocr_is_unavailable(tmp_path, monkeypatch):
    # A configuration error (ValueError) on the requested engine must not abort the fallback chain either.
    controller, attempts, sentinel = _controller_with_engine(
        tmp_path, monkeypatch, usable="windows", requested_error=ValueError
    )

    assert controller._create_configured_ocr_engine() is sentinel
    assert attempts == ["missing_engine", "rapidocr", "windows"]
    assert controller._active_ocr_engine_name == "windows"
    assert controller.config.get("ocr.engine") == "windows"
    controller.floating_status.deleteLater()
    QApplication.instance().processEvents()


def test_window_movement_reconfirms_once_then_schedules_capture(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    controller = Application(app)
    controller._capture_session = _CaptureSession("window", False, False, False, 0x1234)
    controller._highlight_window_rect = (0, 0, 100, 100)
    controller._window_highlight_retries = 0
    controller.monitor_map = []
    reconfirmed = []
    scheduled = []
    monkeypatch.setattr(controller, "_dispose_window_highlight", lambda: None)
    monkeypatch.setattr(
        application_module,
        "get_window_rect_physical",
        lambda _hwnd: (10, 10, 110, 110),
    )
    monkeypatch.setattr(controller, "_after_window_departure", lambda: reconfirmed.append(True))
    monkeypatch.setattr(
        controller,
        "_schedule_capture",
        lambda *args, **kwargs: scheduled.append((args, kwargs)),
    )

    controller._on_window_highlight_finished()
    assert reconfirmed == [True]
    assert scheduled == []
    assert controller._window_highlight_retries == 1

    controller._on_window_highlight_finished()
    assert reconfirmed == [True]
    assert scheduled and scheduled[0][0] == ("window",)
    assert scheduled[0][1]["target_hwnd"] == 0x1234
    controller.floating_status.hide_immediate()
    controller.floating_status.deleteLater()
    app.processEvents()
