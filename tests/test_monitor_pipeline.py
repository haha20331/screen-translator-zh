"""即時監控：管線結果必須能在沒有擷取 session 的情況下顯示（上游缺陷：每一輪結果都被丟掉）。"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from app.application import Application
from app.models import CaptureInfo, TextRegion


class _FakeOverlay:
    def __init__(self) -> None:
        self.shown: list[tuple[list[TextRegion], bool]] = []

    def show_regions(self, capture, regions, animate_new_only=False):
        self.shown.append((list(regions), animate_new_only))
        return True

    def clear_all(self):  # 單次擷取才會呼叫；監控不該呼叫
        raise AssertionError("monitor mode must not clear the whole overlay")

    def hide_all(self, animate=False):
        pass

    def is_visible(self):
        return True


def _payload(texts=("こんにちは",)):
    capture = CaptureInfo(image=np.zeros((40, 120, 3), dtype=np.uint8), bbox=(0, 0, 120, 40), mode="region")
    regions = [
        TextRegion(text=text, translated_text=f"譯:{text}", x=5, y=5 + index * 20, width=60, height=15)
        for index, text in enumerate(texts)
    ]
    return {"capture": capture, "regions": regions, "recognized_count": len(regions), "translated_count": len(regions), "failed_count": 0}


def _controller(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SCREEN_TRANSLATOR_CONFIG", str(tmp_path / "config.json"))
    controller = Application(app)
    controller.overlay_manager = _FakeOverlay()
    controller._monitor_active = True
    controller._capture_session = None  # 監控是在單次擷取結束後才啟動的
    controller._pipeline_succeeded = False
    return controller


def test_monitor_result_is_shown_without_a_capture_session(tmp_path, monkeypatch):
    controller = _controller(tmp_path, monkeypatch)

    controller.on_pipeline_result(_payload())

    assert controller._pipeline_succeeded is True
    assert len(controller.overlay_manager.shown) == 1
    regions, animate_new_only = controller.overlay_manager.shown[0]
    assert regions[0].translated_text == "譯:こんにちは"
    assert animate_new_only is True  # 監控只更新變化的區塊，不整屏清除
    controller.floating_status.deleteLater()


def test_identical_monitor_result_does_not_redraw_but_counts_as_success(tmp_path, monkeypatch):
    controller = _controller(tmp_path, monkeypatch)
    controller.on_pipeline_result(_payload())
    controller._pipeline_succeeded = False

    controller.on_pipeline_result(_payload())

    assert controller._pipeline_succeeded is True  # 內容沒變也是成功，不能累積失敗次數
    assert len(controller.overlay_manager.shown) == 1
    controller.floating_status.deleteLater()


def test_result_is_still_ignored_when_nothing_is_running(tmp_path, monkeypatch):
    controller = _controller(tmp_path, monkeypatch)
    controller._monitor_active = False  # 既沒有監控也沒有擷取 session：陳舊的結果應被丟棄

    controller.on_pipeline_result(_payload())

    assert controller._pipeline_succeeded is False
    assert controller.overlay_manager.shown == []
    controller.floating_status.deleteLater()


def test_result_is_ignored_during_shutdown(tmp_path, monkeypatch):
    controller = _controller(tmp_path, monkeypatch)
    controller._shutting_down = True

    controller.on_pipeline_result(_payload())

    assert controller.overlay_manager.shown == []
    controller.floating_status.deleteLater()
