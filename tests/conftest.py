"""測試的「簡易版」。

整套測試約要兩個半小時（即時動畫、真實模型、逐步等待的 UI 測試），日常只跑簡易版：

    python -m pytest -q -m "not slow"

`tests/slow_tests.txt` 列出單一測試要好幾秒以上的測試（一行一個 node id），這裡把它們標成 ``slow``，簡易版就略過它們，
30 分鐘內跑得完。要完整的整套就不加 ``-m``（只在需要時才跑）。清單用 ``scripts/update_slow_tests.py`` 從
``pytest --durations=0`` 的輸出更新。
"""

from __future__ import annotations

from pathlib import Path

import pytest

SLOW_LIST = Path(__file__).with_name("slow_tests.txt")


def slow_test_ids() -> set[str]:
    try:
        lines = SLOW_LIST.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()}


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: 單一測試要好幾秒以上（即時動畫、真實模型…）；簡易版用 -m 'not slow' 略過")


def pytest_collection_modifyitems(config, items) -> None:
    slow = slow_test_ids()
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _really_delete_widgets_after_each_test():
    """每個測試結束就把 ``deleteLater()`` 排進去的視窗真的刪掉。

    測試裡沒有事件迴圈，``deleteLater()`` 的視窗不會自己消失，整套連續跑時越積越多，
    而每次 ``apply_style``（``qapp`` 夾具）都要把樣式重新套到所有還活著的視窗上：實測每多一個設定視窗就多約 0.14 秒，
    跑到後半段，光是一個測試的準備動作就要一分多鐘，整套因此從約 16 分鐘拖到兩個半小時。
    """
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
    except ImportError:  # 沒裝 PySide6 的環境（純邏輯測試）
        return
    app = QCoreApplication.instance()
    if app is not None:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def _remove_leaked_capture_filters_after_each_test():
    """每個 ``Application(qapp)`` 都會在整個程式上裝一個「新視窗排除擷取」的事件過濾器，測試結束時沒人拿掉。

    過濾器是 Python 寫的，程式裡每一個事件（滑鼠、繪圖、計時）都要先過一遍所有過濾器：越積越多，
    後面的測試就越來越慢，時間敏感的動畫測試（例如設定頁關閉動畫要停在 10%～90% 之間）因此偶爾錯過那個區間。
    """
    yield
    try:
        from PySide6.QtCore import QCoreApplication
    except ImportError:  # 沒裝 PySide6 的環境（純邏輯測試）
        return
    app = QCoreApplication.instance()
    if app is None:
        return
    from utils.window_affinity import CaptureExclusionFilter

    for leaked in app.findChildren(CaptureExclusionFilter):
        app.removeEventFilter(leaked)
        leaked.deleteLater()


@pytest.fixture(autouse=True)
def _no_real_gpu_scan(monkeypatch):
    """設定頁開啟時會在背景偵測 GPU（DXGI 與 Vulkan 初始化，這台電腦第一次要半秒以上）；測試一律用固定的結果，
    不碰真正的顯示卡，結果才不會因電腦而異。要測偵測本身的測試在模組載入時就先拿到真正的函式。"""
    from services.translation import gpu_info

    monkeypatch.setattr(gpu_info, "detect_gpus", lambda: gpu_info.GpuReport((), False))
