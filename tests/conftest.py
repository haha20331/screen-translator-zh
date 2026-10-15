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
