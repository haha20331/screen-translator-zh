"""簡易版測試：慢測試清單的產生與標記。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest as conftest_module

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = """\
============================= slowest durations =============================
12.50s call     tests/test_a.py::test_slow
0.50s setup    tests/test_a.py::test_slow
0.10s teardown tests/test_a.py::test_slow
2.00s call     tests/test_a.py::test_medium[p1]
0.30s call     tests/test_b.py::test_fast
(3 durations < 0.005s hidden.  Use -vv to show these durations.)
======================= 1 failed, 986 passed in 8867.14s =======================
"""


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("update_slow_tests", ROOT / "scripts" / "update_slow_tests.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["update_slow_tests"] = module
    spec.loader.exec_module(module)
    return module


def test_a_tests_time_is_setup_plus_call_plus_teardown(script):
    durations = script.parse_durations(SAMPLE)

    assert durations == {"tests/test_a.py::test_slow": pytest.approx(13.1), "tests/test_a.py::test_medium[p1]": 2.0, "tests/test_b.py::test_fast": 0.3}


def test_the_threshold_decides_which_tests_are_left_out_of_the_quick_run(script):
    durations = script.parse_durations(SAMPLE)

    slow, quick = script.split_by_threshold(durations, 3.0)

    assert [node for node, _ in slow] == ["tests/test_a.py::test_slow"]
    assert quick == pytest.approx(2.3)  # 簡易版只剩下沒列進清單的測試
    assert [node for node, _ in script.split_by_threshold(durations, 1.0)[0]] == ["tests/test_a.py::test_slow", "tests/test_a.py::test_medium[p1]"]


def test_the_list_file_has_one_node_id_per_line_with_the_time_as_a_comment(script, tmp_path):
    text = script.format_list([("tests/test_b.py::test_two", 5.0), ("tests/test_a.py::test_one", 9.25)], 3.0)

    lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert lines == ["tests/test_a.py::test_one  # 9.2s", "tests/test_b.py::test_two  # 5.0s"]  # 依名稱排序，差異容易看
    path = tmp_path / "slow_tests.txt"
    path.write_text(text, encoding="utf-8")
    original = conftest_module.SLOW_LIST
    conftest_module.SLOW_LIST = path
    try:
        assert conftest_module.slow_test_ids() == {"tests/test_a.py::test_one", "tests/test_b.py::test_two"}  # 註解與空行不算
    finally:
        conftest_module.SLOW_LIST = original


def test_listed_tests_get_the_slow_marker_and_others_do_not(monkeypatch):
    monkeypatch.setattr(conftest_module, "slow_test_ids", lambda: {"tests/test_a.py::test_slow"})
    slow_item = SimpleNamespace(nodeid="tests/test_a.py::test_slow", markers=[], add_marker=lambda marker: slow_item.markers.append(marker))
    fast_item = SimpleNamespace(nodeid="tests/test_b.py::test_fast", markers=[], add_marker=lambda marker: fast_item.markers.append(marker))

    conftest_module.pytest_collection_modifyitems(None, [slow_item, fast_item])

    assert [marker.name for marker in slow_item.markers] == ["slow"] and fast_item.markers == []


def test_a_missing_list_means_nothing_is_slow(monkeypatch, tmp_path):
    monkeypatch.setattr(conftest_module, "SLOW_LIST", tmp_path / "nope.txt")

    assert conftest_module.slow_test_ids() == set()


def test_every_file_in_the_checked_in_list_still_exists():
    stale = [node for node in conftest_module.slow_test_ids() if not (ROOT / node.split("::")[0]).exists()]

    assert not stale, f"slow_tests.txt 裡有已經不存在的測試檔，請用 scripts/update_slow_tests.py 重新產生：{stale}"
