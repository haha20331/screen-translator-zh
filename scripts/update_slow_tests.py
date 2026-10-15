"""從 ``pytest --durations=0`` 的輸出更新 ``tests/slow_tests.txt``（簡易版測試要略過的清單）。

用法::

    python -m pytest -q -p no:cacheprovider --durations=0 > durations.log        # 整套（約兩個半小時；可以分幾個行程平行跑再合併 log）
    python scripts/update_slow_tests.py durations.log [更多 log …] --threshold 3

每個測試的耗時 = setup + call + teardown。耗時達門檻（預設 3 秒）的測試列進清單；
最後印出「簡易版」預估的總時間（沒列進清單的測試耗時加總），確認在 30 分鐘內。
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "tests" / "slow_tests.txt"
_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s+(setup|call|teardown)\s+(\S+)")


def parse_durations(text: str) -> dict[str, float]:
    """``pytest --durations=0`` 輸出 → ``{node id: setup + call + teardown 秒數}``。"""
    totals: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        match = _LINE.match(line)
        if match:
            totals[match.group(3)] += float(match.group(1))
    return dict(totals)


def split_by_threshold(durations: dict[str, float], threshold: float) -> tuple[list[tuple[str, float]], float]:
    """回傳（要列進清單的測試，由慢到快）與「簡易版」預估總秒數。"""
    slow = sorted(((node, seconds) for node, seconds in durations.items() if seconds >= threshold), key=lambda item: -item[1])
    quick = sum(seconds for seconds in durations.values() if seconds < threshold)
    return slow, quick


def format_list(slow: list[tuple[str, float]], threshold: float) -> str:
    header = (
        "# 簡易版測試（pytest -m \"not slow\"）略過的測試：單一測試耗時達 %g 秒以上。\n"
        "# 由 scripts/update_slow_tests.py 從 pytest --durations=0 的輸出產生，請不要手動維護。\n" % threshold
    )
    return header + "".join(f"{node}  # {seconds:.1f}s\n" for node, seconds in sorted(slow, key=lambda item: item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="pytest --durations=0 的輸出檔（可以多個，會合併）")
    parser.add_argument("--threshold", type=float, default=3.0, help="達到幾秒就算慢（預設 3）")
    parser.add_argument("--dry-run", action="store_true", help="只印預估，不寫檔")
    args = parser.parse_args()
    durations: dict[str, float] = {}
    for log in args.logs:
        durations.update(parse_durations(log.read_text(encoding="utf-8", errors="replace")))
    slow, quick = split_by_threshold(durations, args.threshold)
    total = sum(durations.values())
    print(f"{len(durations)} 個測試，全部約 {total / 60:.0f} 分鐘；慢的（≥ {args.threshold:g} 秒）{len(slow)} 個，佔 {sum(s for _, s in slow) / 60:.0f} 分鐘")
    print(f"簡易版（略過慢的）{len(durations) - len(slow)} 個測試，預估約 {quick / 60:.1f} 分鐘")
    if not args.dry_run:
        LIST.write_text(format_list(slow, args.threshold), encoding="utf-8")
        print(f"已寫入 {LIST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
