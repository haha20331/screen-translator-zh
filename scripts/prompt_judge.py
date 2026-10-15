"""盲測評審：把兩個提示詞版本的譯文去掉標籤、隨機排順序，交給評審（另一個不知道哪個是新版的 Claude 子代理）打分，再還原。

    python scripts/prompt_judge.py build A.json B.json --out DIR/tag [--splits dev] [--max 120]
        → DIR/tag.pairs.json（給評審看的，沒有版本名稱）與 DIR/tag.key.json（誰是 1 號、誰是 2 號，評審看不到）
    python scripts/prompt_judge.py score DIR/tag.key.json DIR/tag.verdicts.json
        → 勝負表、平均保真度／通順度、符號檢定（只算兩邊譯文不同的句子）

評審回傳的 verdicts 檔格式：``{"<id>": {"winner": "1"|"2"|"tie", "fidelity": [x, y], "fluency": [x, y]}}``，
分數 1～5：保真度＝意思有沒有完整、正確（漏譯、加譯、否定／主詞／人名譯錯都扣分）；
通順度＝是不是自然的台灣中文字幕（不像直譯、沒有日文標點與殘留外文）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def _norm(text: str) -> str:
    return "".join(text.split())


def build(args) -> None:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    splits = set(args.splits.split(",")) if args.splits else None
    index_b = {(r["scene"], r["i"]): r for r in b["rows"]}
    by_scene: dict[str, list[dict]] = {}
    for row in a["rows"]:
        by_scene.setdefault(row["scene"], []).append(row)
    pairs, key = [], {}
    for row in a["rows"]:
        if splits and row["split"] not in splits:
            continue
        other = index_b.get((row["scene"], row["i"]))
        if other is None or _norm(row["out"]) == _norm(other["out"]):
            continue  # 兩邊一樣不必評
        item_id = f"{a['meta']['model'][:9]}-{'on' if a['meta']['lyrics_mode'] else 'off'}-{row['scene']}-{row['i']}"
        flip = int(hashlib.sha256((args.salt + item_id).encode()).hexdigest(), 16) % 2 == 1  # 固定但看不出規律的隨機
        first, second = (other, row) if flip else (row, other)
        context = [r["src"] for r in by_scene[row["scene"]][max(0, row["i"] - 2) : row["i"]]]
        pairs.append({
            "id": item_id,
            "type": "歌詞" if row["kind"] == "lyrics" else "台詞",
            "language": "日文" if row["lang"] == "ja" else "英文",
            "previous_lines": context,
            "source": row["src"],
            "human_reference": row["ref"] if row["ref"] and row["ref"] != "[SKIP]" else None,
            "skip_expected": row["ref"] == "[SKIP]",
            "candidate_1": first["out"],
            "candidate_2": second["out"],
        })
        model = a["meta"]["model"].replace("-instruct", "").replace("-gguf", "")
        key[item_id] = {
            "1": a["meta"]["variant"] if first is row else b["meta"]["variant"],
            "2": a["meta"]["variant"] if second is row else b["meta"]["variant"],
            "group": f"{model} 歌詞{'開' if a['meta']['lyrics_mode'] else '關'} {pairs[-1]['type']}",
        }
    # 同一個設定的多個結果檔可以重複呼叫 build；--append 把新的接在舊的後面
    out = Path(args.out)
    pairs_path, key_path = out.with_suffix(".pairs.json"), out.with_suffix(".key.json")
    if args.append and pairs_path.exists():
        pairs = json.loads(pairs_path.read_text(encoding="utf-8")) + pairs
        key = {**json.loads(key_path.read_text(encoding="utf-8")), **key}
    if args.max and len(pairs) > args.max and not args.append:
        step = len(pairs) / args.max
        pairs = [pairs[int(i * step)] for i in range(args.max)]
        key = {p["id"]: key[p["id"]] for p in pairs}
    pairs_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(pairs)} 組 → {pairs_path.name}（答案：{key_path.name}）")


def sign_test_p(wins: int, losses: int) -> float:
    """雙尾符號檢定（平手不算）。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def score(args) -> None:
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
    variants = sorted({v for pair in key.values() for name, v in pair.items() if name in ("1", "2")})
    wins = {v: 0 for v in variants}
    ties = 0
    fidelity: dict[str, list[float]] = {v: [] for v in variants}
    fluency: dict[str, list[float]] = {v: [] for v in variants}
    groups: dict[str, dict[str, int]] = {}
    missing = 0
    for item_id, names in key.items():
        verdict = verdicts.get(item_id)
        if not verdict:
            missing += 1
            continue
        group = groups.setdefault(names["group"], {v: 0 for v in variants} | {"tie": 0})
        winner = str(verdict.get("winner", "tie"))
        if winner in ("1", "2"):
            wins[names[winner]] += 1
            group[names[winner]] += 1
        else:
            ties += 1
            group["tie"] += 1
        for number in ("1", "2"):
            index = int(number) - 1
            if verdict.get("fidelity"):
                fidelity[names[number]].append(float(verdict["fidelity"][index]))
            if verdict.get("fluency"):
                fluency[names[number]].append(float(verdict["fluency"][index]))
    judged = len(key) - missing
    print(f"評了 {judged} 組（缺 {missing}），平手 {ties}")
    for variant in variants:
        f1 = statistics.mean(fidelity[variant]) if fidelity[variant] else float("nan")
        f2 = statistics.mean(fluency[variant]) if fluency[variant] else float("nan")
        print(f"  {variant}: 勝 {wins[variant]}，保真度 {f1:.2f}，通順度 {f2:.2f}")
    if len(variants) == 2:
        first, second = variants
        print(f"  符號檢定 p = {sign_test_p(wins[first], wins[second]):.3f}（{first} {wins[first]} : {wins[second]} {second}）")
    for name, counts in sorted(groups.items()):
        print(f"  [{name}] " + "，".join(f"{k} {v}" for k, v in counts.items()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("a")
    p_build.add_argument("b")
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--splits", default="")
    p_build.add_argument("--max", type=int, default=0)
    p_build.add_argument("--salt", default="judge")
    p_build.add_argument("--append", action="store_true")
    p_build.set_defaults(func=build)
    p_score = sub.add_parser("score")
    p_score.add_argument("key")
    p_score.add_argument("verdicts")
    p_score.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
