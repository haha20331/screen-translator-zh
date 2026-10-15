"""提示詞調整的評分與比較：讀 ``tune_prompts.py`` 存的 JSON。

    python scripts/prompt_metrics.py summary runs/*.json [--splits dev,log-dev]
    python scripts/prompt_metrics.py diff runs/a.json runs/b.json [--splits dev] [--all]

自動指標只是輔助（沒有哪個指標能判斷「通順」），最後的好壞還是要讀並排的譯文：

- ``chrF``：譯文與參考譯文的字元 1～3 連字 F 值（β=2，偏重召回），0～100。只有自己寫的題目有參考譯文。
  參考譯文不是唯一答案，所以只看「同一組題目、不同版本」的相對變化。
- 略過擬聲詞：該略過的略過了幾句（召回）、不該略過的被略過了幾句（誤略過）。略過的譯文是原文原樣、且翻譯器沒標失敗。
- ``未譯``：翻譯器判定失敗（譯文空、抄回原文、不是中文），畫面上會顯示原文。
- ``假名／外文``：譯文裡還有假名、韓俄文、原文沒有的英文單字、字間空格、亂碼符號。
- ``文言``：譯文含 之乃吾汝爾兮矣焉亦皆猶豈遂勿莫 這些字。
- ``多主詞／少主詞``：譯文有人稱代名詞而參考譯文沒有（多）／參考有而譯文沒有（少）。中文問句與命令句常需要補「你」才通順，
  所以這不是錯誤指標，只用來看提示詞對「補主詞」傾向的影響。
- ``頓號``：譯文裡的日文頓號「、」（中文用逗號「，」；小模型常照抄日文標點，讀起來像日文直譯）。
- ``英文殘留``：英文原文的譯文裡還留著小寫英文單字（sir、reckless、probably not）；縮寫（HP、E）不算。
- ``新字體``：譯文裡有日文新字體漢字（闘、戦、気…，清單見 ``JAPANESE_ONLY_KANJI``）。
- ``SKIP混入``：「SKIP」字樣夾在譯文裡（模型把略過標記寫進句子中間）。
- ``客套回覆``：譯文是道歉或要求更多上下文（對不起、抱歉、無法翻譯…），不是翻譯。
- ``觸控``：譯文有「觸控」（OpenCC 把「觸摸」轉成台灣的觸控螢幕用詞，歌詞裡的「觸れる」被譯成觸控衣袖）。
- ``太長／太短``：字數比參考譯文的 1.8 倍長／0.5 倍短。
- ``前文洩漏``：譯文和上一句譯文有 6 個字以上的連續重疊，而原文和上一句不同（小模型把前文照抄進這一句）。
- ``重翻``：進模型不只一次的句子比例（外文／亂碼觸發重翻，每次重翻多花一次運算）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP = "[SKIP]"
CLASSICAL = "之乃吾汝爾兮矣焉亦皆猶豈遂勿莫"
PRONOUNS = "我你妳他她它祂牠咱"


# ----------------------------------------------------------------------------- 指標
def _normalize(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")


def _ngrams(text: str, n: int) -> Counter:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf(hypothesis: str, reference: str, max_n: int = 3, beta: float = 2.0) -> float:
    """字元連字 F 值（0～100）。標點與空白不算。"""
    hyp, ref = _normalize(hypothesis), _normalize(reference)
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        hyp_n, ref_n = _ngrams(hyp, n), _ngrams(ref, n)
        if not hyp_n or not ref_n:
            continue
        overlap = sum((hyp_n & ref_n).values())
        precisions.append(overlap / sum(hyp_n.values()))
        recalls.append(overlap / sum(ref_n.values()))
    if not precisions:
        return 0.0
    precision, recall = statistics.mean(precisions), statistics.mean(recalls)
    if precision + recall == 0:
        return 0.0
    return 100.0 * (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


_LATIN_WORD = re.compile(r"[A-Za-z]*[a-z][A-Za-z]*")


def latin_words(text: str) -> list[str]:
    """含小寫字母、三個字母以上的英文單字（縮寫與單一字母不算）。"""
    return [word for word in _LATIN_WORD.findall(text) if len(word) >= 3]


# 日文新字體漢字：台灣繁體不用、也不是簡體字形（簡體字形已被 OpenCC 轉成繁體），翻譯結果裡出現就是日文漢字漏進中文。
# OpenCC 的 jp2t 會連「為→爲、真→眞」這種台灣正體本來就對的字也轉掉，所以不能拿它判斷，用明確的清單。
JAPANESE_ONLY_KANJI = frozenset("気戦闘覚観続変読売発対応楽様実帰歓験縁絵経広拡帯団転伝当鉄図圧遅関桜黙臓栄恵巻歩戻沢剣険検県権歯処")


def has_japanese_forms(text: str) -> bool:
    return any(ch in JAPANESE_ONLY_KANJI for ch in text)


def has_pronoun(text: str) -> bool:
    return any(ch in PRONOUNS for ch in text)


def classical_chars(text: str) -> str:
    return "".join(ch for ch in text if ch in CLASSICAL)


def longest_common_run(a: str, b: str) -> int:
    """兩段文字最長的連續相同字數（動態規劃，句子很短）。"""
    a, b = _normalize(a), _normalize(b)
    best = 0
    previous = [0] * (len(b) + 1)
    for ch in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            if ch == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def classify(row: dict, previous: dict | None) -> dict:
    """一句的各項判定（見模組說明）。"""
    from services.translation.offline_common import looks_not_chinese
    from utils import chinese
    from utils.language_utils import has_kana

    source, reference, output = row["src"], row["ref"], row["out"]
    skipped = output.strip() == source.strip() and not row["failed"]
    wants_skip = reference == SKIP
    flags: dict[str, bool] = {
        "skipped": skipped,
        "wants_skip": wants_skip,
        "false_skip": skipped and bool(reference) and not wants_skip,
        "missed_skip": wants_skip and not skipped,
        "failed": bool(row["failed"]),
        "retried": row["calls"] > 1,
    }
    shown = not skipped and not row["failed"]
    flags["kana_foreign"] = shown and (has_kana(output) or looks_not_chinese(source, output))
    flags["classical"] = shown and bool(classical_chars(output))
    flags["subject_added"] = shown and bool(reference) and not wants_skip and has_pronoun(output) and not has_pronoun(reference)
    flags["subject_dropped"] = shown and bool(reference) and not wants_skip and has_pronoun(reference) and not has_pronoun(output)
    flags["ideographic_comma"] = shown and "、" in output
    flags["skip_leak"] = not skipped and "SKIP" in output
    flags["meta_reply"] = not skipped and chinese.is_meta_reply(output)  # 和正式程式碼同一個判斷
    flags["touch"] = shown and "觸控" in output
    flags["latin_left"] = shown and source.isascii() and bool(latin_words(output))
    flags["japanese_forms"] = shown and has_japanese_forms(output)
    scored = shown and bool(reference) and not wants_skip
    ratio = len(_normalize(output)) / max(1, len(_normalize(reference))) if scored else 1.0
    flags["too_long"] = scored and ratio > 1.8
    flags["too_short"] = scored and ratio < 0.5
    flags["leak"] = (
        shown
        and previous is not None
        and previous["src"] != source
        and longest_common_run(output, previous["out"]) >= 6
        and not previous["failed"]
    )
    return flags


def summarize(rows: list[dict]) -> dict:
    """一批句子（同一個檔、同一個 split）的彙總。"""
    previous_by_scene: dict[str, dict] = {}
    counts: Counter = Counter()
    chrf_scores: list[float] = []
    times = [row["ms"] for row in rows]
    for row in rows:
        previous = previous_by_scene.get(row["scene"])
        flags = classify(row, previous)
        previous_by_scene[row["scene"]] = row
        for name, value in flags.items():
            counts[name] += int(value)
        if row["ref"] and row["ref"] != SKIP and not flags["skipped"] and not flags["failed"]:
            chrf_scores.append(chrf(row["out"], row["ref"]))
        elif row["ref"] and row["ref"] != SKIP:
            chrf_scores.append(0.0)  # 該翻卻沒翻（略過或失敗）算零分，不然略過越多分數越好看
    return {
        "n": len(rows),
        "chrf": statistics.mean(chrf_scores) if chrf_scores else None,
        "counts": counts,
        "median_ms": statistics.median(times) if times else 0,
    }


# ----------------------------------------------------------------------------- 讀檔
def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label(data: dict) -> str:
    meta = data["meta"]
    model = meta["model"].replace("-instruct", "").replace("-gguf", "").replace("qwen2.5-", "q2.5-")
    return f"{meta['variant']} | {model} | 歌詞{'開' if meta['lyrics_mode'] else '關'}"


def _group(rows: list[dict], splits: set[str] | None) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if splits and row["split"] not in splits:
            continue
        groups[row["split"]].append(row)
    return groups


# ----------------------------------------------------------------------------- 命令
def cmd_summary(args) -> None:
    splits = set(args.splits.split(",")) if args.splits else None
    header = ["版本 | 模型 | 模式", "批", "句", "chrF", "略過(收/漏/誤)", "未譯", "假名外文", "文言", "頓號", "英文殘留", "新字體", "SKIP混入", "客套回覆", "觸控", "多主詞/少主詞", "太長/太短", "前文洩漏", "重翻", "中位ms"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for path in args.files:
        data = load(path)
        for split, rows in sorted(_group(data["rows"], splits).items()):
            s = summarize(rows)
            c = s["counts"]
            skip_cell = f"{c['skipped'] - c['false_skip']}/{c['missed_skip']}/{c['false_skip']}" if c["wants_skip"] or c["false_skip"] else f"-/-/{c['false_skip']}"
            chrf_cell = f"{s['chrf']:.1f}" if s["chrf"] is not None else "-"
            print(
                f"| {label(data)} | {split} | {s['n']} | {chrf_cell} | {skip_cell} | {c['failed']} | {c['kana_foreign']} | {c['classical']} | "
                f"{c['ideographic_comma']} | {c['latin_left']} | {c['japanese_forms']} | {c['skip_leak']} | {c['meta_reply']} | {c['touch']} | {c['subject_added']}/{c['subject_dropped']} | {c['too_long']}/{c['too_short']} | {c['leak']} | "
                f"{100 * c['retried'] / max(1, s['n']):.0f}% | {s['median_ms']:.0f} |"
            )


def cmd_diff(args) -> None:
    splits = set(args.splits.split(",")) if args.splits else None
    a, b = load(args.a), load(args.b)
    index = {(r["scene"], r["i"]): r for r in b["rows"]}
    shown = changed = total = 0
    print(f"A = {label(a)}\nB = {label(b)}\n")
    for row in a["rows"]:
        if splits and row["split"] not in splits:
            continue
        other = index.get((row["scene"], row["i"]))
        if other is None:
            continue
        total += 1
        if row["out"] == other["out"]:
            continue
        changed += 1
        if not args.all and shown >= args.limit:
            continue
        shown += 1
        ref = row["ref"]
        score_a = f" ({chrf(row['out'], ref):.0f})" if ref and ref != SKIP else ""
        score_b = f" ({chrf(other['out'], ref):.0f})" if ref and ref != SKIP else ""
        print(f"[{row['scene']}#{row['i']}] {row['src']}")
        if ref:
            print(f"   參考: {ref}")
        print(f"   A: {row['out']}{score_a}")
        print(f"   B: {other['out']}{score_b}")
    print(f"\n共 {total} 句，譯文不同 {changed} 句" + (f"（只列前 {shown} 句，--all 列全部）" if changed > shown else ""))


def _paired_scores(a: dict, b: dict, splits: set[str] | None) -> list[tuple[float, float]]:
    """兩個結果檔同一句的 chrF（沒有參考譯文的句子不算；該翻卻略過／失敗算零分）。"""
    index = {(r["scene"], r["i"]): r for r in b["rows"]}
    pairs = []
    for row in a["rows"]:
        if splits and row["split"] not in splits:
            continue
        other = index.get((row["scene"], row["i"]))
        if other is None or not row["ref"] or row["ref"] == SKIP:
            continue

        def one(r):
            skipped = r["out"].strip() == r["src"].strip() and not r["failed"]
            return 0.0 if (skipped or r["failed"]) else chrf(r["out"], r["ref"])

        pairs.append((one(row), one(other)))
    return pairs


def cmd_paired(args) -> None:
    import random

    splits = set(args.splits.split(",")) if args.splits else None
    a, b = load(args.a), load(args.b)
    pairs = _paired_scores(a, b, splits)
    diffs = [y - x for x, y in pairs]
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(args.rounds))
    low, high = means[int(0.025 * args.rounds)], means[int(0.975 * args.rounds)]
    better = sum(1 for d in diffs if d > 1)
    worse = sum(1 for d in diffs if d < -1)
    verdict = "B 較好" if low > 0 else "B 較差" if high < 0 else "沒有顯著差異"
    print(f"A = {label(a)}\nB = {label(b)}\n{len(diffs)} 句：chrF 平均 A {statistics.mean(x for x, _ in pairs):.1f} → B {statistics.mean(y for _, y in pairs):.1f}，"
          f"差 {statistics.mean(diffs):+.1f}（95% 信賴區間 {low:+.1f} ～ {high:+.1f}）→ {verdict}；B 較好 {better} 句、較差 {worse} 句")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summary", help="每個結果檔、每批題目的彙總表")
    p_sum.add_argument("files", nargs="+")
    p_sum.add_argument("--splits", default="")
    p_sum.set_defaults(func=cmd_summary)
    p_diff = sub.add_parser("diff", help="兩個結果檔逐句並排，只列譯文不同的句子")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_diff.add_argument("--splits", default="")
    p_diff.add_argument("--all", action="store_true")
    p_diff.add_argument("--limit", type=int, default=60)
    p_diff.set_defaults(func=cmd_diff)
    p_pair = sub.add_parser("paired", help="兩個結果檔的 chrF 配對自助法（句子重抽樣 95% 信賴區間）")
    p_pair.add_argument("a")
    p_pair.add_argument("b")
    p_pair.add_argument("--splits", default="")
    p_pair.add_argument("--rounds", type=int, default=2000)
    p_pair.set_defaults(func=cmd_paired)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
