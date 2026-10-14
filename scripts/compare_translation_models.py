"""比較離線翻譯的模型（llama.cpp，Qwen2.5 的 0.5B／1.5B／3B／7B 與不同量化）。

同一份提示詞、同一批句子、每句單獨翻譯（和即時監控一樣），每個模型在**獨立的行程**裡跑，
這樣記憶體用量才量得準、也不會互相影響。量測：

- 載入＋暖機時間、記憶體（常駐，載入模型後）
- 每句翻譯時間（中位數／平均／最慢），日文歌詞、日文一般句、英文各一組
- 輸出品質的粗略指標：譯文殘留假名的句數、擬聲詞略過的句數、與基準模型（第一個）譯文相同的句數
- 並排列出每句的譯文，給人眼比較（機器指標沒有人讀可靠）

用法（模型要先用 scripts/fetch_translation_models.py 放進 models/translation/；沒下載的會自動略過）::

    python scripts/compare_translation_models.py
    python scripts/compare_translation_models.py --models qwen2.5-0.5b-instruct-gguf-q8_0 qwen2.5-7b-instruct-gguf-q4_k_m
    python scripts/compare_translation_models.py --threads 4 6 8 --reps 3
    python scripts/compare_translation_models.py --reps 3 --report MODEL_COMPARISON.md     # 另外存成 Markdown 報表

換模型比較時，翻譯快取是分開存的（鍵含模型名稱），本腳本也不使用快取，所以不用先清快取。
句子是自編的（不含有版權的歌詞）；想用自己的畫面驗證，請用 ``scripts/translate_benchmark.py --model ...``。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORPUS = {
    "日文歌詞": [
        "夜空に浮かぶ星を数えて", "君の名前を呼び続けた", "雨の中で笑っていたあの日", "遠い街へ旅立つ朝に",
        "風に乗せて届けたい想い", "繋いだ手を離さないで", "夢の続きを探しに行こう", "涙の数だけ強くなれる",
        "静かな海に沈む夕日", "約束の場所で待っている", "さよならの代わりにありがとう",
    ],
    "日文一般": [
        "今日はいい天気ですね。", "ファイルを保存しますか？", "HPが50回復した！", "次のステージに進みますか？",
        "お腹がすいたので何か食べよう", "電車が遅れているようです", "この道をまっすぐ行ってください",
        "設定を変更しました", "ネットワークに接続できません", "はい、わかりました",
    ],
    "英文": [
        "Press any key to continue", "You found a rusty key.", "Save changes before closing?",
        "I will be there in five minutes", "Where are you going?", "The door is locked",
        "Thank you for your help", "Let's meet again tomorrow", "Please wait a moment", "Game over, try again?",
    ],
    "擬聲詞": ["ドン！", "ガシャーン！", "ははは", "Boom!", "Haha"],
    # 原文沒有主詞：譯文也不該自己加「我」「你」（提示詞有這條規則，小模型不一定每句都聽）
    "無主詞": ["会いたい", "帰りたい", "行ってきます", "好きだよ", "君を愛している"],
    "有主詞": ["私は学生です", "彼は走っている"],
}

# 報表（--report）逐句並排比較的 10 句：歌詞、介面／日常、無主詞、英文各取幾句
REPORT_SENTENCES = [
    "夜空に浮かぶ星を数えて",
    "君の名前を呼び続けた",
    "さよならの代わりにありがとう",
    "次のステージに進みますか？",
    "電車が遅れているようです",
    "お腹がすいたので何か食べよう",
    "会いたい",
    "行ってきます",
    "I will be there in five minutes",
    "Let's meet again tomorrow",
]

# 預設比較的模型（由小到大）；沒有下載的會略過
DEFAULT_MODELS = [
    "qwen2.5-0.5b-instruct-gguf-q8_0",
    "qwen2.5-1.5b-instruct-gguf-q4_k_m",
    "qwen2.5-1.5b-instruct-gguf-q8_0",
    "qwen2.5-3b-instruct-gguf-q4_k_m",
    "qwen2.5-7b-instruct-gguf-q4_k_m",
]


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:  # Linux / WSL
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    if sys.platform == "win32":  # Windows：不靠 psutil，直接問系統這個行程的工作集大小
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # 沒設型別的話 64 位元的代號會被截斷
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1e6
    return None


def run_worker(name: str, threads: int, reps: int, out: Path) -> None:
    """在目前行程裡跑一個模型，把結果寫成 JSON。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows 上日誌檔還開著，清不掉也不要報錯
        os.environ["LOCALAPPDATA"] = tmp  # 日誌與設定都放暫存資料夾，不寫進使用者真正的資料（要在匯入 app 之前設）
        sys.path.insert(0, str(ROOT))
        import services.translation.offline_translator  # noqa: F401
        from app.config import AppConfig
        from services.translation.factory import create_translator
        from utils.language_utils import has_kana

        config = AppConfig(Path(tmp) / "config.json")
        section = config.section("translation")
        section["offline"].update(model_name=name, threads=threads)
        translator = create_translator("offline", section, cache=None)
        begin = time.perf_counter()
        translator.warmup()
        load_seconds = time.perf_counter() - begin
        rss = _rss_mb()

        from services.translation.offline_common import looks_not_chinese

        calls = [0]
        engine_complete = translator._llama.complete

        def counting_complete(*args, **kwargs):
            calls[0] += 1
            return engine_complete(*args, **kwargs)

        translator._llama.complete = counting_complete  # 數每句進了幾次模型（大於 1 = 有重翻）

        # 重複量測要「整批句子」輪流跑（外圈是第幾輪），不能同一句連續翻好幾次：連續翻同一句時提示詞完全相同，
        # llama.cpp 會直接重用上一次的 KV 快取、幾乎不用預填，量出來會比真實的字幕（每句都不同）樂觀很多
        collected: dict[str, dict] = {}
        for _ in range(reps):
            for group, sentences in CORPUS.items():
                for sentence in sentences:
                    calls[0] = 0
                    start = time.perf_counter()
                    output = translator.translate([sentence], "auto", "zh-hant")[0]
                    elapsed = (time.perf_counter() - start) * 1000
                    entry = collected.setdefault(sentence, {"group": group, "times": []})
                    entry.update(output=output, calls=calls[0])
                    entry["times"].append(elapsed)
                    translator.reset_context()
        rows = [
            {
                "group": entry["group"],
                "source": sentence,
                "output": entry["output"],
                "ms": entry["times"][0] if reps == 1 else statistics.median(entry["times"]),
                "calls": entry["calls"],
                "kana": has_kana(entry["output"]),
                "not_chinese": looks_not_chinese(sentence, entry["output"]) and entry["group"] != "擬聲詞",
                "skipped": entry["output"] == sentence and entry["group"] == "擬聲詞",
            }
            for sentence, entry in collected.items()
        ]
    out.write_text(
        json.dumps(
            {"name": name, "threads": threads, "load_seconds": load_seconds, "rss_mb": rss,
             "model": translator.model_name, "device": translator.device, "rows": rows},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _stats(rows: list[dict]) -> str:
    ms = [row["ms"] for row in rows]
    return f"中位 {statistics.median(ms):5.0f}  平均 {statistics.mean(ms):5.0f}  最慢 {max(ms):5.0f}"


def report(results: list[dict]) -> None:
    print("\n=== 速度與記憶體（每句單獨翻譯，ms）")
    for result in results:
        rows = result["rows"]
        rss = f"{result['rss_mb']:.0f} MB" if result["rss_mb"] else "?"
        print(f"\n[{result['name']}  threads={result['threads']}  {result.get('device', 'cpu')}]  載入+暖機 {result['load_seconds']:.1f}s  常駐記憶體 {rss}")
        for group in CORPUS:
            print(f"  {group:6s} {_stats([r for r in rows if r['group'] == group])}")
        print(f"  全部   {_stats(rows)}")

    baseline = {row["source"]: row["output"] for row in results[0]["rows"]}
    print(f"\n=== 品質的粗略指標（以 {results[0]['name']} 為基準）")
    for result in results:
        rows = result["rows"]
        spoken = [r for r in rows if r["group"] != "擬聲詞"]
        same = sum(1 for r in spoken if r["output"] == baseline[r["source"]])
        skipped = sum(1 for r in rows if r["skipped"])
        print(
            f"[{result['name']}]  譯文含假名 {sum(r['kana'] for r in spoken)}/{len(spoken)}  "
            f"擬聲詞略過 {skipped}/{len(CORPUS['擬聲詞'])}  與基準譯文相同 {same}/{len(spoken)}"
        )

    print("\n=== 並排譯文（人眼比較）")
    for index, row in enumerate(results[0]["rows"]):
        print(f"\n{row['group']}｜{row['source']}")
        for result in results:
            print(f"  {result['name']:14s} {result['rows'][index]['output']}")


def short_name(name: str) -> str:
    """qwen2.5-1.5b-instruct-gguf-q4_k_m → 1.5B Q4_K_M（報表欄位用的短名稱）。"""
    size, _, quant = name.removeprefix("qwen2.5-").partition("-instruct-gguf-")
    return f"{size.upper()} {quant.upper()}" if quant else name


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_report(results: list[dict], path: Path, reps: int) -> None:
    """把比較結果存成 Markdown：環境、耗時、品質指標、逐句並排比較。人工的觀察與建議由人另外補在最後。"""
    import datetime
    import platform

    try:
        import llama_cpp

        engine = f"llama-cpp-python {llama_cpp.__version__}"
    except ImportError:
        engine = "llama-cpp-python"
    names = [result["name"] for result in results]
    short = {name: short_name(name) for name in names}
    lines = [
        "# 離線翻譯模型比較",
        "",
        f"- 日期：{datetime.date.today().isoformat()}",
        f"- 環境：{platform.system()} {platform.release()}，{platform.processor() or platform.machine()}，{os.cpu_count()} 個邏輯處理器；{engine}；Python {platform.python_version()}",
        f"- 方法：每個模型各在獨立行程裡跑同一批 {sum(len(v) for v in CORPUS.values())} 句、每句單獨翻譯（和即時監控一樣）、"
        f"執行緒 {results[0]['threads']}、整批句子輪流重複 {reps} 輪、每句取中位數（不連續翻同一句：同一句的提示詞完全相同時 llama.cpp 會直接重用快取，量出來太樂觀）；"
        "不使用翻譯快取；提示詞、上下文與歌詞模式關閉、擬聲詞略過與「避免譯文夾雜外文／不是中文時重翻」開啟（預設）",
        "- 注意：耗時是在一台同時還開著其他程式的電腦上量的，同一個設定前後可以差一倍；請看比例，不要看絕對值",
        "",
        "## 耗時與記憶體",
        "",
        "每句耗時是「從交給翻譯器到拿到譯文」，含重翻；單位毫秒。",
        "",
        "| 模型 | 載入+暖機 | 常駐記憶體 | 全部（中位／平均／最慢） | 日文歌詞 | 日文一般 | 英文 | 無主詞 | 擬聲詞 | 平均進模型次數／句 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        rows = result["rows"]

        def median(group):
            values = [r["ms"] for r in rows if r["group"] == group]
            return f"{statistics.median(values):.0f}" if values else "-"

        everything = [r["ms"] for r in rows]
        rss = f"{result['rss_mb'] / 1000:.1f} GB" if result["rss_mb"] else "?"
        average_calls = statistics.mean(r["calls"] for r in rows)
        lines.append(
            f"| {short[result['name']]} | {result['load_seconds']:.1f} 秒 | {rss} | "
            f"**{statistics.median(everything):.0f}**／{statistics.mean(everything):.0f}／{max(everything):.0f} | "
            f"{median('日文歌詞')} | {median('日文一般')} | {median('英文')} | {median('無主詞')} | {median('擬聲詞')} | {average_calls:.2f} |"
        )
    baseline = results[0]
    default = next((r for r in results if r["name"].endswith("1.5b-instruct-gguf-q4_k_m")), baseline)
    base_out = {r["source"]: r["output"] for r in default["rows"]}
    lines += [
        "",
        "「平均進模型次數／句」大於 1 代表有些句子因為譯文夾雜假名、外文或原文沒有的英文單字，改用取樣重翻了（每次重翻多花一次模型運算的時間）。",
        "",
        "## 譯文品質的機器指標（粗略）",
        "",
        f"基準：{short[default['name']]}（最快的實用模型，拿來當基準）。譯文相同只是「和基準一樣」，不代表比較好或比較差；真正的品質請看下一節的並排比較。",
        "",
        "| 模型 | 譯文仍夾雜假名／外文／多餘英文單字 | 擬聲詞略過 | 和基準譯文相同 |",
        "|---|---|---|---|",
    ]
    for result in results:
        rows = [r for r in result["rows"] if r["group"] != "擬聲詞"]
        mixed = sum(r["not_chinese"] for r in rows)
        skipped = sum(r["skipped"] for r in result["rows"])
        same = sum(r["output"] == base_out[r["source"]] for r in rows)
        lines.append(f"| {short[result['name']]} | {mixed}／{len(rows)} | {skipped}／{len(CORPUS['擬聲詞'])} | {same}／{len(rows)} |")

    lines += ["", "## 逐句並排比較（10 句）", "", "括號內是該模型翻這一句的耗時（毫秒）。", ""]
    lines.append("| 原文 | " + " | ".join(short[name] for name in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for sentence in REPORT_SENTENCES:
        cells = []
        for result in results:
            row = next((r for r in result["rows"] if r["source"] == sentence), None)
            cells.append(f"{_cell(row['output'])}<br>({row['ms']:.0f})" if row else "-")
        lines.append(f"| {_cell(sentence)} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"報表已存到 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="要比較的模型（第一個是品質比較的基準）")
    parser.add_argument("--threads", nargs="+", type=int, default=[6], help="執行緒數（每個設定各跑一次）")
    parser.add_argument("--reps", type=int, default=1, help="整批句子輪流重複幾輪、每句取中位數（預設 1：只量一輪）")
    parser.add_argument("--report", type=Path, help="另外把結果存成 Markdown 報表（耗時表、品質指標、10 句並排比較）")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows 預設編碼（cp950）印不出所有字；導向檔案時用 UTF-8，終端機則不丟例外
        sys.stdout.reconfigure(errors="replace") if sys.stdout.isatty() else sys.stdout.reconfigure(encoding="utf-8")

    if args.worker:
        run_worker(args.worker, args.threads[0], args.reps, args.out)
        return

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for threads in args.threads:
            for name in args.models:
                out = Path(tmp) / f"{name}-{threads}.json"
                print(f"執行 {name}（threads={threads}）…", flush=True)
                command = [sys.executable, __file__, "--worker", name, "--threads", str(threads), "--reps", str(args.reps), "--out", str(out)]
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0 or not out.exists():
                    print(f"  略過 {name}：執行失敗（模型沒下載？請執行 scripts/fetch_translation_models.py --model {name}）")
                    continue
                results.append(json.loads(out.read_text(encoding="utf-8")))
    if results:
        report(results)
        if args.report:
            write_report(results, args.report, args.reps)


if __name__ == "__main__":
    main()
