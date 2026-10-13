"""比較離線翻譯（標準）的推論引擎：CTranslate2 int8 與 llama.cpp（GGUF Q4_K_M／Q8_0）。

同一份提示詞、同一批句子、每句單獨翻譯（和即時監控一樣），每個設定在**獨立的行程**裡跑，
這樣記憶體用量才量得準、也不會互相影響。量測：

- 載入＋暖機時間、記憶體（常駐，載入模型後）
- 每句翻譯時間（中位數／平均／最慢），日文歌詞、日文一般句、英文各一組
- 輸出品質的粗略指標：譯文殘留假名的句數、擬聲詞略過的句數、與 CTranslate2 譯文相同的句數
- 並排列出每句的譯文，給人眼比較（機器指標沒有人讀可靠）

用法（在有 ctranslate2 與 llama-cpp-python 的環境，且模型已放進 models/translation/）::

    python scripts/compare_inference_engines.py
    python scripts/compare_inference_engines.py --threads 4 6 8 --reps 3

句子是自編的（不含有版權的歌詞）；想用自己的畫面驗證，請用 ``scripts/translate_benchmark.py --engine ...``。
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
}

# (顯示名稱, offline 設定)
CONFIGS = {
    "ct2-int8": {"engine": "ctranslate2", "model_name": "auto"},
    "llama-q4_k_m": {"engine": "llama.cpp", "model_name": "qwen2.5-1.5b-instruct-gguf-q4_k_m"},
    "llama-q8_0": {"engine": "llama.cpp", "model_name": "qwen2.5-1.5b-instruct-gguf-q8_0"},
}


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
    """在目前行程裡跑一個設定，把結果寫成 JSON。"""
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
        section["offline"].update(CONFIGS[name], threads=threads)
        translator = create_translator("offline", section, cache=None)
        begin = time.perf_counter()
        translator.warmup()
        load_seconds = time.perf_counter() - begin
        rss = _rss_mb()

        rows = []
        for group, sentences in CORPUS.items():
            for sentence in sentences:
                times, output = [], ""
                for _ in range(reps):
                    start = time.perf_counter()
                    output = translator.translate([sentence], "auto", "zh-hant")[0]
                    times.append((time.perf_counter() - start) * 1000)
                    translator.reset_context()
                rows.append(
                    {
                        "group": group,
                        "source": sentence,
                        "output": output,
                        "ms": times[0] if reps == 1 else statistics.median(times),
                        "kana": has_kana(output),
                        "skipped": output == sentence and group == "擬聲詞",
                    }
                )
    out.write_text(
        json.dumps(
            {"name": name, "threads": threads, "load_seconds": load_seconds, "rss_mb": rss,
             "model": translator.model_name, "rows": rows},
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
        print(f"\n[{result['name']}  threads={result['threads']}  模型={result['model']}]  載入+暖機 {result['load_seconds']:.1f}s  常駐記憶體 {rss}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=sorted(CONFIGS))
    parser.add_argument("--threads", nargs="+", type=int, default=[6], help="執行緒數（每個設定各跑一次）")
    parser.add_argument("--reps", type=int, default=1, help="每句重複幾次取中位數（預設 1：第一次，和實際使用一樣）")
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
            for name in args.configs:
                out = Path(tmp) / f"{name}-{threads}.json"
                print(f"執行 {name}（threads={threads}）…", flush=True)
                command = [sys.executable, __file__, "--worker", name, "--threads", str(threads), "--reps", str(args.reps), "--out", str(out)]
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0 or not out.exists():
                    print(f"  略過 {name}：執行失敗（模型沒下載或缺少套件？）")
                    continue
                results.append(json.loads(out.read_text(encoding="utf-8")))
    if results:
        report(results)


if __name__ == "__main__":
    main()
