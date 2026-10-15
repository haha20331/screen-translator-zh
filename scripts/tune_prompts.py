"""提示詞調整：用題庫（``prompt_corpus``）把每個場景依序翻過，把原文、參考譯文、譯文、耗時、進模型次數存成 JSON。

一次載入一個模型，依序跑多個提示詞版本（``prompt_variants``）與歌詞模式的開／關。設定固定為：
上下文開（兩句）、略過擬聲詞開、避免夾雜外文（生成時擋詞＋重翻）開，貪婪解碼，和程式實際執行時一樣。
翻譯器、引擎都是正式的 ``OfflineTranslator``；快取關閉，每個場景開始前清掉前文。

用法（Windows 原生 Python，才有 llama-cpp-python）::

    python scripts/tune_prompts.py --model qwen2.5-3b-instruct-gguf-q4_k_m --variants v4,production \\
        --splits dev --out val_data/prompt_tuning/runs

結果檔（內含紀錄檔來的真實歌詞）一律放在 ``val_data/``（不進版控）。評分與比較見 ``scripts/prompt_metrics.py``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import prompt_corpus  # noqa: E402
import prompt_variants  # noqa: E402

DEFAULT_LOGS = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ScreenTranslator", "output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="模型資料夾名稱，例如 qwen2.5-3b-instruct-gguf-q4_k_m")
    parser.add_argument("--variants", default="production", help="提示詞版本，逗號分隔（見 prompt_variants.VARIANTS）")
    parser.add_argument("--modes", default="off,on", help="歌詞模式：off、on，逗號分隔")
    parser.add_argument("--splits", default="dev,holdout,log-dev,log-holdout")
    parser.add_argument("--out", default=str(ROOT / "val_data" / "prompt_tuning" / "runs"))
    parser.add_argument("--logs", default=DEFAULT_LOGS, help="程式的輸出紀錄資料夾（取真實歌詞用）")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--gpu-layers", type=int, default=-1, help="-1 = 有 GPU 版就全上（和程式預設一樣），0 = 只用 CPU")
    parser.add_argument("--context-size", type=int, default=2)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--no-skip", action="store_true", help="關掉「略過擬聲詞」")
    parser.add_argument("--no-retry", action="store_true", help="關掉「避免譯文夾雜外文」")
    parser.add_argument("--label", default="", help="結果檔與 meta 用的版本名稱（預設用版本名；關掉上下文等旗標的對照組用它區分）")
    parser.add_argument("--max-scenes", type=int, default=0, help="每個批次最多跑幾個場景（冒煙測試用）")
    args = parser.parse_args()

    splits = [item for item in args.splits.split(",") if item]
    scenes = [s for s in prompt_corpus.SCENES if s["split"] in splits]
    if any(item.startswith("log-") for item in splits):
        scenes += [s for s in prompt_corpus.log_scenes(args.logs) if s["split"] in splits]
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    # 在匯入 app 之前先把使用者資料夾指到暫存資料夾，不碰真正的設定、快取、輸出紀錄
    scratch = tempfile.mkdtemp(prefix="ocrtune_")
    os.environ["LOCALAPPDATA"] = scratch
    os.environ["SCREEN_TRANSLATOR_CONFIG"] = str(Path(scratch) / "config.json")

    from app.config import AppConfig
    from services.translation import offline_translator as module
    from services.translation.factory import create_translator

    config = AppConfig(Path(scratch) / "config.json")
    section = config.section("translation")
    section.update(
        use_context=not args.no_context,
        context_size=args.context_size,
        skip_sounds=not args.no_skip,
        retry_non_chinese=not args.no_retry,
    )
    section["offline"].update(model_name=args.model, threads=args.threads, gpu_layers=args.gpu_layers)
    translator = create_translator("offline", section, cache=None)
    started = time.perf_counter()
    translator.warmup()
    print(f"載入＋預熱 {time.perf_counter() - started:.1f} 秒，裝置＝{translator.device}，{len(scenes)} 個場景、"
          f"{sum(len(s['lines']) for s in scenes)} 句", flush=True)

    engine = translator._llama
    counter = {"calls": 0}
    original_complete = engine.complete

    def counting_complete(*a, **k):
        counter["calls"] += 1
        return original_complete(*a, **k)

    engine.complete = counting_complete
    original_functions = (module.static_prompt, module.history_prompt, module.user_prompt)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model.replace("-instruct", "").replace("-gguf", "")
    for variant in [v for v in args.variants.split(",") if v]:
        module.static_prompt, module.history_prompt, module.user_prompt = original_functions
        spec = prompt_variants.get(variant)
        if spec is not None:
            prompt_variants.install(module, spec)
        for mode in [m for m in args.modes.split(",") if m]:
            translator.config["lyrics_mode"] = mode == "on"
            rows = []
            run_started = time.perf_counter()
            for scene in scenes:
                translator.reset_context()
                for index, (source, reference) in enumerate(scene["lines"]):
                    counter["calls"] = 0
                    began = time.perf_counter()
                    try:
                        output = translator.translate([source], "auto", "zh-hant")[0]
                        failed = 0 in translator.last_failed_indices
                    except Exception as exc:  # 單句失敗不要讓整批白跑
                        output, failed = f"<<錯誤：{exc}>>", True
                    rows.append({
                        "scene": scene["id"], "kind": scene["kind"], "lang": scene["lang"], "split": scene["split"],
                        "i": index, "src": source, "ref": reference, "out": output, "failed": failed,
                        "calls": counter["calls"], "ms": round((time.perf_counter() - began) * 1000, 1),
                    })
            meta = {
                "model": args.model, "variant": args.label or variant, "lyrics_mode": mode == "on", "device": translator.device,
                "context": not args.no_context, "context_size": args.context_size, "skip_sounds": not args.no_skip,
                "retry_non_chinese": not args.no_retry, "threads": args.threads, "gpu_layers": args.gpu_layers,
                "prompt_version": module.PROMPT_VERSION, "seconds": round(time.perf_counter() - run_started, 1),
            }
            path = out_dir / f"{args.label or variant}__{short}__lyrics-{mode}.json"
            path.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
            median = sorted(r["ms"] for r in rows)[len(rows) // 2] if rows else 0
            print(f"  {args.label or variant} / 歌詞模式 {mode}: {len(rows)} 句，{meta['seconds']} 秒（中位 {median:.0f} ms）→ {path.name}", flush=True)
    translator.close()
    sys.stdout.flush()
    os._exit(0)  # llama.cpp 的 Vulkan 版結束時偶爾卡住；結果都已經存好


if __name__ == "__main__":
    main()
