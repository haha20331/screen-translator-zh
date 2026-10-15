"""整條管線（OCR → 合併 → 翻譯）每張圖要多久，用來比較 CPU 版與 GPU（Vulkan）版的 llama-cpp-python。

``compare_translation_models.py`` 只量「翻譯」；這支把 OCR 也算進去，量的是使用者實際等的時間：
一張字幕截圖從交給管線到拿到譯文。耗時拆成三份：

- OCR（含縮放、合併、取色等前後處理）：總時間減掉翻譯
- 翻譯：``translator.translate`` 本身
- 合計：整張圖

測試圖片用腳本自己產生（日文歌詞、日文一般、英文各一批句子，淺色、深色、影片感漸層加雜訊三種背景，三種字級），
也可以加上自己的截圖資料夾（``--images``）。先整批跑一輪不計時（OCR 與模型的第一次使用成本），再計時 ``--rounds`` 輪，
每張取中位數。每個模型／裝置組合請各用獨立的行程跑（這支腳本一次只跑一組）；CPU 版與 Vulkan 版是不同的套件，
要分別在兩個虛擬環境裡執行（見 ``requirements-vulkan.txt``）。

用法::

    python scripts/benchmark_pipeline.py --model qwen2.5-3b-instruct-gguf-q4_k_m --gpu-layers 0  --label cpu --out cpu_3b.json
    python scripts/benchmark_pipeline.py --model qwen2.5-3b-instruct-gguf-q4_k_m --gpu-layers -1 --label vulkan --out gpu_3b.json
    python scripts/benchmark_pipeline.py --merge cpu_3b.json gpu_3b.json        # 彙整成 Markdown 表格
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

JAPANESE_FONTS = ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc")
ENGLISH_FONTS = ("segoeui.ttf", "arial.ttf")
FONT_SIZES = (26, 32, 40)
THEMES = ("light", "dark", "video")


# ----------------------------------------------------------------------------- 產生測試圖片
def default_font_dir() -> Path:
    return Path(r"C:\Windows\Fonts") if os.name == "nt" else Path("/mnt/c/Windows/Fonts")


def _load_font(font_dir: Path, candidates: tuple[str, ...], size: int):
    from PIL import ImageFont

    for name in candidates:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise SystemExit(f"找不到字型 {candidates}（目錄：{font_dir}）；用 --font-dir 指定")


def benchmark_sentences() -> list[tuple[str, str]]:
    """``(語言, 句子)``：日文歌詞、日文一般、英文（自編句子，不含有版權的歌詞）。"""
    from compare_translation_models import CORPUS

    sentences = [("ja", text) for group in ("日文歌詞", "日文一般") for text in CORPUS[group]]
    sentences += [("en", text) for text in CORPUS["英文"]]
    return sentences


def generate_images(directory: Path, font_dir: Path, seed: int = 7) -> list[Path]:
    """把每個句子畫成一張字幕圖（字級與背景輪流換），存進 ``directory``，回傳檔案清單。結果只看 ``seed``，每次相同。"""
    import numpy as np
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for index, (language, text) in enumerate(benchmark_sentences()):
        size = FONT_SIZES[index % len(FONT_SIZES)]
        theme = THEMES[index % len(THEMES)]
        font = _load_font(font_dir, JAPANESE_FONTS if language == "ja" else ENGLISH_FONTS, size)
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
        width, height = max(520, right - left + 90), (bottom - top) + int(size * 1.6)
        if theme == "light":
            background = np.full((height, width, 3), 245, dtype=np.uint8)
            ink, stroke = (25, 25, 25), None
        elif theme == "dark":
            background = np.full((height, width, 3), 28, dtype=np.uint8)
            ink, stroke = (230, 230, 230), None
        else:  # 影片感：漸層加雜訊，白字黑邊
            ramp = np.linspace(40, 180, width, dtype=np.float32)[None, :, None].repeat(height, 0).repeat(3, 2)
            background = np.clip(ramp + rng.normal(0, 14, (height, width, 3)), 0, 255).astype(np.uint8)
            ink, stroke = (255, 255, 255), (0, 0, 0)
        image = Image.fromarray(background)
        draw = ImageDraw.Draw(image)
        origin = ((width - (right - left)) // 2 - left, (height - (bottom - top)) // 2 - top)
        if stroke:
            draw.text(origin, text, font=font, fill=ink, stroke_width=max(2, size // 12), stroke_fill=stroke)
        else:
            draw.text(origin, text, font=font, fill=ink)
        path = directory / f"gen_{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- 量測
class TimedOCR:
    """包住 OCR 引擎，累計 ``recognize`` 花的時間；其他屬性原樣轉給引擎。"""

    def __init__(self, engine) -> None:
        self._engine = engine
        self.seconds = 0.0

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def recognize(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._engine.recognize(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start


def measure(images: list[tuple[str, object]], ocr_engine, translator, config, rounds: int = 2) -> dict:
    """對每張圖跑一次真實的管線（``workers.translation_worker.PipelineTask.run``，不經過畫面），回傳每張圖的耗時。

    ``images``：``[(名稱, BGR numpy 陣列)]``。先整批跑一輪不計時，再計時 ``rounds`` 輪；每張取中位數。
    翻譯失敗或辨識不到文字的圖片記在 ``failed``，不計入耗時。"""
    from PySide6.QtCore import QCoreApplication

    from app.models import CaptureInfo
    from workers.translation_worker import PipelineTask

    QCoreApplication.instance() or QCoreApplication([])
    timed_ocr = TimedOCR(ocr_engine)
    spent = {"translate": 0.0}
    original_translate = translator.translate

    def timed_translate(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_translate(*args, **kwargs)
        finally:
            spent["translate"] += time.perf_counter() - start

    translator.translate = timed_translate
    per_image: dict[str, dict[str, list[float]]] = {name: {"total": [], "ocr": [], "translate": []} for name, _ in images}
    failed: set[str] = set()
    cpu_start = wall_start = 0.0
    try:
        for round_index in range(1 + max(1, rounds)):
            if round_index == 1:
                cpu_start, wall_start = time.process_time(), time.perf_counter()
            for name, bgr in images:
                height, width = bgr.shape[:2]
                outcome: list = []
                task = PipelineTask(
                    capture=CaptureInfo(image=bgr, bbox=(0, 0, width, height), mode="region"),
                    ocr_engine=timed_ocr, translator=translator, config=config, gate=None,
                )
                task.result.connect(lambda payload, sink=outcome: sink.append(payload))
                task.error.connect(lambda message, sink=outcome: sink.append(None))
                timed_ocr.seconds = spent["translate"] = 0.0
                start = time.perf_counter()
                task.run()
                total = (time.perf_counter() - start) * 1000
                getattr(translator, "reset_context", lambda: None)()
                if not outcome or outcome[-1] is None or not outcome[-1].get("regions"):
                    failed.add(name)
                    continue
                if round_index > 0:
                    per_image[name]["total"].append(total)
                    per_image[name]["ocr"].append(total - spent["translate"] * 1000)  # OCR 與縮放、合併、取色等前後處理
                    per_image[name]["translate"].append(spent["translate"] * 1000)
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
    finally:
        translator.translate = original_translate
    rows = {
        name: {part: statistics.median(values) for part, values in parts.items()}
        for name, parts in per_image.items()
        if name not in failed and parts["total"]
    }
    summary = {
        part: {"median": statistics.median(row[part] for row in rows.values()), "mean": statistics.mean(row[part] for row in rows.values())}
        for part in ("ocr", "translate", "total")
    } if rows else {}
    return {"images": len(images), "measured": len(rows), "failed": sorted(failed), "summary": summary,
            "per_image": rows, "cpu_cores": round(cpu / wall, 2) if wall else 0.0}


def load_images(directories: list[Path]) -> list[tuple[str, object]]:
    import numpy as np
    from PIL import Image

    images = []
    for directory in directories:
        for path in sorted(directory.glob("*.png")):
            images.append((f"{directory.name}/{path.name}", np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()))
    return images


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["LOCALAPPDATA"] = tmp  # 日誌與設定放暫存資料夾，不寫進使用者真正的資料（要在匯入 app 之前設）
        import services.ocr.rapidocr_ocr  # noqa: F401  （註冊 OCR 引擎）
        import services.translation.offline_translator  # noqa: F401
        from app.config import AppConfig
        from services.ocr.base import create_ocr_engine
        from services.translation import llama_backend
        from services.translation.factory import create_translator

        config = AppConfig(Path(tmp) / "config.json")
        section = config.section("translation")
        section["use_context"] = False
        section["lyrics_mode"] = False
        section["offline"].update(model_name=args.model, threads=args.threads, gpu_layers=args.gpu_layers)
        directories = [Path(path) for path in args.images]
        if args.generate:
            directories.append(Path(tmp) / "generated")
            generate_images(directories[-1], Path(args.font_dir) if args.font_dir else default_font_dir())
        images = load_images(directories)
        if not images:
            raise SystemExit("沒有圖片：用 --generate 產生，或用 --images 指定資料夾")
        ocr_engine = create_ocr_engine("rapidocr", config.section("ocr"), config)
        translator = create_translator("offline", section, cache=None)
        started = time.perf_counter()
        ocr_engine.warmup()
        translator.warmup()
        load_seconds = time.perf_counter() - started
        result = measure(images, ocr_engine, translator, config, rounds=args.rounds)
        try:
            import llama_cpp

            engine = f"llama-cpp-python {llama_cpp.__version__}"
        except ImportError:
            engine = "llama-cpp-python"
        result.update(
            label=args.label, model=args.model, gpu_layers=args.gpu_layers, threads=args.threads, device=translator.device,
            engine=engine, gpu_offload=llama_backend.gpu_offload_supported(), load_s=round(load_seconds, 1), rounds=args.rounds,
        )
        translator.close()
        return result


# ----------------------------------------------------------------------------- 彙整
def merge_table(results: list[dict]) -> str:
    """把多次執行的結果彙整成 Markdown 表格：同一組（模型、標籤）的多次結果取中位數。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for result in results:
        key = (result["model"], result["label"])
        if key not in groups:
            order.append(key)
        groups[key].append(result)

    from services.translation.model_names import display_name as short

    lines = [
        "| 模型 | 版本 | 裝置 | OCR（含縮放、合併） | 翻譯 | 合計（中位／平均） | 佔用 CPU（核心） | 載入＋預熱 | 次數 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, label in order:
        runs = [r for r in groups[(model, label)] if r.get("summary")]
        if not runs:
            continue
        pick = lambda part, stat: statistics.median(r["summary"][part][stat] for r in runs)  # noqa: E731
        first = runs[0]
        lines.append(
            f"| {short(model)} | {label}（{first['engine'].replace('llama-cpp-python ', '')}） | {first['device'].upper()} | "
            f"{pick('ocr', 'median'):.0f} | {pick('translate', 'median'):.0f} | "
            f"**{pick('total', 'median'):.0f}**／{pick('total', 'mean'):.0f} | "
            f"{statistics.median(r['cpu_cores'] for r in runs):.1f} | {statistics.median(r['load_s'] for r in runs):.0f} 秒 | {len(runs)} |"
        )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace") if sys.stdout.isatty() else sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-3b-instruct-gguf-q4_k_m")
    parser.add_argument("--gpu-layers", type=int, default=0, help="0 = 只用 CPU，-1 = 全部交給 GPU（要裝 Vulkan／CUDA 版）")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--label", default="cpu", help="結果表格裡的版本名稱（例如 cpu、vulkan）")
    parser.add_argument("--images", nargs="*", default=[], help="另外加入的截圖資料夾（*.png）")
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True, help="用內建句子產生字幕圖（預設開）")
    parser.add_argument("--font-dir", default="", help="字型資料夾（預設 Windows 的 Fonts）")
    parser.add_argument("--rounds", type=int, default=2, help="不計時的暖機一輪之後，計時幾輪")
    parser.add_argument("--out", type=Path, help="結果存成 JSON")
    parser.add_argument("--merge", nargs="+", type=Path, help="不跑測試，把這些 JSON 彙整成 Markdown 表格")
    args = parser.parse_args()
    if args.merge:
        print(merge_table([json.loads(path.read_text(encoding="utf-8")) for path in args.merge]))
        return
    result = run(args)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = result["summary"]
    print(
        f"{result['label']} {result['model']} [{result['device']}] {result['engine']}：{result['measured']}/{result['images']} 張，"
        f"OCR {summary['ocr']['median']:.0f} ms ＋ 翻譯 {summary['translate']['median']:.0f} ms ＝ 合計 {summary['total']['median']:.0f} ms（中位），"
        f"CPU {result['cpu_cores']} 核心，載入 {result['load_s']} 秒，辨識失敗 {len(result['failed'])} 張"
    )


if __name__ == "__main__":
    main()
