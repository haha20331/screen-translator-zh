"""端到端驗證：圖片 → OCR → 行合併 → 離線翻譯 → 繁體，並與參考譯文比對。

資料夾內每張圖片可搭配同名 ``.txt``（``1.png`` ↔ ``1.txt``），內容是**參考譯文**（繁體中文）。
相似度是字元層級的粗略指標（參考譯文本身也只是「一種」合理的譯法），用來比較不同設定，
不是絕對的品質分數；請同時閱讀實際輸出。

用法::

    python scripts/translate_benchmark.py --images val_data
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import services.ocr.rapidocr_ocr  # noqa: E402, F401
import services.translation.offline_translator  # noqa: E402, F401
from app.config import AppConfig  # noqa: E402
from app.models import CaptureInfo  # noqa: E402
from services.ocr.base import create_ocr_engine  # noqa: E402
from services.translation.factory import create_translator  # noqa: E402
from workers.translation_worker import PipelineTask  # noqa: E402

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text))


def similarity(candidate: str, reference: str) -> float:
    a, b = _normalize(candidate), _normalize(reference)
    return difflib.SequenceMatcher(None, a, b).ratio() if a or b else 1.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows 預設編碼（cp950）印不出所有字；導向檔案時用 UTF-8，終端機則不丟例外
        sys.stdout.reconfigure(errors="replace") if sys.stdout.isatty() else sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="含圖片（與可選的同名參考譯文 .txt）的資料夾")
    parser.add_argument(
        "--model",
        default="auto",
        help="離線翻譯的模型名稱（見 scripts/fetch_translation_models.py --list）；預設 auto = 3B Q4_K_M（沒有下載就用 1.5B）",
    )
    parser.add_argument("--threads", type=int, default=0, help="推論執行緒數（0 = 自動）")
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if not paths:
        raise SystemExit(f"{args.images} 內沒有圖片")

    app = QApplication.instance() or QApplication([])  # noqa: F841 - PipelineTask 是 QThread
    with tempfile.TemporaryDirectory() as tmp:
        config = AppConfig(Path(tmp) / "config.json")
        offline = config.section("translation")["offline"]
        offline["threads"] = args.threads
        offline["model_name"] = args.model
        engine = create_ocr_engine(config.get("ocr.engine", "rapidocr"), config.section("ocr"), config)
        translator = create_translator(
            config.get("translation.service", "offline"), config.section("translation"), cache=None
        )
        start = time.perf_counter()
        engine.warmup()
        if hasattr(translator, "warmup"):
            translator.warmup()
        print(f"模型載入 {time.perf_counter() - start:.1f}s（OCR={engine.name}，翻譯={translator.name}）")

        scores: list[float] = []
        timings: list[float] = []
        for path in paths:
            bgr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
            capture = CaptureInfo(image=bgr, bbox=(0, 0, bgr.shape[1], bgr.shape[0]), mode="region")
            task = PipelineTask(capture=capture, ocr_engine=engine, translator=translator, config=config)
            results: list[dict] = []
            errors: list[str] = []
            task.result.connect(results.append)
            task.error.connect(errors.append)
            begin = time.perf_counter()
            task.run()
            elapsed = time.perf_counter() - begin
            timings.append(elapsed)

            print(f"\n== {path.name}  {bgr.shape[1]}x{bgr.shape[0]}  {elapsed * 1000:.0f} ms")
            if not results:
                print(f"  （無結果：{'; '.join(errors) or '未知原因'}）")
                scores.append(0.0)
                continue
            regions = results[0]["regions"]
            for region in regions:
                print(f"  {region.text}\n    -> {region.translated_text}")
            translated = "".join(region.translated_text for region in regions)
            reference_path = path.with_suffix(".txt")
            if reference_path.exists():
                reference = reference_path.read_text(encoding="utf-8").strip()
                score = similarity(translated, reference)
                scores.append(score)
                print(f"  參考：{reference!r}\n  相似度 {score * 100:.0f}%")

        print()
        if scores:
            print(f"平均相似度 {statistics.mean(scores) * 100:.0f}%（{len(scores)} 張有參考譯文）")
        print(f"每張耗時（OCR+合併+翻譯）中位數 {statistics.median(timings) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
