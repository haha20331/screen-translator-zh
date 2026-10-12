"""OCR 準確度與速度的快速評估（合成樣本）。

用系統字型把已知文字渲染成圖片，交給 OCR 後端辨識，計算字元錯誤率（CER）與單張耗時。
合成樣本只能當第一道篩選：沒有抗鋸齒差異、壓縮雜訊與複雜背景，實際效果仍要用真實截圖確認。

用法::

    python scripts/ocr_benchmark.py --font-dir /mnt/c/Windows/Fonts
    python scripts/ocr_benchmark.py --font-dir C:\\Windows\\Fonts --engine windows --save-dir out

真實截圖：把圖片放進資料夾（可選：同名 .txt 放標準答案，每行一句），執行::

    python scripts/ocr_benchmark.py --images my_samples
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ocr.base import create_ocr_engine  # noqa: E402

# (類別, 字型候選, 文字行)
SAMPLES: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("en_ui", ("segoeui.ttf", "arial.ttf"), ["File  Edit  View  Help", "Save changes", "Settings > Display > Brightness"]),
    ("en_text", ("segoeui.ttf", "arial.ttf"), [
        "The quick brown fox jumps over the lazy dog.",
        "Press Ctrl+Shift+A to capture the selected region.",
        "Version 2.4.1 (build 20261012) - 3 updates available",
    ]),
    ("ja_ui", ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"), ["ファイル　編集　表示　ヘルプ", "変更を保存", "設定 > ディスプレイ > 明るさ"]),
    ("ja_text", ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"), [
        "吾輩は猫である。名前はまだ無い。",
        "選択した領域をキャプチャするには Ctrl+Shift+A を押してください。",
        "ネットワークに接続できませんでした。もう一度お試しください。",
    ]),
    ("mixed", ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"), [
        "Save 保存する (Ctrl+S)",
        "Player1 のHPが 120 回復しました",
        "Chapter 3: 約束の場所 - Start Game",
    ]),
]
SIZES = (9, 10, 12, 14, 18, 28)
THEMES = {
    "light": ("#202020", "#ffffff"),
    "dark": ("#d4d4d4", "#1e1e1e"),
    "button": ("#ffffff", "#2878e8"),
    # 漸層加高斯雜訊，模擬影片、遊戲畫面這類不乾淨的背景
    "noisy": ("#f2f2f2", "noisy"),
}


@dataclass
class Result:
    category: str
    size: int
    theme: str
    cer: float
    lines_expected: int
    lines_found: int
    seconds: float


def _font(font_dir: Path, candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    for name in candidates:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise SystemExit(f"找不到字型 {candidates}（目錄：{font_dir}）")


def _background(width: int, height: int, bg: str) -> Image.Image:
    if bg != "noisy":
        return Image.new("RGB", (width, height), bg)
    rng = np.random.default_rng(width * 1000 + height)
    ramp = np.linspace(40, 120, width, dtype=np.float32)[None, :, None] * np.ones((height, 1, 3), np.float32)
    ramp[..., 2] += 30
    noisy = ramp + rng.normal(0, 12, ramp.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")


def render(lines: list[str], font: ImageFont.FreeTypeFont, fg: str, bg: str) -> Image.Image:
    pad = 14
    line_h = int(font.size * 1.6)
    width = max(int(font.getlength(line)) for line in lines) + pad * 2
    image = _background(width, line_h * len(lines) + pad * 2, bg)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((pad, pad + index * line_h), line, font=font, fill=fg)
    return image


def _normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def run(engine, font_dir: Path, save_dir: Path | None, repeats: int) -> list[Result]:
    results: list[Result] = []
    for category, fonts, lines in SAMPLES:
        for size in SIZES:
            font = _font(font_dir, fonts, size)
            for theme, (fg, bg) in THEMES.items():
                image = render(lines, font, fg, bg)
                bgr = np.asarray(image)[:, :, ::-1].copy()
                timings = []
                found = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    found = engine.recognize(bgr, lang="auto")
                    timings.append(time.perf_counter() - start)
                ordered = sorted(found, key=lambda line: (line.box[1], line.box[0]))
                got = _normalize("".join(line.text for line in ordered))
                want = _normalize("".join(lines))
                cer = edit_distance(got, want) / max(1, len(want))
                results.append(Result(category, size, theme, cer, len(lines), len(found), statistics.median(timings)))
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    image.save(save_dir / f"{category}_{size}_{theme}.png")
    return results


def report(results: list[Result]) -> None:
    print(f"{'category':9s} {'size':>4s} " + " ".join(f"{t:>14s}" for t in THEMES))
    print("-" * (15 + 15 * len(THEMES)))
    for category, _, _ in SAMPLES:
        for size in SIZES:
            cells = []
            for theme in THEMES:
                r = next(x for x in results if (x.category, x.size, x.theme) == (category, size, theme))
                line_flag = "" if r.lines_found == r.lines_expected else f" L{r.lines_found}/{r.lines_expected}"
                cells.append(f"{r.cer * 100:5.1f}%{line_flag}".rjust(14))
            print(f"{category:9s} {size:>4d} " + " ".join(cells))
    print()
    for category, _, _ in SAMPLES:
        subset = [r for r in results if r.category == category]
        print(f"{category:9s} mean CER {statistics.mean(r.cer for r in subset) * 100:5.2f}%   "
              f"median {statistics.median(r.seconds for r in subset) * 1000:6.0f} ms/image")
    print(f"{'ALL':9s} mean CER {statistics.mean(r.cer for r in results) * 100:5.2f}%   "
          f"median {statistics.median(r.seconds for r in results) * 1000:6.0f} ms/image")
    print("(L a/b = 偵測到的行數 / 預期行數，行數不符時才顯示)")


def run_images(engine, folder: Path) -> None:
    """對真實截圖逐張辨識並印出結果；有同名 .txt 時另算 CER。"""
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if not paths:
        raise SystemExit(f"{folder} 內沒有圖片")
    for path in paths:
        bgr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        start = time.perf_counter()
        found = engine.recognize(bgr, lang="auto")
        elapsed = time.perf_counter() - start
        print(f"\n== {path.name}  {bgr.shape[1]}x{bgr.shape[0]}  {elapsed * 1000:.0f} ms  {len(found)} lines")
        ordered = sorted(found, key=lambda line: (line.box[1], line.box[0]))
        for line in ordered:
            x, y, w, h = line.box
            print(f"  ({x},{y},{w},{h}) {line.confidence:.2f}  {line.text}")
        truth = path.with_suffix(".txt")
        if truth.exists():
            want = _normalize(truth.read_text(encoding="utf-8"))
            got = _normalize("".join(line.text for line in ordered))
            print(f"  CER {edit_distance(got, want) / max(1, len(want)) * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--font-dir", type=Path, help="含 segoeui/meiryo 等字型的資料夾（合成樣本模式必填）")
    parser.add_argument("--images", type=Path, help="改為辨識此資料夾內的真實截圖")
    parser.add_argument("--engine", default="rapidocr")
    parser.add_argument("--repeats", type=int, default=3, help="每張圖重複次數（取耗時中位數）")
    parser.add_argument("--save-dir", type=Path, default=None, help="另存渲染出的樣本圖")
    args = parser.parse_args()

    import services.ocr.rapidocr_ocr  # noqa: F401
    import services.ocr.windows_ocr  # noqa: F401

    engine = create_ocr_engine(args.engine, {})
    engine.warmup()
    if args.images is not None:
        run_images(engine, args.images)
        return
    if args.font_dir is None:
        parser.error("合成樣本模式需要 --font-dir")
    report(run(engine, args.font_dir, args.save_dir, max(1, args.repeats)))


if __name__ == "__main__":
    main()
