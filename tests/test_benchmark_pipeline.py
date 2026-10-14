"""scripts/benchmark_pipeline.py：整條管線（OCR → 翻譯）每張圖的耗時、測試圖片的產生、結果彙整。"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from app.config import AppConfig
from services.ocr.base import OCRLine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("benchmark_pipeline", ROOT / "scripts" / "benchmark_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _OCR:
    """假的 OCR：每次花 6 毫秒；圖片全黑（像素總和為 0）就辨識不到文字。"""

    benefits_from_upscale = False

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image, lang="auto"):
        self.calls += 1
        time.sleep(0.006)
        return [OCRLine("紡ぐたったひとつ絆", (10, 10, 200, 40), 0.95)] if image.sum() else []


class _Translator:
    han_is_source = True
    last_failed_count = 0
    unsupported_direction = False

    def __init__(self) -> None:
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        time.sleep(0.012)
        return ["只有一個羈絆" for _ in texts]


def _images(count=3, black=()):
    return [
        (f"img{i}.png", np.zeros((60, 300, 3), dtype=np.uint8) if i in black else np.full((60, 300, 3), 200, dtype=np.uint8))
        for i in range(count)
    ]


def test_every_image_goes_through_the_real_pipeline_once_per_round_plus_one_untimed_warm_up_round(script, tmp_path):
    ocr, translator = _OCR(), _Translator()

    result = script.measure(_images(3), ocr, translator, AppConfig(tmp_path / "config.json"), rounds=2)

    assert len(translator.calls) == 3 * (1 + 2)  # 暖機一輪（不計時）＋計時兩輪
    assert ocr.calls == 3 * (1 + 2)
    assert result["images"] == 3 and result["measured"] == 3 and result["failed"] == []


def test_the_time_is_split_into_ocr_and_translation_and_adds_up(script, tmp_path):
    result = script.measure(_images(3), _OCR(), _Translator(), AppConfig(tmp_path / "config.json"), rounds=2)
    summary = result["summary"]

    assert 10 <= summary["translate"]["median"] < 200  # 假翻譯器睡 12 毫秒
    assert summary["ocr"]["median"] >= 5  # 假 OCR 睡 6 毫秒，另外還有縮放、合併等前後處理
    assert summary["total"]["median"] == pytest.approx(summary["ocr"]["median"] + summary["translate"]["median"], rel=0.05, abs=1.0)
    for row in result["per_image"].values():
        assert row["total"] == pytest.approx(row["ocr"] + row["translate"], abs=1.0)
    assert result["cpu_cores"] >= 0


def test_images_without_recognizable_text_are_reported_and_left_out_of_the_times(script, tmp_path):
    translator = _Translator()

    result = script.measure(_images(3, black={1}), _OCR(), translator, AppConfig(tmp_path / "config.json"), rounds=1)

    assert result["failed"] == ["img1.png"] and result["measured"] == 2
    assert set(result["per_image"]) == {"img0.png", "img2.png"}
    assert len(translator.calls) == 2 * 2  # 辨識不到文字的那張根本沒有進翻譯


def test_the_translator_is_restored_after_measuring(script, tmp_path):
    translator = _Translator()
    original = translator.translate

    script.measure(_images(1), _OCR(), translator, AppConfig(tmp_path / "config.json"), rounds=1)

    assert translator.translate == original


def test_the_timed_ocr_wrapper_forwards_everything_else(script):
    engine = _OCR()
    wrapped = script.TimedOCR(engine)

    assert wrapped.benefits_from_upscale is False
    wrapped.recognize(np.ones((4, 4, 3), dtype=np.uint8))
    assert engine.calls == 1 and wrapped.seconds >= 0.005


# ----------------------------------------------------------------------------- 結果彙整
def _result(model, label, device, ocr, translate, cores, load, engine="llama-cpp-python 0.3.36"):
    total = ocr + translate
    return {
        "model": model, "label": label, "device": device, "engine": engine, "cpu_cores": cores, "load_s": load,
        "summary": {"ocr": {"median": ocr, "mean": ocr}, "translate": {"median": translate, "mean": translate}, "total": {"median": total, "mean": total + 10}},
    }


def test_the_merged_table_takes_the_median_of_repeated_runs_and_keeps_the_input_order(script):
    model = "qwen2.5-3b-instruct-gguf-q4_k_m"
    table = script.merge_table(
        [
            _result(model, "cpu", "cpu", 300, 700, 5.9, 12),
            _result(model, "vulkan", "gpu", 300, 600, 0.5, 23, engine="llama-cpp-python 0.3.33"),
            _result(model, "cpu", "cpu", 340, 740, 5.7, 13),
            _result(model, "cpu", "cpu", 320, 720, 5.8, 12),
        ]
    )
    rows = table.splitlines()

    assert rows[0].startswith("| 模型 | 版本 | 裝置 |") and len(rows) == 4
    assert rows[2].startswith("| 3B Q4_K_M | cpu（0.3.36） | CPU | 320 | 720 | **1040**／1050 | 5.8 | 12 秒 | 3 |")
    assert rows[3].startswith("| 3B Q4_K_M | vulkan（0.3.33） | GPU | 300 | 600 | **900**／910 | 0.5 | 23 秒 | 1 |")


def test_runs_without_any_measured_image_are_skipped_in_the_table(script):
    assert len(script.merge_table([{"model": "m", "label": "cpu", "summary": {}}]).splitlines()) == 2  # 只有表頭


# ----------------------------------------------------------------------------- 產生測試圖片
@pytest.fixture(scope="module")
def font_dir(script):
    directory = script.default_font_dir()
    if not any((directory / name).exists() for name in script.JAPANESE_FONTS) or not any((directory / name).exists() for name in script.ENGLISH_FONTS):
        pytest.skip("找不到日文與英文字型")
    return directory


def test_one_image_is_generated_for_every_sentence_and_the_result_is_reproducible(script, font_dir, tmp_path):
    first = script.generate_images(tmp_path / "a", font_dir)
    second = script.generate_images(tmp_path / "b", font_dir)

    assert len(first) == len(script.benchmark_sentences()) == 31  # 日文歌詞 11 ＋ 日文一般 10 ＋ 英文 10
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]  # 同一個種子，每次一樣
    from PIL import Image

    sizes = {Image.open(p).size for p in first}
    assert len(sizes) > 5 and all(w >= 520 and h >= 40 for w, h in sizes)  # 寬度隨句子長度、高度隨字級變化


def test_the_generated_images_can_actually_be_read_by_the_ocr(script, font_dir, tmp_path):
    pytest.importorskip("rapidocr")
    import services.ocr.rapidocr_ocr  # noqa: F401
    from services.ocr.base import create_ocr_engine

    config = AppConfig(tmp_path / "config.json")
    engine = create_ocr_engine("rapidocr", config.section("ocr"), config)
    paths = script.generate_images(tmp_path / "gen", font_dir)
    images = script.load_images([tmp_path / "gen"])

    assert len(images) == len(paths)
    found = sum(1 for _, bgr in images[:3] + images[-3:] if engine.recognize(bgr, lang="auto"))
    assert found >= 5  # 日文與英文各幾張都辨識得到文字（背景有雜訊的偶爾會漏）
