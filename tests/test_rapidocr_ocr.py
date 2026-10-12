"""RapidOCR 后端：结果转换、参数组装，以及（有安装时）一次真实辨识。"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from app.config import AppConfig
from app.models import CaptureInfo
from services.ocr.base import OCRLine, list_ocr_engines
from services.ocr.rapidocr_ocr import RapidOCREngine, _output_to_lines, upscale_factor
from workers.translation_worker import PipelineTask


def _output(texts, boxes, scores):
    return SimpleNamespace(txts=texts, boxes=boxes, scores=scores)


def test_engine_is_registered_and_opts_out_of_upscaling():
    import services.ocr.rapidocr_ocr  # noqa: F401

    assert "rapidocr" in list_ocr_engines()
    assert RapidOCREngine.benefits_from_upscale is False


def test_polygon_is_converted_to_axis_aligned_box():
    polygon = [[17.2, 16.0], [316.6, 18.5], [316.6, 43.1], [17.4, 41.4]]
    lines = _output_to_lines(_output(("Hello",), (polygon,), (0.99,)))

    assert len(lines) == 1
    line = lines[0]
    assert isinstance(line, OCRLine)
    assert line.text == "Hello"
    assert line.box == (17, 16, 300, 28)  # x: floor(17.2)..ceil(316.6), y: floor(16.0)..ceil(43.1)
    assert line.confidence == pytest.approx(0.99)
    assert line.angle == pytest.approx(0.47, abs=0.05)  # 上缘 (17.2,16)->(316.6,18.5)


def test_blank_texts_are_dropped_and_text_is_stripped():
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    lines = _output_to_lines(_output(("  ", " ok "), (box, box), (0.9, 0.8)))

    assert [line.text for line in lines] == ["ok"]


@pytest.mark.parametrize("output", [_output(None, None, None), _output((), (), ()), SimpleNamespace()])
def test_empty_output_gives_no_lines(output):
    assert _output_to_lines(output) == []


def test_degenerate_box_keeps_minimum_size():
    box = [[5, 5], [5, 5], [5, 5], [5, 5]]
    (line,) = _output_to_lines(_output(("x",), (box,), (0.5,)))

    assert line.box == (5, 5, 1, 1)
    assert line.angle == 0.0


def test_build_params_defaults_cap_threads_and_avoid_upscaling():
    params = RapidOCREngine({})._build_params()

    assert params["Det.limit_type"] == "max"
    assert params["Det.limit_side_len"] == 960
    assert params["Global.use_cls"] is False
    assert 1 <= params["EngineConfig.onnxruntime.intra_op_num_threads"] <= 4
    assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1


def test_build_params_honours_configuration():
    engine = RapidOCREngine({"rapidocr": {"limit_side_len": 1280, "threads": 2}})
    params = engine._build_params()

    assert params["Det.limit_side_len"] == 1280
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 2


@pytest.mark.parametrize(
    "image",
    [None, np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 2), dtype=np.uint8)],
)
def test_recognize_rejects_unsupported_images(image):
    with pytest.raises(TypeError):
        RapidOCREngine({}).recognize(image)


def test_recognize_returns_empty_for_empty_image():
    assert RapidOCREngine({}).recognize(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_recognize_drops_alpha_channel_before_inference(monkeypatch):
    seen = {}

    def fake_engine(image):
        seen["shape"] = image.shape
        seen["contiguous"] = image.flags["C_CONTIGUOUS"]
        return _output(None, None, None)

    engine = RapidOCREngine({"rapidocr": {"upscale_below": 0}})
    monkeypatch.setattr(engine, "_get_engine", lambda: fake_engine)

    engine.recognize(np.zeros((20, 30, 4), dtype=np.uint8))

    assert seen == {"shape": (20, 30, 3), "contiguous": True}


@pytest.mark.parametrize(
    ("height", "width", "below", "max_scale", "expected"),
    [
        (99, 573, 240, 2.0, 2.0),  # 短邊很小：受 max_scale 限制
        (170, 619, 240, 2.0, pytest.approx(240 / 170)),  # 放大到短邊約 240
        (239, 800, 240, 2.0, 1.0),  # 倍率 < 1.05 不值得放大
        (240, 800, 240, 2.0, 1.0),  # 剛好達標
        (190, 664, 0, 2.0, 1.0),  # below=0 關閉
        (99, 573, 240, 1.0, 1.0),  # max_scale<=1 關閉
        (0, 100, 240, 2.0, 1.0),  # 退化尺寸
    ],
)
def test_upscale_factor_rules(height, width, below, max_scale, expected):
    assert upscale_factor(height, width, below, max_scale) == expected


def test_recognize_upscales_small_images_and_maps_boxes_back(monkeypatch):
    seen = {}
    polygon = [[40.0, 20.0], [200.0, 20.0], [200.0, 60.0], [40.0, 60.0]]  # 放大後（2x）的座標

    def fake_engine(image):
        seen["shape"] = image.shape
        return _output(("字幕",), (polygon,), (0.9,))

    engine = RapidOCREngine({})
    monkeypatch.setattr(engine, "_get_engine", lambda: fake_engine)

    (line,) = engine.recognize(np.zeros((99, 300, 3), dtype=np.uint8))

    assert seen["shape"] == (198, 600, 3)
    assert line.box == (20, 10, 80, 20)  # 還原回原圖尺度


def test_recognize_leaves_large_images_alone(monkeypatch):
    seen = {}

    def fake_engine(image):
        seen["shape"] = image.shape
        return _output(None, None, None)

    engine = RapidOCREngine({})
    monkeypatch.setattr(engine, "_get_engine", lambda: fake_engine)

    engine.recognize(np.zeros((400, 800, 3), dtype=np.uint8))

    assert seen["shape"] == (400, 800, 3)


def test_availability_reason_reports_missing_dependency(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "onnxruntime" else real_find_spec(name, *a, **k),
    )

    reason = RapidOCREngine.availability_reason()

    assert reason is not None and "onnxruntime" in reason
    assert RapidOCREngine.available() is False


@pytest.mark.skipif(not RapidOCREngine.available(), reason="rapidocr / onnxruntime 未安装")
def test_recognizes_rendered_english_text():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (420, 80), "white")
    ImageDraw.Draw(image).text((12, 16), "Hello World 2026", font=ImageFont.load_default(size=36), fill="black")
    bgr = np.asarray(image)[:, :, ::-1].copy()

    engine = RapidOCREngine({})
    try:
        lines = engine.recognize(bgr, lang="auto")
    finally:
        engine.close()

    joined = "".join(line.text for line in lines).replace(" ", "")
    assert "HelloWorld" in joined
    assert all(0 <= line.confidence <= 1 for line in lines)


class _RecordingOCR:
    """Captures the image handed to the engine by the pipeline."""

    def __init__(self, benefits_from_upscale: bool) -> None:
        if not benefits_from_upscale:
            self.benefits_from_upscale = False
        self.shapes: list[tuple[int, ...]] = []

    def recognize(self, image, lang=None):
        self.shapes.append(image.shape)
        return []


@pytest.mark.parametrize(
    ("benefits", "expect_upscaled"),
    [(True, True), (False, False)],
)
def test_pipeline_upscales_only_for_engines_that_benefit(tmp_path, benefits, expect_upscaled):
    capture = CaptureInfo(image=np.zeros((60, 120, 3), dtype=np.uint8), bbox=(0, 0, 120, 60), mode="region")
    engine = _RecordingOCR(benefits)
    task = PipelineTask(
        capture=capture,
        ocr_engine=engine,
        translator=None,
        config=AppConfig(tmp_path / "config.json"),
    )

    task.run()  # 同步执行，不启动线程

    assert engine.shapes
    first = engine.shapes[0]
    assert (first[:2] != (60, 120)) is expect_upscaled
