"""RapidOCR 后端：PP-OCRv6 多语言模型 + ONNX Runtime（纯 CPU，完全离线）。

同一个模型同时覆盖英文、日文与中文，不需要先判断语言再分流。模型文件随 ``rapidocr``
套件一起安装，首次使用不会联网下载。
"""

from __future__ import annotations

import importlib.util
import math
import os
import threading
from typing import Any

import numpy as np

from app.logger import get_logger
from services.ocr.base import OCREngine, OCRLine, register_engine

log = get_logger("ocr.rapidocr")

# 偵測階段的長邊上限。RapidOCR 預設的 limit_type="min" 會把小圖放大到短邊 736，
# 640x140 的小框光偵測就要 1 秒以上；改成「只縮小不放大」後約 0.15 秒。
_DEFAULT_LIMIT_SIDE = 960
# ONNX Runtime 預設會用滿所有核心，在多核心機器上反而因執行緒競爭變慢。
_DEFAULT_MAX_THREADS = 4
# 框選區域通常很小（字幕條、對話框）。短邊低於此值時先放大再辨識：低解析影格與藝術字
# 的錯誤率明顯下降（val_data 的字幕截圖 23% -> 8%），其餘樣本不受影響。0 表示關閉。
_DEFAULT_UPSCALE_BELOW = 240
_DEFAULT_MAX_UPSCALE = 2.0


class RapidOCREngine(OCREngine):
    name = "rapidocr"
    # 管線不需要替我們放大；小圖的放大由本引擎自己決定並還原座標（見 upscale_factor）。
    benefits_from_upscale = False

    def __init__(self, ocr_config: dict, app_config: Any = None, logger: Any = None) -> None:
        super().__init__(ocr_config, app_config, logger)
        self._engine: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- availability
    @classmethod
    def availability_reason(cls) -> str | None:
        for module, hint in (("rapidocr", "rapidocr"), ("onnxruntime", "onnxruntime")):
            if importlib.util.find_spec(module) is None:
                return f"RapidOCR 依賴不可用：請安裝 {hint}（pip install rapidocr onnxruntime）"
        return None

    @classmethod
    def available(cls) -> bool:
        return cls.availability_reason() is None

    # ------------------------------------------------------------- engine
    def _settings(self) -> dict:
        section = self.config.get("rapidocr", {})
        return section if isinstance(section, dict) else {}

    def _build_params(self) -> dict[str, Any]:
        settings = self._settings()
        limit_side = int(settings.get("limit_side_len", _DEFAULT_LIMIT_SIDE))
        threads = int(settings.get("threads", 0))
        if threads <= 0:
            threads = min(_DEFAULT_MAX_THREADS, os.cpu_count() or 1)
        return {
            "Global.log_level": "warning",
            # 只處理橫排文字，不需要 180 度方向分類。
            "Global.use_cls": False,
            "Det.limit_type": "max",
            "Det.limit_side_len": limit_side,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }

    def _get_engine(self) -> Any:
        if self._engine is None:
            reason = self.availability_reason()
            if reason is not None:
                raise RuntimeError(reason)
            from rapidocr import RapidOCR

            self._engine = RapidOCR(params=self._build_params())
        return self._engine

    def warmup(self) -> None:
        """载入模型并跑一次小图，避免第一次真正识别时才付出初始化成本。"""
        with self._lock:
            engine = self._get_engine()
            engine(_warmup_image())

    # ------------------------------------------------------------- recognize
    def recognize(self, image_bgr, lang: str | None = None) -> list[OCRLine]:
        # lang 被忽略：多语言模型一次辨识英文与日文。
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
            raise TypeError("RapidOCR 需要 H×W×3 的 BGR numpy 影像")
        if image_bgr.size == 0:
            return []
        image = np.ascontiguousarray(image_bgr[:, :, :3])
        settings = self._settings()
        scale = upscale_factor(
            image.shape[0],
            image.shape[1],
            below=int(settings.get("upscale_below", _DEFAULT_UPSCALE_BELOW)),
            max_scale=float(settings.get("max_upscale", _DEFAULT_MAX_UPSCALE)),
        )
        if scale > 1.0:
            image = _upscale(image, scale)
        with self._lock:
            output = self._get_engine()(image)
        return _output_to_lines(output, scale)

    def close(self) -> None:
        with self._lock:
            self._engine = None


def upscale_factor(height: int, width: int, below: int, max_scale: float) -> float:
    """短邊小於 `below` 像素時的放大倍率（最多 `max_scale`）；不需要放大時回傳 1.0。"""
    short_side = min(height, width)
    if below <= 0 or max_scale <= 1.0 or short_side <= 0 or short_side >= below:
        return 1.0
    factor = min(max_scale, below / short_side)
    return factor if factor >= 1.05 else 1.0


def _upscale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    from PIL import Image

    height, width = image_bgr.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(image_bgr[:, :, ::-1]).resize(size, Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized)[:, :, ::-1])


def _warmup_image() -> np.ndarray:
    """带一行文字的小图；全白图会让 RapidOCR 在日志里报「偵測結果為空」。"""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (200, 56), "white")
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # 旧版 Pillow 的 load_default 不支持 size
        font = ImageFont.load_default()
    ImageDraw.Draw(image).text((10, 8), "OCR 42", font=font, fill="black")
    return np.asarray(image)[:, :, ::-1].copy()


def _output_to_lines(output: Any, scale: float = 1.0) -> list[OCRLine]:
    """RapidOCR 輸出轉 OCRLine；`scale` 是辨識前放大的倍率，座標會除回原圖尺度。"""
    texts = getattr(output, "txts", None)
    boxes = getattr(output, "boxes", None)
    scores = getattr(output, "scores", None)
    if not texts or boxes is None or scores is None:
        return []
    lines: list[OCRLine] = []
    for text, polygon, score in zip(texts, boxes, scores):
        text = str(text).strip()
        if not text:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if scale != 1.0:
            points = points / np.float32(scale)
        left = int(math.floor(points[:, 0].min()))
        top = int(math.floor(points[:, 1].min()))
        right = int(math.ceil(points[:, 0].max()))
        bottom = int(math.ceil(points[:, 1].max()))
        lines.append(
            OCRLine(
                text=text,
                box=(left, top, max(1, right - left), max(1, bottom - top)),
                confidence=float(score),
                angle=_top_edge_angle(points),
            )
        )
    return lines


def _top_edge_angle(points: np.ndarray) -> float:
    """文字框上緣相對水平線的角度（度）；點數不足四個時視為 0。"""
    if len(points) < 2:
        return 0.0
    dx = float(points[1][0] - points[0][0])
    dy = float(points[1][1] - points[0][1])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


register_engine(RapidOCREngine)
