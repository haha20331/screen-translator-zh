"""截图 -> OCR -> 合并 -> 翻译 -> 覆盖层数据的管线任务（QThread）。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QThread, Signal

from app.config import AppConfig
from app.logger import get_logger
from app.models import CaptureInfo, TextRegion
from services.ocr.base import OCREngine
from services.translation.base import TranslationError, TranslationInterrupted, Translator
from utils.image_utils import (
    detect_colors,
    dominant_color,
    ensure_text_contrast,
    resize_for_ocr,
    sanitize_background,
)
from utils.language_utils import needs_translation
from utils.layout_utils import (
    WRAP_ALIGN_RATIO,
    WRAP_GAP_RATIO,
    companion_translation_texts,
    merge_wrapped_labels,
    ocr_lines_to_regions,
    uses_translation_panel,
    vertical_regions,
)
from utils.stability import CONFIRMED, signature
from utils.text_utils import clean_text, normalize_ocr_text, protect_texts, restore_texts

log = get_logger("worker")


@dataclass
class TranslationJob:
    """辨識整理好、等著翻譯的一筆字幕：翻譯需要的東西都在這裡，不再需要辨識那一側的任何狀態。"""

    capture: CaptureInfo
    regions: list[TextRegion]
    texts: list[str]  # 每個文本塊的原文（和 regions 一一對應）
    unique_texts: list[str]
    translatable: list[str]  # 真的要送去翻譯的（已去重、略過已經是中文的）
    source: str
    target: str
    merge_original: bool  # 譯文放在原文位置時，要把原文併進譯文
    key: tuple = field(default_factory=tuple)  # 這組文字的指紋（排程用）


def translate_job(
    job: TranslationJob,
    translator: Translator,
    stop: Callable[[], bool],
    status: Callable[[str], None],
) -> dict | None:
    """翻譯一筆字幕，回傳要交給畫面的結果；中途被要求停止就回傳 None。

    辨識與翻譯串在一起（``PipelineTask``）或分開（``TranslationTask``）時都用同一份，兩種模式的譯文一模一樣。
    """
    regions = job.regions
    texts = job.texts
    if getattr(translator, "is_ready", lambda: True)():
        status(f"正在翻譯 {len(regions)} 個文本塊...")
    else:
        # 離線模型還在載入（啟動或剛換模型）：下面的 translate 會等它載好，先說明在等什麼
        status("翻譯模型載入中…")
    # 图标、文件名以及已经是目标语言的文字保持原样。旧逻辑把中文
    # 中文再翻一遍、把文件夹图标“O/0”翻成方块，是整页发脏的主因。
    restored_map = {text: text for text in job.unique_texts}
    failed = 0
    if job.translatable:
        protected, mapping = protect_texts(job.translatable)
        translated = translator.translate(protected, job.source, job.target)
        restored_unique = [clean_text(text) for text in restore_texts(translated, mapping)]
        restored_map.update(zip(job.translatable, restored_unique))
        failed = getattr(translator, "last_failed_count", 0)
    restored = [restored_map[text] for text in texts]

    # 譯文放在框下方時原文本來就留在畫面上，不需要再把原文併進譯文
    if job.merge_original:
        restored = [
            f"{original}\n{translated_text}" if translated_text.strip() != original.strip() else original
            for original, translated_text in zip(texts, restored)
        ]

    if stop():
        return None

    for region, translated_text in zip(regions, restored):
        region.translated_text = translated_text

    if failed:
        status(f"翻譯完成：{failed} 個文本塊失敗，已保留原文")
    if job.translatable and getattr(translator, "unsupported_direction", False):
        status(
            "Mock 演示翻譯只支援輸出中文，當前目標語言下僅保留原文；"
            "請在 設定 → 翻譯 中填寫 OpenAI/DeepL/Google API Key 使用真實翻譯"
        )

    status("正在生成覆蓋層...")
    return {
        "capture": job.capture,
        "regions": regions,
        "recognized_count": len(regions),
        "translation_candidate_count": len(job.translatable),
        "translated_count": max(0, len(job.translatable) - failed),
        "failed_count": failed,
        # 即時監控（辨識與翻譯分開進行）據此回報「顯示的是哪一組文字」
        "gate_texts": list(job.translatable),
        "job_key": job.key,
    }


class PipelineTask(QThread):
    status = Signal(str)
    error = Signal(str)
    result = Signal(object)
    # 辨識與翻譯分開進行（defer_translation）：辨識整理好之後交出這一筆等翻譯的字幕，這個任務就結束
    job = Signal(object)

    def __init__(
        self,
        capture: CaptureInfo,
        ocr_engine: OCREngine,
        translator: Translator,
        config: AppConfig,
        parent=None,
        gate: Callable[[list[str]], str] | None = None,
        vertical: bool = False,
        defer_translation: bool = False,
    ) -> None:
        super().__init__(parent)
        self.capture = capture
        self.ocr_engine = ocr_engine
        self.translator = translator
        self.config = config
        # 即時監控的穩定化：拿要翻譯的文字問「該不該翻」，回傳 stability.SAME / PENDING / CONFIRMED
        self.gate = gate
        # 這次擷取的文字是直排（由上往下、欄由右往左）：版面照直排處理，需要能辨識直排的 OCR 引擎
        self.vertical = bool(vertical)
        self.defer_translation = bool(defer_translation)
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if self.vertical and not getattr(self.ocr_engine, "supports_vertical", False):
                self.error.emit("目前的 OCR 引擎不能辨識直向文字，請到「設定 → OCR」改用 rapidocr")
                return
            self._emit_status("正在識別...")
            # 小图先放大再 OCR：游戏截图/弹窗内的小字识别率提升数倍。
            # 放大倍数越高识别率越好但 OCR 更慢，可在设置里调小换速度。
            if getattr(self.ocr_engine, "benefits_from_upscale", True):
                image, scale = resize_for_ocr(
                    self.capture.image,
                    min_side=int(self.config.get("ocr.ocr_min_side", 2200)),
                    max_side=int(self.config.get("ocr.ocr_max_side", 4096)),
                    max_upscale=float(self.config.get("ocr.max_upscale", 3.0)),
                )
            else:
                image, scale = self.capture.image, 1.0
            lines = self.ocr_engine.recognize(
                image, lang=str(self.config.get("ocr.lang", "auto"))
            )
            # Upscaling normally helps small text, but some Windows OCR packs
            # occasionally return an empty result for the resized bitmap. Retry
            # the untouched capture before telling the user there is no text.
            if not lines and scale != 1.0:
                image = self.capture.image
                scale = 1.0
                lines = self.ocr_engine.recognize(
                    image, lang=str(self.config.get("ocr.lang", "auto"))
                )
            if self._stop:
                return
            for line in lines:
                line.text = normalize_ocr_text(clean_text(line.text))
            if scale != 1.0:
                for line in lines:
                    x, y, w, h = line.box
                    line.box = (
                        int(round(x / scale)),
                        int(round(y / scale)),
                        int(round(w / scale)),
                        int(round(h / scale)),
                    )
            if not lines:
                self.error.emit("沒有識別到文字（檢查 OCR 引擎設定或置信度閾值）")
                return

            target = str(self.config.get("translation.target_language", "zh-hant"))
            han_is_source = bool(getattr(self.translator, "han_is_source", False))
            min_confidence = float(self.config.get("ocr.min_confidence", 0.6))
            y_tolerance_ratio = float(self.config.get("ocr.merge_y_tolerance_ratio", 0.3))
            x_gap_ratio = float(self.config.get("ocr.merge_x_gap_ratio", 0.8))
            max_width_ratio = float(self.config.get("ocr.merge_wrapped_max_width_ratio", 1.0))
            merge_centered = bool(self.config.get("ocr.merge_centered_lines", False))
            wrap_gap_ratio = float(self.config.get("ocr.merge_wrapped_gap_ratio", WRAP_GAP_RATIO))
            wrap_align_ratio = float(self.config.get("ocr.merge_wrapped_align_ratio", WRAP_ALIGN_RATIO))
            if self.vertical:
                image_height, image_width = self.capture.image.shape[:2]
                regions = vertical_regions(
                    lines,
                    image_width=image_width,
                    image_height=image_height,
                    offset_x=self.capture.offset_x,
                    offset_y=self.capture.offset_y,
                    min_confidence=min_confidence,
                    y_tolerance_ratio=y_tolerance_ratio,
                    x_gap_ratio=x_gap_ratio,
                    target_language=target,
                    max_width_ratio=max_width_ratio,
                    merge_centered=merge_centered,
                    han_is_source=han_is_source,
                    gap_ratio=wrap_gap_ratio,
                    align_ratio=wrap_align_ratio,
                )
                if not regions:
                    self.error.emit("沒有識別到文字（置信度過濾後為空）")
                    return
            else:
                regions = ocr_lines_to_regions(
                    lines,
                    offset_x=self.capture.offset_x,
                    offset_y=self.capture.offset_y,
                    min_confidence=min_confidence,
                    y_tolerance_ratio=y_tolerance_ratio,
                    x_gap_ratio=x_gap_ratio,
                )
                if not regions:
                    self.error.emit("沒有識別到文字（置信度過濾後為空）")
                    return
                regions = merge_wrapped_labels(
                    regions,
                    target_language=target,
                    capture_width=self.capture.bbox[2] - self.capture.bbox[0],
                    max_width_ratio=max_width_ratio,
                    merge_centered=merge_centered,
                    han_is_source=han_is_source,
                    gap_ratio=wrap_gap_ratio,
                    align_ratio=wrap_align_ratio,
                )

            source = str(self.config.get("translation.source_language", "auto"))
            texts = [region.text for region in regions]
            # 同批去重：重复文本只请求一次，减少免费服务限流压力
            unique_texts = list(dict.fromkeys(texts))
            # 雙語字幕裡疊在日文旁的那行中文譯文：本來就看得懂，不要再當日文翻一遍
            skipped = (
                companion_translation_texts(regions)
                if han_is_source and target.lower().startswith("zh")
                and bool(self.config.get("ocr.skip_companion_translation", True))
                else set()
            )
            translatable = [
                text
                for text in unique_texts
                if text not in skipped and needs_translation(text, target, han_is_source=han_is_source)
            ]

            if self.gate is not None:
                verdict = self.gate(translatable)
                if verdict != CONFIRMED:
                    # 與已顯示的相同（不必再翻），或是還沒穩定的新文字（先不翻）：不動畫面
                    self.result.emit({"capture": self.capture, "regions": [], "verdict": verdict})
                    return
            if self._stop:
                return

            # 背景亮度 -> 自动文字颜色
            auto_color = bool(self.config.get("overlay.use_auto_text_color", True))
            default_color = str(self.config.get("overlay.text_color", "#FFFFFF"))
            # 整图页面基调色，用于交叉校验块级背景识别
            global_background = dominant_color(self.capture.image)
            for region in regions:
                # Windows OCR 的框通常紧贴字形；向外采一小圈，避免把黑色
                # 笔画误判成背景，也与覆盖层的原字擦除范围保持一致。
                sample_margin = max(2, min(6, int(round(region.height * 0.18))))
                background_hex, text_hex, lum = detect_colors(
                    self.capture.image,
                    region.x - self.capture.offset_x - sample_margin,
                    region.y - self.capture.offset_y - sample_margin,
                    region.width + sample_margin * 2,
                    region.height + sample_margin * 2,
                )
                region.source_luminance = lum
                region.background_color = sanitize_background(background_hex, global_background)
                region.text_color = text_hex if auto_color else default_color
                if auto_color:
                    region.text_color = ensure_text_contrast(region.text_color, region.background_color)

            job = TranslationJob(
                capture=self.capture,
                regions=regions,
                texts=texts,
                unique_texts=unique_texts,
                translatable=translatable,
                source=source,
                target=target,
                merge_original=bool(self.config.get("translation.keep_original", False))
                and not uses_translation_panel(self.config.get("overlay.placement", "below"), self.capture.mode),
                key=signature(translatable),
            )
            if self.defer_translation:
                self.job.emit(job)
                return
            payload = translate_job(job, self.translator, lambda: self._stop, self._emit_status)
            if payload is not None:
                self.result.emit(payload)
        except TranslationError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            log.exception("管線處理失敗")
            self.error.emit(f"處理失敗：{exc}")
    def _emit_status(self, text: str) -> None:
        self.status.emit(text)


class TranslationTask(QThread):
    """只負責翻譯一筆字幕（辨識與翻譯分開進行時用）。

    ``cancel()``：畫面上已經有更新的字幕，這一筆不必翻了——翻譯會在下一個詞（或下一句）停下來，
    不送出結果。翻完才被取消的，結果也不送出（快取裡已經有了）。"""

    status = Signal(str)
    error = Signal(str)
    result = Signal(object)

    def __init__(self, job: TranslationJob, translator: Translator, parent=None) -> None:
        super().__init__(parent)
        self.job = job
        self.translator = translator
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop

    def run(self) -> None:
        if self._stop:
            return  # 排隊的時候就已經過時了
        try:
            interruptible = getattr(self.translator, "interruptible", None)
            scope = interruptible(lambda: self._stop) if callable(interruptible) else nullcontext()
            with scope:
                payload = translate_job(self.job, self.translator, lambda: self._stop, self.status.emit)
            if payload is not None and not self._stop:
                self.result.emit(payload)
        except TranslationInterrupted:
            return  # 被放棄的翻譯：沒有結果、不算錯誤
        except TranslationError as exc:
            if not self._stop:
                self.error.emit(str(exc))
        except Exception as exc:
            log.exception("翻譯任務失敗")
            if not self._stop:
                self.error.emit(f"處理失敗：{exc}")
