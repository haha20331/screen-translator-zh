"""翻譯模型的背景預載：啟動、換模型存檔後立刻載入並預熱，載入期間讓使用者看得到、監控先不送翻譯。"""

from __future__ import annotations

import os
import threading
import time

from app.logger import get_logger
from ui.motion import STATUS_HOLD

log = get_logger("application")

_BUBBLE_UPDATING = "模型更換中，請稍候…"  # 浮動提示氣泡很窄（約十二個字），文字要短
_BUBBLE_READY = "模型已就緒"
_BUBBLE_FAILED = "模型載入失敗"


class ModelLoadingMixin:
    """離線翻譯模型要載入數 GB、預熱要幾秒到幾十秒（Vulkan 第一次還要編譯）。

    設定存檔換了模型後馬上在背景載入，並告訴使用者「模型更換中」；
    載入期間即時監控不送新的翻譯（送了也只是卡在載入上，重新整理還會跳出「舊任務尚未結束」），
    載入完成後自動接著翻譯。其他（線上）翻譯服務沒有載入問題，不受影響。
    """

    def _init_model_loading(self) -> None:
        self._warming: list = []  # 正在背景預熱的翻譯器（同一個翻譯器預熱兩次就列兩次）
        self._warm_label = ""  # 上一次預熱的模型標記，用來分辨「更換模型」與「第一次載入」
        self._model_notice_for = None  # 已經告訴使用者「載入中」的翻譯器：載入完成時要回報
        self._model_bubble = False  # 浮動氣泡是不是我們叫出來的（擷取流程也用同一個氣泡）
        self._warmup_finished.connect(self._on_warmup_finished)

    def _translator_warming(self) -> bool:
        """目前的翻譯器是不是還在背景載入／預熱（監控這時候不送新的翻譯）。"""
        translator = self.translator
        return translator is not None and any(item is translator for item in self._warming)

    def _warmup(self) -> str:
        """背景預熱 OCR 與離線翻譯模型，避免第一次截圖時卡很久。

        回傳要給使用者看的「模型載入中」提示（沒有要等的時候是空字串），讓呼叫的人併進自己的狀態列文字。"""
        label = self._translator_label()
        previous, self._warm_label = self._warm_label, label
        if os.environ.get("SCREEN_TRANSLATOR_NO_WARMUP") == "1" or os.environ.get(
            "SCREEN_TRANSLATOR_SELFTEST"
        ) == "1":
            return ""
        ocr_engine = self.ocr_engine
        translator = self.translator
        translator_warmup = getattr(translator, "warmup", None)
        loads_model = callable(translator_warmup)
        note = ""
        if loads_model:
            self._warming.append(translator)
            # 模型已經在記憶體裡（只是設定改了、重算固定前綴要幾秒）就不吵使用者
            if not getattr(translator, "is_ready", lambda: True)():
                note = self._announce_model_loading(translator, changed=bool(previous) and previous != label)

        def job() -> None:
            started = time.monotonic()
            try:
                if ocr_engine is not None:
                    ocr_engine.warmup()
            except Exception as exc:
                log.warning("OCR 模型預熱失敗：%s", exc)
            error = ""
            if loads_model:
                try:
                    translator_warmup()
                except Exception as exc:
                    log.warning("翻譯模型預熱失敗：%s", exc)
                    error = str(exc) or type(exc).__name__
                try:
                    self._warmup_finished.emit(translator, error, time.monotonic() - started)
                except RuntimeError:  # 程式已經在結束，控制器已被刪除
                    pass

        threading.Thread(target=job, daemon=True, name="ocr-warmup").start()
        return note

    def _announce_model_loading(self, translator, changed: bool) -> str:
        self._model_notice_for = translator
        name = self._model_display_name(translator)
        if changed:
            # 使用者剛在設定換了模型：中央上方的氣泡一起提示（擷取中的話氣泡是擷取流程的，不去搶）
            if self._capture_session is None and not self._busy:
                self.floating_status.show_fade(_BUBBLE_UPDATING)
                self._model_bubble = True
            return f"翻譯模型更換中：正在載入 {name}，完成後自動繼續翻譯"
        return f"正在載入翻譯模型 {name}，完成後自動繼續翻譯"

    @staticmethod
    def _model_display_name(translator) -> str:
        name = getattr(translator, "display_model_name", None)
        if callable(name):
            try:
                return str(name())
            except Exception:
                pass
        return str(getattr(translator, "name", "") or "翻譯模型")

    def _on_warmup_finished(self, translator, error: str, seconds: float) -> None:
        """（主執行緒）背景預熱結束：回報結果，監控下一輪起照常翻譯。"""
        for index, item in enumerate(self._warming):
            if item is translator:
                del self._warming[index]
                break
        if self._shutting_down or translator is not self._model_notice_for:
            return  # 沒有告訴過使用者載入中（模型本來就在記憶體裡），或已經換成別的翻譯器了
        if any(item is translator for item in self._warming):
            return  # 同一個翻譯器還有一次預熱在跑，等它
        self._model_notice_for = None
        name = self._model_display_name(translator)
        if error:
            self.set_status(f"翻譯模型載入失敗：{error.strip().splitlines()[0] if error.strip() else error}")
            self._finish_model_bubble(_BUBBLE_FAILED, hold_ms=STATUS_HOLD * 3)
            return
        device = {"gpu": "GPU", "cpu": "CPU"}.get(str(getattr(translator, "device", "") or ""), "")
        detail = "，".join(part for part in (device, f"{seconds:.0f} 秒") if part)
        self.set_status(f"翻譯模型已就緒：{name}（{detail}）")
        self._finish_model_bubble(_BUBBLE_READY, hold_ms=STATUS_HOLD)

    def _finish_model_bubble(self, text: str, hold_ms: int) -> None:
        if not self._model_bubble:
            return
        self._model_bubble = False
        if self._capture_session is not None or self._busy:
            return  # 氣泡已被擷取流程接手，它自己會收
        self.floating_status.set_text(text)
        self.floating_status.hide_fade(delay_ms=hold_ms)
