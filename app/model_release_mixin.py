"""停止即時監控一段時間後，把翻譯模型從記憶體放掉；之後用到時再自動載回。

3B 模型載入後會佔掉約 2.8 GB 記憶體（用顯示卡時處理程序約 4.6 GB），關掉監控之後留著沒有意義。
停止監控超過 ``monitor.release_model_after_seconds`` 秒（預設 10，0 = 不釋放）就釋放；
再次開始擷取或監控時，會先在背景把模型載回來（和使用者框選範圍的時間重疊），載好之前監控不送翻譯。
"""

from __future__ import annotations

import gc

from PySide6.QtCore import QTimer

from app.logger import get_logger

log = get_logger("application")


class ModelReleaseMixin:
    def _init_model_release(self) -> None:
        self._model_release_wanted = False  # 監控停了、模型還沒放掉：之後只要閒下來就要（重新）倒數
        self._model_release_timer = QTimer(self)
        self._model_release_timer.setSingleShot(True)
        self._model_release_timer.timeout.connect(self._release_idle_model)

    def _release_model_after_seconds(self) -> float:
        try:
            return max(0.0, float(self.config.get("monitor.release_model_after_seconds", 10)))
        except (TypeError, ValueError):
            return 10.0

    def _can_release_model(self) -> bool:
        return callable(getattr(self.translator, "close", None))

    def _arm_model_release(self) -> None:
        """監控剛停止：倒數幾秒後釋放模型。"""
        seconds = self._release_model_after_seconds()
        if seconds <= 0 or self._shutting_down or not self._can_release_model():
            return
        self._model_release_wanted = True
        self._restart_model_release_timer(seconds)

    def _restart_model_release_timer(self, seconds: float) -> None:
        self._model_release_timer.start(max(1, int(seconds * 1000)))

    def _cancel_model_release(self) -> None:
        """重新開始即時監控：不必釋放了。"""
        self._model_release_wanted = False
        self._model_release_timer.stop()

    def _resume_model_release(self) -> None:
        """擷取或載入結束、又閒下來了：監控停了還沒釋放的話，重新倒數（例如擷取被取消，沒有接著進入監控）。"""
        if self._model_release_wanted and not self._monitor_active and not self._shutting_down:
            seconds = self._release_model_after_seconds()
            if seconds > 0:
                self._restart_model_release_timer(seconds)

    def _model_is_idle(self) -> bool:
        worker = getattr(self, "worker", None)
        return not (
            self._monitor_active
            or self._busy
            or self._capture_session is not None
            or self._translator_warming()
            or (worker is not None and worker.isRunning())
        )

    def _release_idle_model(self) -> None:
        if not self._model_release_wanted or self._shutting_down:
            return
        if not self._model_is_idle():
            self._resume_model_release()  # 正在擷取、翻譯或載入：晚一點再看
            return
        self._model_release_wanted = False
        translator = self.translator
        close = getattr(translator, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            log.exception("釋放翻譯模型失敗")
            return
        gc.collect()  # 模型的緩衝區是 ctypes 與 llama.cpp 的記憶體，物件放掉後才還給系統
        log.info("停止即時監控超過 %g 秒，已釋放翻譯模型的記憶體（下次翻譯會自動載回）", self._release_model_after_seconds())
        self.set_status("已釋放翻譯模型的記憶體，下次擷取時會自動載回")

    def _ensure_model_loaded(self) -> str:
        """要用模型了：還沒載入（例如剛被釋放）就在背景載回來；回傳要給使用者看的「載入中」提示，沒有就是空字串。"""
        self._model_release_timer.stop()  # 擷取的這段時間不釋放；擷取結束時 _set_busy(False) 會視情況重新倒數
        translator = self.translator
        if (
            translator is None
            or not callable(getattr(translator, "warmup", None))
            or getattr(translator, "is_ready", lambda: True)()
            or self._translator_warming()
        ):
            return ""
        return self._warmup() or ""
