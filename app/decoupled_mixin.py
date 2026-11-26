"""辨識與翻譯分開進行（設定 → 翻譯 →「辨識與翻譯分開進行」）。

預設的即時監控是「截圖 → 辨識 → 翻譯 → 顯示」排成一串：翻譯一句要 0.5～1.3 秒，這段時間完全不看畫面，
字幕換得快就會漏掉、或是新字幕要等舊的翻完才開始辨識。

開啟之後：

* 辨識（``PipelineTask``）照常一直看畫面，確認有新字幕就把它交給這裡排隊，自己馬上結束、排下一輪；
* 翻譯（``TranslationTask``）另外在背景做，同一時間只翻一筆（模型一次只能翻一句）；
* 翻譯途中又確認了更新的字幕：還沒開始的舊字幕直接蓋掉，正在翻的那句立刻放棄（下一個詞就停），
  翻完才發現過時的結果不顯示——畫面上永遠是最新的一句。

誰該翻、誰過時由 :class:`utils.translation_scheduler.TranslationScheduler` 決定，這裡只負責接線。
單次擷取（框選、全螢幕、視窗）不受影響，照舊一次做完。
"""

from __future__ import annotations

from app.logger import get_logger
from utils.translation_scheduler import QUEUED, START, TranslationScheduler
from workers.translation_worker import TranslationJob, TranslationTask

log = get_logger("application")


class DecoupledTranslationMixin:
    def _init_decoupled_translation(self) -> None:
        self._translation_scheduler = TranslationScheduler()
        self._translation_task: TranslationTask | None = None
        self._translation_errors = 0  # 連續幾筆翻譯失敗（辨識每一輪都成功，所以不能用辨識那邊的計數）

    # ------------------------------------------------------------------ 狀態
    def _decoupled_enabled(self) -> bool:
        """這一輪辨識要不要把翻譯交給另一條路：只有即時監控、而且設定開著。"""
        return bool(self._monitor_active and self.config.get("translation.decouple_ocr", False))

    def _translation_busy(self) -> bool:
        """有字幕正在翻或排著隊（畫面沒變的話，這時候不必再辨識同一格畫面）。"""
        return self._translation_scheduler.busy

    def _translation_task_running(self) -> bool:
        task = self._translation_task
        return task is not None and task.isRunning()

    # ------------------------------------------------------------------ 辨識交來一筆字幕
    def _on_translation_job(self, job: TranslationJob) -> None:
        sender = self.sender()
        if self._shutting_down or not self._monitor_active or (sender is not None and sender is not self.worker):
            return
        self._pipeline_succeeded = True  # 辨識這一輪成功了（翻譯成不成功另外算）
        self._cancel_no_text_timer()  # 這一輪抓到文字了：不用因為「沒有字幕」關掉譯文
        verdict = self._translation_scheduler.submit(job.key, job)
        if verdict == START:
            self._start_translation(job)
        elif verdict == QUEUED:
            self.set_status("監控中：有更新的字幕，放棄還沒翻完的舊字幕")
            self._interrupt_stale_translation()

    def _interrupt_stale_translation(self) -> None:
        """正在翻的那一筆已經過時了：叫它在下一個詞停下來。"""
        if self._translation_scheduler.should_stop_running() and self._translation_task is not None:
            self._translation_task.cancel()

    def _start_translation(self, job: TranslationJob) -> None:
        try:
            task = TranslationTask(job, self.translator)
            task.status.connect(self._on_translation_status)
            task.error.connect(self._on_translation_error)
            task.result.connect(self._on_translation_result)
            task.finished.connect(self._on_translation_finished)
            self._translation_task = task
            task.start()
        except Exception as exc:
            log.exception("啟動翻譯任務失敗")
            self._translation_task = None
            self._translation_scheduler.cancel_all()
            self._translation_scheduler.finished()
            self._translation_errors += 1
            self.set_status(f"監控出錯：無法啟動翻譯：{exc}")

    # ------------------------------------------------------------------ 翻譯的回報
    def _is_current_translation(self) -> bool:
        """目前這個信號是不是來自還在跑的那筆翻譯，而且監控還開著。"""
        sender = self.sender()
        return not (
            self._shutting_down
            or not self._monitor_active
            or (sender is not None and sender is not self._translation_task)
        )

    def _on_translation_status(self, text: str) -> None:
        if not self._is_current_translation():
            return
        task = self._translation_task
        if task is not None and self._translation_scheduler.is_current(task.job.key):
            self.set_status(text)

    def _on_translation_result(self, payload: dict) -> None:
        if not self._is_current_translation():
            return
        if not self._translation_scheduler.is_current(payload.get("job_key")):
            return  # 畫面上已經有更新的字幕：這一句不顯示（譯文已經存進快取，之後再出現就直接用）
        self._translation_errors = 0
        self._present_result(payload, commit_texts=payload.get("gate_texts"))

    def _on_translation_error(self, message: str) -> None:
        if not self._is_current_translation():
            return
        task = self._translation_task
        if task is None or not self._translation_scheduler.is_current(task.job.key):
            return  # 過時的那一句出錯不用管
        self._pipeline_error = message
        self.set_status(f"監控出錯：{message}")
        self._translation_errors += 1
        if self._translation_errors >= int(self.config.get("monitor.max_consecutive_errors", 3)):
            self.stop_monitor(reason="連續翻譯失敗，已停止即時監控")

    def _on_translation_finished(self) -> None:
        sender = self.sender()
        task = self._translation_task
        if sender is not None and sender is not task:
            return
        self._translation_task = None
        if task is not None:
            task.deleteLater()
        following = self._translation_scheduler.finished()
        if self._shutting_down:
            self._maybe_finalize_shutdown()
            return
        if following is None:
            self._resume_model_release()
            return
        if self._monitor_active:
            self._start_translation(following[1])
        else:  # 監控已經停了：排著的不必翻
            self._translation_scheduler.cancel_all()
            self._translation_scheduler.finished()

    # ------------------------------------------------------------------ 作廢與結束
    def _cancel_translation_jobs(self) -> None:
        """畫面上的內容作廢了（停止監控、重新整理、字幕消失、換設定）：排隊的丟掉，正在翻的放棄。"""
        if self._translation_scheduler.cancel_all() and self._translation_task is not None:
            self._translation_task.cancel()

    def _wait_for_translation_before_shutdown(self) -> None:
        """離開程式：叫翻譯停下來，等它真的結束才能放掉 QThread。"""
        self._cancel_translation_jobs()
        task = self._translation_task
        if task is not None and task.isRunning():
            from PySide6.QtCore import Qt

            task.cancel()
            task.finished.connect(self._maybe_finalize_shutdown, Qt.ConnectionType.UniqueConnection)
