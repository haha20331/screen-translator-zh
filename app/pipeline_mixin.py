"""翻译管线协调：启动 worker、接收状态/错误/结果、收尾与回滚。"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt

from app.logger import get_logger
from app.models import CaptureInfo, TextRegion
from ui.motion import STATUS_HOLD
from utils.stability import CONFIRMED, PENDING
from workers.translation_worker import PipelineTask

log = get_logger("application")


class PipelineMixin:
    """翻译管线协调：启动 worker、接收状态/错误/结果、收尾与回滚。"""

    def _commit_stable_text(self) -> None:
        """放行的那組文字已經顯示出來了：之後辨識到一樣的文字就不再翻譯。"""
        if self._monitor_active and self._monitor_stabilizer is not None:
            self._monitor_stabilizer.commit()

    def _reset_translation_context(self) -> None:
        """新的擷取、監控開始／停止＝換了一段內容，先前記住的上下文不該再影響翻譯。"""
        if self.translator is not None:
            self.translator.reset_context()

    def _start_pipeline(self, capture: CaptureInfo) -> None:
        if not self._monitor_active:  # 監控期間每一輪都是同一段內容的延續，要保留上下文
            self._reset_translation_context()
        if not self._cancel_worker():
            self._abort_capture(
                "上一次翻譯仍在結束，請稍後重試",
                failed=True,
                error_title="任務仍在執行",
                error_message="舊任務尚未安全結束，沒有啟動新的翻譯。",
            )
            return
        try:
            config_snapshot = copy.copy(self.config)
            config_snapshot.data = copy.deepcopy(self.config.data)
            self.worker = PipelineTask(
                capture=capture,
                ocr_engine=self.ocr_engine,
                translator=self.translator,
                config=config_snapshot,
                gate=self._monitor_stabilizer.observe if self._monitor_active and self._monitor_stabilizer else None,
            )
            self.worker.status.connect(self._on_worker_status)
            self.worker.error.connect(self._on_worker_error)
            self.worker.result.connect(self.on_pipeline_result)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start()
        except Exception as exc:
            log.exception("啟動翻譯任務失敗")
            self._abort_capture(
                "無法啟動翻譯任務",
                failed=True,
                error_title="處理失敗",
                error_message=str(exc),
            )

    def _on_worker_status(self, text: str) -> None:
        signal_sender = self.sender()
        if (
            self._shutting_down
            or (self._capture_session is None and not self._monitor_active)
            or (signal_sender is not None and signal_sender is not self.worker)
        ):
            return
        self.set_status(text)
        if not self._monitor_active:
            self.floating_status.set_text(text)

    def _on_worker_error(self, message: str) -> None:
        signal_sender = self.sender()
        if (
            self._shutting_down
            or (self._capture_session is None and not self._monitor_active)
            or (signal_sender is not None and signal_sender is not self.worker)
        ):
            return
        if self._monitor_active:
            # 画面里没有文字是常态（视频/空白），不算错误，照常继续轮询
            if message.startswith("沒有識別到文字"):
                self._monitor_no_text = True
                self._monitor_idle_hash = self._monitor_frame_hash
                self._arm_no_text_timer()
                self.set_status("監控中：當前畫面沒有可翻譯的文字")
            else:
                self._pipeline_error = message
                self.set_status(f"監控出錯：{message}")
            return
        self._pipeline_error = message
        self.set_status(f"處理失敗：{message}")

    def _on_worker_finished(self) -> None:
        signal_sender = self.sender()
        if signal_sender is not None and signal_sender is not self.worker:
            return
        finished_worker = self.worker
        if finished_worker is None:
            return
        self.worker = None
        finished_worker.deleteLater()
        if self._shutting_down:
            return
        if self._monitor_active:
            if self._pipeline_succeeded or self._monitor_no_text:
                self._monitor_errors = 0
            else:
                self._monitor_errors += 1
                if self._monitor_errors >= int(
                    self.config.get("monitor.max_consecutive_errors", 3)
                ):
                    self.stop_monitor(reason="連續翻譯失敗，已停止即時監控")
                    return
            self._monitor_no_text = False
            self._monitor_schedule_next()
            return
        if self._pipeline_succeeded:
            session = self._capture_session
            self._capture_session = None
            if self.window is not None and session is not None:
                self.window.finish_capture(session.mode)
            self._set_busy(False)
            self.floating_status.hide_fade(delay_ms=STATUS_HOLD)
            self._auto_start_monitor()
            self._transcript_finish_single_shot()  # 沒有接著監控：這一句的結束時間不可知，直接寫出
            return
        message = self._pipeline_error or "處理已取消"
        show_dialog = bool(self._pipeline_error)
        self._abort_capture(
            message,
            failed=show_dialog,
            error_title="處理失敗" if show_dialog else "",
            error_message=message if show_dialog else "",
        )

    def _cancel_worker(self) -> bool:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            if not self.worker.wait(1000):
                return False
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        return True

    def on_pipeline_result(self, payload: dict) -> None:
        signal_sender = self.sender()
        if (
            self._shutting_down
            # 即時監控是在單次擷取結束（session 已清空）之後才啟動的，沒有 session 是常態；
            # 上游漏了這個例外，導致每一輪監控結果都被丟掉並被判為連續失敗。
            or (self._capture_session is None and not self._monitor_active)
            or (signal_sender is not None and signal_sender is not self.worker)
        ):
            return
        capture: CaptureInfo = payload["capture"]
        verdict = payload.get("verdict", CONFIRMED)
        self._cancel_no_text_timer()  # 這一輪抓到文字了：不用因為「沒有字幕」關掉譯文
        if verdict != CONFIRMED:
            # 穩定化：文字與已顯示的相同（不必再翻譯），或是新文字還沒穩定。畫面維持原樣，也不算失敗。
            self._pipeline_succeeded = True
            if verdict == PENDING:
                self.set_status("監控中：等待字幕穩定")
            return
        regions: list[TextRegion] = payload["regions"]
        recognized_count = int(payload.get("recognized_count", len(regions)))
        translated_count = int(
            payload.get("translated_count", max(0, len(regions)))
        )
        failed_count = int(payload.get("failed_count", 0))
        if self.overlay_manager is None:
            return
        if not regions:
            self._pipeline_error = "沒有識別到可翻譯的文字"
            return
        # 监控模式下，内容没变就保留当前覆盖层不重绘，避免整屏闪烁
        if self._monitor_active:
            from app.monitor_utils import known_translations, region_signature

            signature = region_signature(capture, regions)
            self._monitor_known = known_translations(regions)
            if signature == self._monitor_signature:
                self._pipeline_succeeded = True
                self._commit_stable_text()
                return
            self._monitor_signature = signature
        # Preserve the previous translation until a new result actually exists;
        # that lets cancellation/OCR errors restore it without stale ghost windows.
        # 监控模式不整屏清屏（避免闪烁），只更新变化的文本块。
        if not self._monitor_active:
            self.overlay_manager.clear_all()
        overlay_visible = self.overlay_manager.show_regions(
            capture, regions, animate_new_only=self._monitor_active
        )
        self._sync_overlay_state(overlay_visible)
        if self._monitor_active and self._monitor_region_selector is not None:
            self._monitor_region_selector.raise_()
        self._pipeline_succeeded = True
        self._commit_stable_text()
        self._transcript_show(regions, single_shot=not self._monitor_active)
        if failed_count:
            self.set_status(
                f"識別 {recognized_count} 個文本塊，已翻譯 {translated_count} 個，"
                f"{failed_count} 個保留原文"
            )
        else:
            self.set_status(
                f"翻譯完成：識別 {recognized_count} 個文本塊，"
                f"已翻譯 {translated_count} 個"
            )
        self.floating_status.set_text("翻譯完成")
        self._save_history(capture, regions)
        original = "\n".join(str(getattr(r, "text", "") or "") for r in regions)
        translated = "\n".join(str(getattr(r, "translated_text", "") or "") for r in regions)
        self._last_original_text = original
        self._last_translated_text = translated
