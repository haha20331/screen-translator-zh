"""文字輸出：把顯示過的字幕（原文、譯文、四角點、時間、模型）寫進 txt。

只負責接線：字幕「開始、結束」的時機從監控與管線的各個地方呼叫進來，內容整理在
``services/transcript.py``，真正的檔案寫入在 ``services/transcript_writer.py`` 的專用執行緒，
所以不會拖慢辨識與翻譯。這裡的任何錯誤都只記錄、不影響畫面上的翻譯。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from app.logger import get_logger
from app.models import TextRegion
from app.monitor_utils import _without_kept_original
from services.transcript import DEFAULT_MIN_SECONDS, Row, TranscriptRecorder, resolve_directory
from services.transcript_writer import TranscriptWriter

log = get_logger("application")


class TranscriptMixin:
    """文字輸出：字幕開始與結束的接線、輸出資料夾、關閉時寫完。"""

    def _init_transcript(self) -> None:
        self._transcript_writer = TranscriptWriter(
            self.transcript_directory(), on_problem=self._transcript_problem.emit
        )
        self._transcript = TranscriptRecorder(self._transcript_writer, min_seconds=self._transcript_min_seconds())
        self._transcript_problem.connect(self._on_transcript_problem)

    # ------------------------------------------------------------------ 設定
    def transcript_directory(self) -> Path:
        default = Path(self.config.path.parent) / "output"
        return resolve_directory(str(self.config.get("output.directory", "") or ""), default)

    def _transcript_enabled(self) -> bool:
        return bool(self.config.get("output.enabled", True))

    def _transcript_min_seconds(self) -> float:
        try:
            return max(0.0, float(self.config.get("output.min_seconds", DEFAULT_MIN_SECONDS)))
        except (TypeError, ValueError):
            return DEFAULT_MIN_SECONDS

    def _apply_transcript_settings(self) -> None:
        """設定儲存後：換資料夾、換「短暫」門檻；關掉輸出時，正在顯示的那句照常寫完（它是在開啟時開始的）。"""
        self._transcript_writer.set_directory(self.transcript_directory())
        self._transcript.min_seconds = self._transcript_min_seconds()
        if not self._transcript_enabled():
            self._transcript_end("stop")

    def open_output_folder(self) -> None:
        """托盤與設定頁的「開啟輸出資料夾」：資料夾不存在就先建立；先把佇列裡的字幕寫完，打開時看得到最新的。"""
        directory = self.transcript_directory()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._transcript_writer.flush(timeout=1.0)
        except OSError as exc:
            log.warning("無法建立輸出資料夾 %s：%s", directory, exc)
            self.set_status(f"無法建立輸出資料夾：{exc}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            self.set_status(f"無法開啟資料夾：{directory}")

    # ------------------------------------------------------------------ 字幕的開始與結束
    def _transcript_model_label(self) -> str:
        translator = getattr(self, "translator", None)
        try:
            label = translator.model_label() if translator is not None else ""
        except Exception:
            label = ""
        return label or str(self.config.get("translation.service", ""))

    def _transcript_show(self, regions: list[TextRegion], *, single_shot: bool) -> None:
        """畫面上顯示了一組新的字幕（已翻譯完）。"""
        if not self._transcript_enabled():
            return
        try:
            rows = [
                Row.from_box(
                    region.text,
                    _without_kept_original(region.text, region.translated_text or ""),
                    region.x,
                    region.y,
                    region.width,
                    region.height,
                )
                for region in regions
                if (region.text or "").strip()
            ]
            self._transcript.show(rows, self._transcript_model_label(), single_shot=single_shot)
        except Exception:
            log.exception("記錄字幕輸出失敗")

    def _transcript_end(self, reason: str) -> None:
        try:
            self._transcript.end(reason)
        except Exception:
            log.exception("結束字幕輸出失敗")

    def _transcript_text_gone(self) -> None:
        self._transcript.note_text_gone()

    def _transcript_text_back(self) -> None:
        self._transcript.cancel_text_gone()

    def _transcript_finish_single_shot(self) -> None:
        """單次截圖翻譯結束，且沒有接著進入監控：寫出那一句（結束時間留白）。"""
        if not self._monitor_active:
            try:
                self._transcript.finish_single_shot()
            except Exception:
                log.exception("寫出單次翻譯的輸出失敗")

    def _transcript_note_monitor(self, started: bool) -> None:
        if not self._transcript_enabled():
            return
        try:
            if started:
                request = self._monitor_request or {}
                bbox = request.get("bbox")
                box = "(" + ",".join(str(int(v)) for v in bbox) + ")" if bbox else "-"
                self._transcript.comment(f"監控開始 模式={request.get('mode', '-')} 框={box} 模型={self._transcript_model_label()}")
            else:
                self._transcript.comment("監控停止")
        except Exception:
            log.exception("記錄監控開始／停止失敗")

    def _transcript_shutdown(self) -> None:
        """離開程式：寫出正在顯示的那一句，把佇列寫完並關檔。"""
        self._transcript_end("shutdown")
        try:
            self._transcript_writer.close(timeout=3.0)
        except Exception:
            log.exception("關閉輸出檔失敗")

    def _on_transcript_problem(self, message: str) -> None:
        log.warning("%s", message)
        self.set_status(message)
