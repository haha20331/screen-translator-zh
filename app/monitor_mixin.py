"""实时监控：按固定间隔截图、像素门控、调度翻译管线。"""

from __future__ import annotations

# 通过 app.application 命名空间解析 is_window_capturable，使测试对该符号的 monkeypatch 仍然生效。
from app import application as _application
from app.logger import get_logger
from utils.stability import (
    MAX_INTERVAL_MS,
    MAX_NO_TEXT_SECONDS,
    MAX_ROUNDS,
    MIN_INTERVAL_MS,
    MIN_ROUNDS,
    RECOMMENDED_FAST_ROUNDS,
    RECOMMENDED_INTERVAL_MS,
    RECOMMENDED_MAX_WAIT,
    RECOMMENDED_NO_TEXT_SECONDS,
    TextStabilizer,
)

log = get_logger("application")


class MonitorMixin:
    """实时监控：按固定间隔截图、像素门控、调度翻译管线。"""

    def toggle_monitor(self) -> None:
        if self._monitor_active:
            self.stop_monitor()
        else:
            self.start_monitor()

    def _auto_start_monitor(self) -> None:
        """一次翻譯成功後，預設接著進入即時監控（monitor.auto_start）；失敗的翻譯不會自動開始。"""
        if (
            self.config.get("monitor.auto_start", True)
            and self.config.get("monitor.enabled", True)
            and not self._monitor_active
            and not self._shutting_down
            and self._last_capture is not None
        ):
            self.start_monitor()

    def start_monitor(self) -> None:
        if self._monitor_active:
            return
        if self._busy:
            self.set_status("正在處理上一次任務，稍後再開監控")
            return
        if self._last_capture is None:
            self.set_status("先截一張圖，再開啟即時監控")
            return
        if self._last_capture.mode == "window" and (
            not self._last_window_hwnd or not _application.is_window_capturable(self._last_window_hwnd)
        ):
            self.set_status("上次視窗已關閉，請重新框選/視窗截圖後再監控")
            return
        if not bool(self.config.get("monitor.enabled", True)):
            self.set_status("即時監控已在設定中關閉")
            return
        self._monitor_request = {
            "mode": self._last_capture.mode,
            "bbox": tuple(self._last_capture.bbox),
            "hwnd": self._last_window_hwnd,
        }
        if self._last_capture.mode == "region":
            self._create_monitor_region_selector()
        self._reset_translation_context()
        self._monitor_active = True
        self._no_text_timer.stop()
        self._monitor_errors = 0
        self._monitor_no_text = False
        self._monitor_known = {}
        self._monitor_signature = ""
        self._monitor_frame_hash = ""
        self._monitor_idle_hash = ""
        self._monitor_stabilizer = TextStabilizer()
        self._configure_stabilizer()
        self._transcript_note_monitor(started=True)
        self.set_status("即時監控已開啟，頁面一動就會自動重譯")
        if self.window is not None:
            self.window.set_monitor_checked(True)
        if self.tray is not None:
            self.tray.set_monitor_checked(True)
        self._monitor_cycle()

    def stop_monitor(self, reason: str = "") -> None:
        if not self._monitor_active:
            return
        self._monitor_active = False
        self._transcript_end("stop")
        self._transcript_note_monitor(started=False)
        self._reset_translation_context()
        self._monitor_timer.stop()
        self._no_text_timer.stop()
        self._monitor_request = None
        self._monitor_known = {}
        self._monitor_signature = ""
        self._monitor_frame_hash = ""
        self._monitor_idle_hash = ""
        self._monitor_stabilizer = None
        self._dispose_monitor_region()
        self._cancel_worker()
        # 監控期間顯示的譯文隨監控結束一起清掉，不留在畫面上
        if self.overlay_manager is not None:
            self.overlay_manager.clear_all()
            self._sync_overlay_state(False)
        if self.window is not None:
            self.window.set_monitor_checked(False)
        if self.tray is not None:
            self.tray.set_monitor_checked(False)
        self.set_status(reason or "已停止即時監控")

    def _monitor_cycle(self) -> None:
        from utils.image_utils import frame_fingerprint

        if not self._monitor_active or self._shutting_down:
            return
        request = self._monitor_request
        if request is None:
            return
        if self._translator_warming():
            # 翻譯模型還在背景載入：這一輪送出去只會卡在載入上（重新整理還會被當成舊任務沒結束）。
            # 這一格畫面不記進去，載入完成後的下一輪照常辨識、翻譯。
            self._monitor_schedule_next()
            return
        try:
            if request["mode"] == "fullscreen":
                capture = self.screenshot_service.capture_fullscreen()
            elif request["mode"] == "window":
                capture = self.screenshot_service.capture_window(request["hwnd"])
                self._last_window_hwnd = request["hwnd"]
            elif request["mode"] == "region":
                capture = self.screenshot_service.capture_bbox(request["bbox"])
                capture.mode = "region"
            else:
                return
        except Exception as exc:
            log.warning("監控截圖失敗：%s", exc)
            self._monitor_errors += 1
            if self._monitor_errors >= int(self.config.get("monitor.max_consecutive_errors", 3)):
                self.stop_monitor(reason="監控截圖連續失敗，已停止")
            else:
                self._monitor_schedule_next()
            return
        capture.mode = request["mode"]
        self._last_capture = capture
        # 像素差异门控：画面逐像素完全没变就跳过整轮 OCR + 翻译管线，覆盖层保持，省资源。
        # 例外：有一組新文字正在等第二次確認時，同一格畫面也要再辨識一次，才能放行。
        self._configure_stabilizer()
        frame_hash = frame_fingerprint(capture.image)
        pending = self._monitor_stabilizer is not None and self._monitor_stabilizer.has_pending
        settled = bool(self._monitor_signature) or self._monitor_idle_hash == frame_hash
        if self._monitor_frame_hash and self._monitor_frame_hash == frame_hash and settled and not pending:
            self._monitor_schedule_next()
            return
        self._monitor_frame_hash = frame_hash
        # 把上一轮译文写回缓存：内容没变过的文本块直接命中缓存，几乎瞬时返回，
        # 只有新出现/变化的文字才真正走网络——这就是“特别快”的来源。
        if (
            self.config.get("monitor.skip_unchanged", True)
            and self._monitor_known
            and self.cache is not None
            and self.translator is not None
        ):
            source = self.translator.cache_namespace(str(self.config.get("translation.source_language", "auto")))
            target = str(self.config.get("translation.target_language", "zh"))
            for text, translated in self._monitor_known.items():
                try:
                    self.cache.set(source, target, text, translated)
                except Exception:
                    pass
        self._pipeline_succeeded = False
        self._pipeline_error = ""
        self._start_pipeline(capture)

    def _refresh_monitor(self) -> None:
        """「重新整理」：監控中就忘掉目前顯示的內容，立刻重新辨識、翻譯、顯示。"""
        if not self._monitor_active:
            return
        if self._monitor_stabilizer is not None:
            self._monitor_stabilizer.reset()
        self._monitor_signature = ""
        self._monitor_frame_hash = ""
        self._monitor_idle_hash = ""
        self.set_status(
            "翻譯模型載入中，完成後自動重新整理" if self._translator_warming() else "重新整理中…"
        )
        self._monitor_timer.stop()
        self._monitor_timer.start(0)

    def _configure_stabilizer(self) -> None:
        """依設定決定要連續辨識到相同結果幾次才顯示（1 = 快速、2 = 穩定，使用者可自訂）。"""
        if self._monitor_stabilizer is None:
            return
        try:
            rounds = int(self.config.get("monitor.confirm_rounds", RECOMMENDED_FAST_ROUNDS))
            max_wait = float(self.config.get("monitor.max_wait_seconds", RECOMMENDED_MAX_WAIT))
        except (TypeError, ValueError):
            rounds, max_wait = RECOMMENDED_FAST_ROUNDS, RECOMMENDED_MAX_WAIT
        self._monitor_stabilizer.configure(max(MIN_ROUNDS, min(MAX_ROUNDS, rounds)), max_wait)

    def _monitor_interval_ms(self) -> int:
        """設定的辨識間隔（毫秒），壞掉的值用推薦值，並限制在允許範圍內。"""
        try:
            interval = int(self.config.get("monitor.interval_ms", RECOMMENDED_INTERVAL_MS))
        except (TypeError, ValueError):
            interval = RECOMMENDED_INTERVAL_MS
        return max(MIN_INTERVAL_MS, min(MAX_INTERVAL_MS, interval))

    def _monitor_schedule_next(self) -> None:
        if not self._monitor_active:
            return
        self._monitor_timer.stop()
        self._monitor_timer.start(self._monitor_interval_ms())

    def _forget_displayed_translations(self) -> None:
        """忘掉畫面上已顯示的譯文與上下文，下一輪重新辨識、重新翻譯。

        換翻譯模型或清除快取之後必須做：監控每一輪會把「已顯示的譯文」寫回快取，
        不忘掉的話，舊模型的譯文會被寫進新模型的快取項目，看起來像新模型翻的。
        """
        self._monitor_known = {}
        self._monitor_signature = ""
        self._monitor_frame_hash = ""
        if self._monitor_stabilizer is not None:
            self._monitor_stabilizer.reset()
        self._reset_translation_context()

    def clear_translation_cache(self) -> int:
        """設定畫面的「清除翻譯快取」：刪除所有已儲存的翻譯，回傳刪了幾句。"""
        removed = self.cache.clear() if self.cache is not None else 0
        self._forget_displayed_translations()
        self.set_status(f"已清除翻譯快取（{removed} 句）")
        return removed

    def translation_cache_count(self) -> int:
        return self.cache.count() if self.cache is not None else 0

    def _arm_no_text_timer(self) -> None:
        """這一輪 OCR 沒抓到任何文字：開始計時，連續這麼久都沒有字幕就把譯文關掉。
        已經在計時（連續好幾輪都沒字）就不重新開始，才是「連續」一段時間；0 = 不自動關閉。"""
        try:
            seconds = float(self.config.get("monitor.hide_after_no_text_seconds", RECOMMENDED_NO_TEXT_SECONDS))
        except (TypeError, ValueError):
            seconds = RECOMMENDED_NO_TEXT_SECONDS
        seconds = max(0.0, min(MAX_NO_TEXT_SECONDS, seconds))
        if seconds <= 0 or not self._monitor_active or self._no_text_timer.isActive():
            return
        self._transcript_text_gone()  # 字幕大概從這一刻就沒了；計時到了才真的關譯文、結束這一筆
        self._no_text_timer.start(int(round(seconds * 1000)))

    def _cancel_no_text_timer(self) -> None:
        """OCR 又抓到字了（不論是新字幕、相同字幕或還在等穩定）：不用關譯文。"""
        self._no_text_timer.stop()
        self._transcript_text_back()

    def _close_translation_after_no_text(self) -> None:
        if not self._monitor_active or self._shutting_down:
            return
        self._transcript_end("gone")
        if self.overlay_manager is not None:
            self.overlay_manager.clear_all()
            self._sync_overlay_state(False)
        # 忘掉目前顯示的內容：同一句字幕之後再出現，才會重新顯示（而不是被當成「已顯示過」）
        if self._monitor_stabilizer is not None:
            self._monitor_stabilizer.reset()
        self._monitor_signature = ""
        self._monitor_known = {}
        self.set_status("監控中：畫面沒有字幕，已關閉譯文")

    def _reschedule_monitor_after_settings(self) -> None:
        """設定改了間隔：正在等下一輪的計時器立刻照新間隔重算（不然要等完舊間隔才生效）。
        正在辨識／翻譯的那一輪不動，它結束時自然會用新間隔排下一輪。"""
        if self._monitor_active and self._monitor_timer.isActive():
            self._monitor_schedule_next()

    def _create_monitor_region_selector(self) -> None:
        self._dispose_monitor_region()
        if self.monitor_map is None or self._monitor_request is None:
            return
        from ui.monitor_region import MonitorRegionSelector

        bbox = self._monitor_request["bbox"]
        accent = str(self.config.get("capture.select_border_color", "#2878E8"))
        selector = MonitorRegionSelector(self.monitor_map, bbox, accent)
        selector.region_changed.connect(self._on_monitor_region_changed)
        selector.region_committed.connect(self._on_monitor_region_committed)
        selector.stop_requested.connect(self.stop_monitor)
        self._monitor_region_selector = selector
        selector.show()
        selector.raise_()

    def _dispose_monitor_region(self) -> None:
        if self._monitor_region_selector is not None:
            try:
                self._monitor_region_selector.close()
            except Exception:
                pass
            self._monitor_region_selector = None

    def _on_monitor_region_changed(self, bbox) -> None:
        if self._monitor_request is not None:
            self._monitor_request["bbox"] = tuple(bbox)
        # 譯文面板跟著框走（框內容沒變時不會重新翻譯，也就不會自己更新位置）
        if self.overlay_manager is not None:
            self.overlay_manager.follow_region(tuple(bbox))

    def _on_monitor_region_committed(self, bbox) -> None:
        if self._monitor_request is not None:
            self._monitor_request["bbox"] = tuple(bbox)
        # 拖拽结束尽快重刷一次；若当前正在跑轮询则不抢占，避免阻塞 UI 线程，
        # 下一轮周期自会采用新位置。
        if self._monitor_active and self.worker is None:
            self._monitor_timer.stop()
            self._monitor_timer.start(80)
