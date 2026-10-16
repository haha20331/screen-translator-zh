"""设置、外观、OCR/翻译器解析、快捷键与开机自启。"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer

# 通过 app.application 命名空间解析 create_ocr_engine，使测试对该符号的 monkeypatch 仍然生效。
from app import application as _application
from app.hotkeys import HotkeyError, HotkeyManager
from app.logger import get_logger
from app.runtime import main_script, sys_executable
from services.ocr.base import OCRUnavailableError
from services.translation.base import Translator
from services.translation.factory import create_translator
from ui.appearance import resolve_tokens
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style
from ui.tray_icon import build_icon

log = get_logger("application")


class SettingsMixin:
    """设置、外观、OCR/翻译器解析、快捷键与开机自启。"""

    def _capture_accent(self) -> str:
        """Resolve the capture accent from the versioned appearance settings."""
        return resolve_tokens(self.config.section("appearance")).accent

    def _refresh_appearance(self) -> None:
        tokens = apply_style(self.qt_app, self.config.section("appearance"))
        self.qt_app.setWindowIcon(build_icon(tokens))
        self.floating_status.refresh_appearance()
        if self.selection is not None:
            self.selection.set_accent(tokens.accent)
        if self.window_highlight is not None:
            self.window_highlight.set_accent(tokens.accent)
        if self.tray is not None:
            self.tray.refresh_appearance()
        if self.window is not None:
            self.window.refresh_appearance()
        if self._settings_page is not None:
            self._settings_page.refresh_appearance()

    def _on_system_color_scheme_changed(self, _scheme) -> None:
        if self.config.get("appearance.palette", "warm_paper") == "system":
            self._refresh_appearance()

    def open_settings(self) -> None:
        if self.window is None or self._busy:
            return
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()
        if self._settings_page is not None:
            if self._settings_page.close_intent == 1:
                self._reopen_settings_after_close = True
                return
            if self._settings_page.exit_pending:
                self._settings_page.cancel_exit()
            self.window.show_settings_page(self._settings_page)
            return
        # A direct or queued open has now reached the only point where it can
        # succeed. Consume the request here, never when merely scheduling it.
        self._reopen_settings_after_close = False
        dialog = SettingsDialog(
            self.config,
            parent=self.window,
            cache_actions=(self.translation_cache_count, self.clear_translation_cache),
            speed_test_guard=self._speed_test_blocker,
            speed_test_hooks=(self._release_translation_model, self._reload_translation_model),
        )
        dialog.setWindowFlags(Qt.WindowType.Widget)
        dialog.setMinimumSize(0, 0)
        self._settings_page = dialog
        dialog.accepted.connect(self._apply_settings)
        dialog.exit_requested.connect(
            lambda result, page=dialog: self._begin_settings_exit(page, result)
        )
        dialog.finished.connect(
            lambda result, page=dialog: self._close_settings_page(page, result)
        )
        self.window.show_settings_page(dialog)

    def _speed_test_blocker(self) -> str:
        """設定頁的「測試速度」能不能開始：回傳空字串 = 可以；否則是不能測的原因。"""
        if self._monitor_active:
            return "即時監控進行中，翻譯會和測試搶 CPU，量出來的數字不準。請先停止即時監控再測。"
        if self._busy:
            return "正在處理翻譯，請等它結束再測。"
        return ""

    def _release_translation_model(self) -> None:
        """速度測試開始前：把應用程式自己載入的離線模型放掉，騰出記憶體給測試載入各個模型（下次翻譯會自動再載入）。"""
        close = getattr(self.translator, "close", None)
        if callable(close):
            close()

    def _reload_translation_model(self) -> None:
        """速度測試結束後：背景把原本的模型載回來並預熱，下一句字幕才不必等。"""
        if not self._shutting_down:
            note = self._warmup()
            if note:
                self.set_status(note)

    def _begin_settings_exit(self, page: SettingsDialog, result: int) -> None:
        if self._settings_page is not page:
            page.complete_exit(result)
            return
        if self.window is None:
            page.complete_exit(result)
            return
        self.window.begin_settings_exit(
            page,
            lambda current=page, value=result: current.complete_exit(value),
        )

    def _close_settings_page(self, page: SettingsDialog, _result: int = 0) -> None:
        if self._settings_page is page:
            self._settings_page = None
        if self.window is not None:
            self.window.remove_settings_page(page)
        if self._reopen_settings_after_close:
            QTimer.singleShot(0, self._try_reopen_settings)

    def _try_reopen_settings(self) -> None:
        """Honor a queued reopen only when capture and teardown are both idle."""
        if (
            not self._reopen_settings_after_close
            or self._busy
            or self.window is None
            or self._settings_page is not None
        ):
            return
        self.open_settings()

    def _apply_settings(self) -> None:
        self._refresh_appearance()
        self.refresh_monitor_map()
        if self.cache is not None:
            self.cache.set_ttl(
                self.config.get("translation.cache_ttl_days", 30),
                self.config.get("translation.cache_max_entries", 2000),
            )
        self.ocr_engine = self._create_configured_ocr_engine()
        self._vertical_ocr_engine = None  # OCR 設定可能改了，直向文字用的備用引擎下次重建
        cleared = self._rebuild_translator()
        self._forget_displayed_translations()  # 可能換了模型：畫面上舊模型的譯文不能被寫進新模型的快取
        loading = self._warmup() or ""  # 換了模型就馬上在背景載入；載入中的提示併進下面的狀態列文字
        if self.overlay_manager is not None:
            self.overlay_manager.apply_style()
        try:
            self._apply_hotkeys()
        except HotkeyError as exc:
            self.show_error("快捷鍵", str(exc))
        self._apply_autostart()
        self._reschedule_monitor_after_settings()
        self._apply_transcript_settings()
        if self.window is not None:
            self.window.reload_values()
        self.set_status(
            "設定已儲存"
            + (f"（換了翻譯模型，已清空翻譯快取 {cleared} 句）" if cleared else "")
            + (f"；{loading}" if loading else "")
        )

    def _translator_label(self) -> str:
        """目前翻譯器用的模型（``offline/<模型名稱>``、``openai/<model>``…）；沒有翻譯器就是空字串。"""
        translator = getattr(self, "translator", None)
        try:
            return str(translator.model_label()) if translator is not None else ""
        except Exception:
            return ""

    def _rebuild_translator(self) -> int:
        """重建翻譯器；**換了模型（或翻譯服務）就把翻譯快取清空**，回傳清掉幾句（沒換或沒有快取回傳 0）。

        快取的鍵本來就帶著模型名稱，不同模型的譯文不會互相蓋過；清空是為了換模型比較時，
        不會看到、也不會混進上一個模型留下的譯文。設定沒變時 ``_build_translator`` 會重用同一個翻譯器，不會清。"""
        previous = self._translator_label()
        self.translator = self._build_translator()
        return self._clear_cache_if_model_changed(previous)

    def _clear_cache_if_model_changed(self, previous: str) -> int:
        current = self._translator_label()
        self._remember_cache_model(current)
        if not previous or not current or previous == current or self.cache is None:
            return 0
        removed = self.cache.clear()
        self._forget_displayed_translations()
        log.info("翻譯模型從 %s 換成 %s，已清空翻譯快取（%d 句）", previous, current, removed)
        return removed

    def _remember_cache_model(self, label: str) -> None:
        """記下快取是用哪個模型存的（存進設定檔）：下次啟動時模型變了（例如下載了 3B、自動選擇的結果不同）也能清空。"""
        if not label or self.config.get("translation.cache_model", "") == label:
            return
        self.config.set("translation.cache_model", label)
        try:
            self.config.save()
        except OSError as exc:
            log.warning("無法儲存快取所屬的模型：%s", exc)

    def reconcile_cache_with_model(self) -> int:
        """啟動時：設定檔記的快取模型和這次的不同，就清空快取。第一次（沒有紀錄）只記下來，不清。"""
        stored = str(self.config.get("translation.cache_model", "") or "")
        return self._clear_cache_if_model_changed(stored)

    def _build_translator(self):
        """建立翻譯器；離線翻譯的設定沒變時重用現有實例，避免重複載入數 GB 的模型。"""
        service = str(self.config.get("translation.service", "offline"))
        section = self.config.section("translation")
        current = getattr(self, "translator", None)
        if (
            service == "offline"
            and current is not None
            and getattr(current, "name", "") == service
            and current.config.get(service) == section.get(service)
        ):
            current.cache = self.cache
            return current
        return create_translator(service, section, cache=self.cache)

    def _create_configured_ocr_engine(self):
        """Resolve the requested backend without making a light build unbootable."""
        requested = str(self.config.get("ocr.engine", "windows") or "windows")
        candidates = list(dict.fromkeys((requested, "rapidocr", "windows", "none")))
        failures: list[str] = []
        for candidate in candidates:
            try:
                engine = _application.create_ocr_engine(
                    candidate, self.config.section("ocr"), self.config
                )
            except (OCRUnavailableError, OSError, RuntimeError, ValueError) as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            self._active_ocr_engine_name = candidate
            if candidate != requested:
                log.warning(
                    "OCR %s unavailable; using %s (%s)",
                    requested,
                    candidate,
                    "; ".join(failures),
                )
                # Keep the visible selectors and the engine actually executing
                # in agreement.
                self.config.set("ocr.engine", candidate)
                try:
                    self.config.save()
                except OSError as exc:
                    log.warning("無法儲存 OCR 回退選擇：%s", exc)
            return engine
        raise OCRUnavailableError("沒有可用的 OCR 引擎：" + "; ".join(failures))

    def _apply_hotkeys(self) -> None:
        try:
            self.hotkey_manager.apply(self.config.hotkeys())
        except HotkeyError as exc:
            log.exception("快捷鍵註冊失敗")
            self.set_status(f"快捷鍵註冊失敗：{exc}")

    def _apply_autostart(self) -> None:
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE,
            )
            if self.config.get("general.startup_with_system", False):
                exe = sys_executable()
                winreg.SetValueEx(
                    key, "ScreenTranslator", 0, winreg.REG_SZ, f'"{exe}" "{main_script()}"'
                )
            else:
                try:
                    winreg.DeleteValue(key, "ScreenTranslator")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        except Exception as exc:
            log.warning("開機啟動設定失敗：%s", exc)
