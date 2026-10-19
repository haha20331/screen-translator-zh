"""配置管理：JSON 读写、默认值合并、原子保存、密钥脱敏。"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from app.logger import app_data_dir
from utils.language_utils import TARGET_LANGUAGE

DEFAULTS: dict[str, Any] = {
    "general": {
        "startup_with_system": False,
        "minimize_to_tray": False,  # 關閉主視窗時：結束程式（False，預設），或收到系統匣繼續在背景執行（True）
        "minimize_hides_to_tray": False,  # 最小化主視窗時：收到系統匣、不佔工作列（True），或照一般視窗縮到工作列（False）
        "save_history": False,
        "history_dir": "",
    },
    "appearance": {
        "schema_version": 1,
        "palette": "warm_paper",
        "accent": "#2878E8",
        "motion_profile": "flow",
        "density": "balanced",
        "surface": "layered",
        "reduce_motion": False,
    },
    "capture": {
        "select_mask_opacity": 84,
        "select_border_color": "#2878E8",
    },
    "ocr": {
        "engine": "rapidocr",
        "lang": "auto",
        "language_mode_version": 2,
        # auto 模式最多串行尝试几个语言包（每个都是一次整图 OCR）
        "auto_max_languages": 3,
        # OCR 前自适应缩放：小图放大到 min_side，超大图缩到 max_side
        "ocr_min_side": 2200,
        "ocr_max_side": 4096,
        "max_upscale": 3.0,
        "min_confidence": 0.6,
        "merge_y_tolerance_ratio": 0.3,
        "merge_x_gap_ratio": 0.8,
        # 相鄰行合併成同一段翻譯（字幕、對白需要完整上下文）。上游只合併寬度 < 擷取寬度 36% 的窄標籤；
        # 這裡放寬到整個擷取範圍，並把置中對齊的相鄰行也視為同一段。
        "merge_wrapped_max_width_ratio": 1.0,
        "merge_centered_lines": True,
        # 雙語字幕：疊在日文字幕旁、純漢字的那行（字幕檔附的中文譯文）不再當日文翻譯
        "skip_companion_translation": True,
        "rapidocr": {
            # 偵測階段的長邊上限（像素）；越大越能抓小字，但越慢
            "limit_side_len": 960,
            # ONNX Runtime 執行緒數；0 = 自動（最多 4）
            "threads": 0,
            # 影像短邊小於此像素時先放大再辨識（0 = 關閉），最多放大 max_upscale 倍
            "upscale_below": 240,
            "max_upscale": 2.0,
        },
    },
    "translation": {
        "service": "offline",  # 只有離線翻譯；舊設定檔裡的其他值會被改回來（見 _purge_online_services）
        "source_language": "auto",
        "target_language": "zh-hant",
        "keep_original": False,
        # 離線翻譯（LLM）的輔助功能，即時生效：
        # 上下文：記住最近 context_size 句已翻好的句子，當成前文協助翻譯這一句（超過 context_ttl_seconds 的不用）。
        # 預設開：盲測與對照（PROMPT_TUNING.md）顯示關掉後 chrF 四組都下降 2.2～3.5 點；只有離線翻譯有效
        "use_context": True,
        "context_size": 2,
        "context_ttl_seconds": 120,
        # 歌詞模式：提示模型這是歌詞，用較自然、有詩意的方式意譯
        "lyrics_mode": False,
        # 只有擬聲詞（音效、笑聲、狀聲詞）的句子不翻譯也不顯示（由提示詞判斷，再用譯文與原文的長相把關）
        "skip_sounds": True,
        # 譯文不是中文（殘留日文假名、不該略過的句子被回報略過）時，改用一般提示重翻一次（預設開）；
        # 關閉就只進模型一次，翻出什麼就輸出什麼，每句少一次模型運算
        "retry_non_chinese": True,
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_delay_seconds": 1.0,
        "request_interval_seconds": 0.0,
        # 離線翻譯（Qwen2.5-Instruct，llama.cpp / GGUF）；model_dir 留空 = 自動尋找 models/translation/
        "offline": {
            # 模型名稱（見 services/translation/offline_translator.OFFLINE_MODELS，由小到大 0.5B／1.5B／3B／7B）。
            # auto：有獨立顯示卡（可用顯示卡記憶體至少 5.5 GB）且放了 7B 就用 7B，否則用 3B Q4_K_M，沒有下載 3B 就用 1.5B Q4_K_M（內顯與 CPU 共用記憶體，不算）
            "model_name": "auto",
            "model_dir": "",
            # 運算裝置（設定頁「運算裝置」）：auto = 有可用的 GPU 就用（沒有 GPU 版的 llama.cpp 或載入失敗就用 CPU）、
            # cpu = 不使用 GPU，或指定的那顆 GPU 的名稱（例如 "NVIDIA GeForce RTX 3060"；同型號的第二顆是 "… #2"）
            "gpu_device": "auto",
            # 用 GPU 時交給它的層數：-1 = 全部。（舊版用 0 表示只用 CPU，現在改成上面選 cpu；載入舊設定檔時會自動轉換）
            "gpu_layers": -1,
            # GPU 載入失敗（顯示卡記憶體不足）時是否改用 CPU。預設是；只有速度測試在電腦記憶體也不夠時才會關掉
            "cpu_fallback": True,
            # 0 = 自動（最多 6；Intel 混合核心 CPU 上執行緒開太多反而更慢）
            "threads": 0,
            "max_new_tokens": 160,
            # 只有漢字、沒有假名的文字視為：ja（日文）或 空字串（不翻譯）
            "han_only_as": "ja",
        },
        "cache_ttl_days": 30,
        "cache_max_entries": 2000,
        # 翻譯快取是用哪個模型存的（程式自己記）：換了模型就清空快取
        "cache_model": "",
    },
    "overlay": {
        "background_color": "#000000",
        "auto_background": True,
        "background_alpha": 160,
        "font_family": "",
        "font_size": 18,
        "text_color": "#FFFFFF",
        "use_auto_text_color": True,
        "padding": 4,
        "border_radius": 4,
        "border_color": "#FFFFFF",
        "border_alpha": 60,
        "show_border": True,
        "min_font_size": 8,
        # below：框選區域時，譯文放在框的下方，原文保持可見；in_place：譯文蓋在原文位置上取代原文
        "placement": "below",
        # 譯文面板（below 模式）的底色不透明度 0–255
        "panel_alpha": 215,
        # 字幕翻得很快、新的一句已經出來時，舊的一句譯文至少還要留幾秒（疊在上面，新的接在它下面）；0 = 立刻換掉
        "min_hold_seconds": 0.5,
    },
    "monitor": {
        "enabled": True,
        # 框選／全螢幕／視窗翻譯完成後，自動接著進入即時監控（預設開啟；監控中隨時可關，也能直接開始新的擷取）
        "auto_start": True,
        # 相鄰兩輪監控的間隔（毫秒），設定畫面可輸入到 0.01 秒（最小 100）。畫面靜止時這基本就是「刷新延遲」
        "interval_ms": 100,
        # 預設值版本：檔案裡沒有這個標記的舊設定，才會把舊預設的間隔（1200／400）換成新預設（見 _normalize_monitor）
        "interval_default_version": 2,
        # 连续出错超过这个次数就自动停掉监控，避免一直空转
        "max_consecutive_errors": 3,
        # 把上一轮译文写回缓存：内容没变过的文本块直接命中缓存，几乎瞬时
        "skip_unchanged": True,
        # 確認次數：辨識到的新字幕要「連續幾次辨識結果相同」才翻譯並顯示。
        # 1 = 快速（系統推薦，辨識到就顯示，但字幕淡入淡出時可能先閃出殘缺的句子）；
        # 2 = 穩定（系統推薦，過渡畫面的殘缺句子不會被翻譯，但換字幕約慢 0.5～1 秒）。
        # 與已顯示的相同時，不論次數都不再翻譯。
        "confirm_rounds": 1,
        # 新文字一直在變（辨識抖動）時，最久等幾秒就放行最新的一組（推薦 2.5 秒）
        "max_wait_seconds": 2.5,
        # 監控中 OCR 連續這麼多秒都抓不到字幕，就自動關閉譯文（推薦 1 秒，0 = 不自動關閉）
        "hide_after_no_text_seconds": 1.0,
        # 停止即時監控超過這麼多秒，就把翻譯模型從記憶體放掉（3B 約 2.8 GB）；下次擷取或監控時自動載回。0 = 不釋放
        "release_model_after_seconds": 10,
        # 監視框的移動方式：grips = 拖左上、右下兩個移動符號（預設，框的中間穿透，點得到後面的畫面）；
        # center = 整個框的中央都能拖，但中間不再穿透。設定 → 即時監控 可改，框在畫面上時馬上生效
        "drag_mode": "grips",
    },
    # 文字輸出：每句字幕（開始、結束時間、原文、譯文、四角點、模型）附加寫進一份 txt，一天一檔。
    # 字幕結束（被下一句取代、消失、停止監控、離開程式）時才寫，寫下的是最完整的那一版。
    "output": {
        "enabled": True,
        # 留空 = %LOCALAPPDATA%\ScreenTranslator\output\；也可以填其他資料夾
        "directory": "",
        # 顯示不到這麼多秒就被下一句取代的字幕（多半是淡入淡出的殘句），狀態欄加上「短暫」；不會丟掉
        "min_seconds": 0.3,
    },
    "hotkeys": {
        "capture_region": "ctrl+shift+a",
        "capture_fullscreen": "ctrl+shift+f",
        "capture_window": "ctrl+shift+w",
        "toggle_overlay": "ctrl+shift+h",
        "refresh": "ctrl+shift+r",
        "monitor_toggle": "ctrl+shift+m",
    },
}

# 這個程式只有離線翻譯，沒有任何線上翻譯服務，也不再儲存任何服務的憑據。舊版設定檔裡殘留的線上服務設定與憑據
# （API Key、App Secret…）在載入時會被刪掉，並立刻把清乾淨的設定存回檔案。
_ONLINE_TRANSLATION_KEYS = (
    "openai",
    "deepl",
    "google",
    "baidu",
    "youdao",
    "azure",
    "mymemory",
    "auto_select_service",
    "mock_mode",
    "glossary",
    "google_free_max_workers",
    "google_free_interval",
    "google_free_partial_retries",
    "google_free_retry_backoff",
    "google_free_fallback_to_mymemory",
)
OFFLINE_SERVICE = "offline"


def _purge_online_services(translation: Any) -> bool:
    """從載入的 ``translation`` 區段刪掉線上服務的設定與憑據；有刪東西、或服務不是離線翻譯時回傳 True。"""
    if not isinstance(translation, dict):
        return False
    purged = False
    for key in _ONLINE_TRANSLATION_KEYS:
        if key in translation:
            del translation[key]
            purged = True
    if translation.get("service", OFFLINE_SERVICE) != OFFLINE_SERVICE:
        purged = True
    translation["service"] = OFFLINE_SERVICE
    return purged


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_appearance(data: dict[str, Any]) -> None:
    defaults = DEFAULTS["appearance"]
    appearance = data.get("appearance")
    if not isinstance(appearance, dict):
        data["appearance"] = copy.deepcopy(defaults)
        return

    allowed = {
        "palette": {"warm_paper", "mist", "midnight", "system"},
        "motion_profile": {"flow", "calm", "minimal"},
        "density": {"spacious", "balanced", "compact"},
        "surface": {"clean", "layered"},
    }
    appearance["schema_version"] = 1
    for key, values in allowed.items():
        if appearance.get(key) not in values:
            appearance[key] = defaults[key]

    accent = str(appearance.get("accent", defaults["accent"])).strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", accent):
        appearance["accent"] = accent.upper()
    else:
        appearance["accent"] = defaults["accent"]
    if not isinstance(appearance.get("reduce_motion"), bool):
        appearance["reduce_motion"] = defaults["reduce_motion"]


def _normalize_translation(data: dict[str, Any], loaded: dict[str, Any] | None = None) -> None:
    """Translation-section migrations: fixed target language, offline-only service, legacy offline settings."""
    translation = data.setdefault("translation", {})
    translation["target_language"] = TARGET_LANGUAGE  # 目標語言只有繁體中文：舊設定檔裡的其他值（簡體、英文…）改回來
    translation["service"] = OFFLINE_SERVICE
    # CTranslate2 引擎與 NLLB「快速翻譯」已移除：舊設定改成 llama.cpp 的對應設定（沿用 auto 的預設模型）
    translation.pop("offline_fast", None)
    offline = translation.setdefault("offline", {})
    for obsolete in ("engine", "device", "compute_type", "beam_size"):
        offline.pop(obsolete, None)
    # 舊版用 gpu_layers = 0 表示只用 CPU：檔案裡沒有 gpu_device 時轉成 gpu_device = "cpu"（之後 gpu_layers 只管層數）
    loaded_offline = ((loaded or {}).get("translation") or {}).get("offline")
    if isinstance(loaded_offline, dict) and "gpu_device" not in loaded_offline and loaded_offline.get("gpu_layers") == 0:
        offline["gpu_device"] = "cpu"
        offline["gpu_layers"] = DEFAULTS["translation"]["offline"]["gpu_layers"]
    if not isinstance(offline.get("gpu_device"), str) or not offline["gpu_device"].strip():
        offline["gpu_device"] = DEFAULTS["translation"]["offline"]["gpu_device"]
    name = str(offline.get("model_name") or "auto")
    lowered = name.lower()
    # 舊版 CTranslate2 時期的模型名稱（qwen2.5-…-ct2-int8、nllb-…）已經沒有意義；其他名稱（包括使用者放在模型資料夾裡的自訂模型）照舊
    if name != "auto" and "gguf" not in lowered and any(tag in lowered for tag in ("ct2", "nllb", "opus-mt")):
        offline["model_name"] = "auto"


def _normalize_overlay(data: dict[str, Any]) -> None:
    overlay = data.setdefault("overlay", {})
    try:
        hold = float(overlay.get("min_hold_seconds", DEFAULTS["overlay"]["min_hold_seconds"]))
    except (TypeError, ValueError):
        hold = DEFAULTS["overlay"]["min_hold_seconds"]
    overlay["min_hold_seconds"] = max(0.0, min(10.0, hold))
    if overlay.get("placement") not in {"below", "in_place"}:
        overlay["placement"] = DEFAULTS["overlay"]["placement"]


_LEGACY_MONITOR_INTERVAL_MS = 1200  # 上游的預設
_PREVIOUS_MONITOR_INTERVAL_MS = 400  # 前一版的預設


def _normalize_monitor_stability(data: dict[str, Any], loaded: dict[str, Any]) -> None:
    """確認次數與最久等待要是合理範圍內的數字；舊版的「快速／穩定」選項換算成次數。"""
    monitor = data.setdefault("monitor", {})
    loaded_monitor = loaded.get("monitor") if isinstance(loaded.get("monitor"), dict) else {}
    legacy = monitor.pop("stability", None)
    monitor.pop("stable_rounds", None)
    if "confirm_rounds" not in loaded_monitor and legacy == "stable":
        monitor["confirm_rounds"] = 2
    defaults = DEFAULTS["monitor"]
    try:
        rounds = int(monitor.get("confirm_rounds", defaults["confirm_rounds"]))
    except (TypeError, ValueError):
        rounds = defaults["confirm_rounds"]
    try:
        wait = float(monitor.get("max_wait_seconds", defaults["max_wait_seconds"]))
    except (TypeError, ValueError):
        wait = defaults["max_wait_seconds"]
    monitor["confirm_rounds"] = max(1, min(10, rounds))
    monitor["max_wait_seconds"] = max(0.5, min(30.0, wait))


def _normalize_monitor(data: dict[str, Any], loaded: dict[str, Any]) -> None:
    """辨識間隔：舊的預設值換成新預設（100 ms），使用者自己選的值保留；其餘限制在合理範圍（100～10000 ms）。

    舊預設有兩個：上游的 1200 ms 與前一版的 400 ms。檔案裡還沒有 ``interval_default_version`` 標記時，
    這兩個值視為「沒改過」換掉；換過（標記存進檔案）之後，使用者選的任何值（包括剛好 0.4 或 1.2 秒）都照用。
    例外：前一版的設定畫面存過的 1200（有 ``interval_migrated`` 標記）一定是使用者自己選的，保留。
    畫面沒變時會直接跳過 OCR 與翻譯，所以縮短間隔不會增加無謂的運算。
    """
    monitor = data.setdefault("monitor", {})
    loaded_monitor = loaded.get("monitor") if isinstance(loaded.get("monitor"), dict) else {}
    default = DEFAULTS["monitor"]["interval_ms"]
    if not loaded_monitor.get("interval_default_version"):
        stored = loaded_monitor.get("interval_ms")
        chosen_in_ui = bool(loaded_monitor.get("interval_migrated"))
        if stored == _PREVIOUS_MONITOR_INTERVAL_MS or (stored == _LEGACY_MONITOR_INTERVAL_MS and not chosen_in_ui):
            monitor["interval_ms"] = default
    monitor["interval_default_version"] = 2
    monitor.pop("interval_migrated", None)
    try:
        interval = int(monitor.get("interval_ms", default))
    except (TypeError, ValueError):
        interval = default
    monitor["interval_ms"] = max(100, min(10_000, interval))
    try:
        hide_after = float(monitor.get("hide_after_no_text_seconds", DEFAULTS["monitor"]["hide_after_no_text_seconds"]))
    except (TypeError, ValueError):
        hide_after = DEFAULTS["monitor"]["hide_after_no_text_seconds"]
    monitor["hide_after_no_text_seconds"] = max(0.0, min(30.0, hide_after))
    try:
        release_after = float(monitor.get("release_model_after_seconds", DEFAULTS["monitor"]["release_model_after_seconds"]))
    except (TypeError, ValueError):
        release_after = DEFAULTS["monitor"]["release_model_after_seconds"]
    monitor["release_model_after_seconds"] = max(0.0, min(3600.0, release_after))
    if monitor.get("drag_mode") not in ("grips", "center"):
        monitor["drag_mode"] = DEFAULTS["monitor"]["drag_mode"]


def _normalize_output(data: dict[str, Any]) -> None:
    defaults = DEFAULTS["output"]
    output = data.get("output")
    if not isinstance(output, dict):
        data["output"] = copy.deepcopy(defaults)
        return
    if not isinstance(output.get("enabled"), bool):
        output["enabled"] = defaults["enabled"]
    directory = output.get("directory", defaults["directory"])
    output["directory"] = directory.strip() if isinstance(directory, str) else defaults["directory"]
    try:
        seconds = float(output.get("min_seconds", defaults["min_seconds"]))
    except (TypeError, ValueError):
        seconds = defaults["min_seconds"]
    output["min_seconds"] = max(0.0, min(10.0, seconds))


def _normalize_ocr(data: dict[str, Any], loaded: dict[str, Any]) -> None:
    """Migrate the old Chinese-only OCR default to real automatic detection."""

    ocr = data.setdefault("ocr", {})
    loaded_ocr = loaded.get("ocr") if isinstance(loaded.get("ocr"), dict) else {}
    legacy_language_mode = "language_mode_version" not in loaded_ocr
    source_language = str(
        data.get("translation", {}).get("source_language", "auto") or "auto"
    ).lower()
    language = str(ocr.get("lang", "auto") or "auto").lower()

    # v0.1 used ``ch``/``zh`` as a hidden default even when the visible source
    # selector said "automatic".  That made Windows OCR parse English pages
    # with the Chinese recognizer and either return garbage or no text.
    if legacy_language_mode and source_language == "auto" and language in {"ch", "zh"}:
        language = "auto"
    elif language == "ch":
        language = "zh"

    allowed = {
        "auto", "zh", "zh-hant", "en", "ja", "ko", "fr", "de", "es",
        "ru", "pt", "it", "vi", "th", "ar",
    }
    ocr["lang"] = language if language in allowed else "auto"
    ocr["language_mode_version"] = 2


class AppConfig:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else self._default_path()
        self.data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load()

    @staticmethod
    def _default_path() -> Path:
        override = os.environ.get("SCREEN_TRANSLATOR_CONFIG")
        if override:
            return Path(override)
        return app_data_dir() / "config.json"

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8-sig") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                purged = _purge_online_services(loaded.get("translation"))
                _deep_merge(self.data, loaded)
                _normalize_appearance(self.data)
                _normalize_translation(self.data, loaded)
                _normalize_ocr(self.data, loaded)
                _normalize_monitor(self.data, loaded)
                _normalize_monitor_stability(self.data, loaded)
                _normalize_overlay(self.data)
                _normalize_output(self.data)
                # This legacy field is retained for compatibility; capture
                # feedback now reads appearance.accent.  Normalize the red
                # debug value so old files no longer carry misleading state.
                capture = self.data.setdefault("capture", {})
                capture["select_border_color"] = "#2878E8"
                if capture.get("select_mask_opacity") == 100:
                    capture["select_mask_opacity"] = 84
                if purged:  # 把清乾淨的設定存回檔案：舊版留下的線上服務憑據不會繼續留在磁碟上
                    self._save_after_purge()
        except (json.JSONDecodeError, OSError) as exc:
            backup = self.path.with_suffix(".json.bak")
            try:
                backup.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                pass
            from app.logger import get_logger

            get_logger("config").warning("配置檔案損壞，已備份到 %s：%s", backup, exc)

    def _save_after_purge(self) -> None:
        try:
            self.save()
        except OSError as exc:
            from app.logger import get_logger

            get_logger("config").warning("無法把清掉線上服務設定的結果存回 %s：%s", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def section(self, *names: str) -> dict:
        node: Any = self.data
        for name in names:
            node = node.get(name, {})
        return node if isinstance(node, dict) else {}

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        fixed = {"translation.target_language": TARGET_LANGUAGE, "translation.service": OFFLINE_SERVICE}
        node[parts[-1]] = fixed.get(dotted, value)

    def hotkeys(self) -> dict[str, str]:
        return dict(self.section("hotkeys"))

    def masked_snapshot(self) -> dict:
        """給診斷包用的設定副本。程式不再儲存任何服務的憑據，所以這裡沒有要遮的欄位（診斷包另外會遮掉像金鑰的欄位名稱）。"""
        return copy.deepcopy(self.data)

    def reset(self) -> None:
        self.data = copy.deepcopy(DEFAULTS)
