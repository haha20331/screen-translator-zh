"""配置管理：JSON 读写、默认值合并、原子保存、密钥脱敏。"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from app.logger import app_data_dir

DEFAULTS: dict[str, Any] = {
    "general": {
        "startup_with_system": False,
        "minimize_to_tray": True,
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
        "service": "offline",
        "auto_select_service": True,
        "mock_mode": "dictionary",
        "glossary": "",
        "source_language": "auto",
        "target_language": "zh-hant",
        "keep_original": False,
        # 離線翻譯・標準（LLM）的輔助功能，即時生效：
        # 上下文：記住最近 context_size 句已翻好的句子，當成前文協助翻譯這一句（超過 context_ttl_seconds 的不用）
        "use_context": False,
        "context_size": 2,
        "context_ttl_seconds": 120,
        # 歌詞模式：提示模型這是歌詞，用較自然、有詩意的方式意譯
        "lyrics_mode": False,
        "timeout_seconds": 30,
        "max_retries": 3,
        "retry_delay_seconds": 1.0,
        "request_interval_seconds": 0.0,
        # 離線翻譯（Qwen2.5-Instruct + CTranslate2）；model_dir 留空 = 自動尋找 models/translation/
        "offline": {
            # auto：有 NVIDIA GPU 時優先用 7B 模型（載入失敗退回 1.5B），沒有 GPU 用 CPU 的 1.5B
            "model_name": "auto",
            "model_dir": "",
            # auto：有 NVIDIA GPU（CUDA）就用 GPU，否則 CPU
            "device": "auto",
            "compute_type": "auto",
            # 0 = 自動（最多 6；Intel 混合核心 CPU 上超過 8 會明顯變慢）
            "threads": 0,
            "beam_size": 1,
            "max_new_tokens": 160,
            # 只有漢字、沒有假名的文字視為：ja（日文）或 空字串（不翻譯）
            "han_only_as": "ja",
        },
        # 離線快速翻譯（NLLB-200 600M，貪婪解碼）：近乎即時，品質較低
        "offline_fast": {
            "model_name": "nllb-200-distilled-600m-ct2-int8",
            "model_dir": "",
            "device": "auto",
            "compute_type": "auto",
            "threads": 0,
            "beam_size": 1,
            "han_only_as": "ja",
        },
        "google_free_max_workers": 6,
        "google_free_interval": 0.02,
        "google_free_partial_retries": 0,
        "google_free_retry_backoff": 0.35,
        "google_free_fallback_to_mymemory": True,
        "cache_ttl_days": 30,
        "cache_max_entries": 2000,
        "openai": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
        "deepl": {
            "api_key": "",
            "base_url": "https://api-free.deepl.com/v2",
            "formality": "default",
        },
        "google": {
            "api_key": "",
            "base_url": "https://translation.googleapis.com/language/translate/v2",
        },
        "baidu": {
            "appid": "",
            "api_key": "",
            "base_url": "https://fanyi-api.baidu.com/api/trans/vip/translate",
        },
        "youdao": {
            "app_key": "",
            "app_secret": "",
            "base_url": "https://openapi.youdao.com/api",
        },
        "azure": {
            "api_key": "",
            "region": "",
            "base_url": "https://api.cognitive.microsofttranslator.com",
        },
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
    },
    "monitor": {
        "enabled": True,
        # 相邻两轮监控的间隔（毫秒）。页面静止时这基本就是“刷新延迟”
        "interval_ms": 400,
        # 连续出错超过这个次数就自动停掉监控，避免一直空转
        "max_consecutive_errors": 3,
        # 把上一轮译文写回缓存：内容没变过的文本块直接命中缓存，几乎瞬时
        "skip_unchanged": True,
        # 字幕顯示方式。stable（預設）：同一組辨識結果連續 stable_rounds 輪相同才翻譯並顯示，
        # 過渡畫面（淡入淡出）的殘缺句子不會被翻譯；fast：辨識到就立刻翻譯顯示。
        # 與已顯示的相同時，兩種模式都不再翻譯。
        "stability": "stable",
        "stable_rounds": 2,
        # 新文字一直在變（辨識抖動）時，最久等幾秒就放行最新的一組
        "max_wait_seconds": 2.5,
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

_KEY_SECTIONS = (
    "translation.openai.api_key",
    "translation.deepl.api_key",
    "translation.google.api_key",
    "translation.baidu.appid",
    "translation.baidu.api_key",
    "translation.youdao.app_key",
    "translation.youdao.app_secret",
    "translation.azure.api_key",
)


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


def _normalize_translation(data: dict[str, Any]) -> None:
    """Clamp legacy anonymous-Google tuning that made large captures slower."""
    translation = data.setdefault("translation", {})
    try:
        workers = int(translation.get("google_free_max_workers", 6) or 6)
    except (TypeError, ValueError):
        workers = 6
    try:
        interval = float(translation.get("google_free_interval", 0.02) or 0.0)
    except (TypeError, ValueError):
        interval = 0.02
    translation["google_free_max_workers"] = max(1, min(8, workers))
    translation["google_free_interval"] = max(0.02, interval)
    # 旧版逐条重试会把 20 个失败尾项串行拖几十秒；失败尾巴现在直接批量兜底。
    translation["google_free_partial_retries"] = 0
    translation.setdefault("google_free_fallback_to_mymemory", True)


def _normalize_overlay(data: dict[str, Any]) -> None:
    overlay = data.setdefault("overlay", {})
    if overlay.get("placement") not in {"below", "in_place"}:
        overlay["placement"] = DEFAULTS["overlay"]["placement"]


_LEGACY_MONITOR_INTERVAL_MS = 1200


def _normalize_monitor_stability(data: dict[str, Any]) -> None:
    monitor = data.setdefault("monitor", {})
    if monitor.get("stability") not in {"stable", "fast"}:
        monitor["stability"] = DEFAULTS["monitor"]["stability"]


def _normalize_monitor(data: dict[str, Any], loaded: dict[str, Any]) -> None:
    """舊預設的輪詢間隔 1200 ms 太慢（每輪結束後固定多等 1.2 秒）；沒被使用者改過就換成新預設。

    畫面沒變時會直接跳過 OCR 與翻譯，所以縮短間隔不會增加無謂的運算。
    """
    loaded_monitor = loaded.get("monitor") if isinstance(loaded.get("monitor"), dict) else {}
    if loaded_monitor.get("interval_ms") == _LEGACY_MONITOR_INTERVAL_MS:
        data.setdefault("monitor", {})["interval_ms"] = DEFAULTS["monitor"]["interval_ms"]


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
            with open(self.path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                _deep_merge(self.data, loaded)
                _normalize_appearance(self.data)
                _normalize_translation(self.data)
                _normalize_ocr(self.data, loaded)
                _normalize_monitor(self.data, loaded)
                _normalize_monitor_stability(self.data)
                _normalize_overlay(self.data)
                # This legacy field is retained for compatibility; capture
                # feedback now reads appearance.accent.  Normalize the red
                # debug value so old files no longer carry misleading state.
                capture = self.data.setdefault("capture", {})
                capture["select_border_color"] = "#2878E8"
                if capture.get("select_mask_opacity") == 100:
                    capture["select_mask_opacity"] = 84
        except (json.JSONDecodeError, OSError) as exc:
            backup = self.path.with_suffix(".json.bak")
            try:
                backup.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                pass
            from app.logger import get_logger

            get_logger("config").warning("配置檔案損壞，已備份到 %s：%s", backup, exc)

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
        node[parts[-1]] = value

    def hotkeys(self) -> dict[str, str]:
        return dict(self.section("hotkeys"))

    def masked_snapshot(self) -> dict:
        snapshot = copy.deepcopy(self.data)
        for key in _KEY_SECTIONS:
            parts = key.split(".")
            node = snapshot
            for part in parts[:-1]:
                node = node.get(part, {})
            if parts[-1] in node and node[parts[-1]]:
                node[parts[-1]] = "***"
        return snapshot

    def api_key(self, service: str) -> str:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "deepl": "DEEPL_API_KEY",
            "google": "GOOGLE_TRANSLATE_API_KEY",
            "baidu": "BAIDU_APP_KEY",
            "youdao": "YOUDAO_APP_KEY",
            "azure": "AZURE_TRANSLATOR_KEY",
        }
        env_value = os.environ.get(env_map.get(service, ""), "")
        if env_value:
            return env_value.strip()
        return str(self.get(f"translation.{service}.api_key", "")).strip()

    def reset(self) -> None:
        self.data = copy.deepcopy(DEFAULTS)
