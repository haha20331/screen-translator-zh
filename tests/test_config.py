"""配置文件读写与默认值合并测试。"""

import json

from app.config import DEFAULTS, AppConfig


def test_defaults_loaded(tmp_path):
    config = AppConfig(tmp_path / "config.json")
    assert config.get("ocr.min_confidence") == 0.6
    assert config.get("ocr.engine") == "rapidocr"
    assert config.get("ocr.lang") == "auto"
    assert config.get("updates.auto_check") is None  # 自动更新已移除
    assert config.get("translation.service") == "offline"
    assert config.get("translation.target_language") == "zh-hant"  # 一定要繁體
    assert config.hotkeys()["capture_region"] == "ctrl+shift+a"
    assert config.get("appearance.palette") == "warm_paper"
    assert config.get("appearance.accent") == "#2878E8"
    assert config.get("appearance.motion_profile") == "flow"


def test_merge_partial(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"ocr": {"min_confidence": 0.8}}', encoding="utf-8")
    config = AppConfig(path)
    assert config.get("ocr.min_confidence") == 0.8
    assert config.get("ocr.engine") == "rapidocr"
    assert config.get("ocr.lang") == "auto"
    assert config.get("appearance.density") == "balanced"


def test_legacy_chinese_default_migrates_to_real_auto_mode(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"ocr":{"lang":"zh"},"translation":{"source_language":"auto"}}',
        encoding="utf-8",
    )

    config = AppConfig(path)

    assert config.get("ocr.lang") == "auto"
    assert config.get("ocr.language_mode_version") == 2


def test_explicit_versioned_ocr_language_is_preserved(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"ocr":{"lang":"zh","language_mode_version":2},'
        '"translation":{"source_language":"auto"}}',
        encoding="utf-8",
    )

    assert AppConfig(path).get("ocr.lang") == "zh"


def test_legacy_google_free_speed_settings_are_normalized(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"translation":{"google_free_max_workers":8,'
        '"google_free_interval":0,"google_free_partial_retries":2}}',
        encoding="utf-8",
    )

    config = AppConfig(path)

    # 并发上限 8、最小间隔 0.02s：大批量文本块的总耗时约为旧上限的一半
    assert config.get("translation.google_free_max_workers") == 8
    assert config.get("translation.google_free_interval") == 0.02
    assert config.get("translation.google_free_partial_retries") == 0
    assert config.get("translation.google_free_fallback_to_mymemory") is True


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig(path)
    config.set("overlay.font_size", 24)
    config.set("hotkeys.capture_region", "ctrl+alt+x")
    config.set("appearance.palette", "midnight")
    config.set("appearance.accent", "#7258D6")
    config.save()
    config2 = AppConfig(path)
    assert config2.get("overlay.font_size") == 24
    assert config2.hotkeys()["capture_region"] == "ctrl+alt+x"
    assert config2.get("appearance.palette") == "midnight"
    assert config2.get("appearance.accent") == "#7258D6"


def test_invalid_appearance_values_fall_back_without_touching_other_sections(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"appearance":{"palette":"laser","accent":"red/blue",'
        '"motion_profile":"warp","density":"tiny","surface":"glass",'
        '"reduce_motion":"no"},"ocr":{"min_confidence":0.77}}',
        encoding="utf-8",
    )

    config = AppConfig(path)

    assert config.get("appearance.palette") == "warm_paper"
    assert config.get("appearance.accent") == "#2878E8"
    assert config.get("appearance.motion_profile") == "flow"
    assert config.get("appearance.density") == "balanced"
    assert config.get("appearance.surface") == "layered"
    assert config.get("appearance.reduce_motion") is False
    assert config.get("ocr.min_confidence") == 0.77


def test_masked_snapshot_hides_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"translation": {"openai": {"api_key": "sk-secret-1234567890"}}}', encoding="utf-8")
    config = AppConfig(path)
    snapshot = config.masked_snapshot()
    assert snapshot["translation"]["openai"]["api_key"] == "***"
    assert "sk-secret-1234567890" not in str(snapshot)


def test_corrupt_config_backed_up(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{nope", encoding="utf-8")
    config = AppConfig(path)
    assert config.get("ocr.min_confidence") == 0.6
    assert path.with_suffix(".json.bak").exists()


def test_legacy_monitor_interval_is_migrated_to_the_faster_default(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"monitor": {"interval_ms": 1200}}', encoding="utf-8")

    assert AppConfig(path).get("monitor.interval_ms") == 100


def test_customised_monitor_interval_is_kept(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"monitor": {"interval_ms": 2500}}', encoding="utf-8")

    assert AppConfig(path).get("monitor.interval_ms") == 2500


# ----------------------------------------------------------------------------- 已移除的 CTranslate2 / NLLB 設定
def _offline_config(tmp_path, translation):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"translation": translation}), encoding="utf-8")
    return AppConfig(path)


def test_old_ctranslate2_and_fast_translator_settings_are_migrated(tmp_path):
    config = _offline_config(
        tmp_path,
        {
            "service": "offline_fast",
            "offline_fast": {"model_name": "nllb-200-distilled-600m-ct2-int8", "device": "auto"},
            "offline": {
                "engine": "ctranslate2",
                "model_name": "qwen2.5-7b-instruct-ct2-int8",
                "device": "cuda",
                "compute_type": "int8_float16",
                "beam_size": 4,
                "threads": 4,
                "model_dir": "D:/models",
            },
        },
    )

    translation = config.section("translation")
    assert translation["service"] == "offline"  # 「快速翻譯」已移除
    assert "offline_fast" not in translation
    offline = translation["offline"]
    assert offline["model_name"] == "auto"  # CTranslate2 的模型名稱沒有意義了
    assert not {"engine", "device", "compute_type", "beam_size"} & set(offline)
    assert offline["threads"] == 4 and offline["model_dir"] == "D:/models"  # 其餘使用者的設定照舊


def test_gguf_model_names_and_other_services_are_left_alone(tmp_path):
    config = _offline_config(tmp_path, {"service": "google_free", "offline": {"model_name": "qwen2.5-3b-instruct-gguf-q4_k_m", "gpu_layers": 20}})

    assert config.get("translation.service") == "google_free"
    assert config.get("translation.offline.model_name") == "qwen2.5-3b-instruct-gguf-q4_k_m"
    assert config.get("translation.offline.gpu_layers") == 20


def test_the_default_offline_settings_only_have_llama_cpp_options():
    offline = DEFAULTS["translation"]["offline"]

    assert set(offline) == {"model_name", "model_dir", "gpu_layers", "cpu_fallback", "threads", "max_new_tokens", "han_only_as"}
    assert offline["model_name"] == "auto" and offline["gpu_layers"] == -1 and offline["cpu_fallback"] is True
    assert "offline_fast" not in DEFAULTS["translation"]
