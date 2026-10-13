"""離線翻譯器：提示詞、輸出整理、模型位置與選擇、語言分組、繁體轉換、失敗判定。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from services.translation import llama_backend
from services.translation import offline_common as module
from services.translation.base import TranslationError, list_translators
from services.translation.cache import TranslationCache
from services.translation.offline_common import model_search_paths
from services.translation.offline_translator import (
    _SHOTS,
    DEFAULT_MODEL_NAME,
    LARGE_MODEL_NAME,
    OFFLINE_MODELS,
    OfflineTranslator,
    build_prompt,
    clean_output,
    llama_model_available,
    model_tag,
    static_prompt,
    user_prompt,
)
from utils import chinese


def _translator(**offline) -> OfflineTranslator:
    return OfflineTranslator({"offline": offline})


# ----------------------------------------------------------------------------- 提示詞與輸出
def test_registered_under_offline_name():
    assert "offline" in list_translators()
    assert OfflineTranslator.han_is_source is True
    assert OfflineTranslator.identical_means_failure is False


@pytest.mark.parametrize(("language", "name"), [("ja", "Japanese"), ("en", "English")])
def test_prompt_has_system_shots_and_target_text(language, name):
    prompt = build_prompt("待翻譯", language)

    assert prompt.startswith("<|im_start|>system\n")
    assert "Traditional Chinese as used in Taiwan" in prompt
    shots = len(_SHOTS[language]) + 1  # 示範 + 最後等待模型續寫的一個
    assert prompt.count("<|im_start|>assistant\n") == shots
    assert prompt.count(f"Translate this {name} text:") == shots
    assert prompt.endswith(f"Translate this {name} text:\n待翻譯<|im_end|>\n<|im_start|>assistant\n")


@pytest.mark.parametrize("language", ["ja", "en"])
def test_prompt_is_static_prefix_plus_user_turn(language):
    assert build_prompt("文字", language) == static_prompt(language) + user_prompt("文字", language)
    # 固定前綴不含待翻譯文字，且停在對話輪次邊界，才能被引擎的 KV 快取重用
    assert "文字" not in static_prompt(language)
    assert static_prompt(language).endswith("<|im_end|>\n")
    assert user_prompt("文字", language).startswith("<|im_start|>user\n")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  你好  ", "你好"),
        ("\n\n第一行\n(Translation: second)\n", "第一行"),
        ("", ""),
        ("   \n  ", ""),
    ],
)
def test_clean_output_keeps_first_non_empty_line(raw, expected):
    assert clean_output(raw) == expected


# ----------------------------------------------------------------------------- 模型位置
def _make_gguf(directory: Path, name: str = "model.gguf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"x")
    return directory / name


def _isolate(tmp_path, monkeypatch) -> Path:
    """只在暫存資料夾裡找模型（不要找到開發機上真的模型）。"""
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])
    return tmp_path / "app" / "models" / "translation"


def test_configured_model_dir_has_priority(tmp_path, monkeypatch):
    models = _isolate(tmp_path, monkeypatch)
    configured = _make_gguf(tmp_path / "mine")
    _make_gguf(models / DEFAULT_MODEL_NAME)

    assert _translator(model_dir=str(configured.parent)).llama_model_path() == configured


def test_default_locations_are_searched_in_order(tmp_path, monkeypatch):
    models = _isolate(tmp_path, monkeypatch)

    assert not llama_model_available(DEFAULT_MODEL_NAME)
    user_copy = _make_gguf(tmp_path / "appdata" / "models" / "translation" / DEFAULT_MODEL_NAME)
    assert _translator().llama_model_path() == user_copy
    app_copy = _make_gguf(models / DEFAULT_MODEL_NAME)
    assert _translator().llama_model_path() == app_copy  # 程式資料夾優先於使用者資料夾


def test_relative_model_dir_is_resolved_against_app_root(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    model = _make_gguf(tmp_path / "app" / "custom" / "model")

    assert _translator(model_dir="custom/model").llama_model_path() == model
    assert tmp_path / "app" / "custom" / "model" in model_search_paths(DEFAULT_MODEL_NAME, "custom/model")


def test_a_folder_without_a_gguf_file_is_ignored(tmp_path, monkeypatch):
    models = _isolate(tmp_path, monkeypatch)
    (models / DEFAULT_MODEL_NAME).mkdir(parents=True)
    (models / DEFAULT_MODEL_NAME / "manifest.json").write_text("{}", encoding="utf-8")  # 只有說明檔，沒有模型

    assert not llama_model_available(DEFAULT_MODEL_NAME)
    with pytest.raises(TranslationError):
        _translator().llama_model_path()


def test_a_split_model_is_loaded_through_its_first_part(tmp_path, monkeypatch):
    models = _isolate(tmp_path, monkeypatch)
    _make_gguf(models / LARGE_MODEL_NAME, "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf")
    first = _make_gguf(models / LARGE_MODEL_NAME, "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")

    assert _translator(model_name=LARGE_MODEL_NAME).llama_model_path() == first  # llama.cpp 讀第一片會自動找其餘的


def test_missing_model_error_lists_search_locations_and_how_to_get_it(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    with pytest.raises(TranslationError) as excinfo:
        _translator().llama_model_path()

    message = str(excinfo.value)
    assert DEFAULT_MODEL_NAME in message and str(tmp_path / "app") in message
    assert f"fetch_translation_models.py --model {DEFAULT_MODEL_NAME}" in message


# ----------------------------------------------------------------------------- 模型選擇
def test_auto_uses_the_default_model_on_cpu_only_builds(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)

    assert _translator().resolved_model_name() == DEFAULT_MODEL_NAME == "qwen2.5-1.5b-instruct-gguf-q4_k_m"


def test_auto_uses_the_large_model_only_with_gpu_support_and_the_model_file(tmp_path, monkeypatch):
    models = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)

    assert _translator().resolved_model_name() == DEFAULT_MODEL_NAME  # 有 GPU 但沒放 7B：用預設的
    _make_gguf(models / LARGE_MODEL_NAME)
    assert _translator().resolved_model_name() == LARGE_MODEL_NAME  # 有 GPU 且放了 7B：用 7B

    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)
    assert _translator().resolved_model_name() == DEFAULT_MODEL_NAME  # 放了 7B 但只有 CPU 版：不自動用（太慢）


@pytest.mark.parametrize("name", [model.name for model in OFFLINE_MODELS])
def test_an_explicit_model_name_is_used_as_is(name, monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)

    assert _translator(model_name=name).resolved_model_name() == name


def test_the_choice_is_made_once_per_translator(monkeypatch):
    instance = _translator()
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)
    assert instance.resolved_model_name() == DEFAULT_MODEL_NAME
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)

    assert instance.resolved_model_name() == DEFAULT_MODEL_NAME  # 設定改了會重建翻譯器；這個實例的快取鍵不能中途變


def test_the_model_list_is_ordered_small_to_large_and_has_distinct_names():
    names = [model.name for model in OFFLINE_MODELS]

    assert len(set(names)) == len(names) and DEFAULT_MODEL_NAME in names and LARGE_MODEL_NAME in names
    sizes = [float(name.split("-")[1].removesuffix("b")) for name in names]
    assert sizes == sorted(sizes)
    assert [size for size in sizes if size < 1.5] and [size for size in sizes if size > 1.5]  # 有比預設更小、更大的可試


def test_every_listed_model_can_be_fetched_by_the_download_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_translation_models.py"
    spec = importlib.util.spec_from_file_location("fetch_translation_models", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    assert {model.name for model in OFFLINE_MODELS} == set(script.MODELS)
    assert script.DEFAULT_MODEL == DEFAULT_MODEL_NAME
    for info in script.MODELS.values():
        assert info["revision"] and len(info["revision"]) == 40  # 固定版本
        assert all(file.endswith(".gguf") for file in info["files"])


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("qwen2.5-1.5b-instruct-gguf-q4_k_m", "1.5b-q4_k_m"),
        ("qwen2.5-0.5b-instruct-gguf-q8_0", "0.5b-q8_0"),
        ("qwen2.5-7b-instruct-gguf-q4_k_m", "7b-q4_k_m"),
        ("my-own-model", "my-own-model"),
    ],
)
def test_the_cache_tag_tells_models_apart(name, tag):
    assert model_tag(name) == tag
    assert _translator(model_name=name).cache_namespace("auto") == f"auto|p2|{tag}"


@pytest.mark.parametrize(
    ("setting", "supported", "expected"),
    [(-1, True, -1), (-1, False, 0), (0, True, 0), (20, True, 20), ("abc", True, -1), ("abc", False, 0), (None, False, 0)],
)
def test_gpu_layers(setting, supported, expected, monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: supported)

    assert _translator(gpu_layers=setting).gpu_layers() == expected


# ----------------------------------------------------------------------------- 翻譯流程
class _Recorder:
    """取代 _generate：記錄每次呼叫，依 mapping 回傳簡體輸出。"""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, texts, language):
        self.calls.append((list(texts), language))
        return [self.mapping.get(text, "") for text in texts]


@pytest.fixture
def translator(monkeypatch):
    instance = _translator()
    recorder = _Recorder(
        {
            "こんにちは": "你好",
            "ファイルを保存しますか？": "要保存文件吗？",
            "Save changes": "保存更改",
            "Settings": "设置",
            "設定": "设定",
            "ご主人さま": "ご主人さま",  # 模型只是把原文抄回來
        }
    )
    monkeypatch.setattr(instance, "_generate", recorder)
    instance.recorder = recorder
    return instance


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_texts_are_grouped_by_language_and_converted_to_taiwan_traditional(translator):
    out = translator.translate(["こんにちは", "Save changes", "ファイルを保存しますか？", "Settings"], "auto", "zh-hant")

    assert out == ["你好", "儲存更改", "要儲存檔案嗎？", "設定"]  # 保存→儲存、文件→檔案、更改→更改
    languages = {language: texts for texts, language in translator.recorder.calls}
    assert languages == {"ja": ["こんにちは", "ファイルを保存しますか？"], "en": ["Save changes", "Settings"]}


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_simplified_target_keeps_simplified_output(translator):
    assert translator.translate(["Settings"], "auto", "zh") == ["设置"]


def test_text_without_a_language_is_passed_through_and_not_a_failure(translator):
    out = translator.translate(["12345", "!!!"], "auto", "zh-hant")

    assert out == ["12345", "!!!"]
    assert translator.recorder.calls == []
    assert translator.last_failed_count == 0


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_unchanged_result_is_not_reported_as_failure_and_is_cached(translator, tmp_path):
    translator.cache = TranslationCache(tmp_path / "cache.json")
    out = translator.translate(["設定"], "auto", "zh-hant")  # 模型回「设定」→ 轉成「設定」，與原文相同

    assert out == ["設定"]
    assert translator.last_failed_count == 0
    assert translator.cache.get(translator.cache_namespace("auto"), "zh-hant", "設定") == "設定"


def test_echoed_japanese_is_kept_and_marked_failed(translator):
    out = translator.translate(["ご主人さま"], "auto", "zh-hant")

    assert out == ["ご主人さま"]
    assert translator.last_failed_count == 1
    assert translator.last_failed_indices == {0}


def test_japanese_output_without_han_characters_is_marked_failed(monkeypatch):
    instance = _translator()
    monkeypatch.setattr(instance, "_generate", _Recorder({"スキ": "skip", "ありがとう": "Thanks"}))

    out = instance.translate(["スキ", "ありがとう"], "auto", "zh-hant")

    assert out == ["スキ", "ありがとう"]  # 保留原文，不顯示英文
    assert instance.last_failed_count == 2


def test_empty_model_output_is_marked_failed(translator):
    out = translator.translate(["未登録の文章です"], "auto", "zh-hant")  # mapping 沒有 → 回傳空字串

    assert out == ["未登録の文章です"]
    assert translator.last_failed_count == 1


@pytest.mark.parametrize("target", ["en", "ja", "ko"])
def test_only_chinese_targets_are_supported(translator, target):
    with pytest.raises(TranslationError, match="中文"):
        translator._translate_batch(["Settings"], "auto", target)


def test_explicit_source_language_overrides_detection(translator):
    translator._translate_batch(["設定"], "en", "zh-hant")

    assert translator.recorder.calls == [(["設定"], "en")]


def test_han_only_text_can_be_excluded_by_configuration(monkeypatch):
    instance = _translator(han_only_as="")
    recorder = _Recorder({})
    monkeypatch.setattr(instance, "_generate", recorder)

    assert instance._translate_batch(["設定"], "auto", "zh-hant") == ["設定"]
    assert recorder.calls == []


# ----------------------------------------------------------------------------- 可用性
def test_availability_reports_the_missing_dependency(monkeypatch):
    monkeypatch.setattr(llama_backend, "available", lambda: False)
    assert "llama-cpp-python" in OfflineTranslator.availability_reason()

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "opencc" else real(name, *a, **k))
    assert "opencc" in OfflineTranslator.availability_reason()

    monkeypatch.setattr(importlib.util, "find_spec", real)
    monkeypatch.setattr(llama_backend, "available", lambda: True)
    assert OfflineTranslator.availability_reason() is None


# ----------------------------------------------------------------------------- 真實模型（有模型才跑）
@pytest.mark.skipif(
    not llama_model_available(DEFAULT_MODEL_NAME) or not chinese.available(),
    reason="未安裝 llama-cpp-python 或預設模型",
)
def test_real_model_translates_to_traditional_chinese():
    instance = OfflineTranslator({"offline": {}})
    out = instance.translate(["ファイルを保存しますか？", "Save changes before closing?", "HPが120回復しました"], "auto", "zh-hant")

    assert "儲存" in out[0] and "檔案" in out[0]
    assert "儲存" in out[1] or "保存" in out[1]
    assert "120" in out[2]
    # 全部是繁體：轉成簡體再轉回來不應改變任何字
    for text in out:
        assert chinese.to_traditional(text) == text
        assert not any("぀" <= char <= "ヿ" for char in text)
    assert instance.last_failed_count == 0
