"""離線翻譯器：提示詞、輸出整理、模型位置、語言分組、繁體轉換、失敗判定與裝置選擇。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from services.translation import ct2_common as module
from services.translation.base import TranslationError, list_translators
from services.translation.cache import TranslationCache
from services.translation.ct2_common import find_model_dir, model_search_paths
from services.translation.offline_translator import (
    DEFAULT_MODEL_NAME,
    LARGE_MODEL_NAME,
    _SHOTS,
    OfflineTranslator,
    build_prompt,
    clean_output,
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
    # 固定前綴不含待翻譯文字，且停在對話輪次邊界，才能被 CTranslate2 快取
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
def _make_model(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "model.bin").write_bytes(b"x")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


def test_configured_model_dir_has_priority(tmp_path, monkeypatch):
    configured = _make_model(tmp_path / "mine")
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])
    _make_model(tmp_path / "app" / "models" / "translation" / DEFAULT_MODEL_NAME)

    assert find_model_dir(DEFAULT_MODEL_NAME, str(configured)) == configured


def test_default_locations_are_searched_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])

    assert find_model_dir(DEFAULT_MODEL_NAME) is None
    user_copy = _make_model(tmp_path / "appdata" / "models" / "translation" / DEFAULT_MODEL_NAME)
    assert find_model_dir(DEFAULT_MODEL_NAME) == user_copy
    app_copy = _make_model(tmp_path / "app" / "models" / "translation" / DEFAULT_MODEL_NAME)
    assert find_model_dir(DEFAULT_MODEL_NAME) == app_copy  # 程式資料夾優先於使用者資料夾


def test_relative_model_dir_is_resolved_against_app_root(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])
    model = _make_model(tmp_path / "app" / "custom" / "model")

    assert find_model_dir(DEFAULT_MODEL_NAME, "custom/model") == model
    assert tmp_path / "app" / "custom" / "model" in model_search_paths(DEFAULT_MODEL_NAME, "custom/model")


def test_incomplete_model_dir_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])
    broken = tmp_path / "app" / "models" / "translation" / DEFAULT_MODEL_NAME
    broken.mkdir(parents=True)
    (broken / "model.bin").write_bytes(b"x")  # 少了 tokenizer.json

    assert find_model_dir(DEFAULT_MODEL_NAME) is None


def test_missing_model_error_lists_search_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])

    with pytest.raises(TranslationError) as excinfo:
        _translator().load_plans(_FakeCT2())

    message = str(excinfo.value)
    assert DEFAULT_MODEL_NAME in message and str(tmp_path / "app") in message


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
    assert translator.cache.get("auto", "zh-hant", "設定") == "設定"


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


# ----------------------------------------------------------------------------- 裝置與可用性
class _FakeCT2:
    def __init__(self, count=0, error=None) -> None:
        self._count, self._error = count, error

    def get_cuda_device_count(self):
        if self._error:
            raise self._error
        return self._count


@pytest.mark.parametrize(
    ("setting", "fake", "expected"),
    [
        ("auto", _FakeCT2(count=1), "cuda"),
        ("auto", _FakeCT2(count=0), "cpu"),
        ("auto", _FakeCT2(error=RuntimeError("no cuda runtime")), "cpu"),
        ("cpu", _FakeCT2(count=1), "cpu"),
        ("cuda", _FakeCT2(count=0), "cuda"),
    ],
)
def test_device_resolution(setting, fake, expected):
    assert _translator(device=setting)._resolve_device(fake) == expected


def test_availability_reason_reports_missing_dependency(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "ctranslate2" else real(name, *a, **k))

    assert "ctranslate2" in OfflineTranslator.availability_reason()


# ----------------------------------------------------------------------------- 推論包裝（假的 CTranslate2）
def test_generate_builds_prompts_and_decodes_results(monkeypatch):
    instance = _translator(beam_size=2, max_new_tokens=50)
    seen = {}

    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens):
            seen.setdefault("prompts", []).append((prompt, add_special_tokens))
            return SimpleNamespace(tokens=["<tok>", prompt[-30:]])

        def decode(self, ids, skip_special_tokens):
            return f"\n  譯文{ids[0]}  \n(註解)"

    class FakeGenerator:
        def generate_batch(self, batches, **kwargs):
            seen["batches"], seen["kwargs"] = batches, kwargs
            return [SimpleNamespace(sequences_ids=[[index]]) for index, _ in enumerate(batches)]

    instance._generator, instance._tokenizer = FakeGenerator(), FakeTokenizer()

    out = instance._generate(["こんにちは", "さようなら"], "ja")

    assert out == ["譯文0", "譯文1"]
    assert all(flag is False for _, flag in seen["prompts"])
    assert seen["prompts"][0][0] == static_prompt("ja")  # 固定前綴只分詞一次
    assert "こんにちは" in seen["prompts"][1][0] and "さようなら" in seen["prompts"][2][0]
    assert len(seen["prompts"]) == 3
    assert seen["kwargs"]["static_prompt"] == ["<tok>", static_prompt("ja")[-30:]]
    assert seen["batches"][0][1].endswith("assistant\n")
    # 第二次呼叫沿用快取的前綴，不重新分詞
    instance._generate(["もう一つ"], "ja")
    assert len(seen["prompts"]) == 4
    assert seen["kwargs"]["beam_size"] == 2
    assert seen["kwargs"]["max_length"] == 50
    assert seen["kwargs"]["include_prompt_in_result"] is False
    assert seen["kwargs"]["end_token"] == "<|im_end|>"


# ----------------------------------------------------------------------------- 真實模型（有模型才跑）
@pytest.mark.skipif(
    find_model_dir(DEFAULT_MODEL_NAME) is None or OfflineTranslator.availability_reason() is not None or not chinese.available(),
    reason="未安裝離線翻譯模型或相依套件",
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


# ----------------------------------------------------------------------------- 依硬體自動選模型
def _models(tmp_path, monkeypatch, *names):
    monkeypatch.setattr(module, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(module, "_app_roots", lambda: [tmp_path / "app"])
    for name in names:
        _make_model(tmp_path / "app" / "models" / "translation" / name)


def test_auto_uses_the_large_model_only_on_cuda(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, LARGE_MODEL_NAME, DEFAULT_MODEL_NAME)

    cpu_plans = _translator().load_plans(_FakeCT2(count=0))
    cuda_plans = _translator().load_plans(_FakeCT2(count=1))

    assert [(plan.model_name, plan.device) for plan in cpu_plans] == [(DEFAULT_MODEL_NAME, "cpu")]
    # GPU：7B 優先；載入失敗（顯示卡記憶體不足等）依序退到 GPU 上的 1.5B、最後 CPU 上的 1.5B
    assert [(plan.model_name, plan.device) for plan in cuda_plans] == [
        (LARGE_MODEL_NAME, "cuda"),
        (DEFAULT_MODEL_NAME, "cuda"),
        (DEFAULT_MODEL_NAME, "cpu"),
    ]


def test_auto_without_the_large_model_still_works_on_cuda(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, DEFAULT_MODEL_NAME)

    plans = _translator().load_plans(_FakeCT2(count=1))

    assert [(plan.model_name, plan.device) for plan in plans][0] == (DEFAULT_MODEL_NAME, "cuda")


def test_explicit_model_name_disables_auto_selection(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, LARGE_MODEL_NAME, DEFAULT_MODEL_NAME)

    plans = _translator(model_name=DEFAULT_MODEL_NAME).load_plans(_FakeCT2(count=1))

    assert [(plan.model_name, plan.device) for plan in plans] == [(DEFAULT_MODEL_NAME, "cuda")]


def test_loading_falls_back_to_the_next_plan_when_a_model_fails(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, LARGE_MODEL_NAME, DEFAULT_MODEL_NAME)
    attempts = []

    class FakeGenerator:
        def __init__(self, path, device, compute_type, inter_threads, intra_threads):
            attempts.append((Path(path).name, device, compute_type))
            if Path(path).name == LARGE_MODEL_NAME:
                raise RuntimeError("CUDA out of memory")

    class FakeTokenizerClass:
        @staticmethod
        def from_file(path):
            return object()

    import sys

    fake_ct2 = SimpleNamespace(Generator=FakeGenerator, get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizerClass))
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))
    instance = _translator()

    instance._load()

    assert attempts == [(LARGE_MODEL_NAME, "cuda", "int8_float16"), (DEFAULT_MODEL_NAME, "cuda", "int8_float16")]
    assert (instance.model_name, instance.device) == (DEFAULT_MODEL_NAME, "cuda")


def test_loading_reports_every_failure_when_no_plan_works(tmp_path, monkeypatch):
    _models(tmp_path, monkeypatch, DEFAULT_MODEL_NAME)

    class BrokenGenerator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no luck")

    import sys

    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Generator=BrokenGenerator, get_cuda_device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda p: object())))
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))

    with pytest.raises(TranslationError, match="no luck"):
        _translator()._load()
