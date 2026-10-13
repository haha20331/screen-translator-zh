"""llama.cpp 推論引擎：引擎選擇、提示詞、模型位置、快取分流、設定介面（用假的 llama 引擎，不載入真模型）。"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.config import DEFAULTS, AppConfig
from services.translation import llama_backend
from services.translation import offline_translator as module
from services.translation.base import TranslationError
from services.translation.ct2_common import SKIP_MARKER
from services.translation.offline_translator import OfflineTranslator
from ui import settings_dialog as settings_module
from ui.settings_dialog import INFERENCE_ENGINES, SettingsDialog
from ui.style import apply_style
from utils import chinese

pytestmark = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")

TRANSLATIONS = {"こんにちは": "你好", "ありがとう": "謝謝", "進む": "前進", "ドン！": SKIP_MARKER, "Hello": "哈囉"}


class FakeLlama:
    """像 LlamaCppEngine：收到完整提示詞，回傳「模型會寫的字」。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []
        self.prefixes: list[str] = []

    def complete(self, prompt: str, max_tokens: int, prefix: str = "") -> str:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        self.prefixes.append(prefix)
        source = re.findall(r"Translate this \w+ text:\n(.*?)<\|im_end\|>", prompt)[-1]
        if "song lyrics" in prompt and source == "進む":  # 歌詞提示讓模型把日文留在譯文裡
            return "進むと"
        return TRANSLATIONS.get(source, "")


def _translator(**section):
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp"}, **section})
    fake = FakeLlama()
    instance._llama = fake
    return instance, fake


# ----------------------------------------------------------------------------- 引擎選擇
@pytest.mark.parametrize(
    ("offline", "engine"),
    [
        ({}, "ctranslate2"),
        ({"engine": "ctranslate2"}, "ctranslate2"),
        ({"engine": "llama.cpp"}, "llama.cpp"),
        ({"engine": "LLAMA.CPP"}, "llama.cpp"),
        ({"engine": "ctranslate2", "model_name": "qwen2.5-1.5b-instruct-gguf-q8_0"}, "llama.cpp"),  # GGUF 模型名稱本身就決定引擎
        ({"model_name": "qwen2.5-7b-instruct-ct2-int8"}, "ctranslate2"),
        ({"engine": ""}, "ctranslate2"),
    ],
)
def test_engine_is_chosen_from_the_settings(offline, engine):
    assert OfflineTranslator({"offline": offline}).engine() == engine


def test_the_default_engine_stays_ctranslate2():
    assert DEFAULTS["translation"]["offline"]["engine"] == "ctranslate2"


def test_the_model_name_falls_back_to_q4_k_m_when_only_the_engine_is_set():
    assert OfflineTranslator({"offline": {"engine": "llama.cpp"}}).llama_model_name() == "qwen2.5-1.5b-instruct-gguf-q4_k_m"
    chosen = OfflineTranslator({"offline": {"model_name": "qwen2.5-1.5b-instruct-gguf-q8_0"}})
    assert chosen.llama_model_name() == "qwen2.5-1.5b-instruct-gguf-q8_0"


# ----------------------------------------------------------------------------- 提示詞與翻譯流程
def test_the_whole_chatml_prompt_goes_to_llama_cpp_and_the_answer_is_cleaned():
    instance, fake = _translator()

    out = instance.translate(["こんにちは"], "auto", "zh-hant")

    assert out == ["你好"]
    prompt = fake.prompts[0]
    assert prompt.startswith("<|im_start|>system\n")  # 固定前綴（系統提示 + 示範）
    assert "Translate this Japanese text:\nこんにちは<|im_end|>\n<|im_start|>assistant\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")  # 讓模型從這裡續寫
    assert fake.max_tokens == [160]


def test_the_fixed_part_of_the_prompt_is_passed_along_so_its_cache_can_be_kept():
    instance, fake = _translator(lyrics_mode=True)

    instance.translate(["こんにちは"], "auto", "zh-hant")
    instance.translate(["Hello"], "auto", "zh-hant")

    ja, en = fake.prefixes[0], fake.prefixes[1]
    assert ja == module.static_prompt("ja", True, True) and en == module.static_prompt("en", True, True)
    assert ja != en  # 換語言固定前綴就不同：這正是要分開存快取的原因
    assert all(prompt.startswith(prefix) for prompt, prefix in zip(fake.prompts, fake.prefixes))


def test_context_and_lyrics_use_the_same_prompts_as_the_other_engine():
    instance, fake = _translator(use_context=True, context_size=1, lyrics_mode=True)
    instance.translate(["こんにちは"], "auto", "zh-hant")

    instance.translate(["ありがとう"], "auto", "zh-hant")

    prompt = fake.prompts[-1]
    assert "song lyrics" in prompt  # 歌詞提示
    assert prompt.index("こんにちは") < prompt.index("ありがとう")  # 前一句當前文，放在待翻句之前
    assert "<|im_start|>assistant\n你好<|im_end|>\n" in prompt


def test_the_kana_retry_and_the_sound_skip_work_on_this_engine_too():
    instance, fake = _translator(lyrics_mode=True)
    assert instance.translate(["進む"], "auto", "zh-hant") == ["前進"]  # 歌詞提示殘留假名 → 改用一般提示重翻
    assert len(fake.prompts) == 2 and "song lyrics" not in fake.prompts[-1]

    sound, _ = _translator()
    assert sound.translate(["ドン！"], "auto", "zh-hant") == ["ドン！"]  # 擬聲詞：原文保留、不算失敗
    assert sound.last_failed_count == 0

    once, once_fake = _translator(lyrics_mode=True, retry_non_chinese=False)
    assert once.translate(["進む"], "auto", "zh-hant") == ["進むと"] and len(once_fake.prompts) == 1


def test_sentences_in_one_batch_are_translated_one_by_one():
    instance, fake = _translator()

    out = instance.translate(["こんにちは", "ありがとう"], "auto", "zh-hant")

    assert out == ["你好", "謝謝"] and len(fake.prompts) == 2


# ----------------------------------------------------------------------------- 快取分流
@pytest.mark.parametrize(
    ("offline", "namespace"),
    [
        ({"engine": "llama.cpp"}, "auto|llama-q4_k_m"),
        ({"model_name": "qwen2.5-1.5b-instruct-gguf-q8_0"}, "auto|llama-q8_0"),
        ({"engine": "ctranslate2"}, "auto"),  # CTranslate2 維持原本的鍵，舊快取不失效
    ],
)
def test_translations_of_different_engines_are_cached_separately(offline, namespace):
    assert OfflineTranslator({"offline": offline}).cache_namespace("auto") == namespace


def test_the_tag_comes_after_the_style_tags():
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp"}, "lyrics_mode": True})
    assert instance.cache_namespace("auto") == "auto|lyrics|llama-q4_k_m"


# ----------------------------------------------------------------------------- 模型位置與載入
def test_find_gguf_picks_the_first_file_by_name(tmp_path):
    assert llama_backend.find_gguf(tmp_path / "missing") is None
    assert llama_backend.find_gguf(tmp_path) is None
    (tmp_path / "b.gguf").write_bytes(b"x")
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert llama_backend.find_gguf(tmp_path).name == "a.gguf"


def test_the_gguf_is_found_in_the_configured_folder(tmp_path):
    model = tmp_path / "my-model" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp", "model_dir": str(model.parent)}})

    assert instance.llama_model_path() == model


def test_a_missing_gguf_explains_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "model_search_paths", lambda name, model_dir="": [tmp_path / name])
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp"}})

    with pytest.raises(TranslationError) as error:
        instance.llama_model_path()

    assert "fetch_translation_models.py --model qwen2.5-1.5b-instruct-gguf-q4_k_m" in str(error.value)
    assert str(tmp_path) in str(error.value)


def test_load_creates_the_llama_engine_with_the_default_threads(tmp_path, monkeypatch):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    created = []

    class FakeEngine:
        def __init__(self, path, threads):
            created.append((path, threads))

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", FakeEngine)
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp", "model_dir": str(model.parent)}})

    instance._load()

    assert created == [(model, 6)]  # 預設 6 執行緒（和 CTranslate2 一樣）
    assert (instance.device, instance.model_name) == ("cpu", "m")
    instance._load()  # 已載入：不會再建一次
    assert len(created) == 1


def test_missing_llama_cpp_python_is_reported(monkeypatch):
    monkeypatch.setattr(llama_backend, "available", lambda: False)
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp"}})

    with pytest.raises(TranslationError, match="llama-cpp-python"):
        instance._load()


def test_a_broken_model_file_is_reported_not_crashed(tmp_path, monkeypatch):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"not a model")

    class Broken:
        def __init__(self, path, threads):
            raise ValueError("invalid magic")

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", Broken)
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp", "model_dir": str(model.parent)}})

    with pytest.raises(TranslationError, match="qwen.gguf"):
        instance._load()


def test_ctranslate2_is_not_required_when_llama_cpp_is_installed(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name in ("ctranslate2", "tokenizers") else real(name, *a, **k))
    monkeypatch.setattr(llama_backend, "available", lambda: True)
    assert OfflineTranslator.availability_reason() is None

    monkeypatch.setattr(llama_backend, "available", lambda: False)
    assert "ctranslate2" in OfflineTranslator.availability_reason()


# ----------------------------------------------------------------------------- 設定介面
@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


@pytest.fixture
def models_present(monkeypatch):
    monkeypatch.setattr(settings_module, "llama_model_available", lambda name, model_dir="": True)


def _commit(dialog):
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()


def test_the_combo_lists_the_engines_and_defaults_to_ctranslate2(qapp, tmp_path, models_present):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    assert [dialog.combo_infer_engine.itemText(i) for i in range(3)] == [label for label, _e, _m in INFERENCE_ENGINES]
    assert dialog.combo_infer_engine.currentIndex() == 0
    assert dialog.pages.widget(3).widget().isAncestorOf(dialog.combo_infer_engine)  # 「翻譯」頁
    dialog.deleteLater()


def test_options_are_disabled_when_the_model_or_the_package_is_missing(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(settings_module, "llama_model_available", lambda name, model_dir="": name.endswith("q4_k_m"))
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    items = [dialog.combo_infer_engine.model().item(i) for i in range(3)]
    assert [item.isEnabled() for item in items] == [True, True, False]
    assert "fetch_translation_models.py --model qwen2.5-1.5b-instruct-gguf-q8_0" in dialog.combo_infer_engine.itemData(2, 3)  # ToolTipRole
    dialog.deleteLater()


def test_the_chosen_engine_is_saved_and_shown_again(qapp, tmp_path, models_present):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    dialog.combo_infer_engine.setCurrentIndex(2)
    _commit(dialog)

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("translation.offline.engine") == "llama.cpp"
    assert saved.get("translation.offline.model_name") == "qwen2.5-1.5b-instruct-gguf-q8_0"
    reopened = SettingsDialog(saved)
    assert reopened.combo_infer_engine.currentIndex() == 2
    dialog.deleteLater()
    reopened.deleteLater()


def test_an_untouched_combo_does_not_overwrite_a_custom_model_name(qapp, tmp_path, models_present):
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.offline.model_name", "qwen2.5-7b-instruct-ct2-int8")
    config.save()
    dialog = SettingsDialog(config)
    assert dialog.combo_infer_engine.currentIndex() == 0  # 顯示成 CTranslate2

    _commit(dialog)

    assert AppConfig(tmp_path / "config.json").get("translation.offline.model_name") == "qwen2.5-7b-instruct-ct2-int8"
    dialog.deleteLater()


@pytest.mark.parametrize(("offline", "index"), [({"engine": "llama.cpp"}, 1), ({"model_name": "qwen2.5-1.5b-instruct-gguf-q8_0"}, 2)])
def test_existing_settings_select_the_matching_item(qapp, tmp_path, models_present, offline, index):
    config = AppConfig(tmp_path / "config.json")
    for key, value in offline.items():
        config.set(f"translation.offline.{key}", value)

    dialog = SettingsDialog(config)

    assert dialog.combo_infer_engine.currentIndex() == index
    dialog.deleteLater()


def test_the_models_exist_check_looks_for_gguf_files(tmp_path, monkeypatch):
    folder = tmp_path / "qwen2.5-1.5b-instruct-gguf-q4_k_m"
    folder.mkdir()
    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(module, "model_search_paths", lambda name, model_dir="": [folder])  # 不要找到開發機上真的模型

    assert not module.llama_model_available("qwen2.5-1.5b-instruct-gguf-q4_k_m")
    (folder / "x.gguf").write_bytes(b"x")
    assert module.llama_model_available("qwen2.5-1.5b-instruct-gguf-q4_k_m")
    monkeypatch.setattr(llama_backend, "available", lambda: False)
    assert not module.llama_model_available("qwen2.5-1.5b-instruct-gguf-q4_k_m")


# ----------------------------------------------------------------------------- 固定前綴的 KV 快取（假的 llama，只驗證流程）
class _Ctx:
    def __init__(self, owner) -> None:
        self.owner = owner

    def kv_cache_clear(self) -> None:
        self.owner.log.append("clear")


class _RawLlama:
    """像 llama_cpp.Llama 的最小替身：一個字元一個 token，記錄被呼叫的動作。"""

    def __init__(self) -> None:
        import numpy as np

        self.input_ids = np.zeros(4096, dtype=np.int32)
        self.n_tokens = 0
        self.log: list[str] = []
        self._ctx = _Ctx(self)

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False):
        return [ord(c) for c in data.decode("utf-8")]

    def reset(self) -> None:
        self.n_tokens = 0
        self.log.append("reset")

    def eval(self, tokens) -> None:
        self.input_ids[self.n_tokens : self.n_tokens + len(tokens)] = tokens
        self.n_tokens += len(tokens)
        self.log.append(f"eval {len(tokens)}")

    def create_completion(self, prompt, **kwargs):
        self.log.append("complete")
        self.kwargs = kwargs
        return {"choices": [{"text": "譯文"}]}


def _engine():
    engine = llama_backend.LlamaCppEngine.__new__(llama_backend.LlamaCppEngine)
    engine._llama = _RawLlama()
    engine._prefix_states = __import__("collections").OrderedDict()
    engine._prefix_cache_ok = True
    engine._capture_kv = lambda: f"kv:{engine._llama.n_tokens}".encode()
    engine._restore_kv = lambda state: engine._llama.log.append(f"restore {state.decode()}")
    return engine


def test_the_prefix_is_computed_once_then_kept_while_it_stays_the_same():
    engine = _engine()

    engine.prepare("AAAA")
    engine.prepare("AAAA")
    engine.complete("AAAA問題", 50, "AAAA")

    assert engine._llama.log == ["reset", "clear", "eval 4", "complete"]  # 只算一次：已經停在這個前綴上就不再動
    assert list(engine._prefix_states) == ["AAAA"]


def test_switching_back_to_a_known_prefix_restores_it_instead_of_computing_it_again():
    engine = _engine()
    engine.prepare("AAAA")
    engine.complete("AAAA一", 50, "AAAA")
    engine.prepare("BBBBB")  # 換語言：第一次要算
    engine._llama.log.clear()

    engine.prepare("AAAA")  # 換回來：還原

    assert engine._llama.log == ["restore kv:4"]
    assert engine._llama.n_tokens == 4
    assert engine._llama.input_ids[:4].tolist() == [ord(c) for c in "AAAA"]  # 函式庫之後才認得這是可重用的前綴


def test_only_the_most_recent_prefixes_are_kept(monkeypatch):
    monkeypatch.setattr(llama_backend, "MAX_PREFIX_STATES", 2)
    engine = _engine()

    for prefix in ("AA", "BB", "CC"):
        engine.prepare(prefix)

    assert list(engine._prefix_states) == ["BB", "CC"]  # 最舊的被丟掉，記憶體有上限


def test_any_failure_turns_the_cache_off_and_translation_goes_on():
    engine = _engine()

    def broken():
        raise RuntimeError("state size mismatch")

    engine._capture_kv = broken
    engine.prepare("AAAA")

    assert engine._prefix_cache_ok is False and not engine._prefix_states
    engine._llama.log.clear()
    assert engine.complete("AAAA一", 50, "AAAA") == "譯文"  # 照函式庫原本的行為繼續翻
    assert engine._llama.log == ["complete"]  # 不再碰快取


def test_complete_uses_greedy_decoding_and_stops_at_the_end_marker():
    engine = _engine()

    assert engine.complete("提示詞", 77) == "譯文"

    assert engine._llama.kwargs == {"max_tokens": 77, "temperature": 0.0, "top_k": 1, "stop": ["<|im_end|>"], "echo": False}
    assert engine._llama.log == ["complete"]  # 沒給固定前綴：不做任何快取動作


def test_a_prompt_that_does_not_start_with_the_prefix_skips_the_cache():
    engine = _engine()

    engine.complete("別的提示詞", 50, "AAAA")

    assert engine._llama.log == ["complete"]


# ----------------------------------------------------------------------------- 真實模型（開發機有下載才跑）
_REAL_MODEL = module.OfflineTranslator({"offline": {"engine": "llama.cpp"}})


def _real_model_present() -> bool:
    try:
        _REAL_MODEL.llama_model_path()
    except TranslationError:
        return False
    return llama_backend.available()


@pytest.mark.skipif(not _real_model_present(), reason="沒有 llama-cpp-python 或 GGUF 模型")
def test_real_model_alternating_languages_keeps_one_cached_prefix_per_language():
    instance = OfflineTranslator({"offline": {"engine": "llama.cpp"}})
    outputs = [instance.translate([text], "auto", "zh-hant")[0] for text in ["今日はいい天気ですね。", "Where are you going?", "ファイルを保存しますか？", "The door is locked"]]

    assert all(outputs) and all(out != text for out, text in zip(outputs, ["今日はいい天気ですね。", "Where are you going?", "ファイルを保存しますか？", "The door is locked"]))
    assert len(instance._llama._prefix_states) == 2 and instance._llama._prefix_cache_ok
