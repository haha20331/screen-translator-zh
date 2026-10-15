"""llama.cpp 引擎：提示詞、載入、GPU 卸載與退回、固定前綴的 KV 快取、設定介面（用假的引擎，不載入真模型）。"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from services.translation import llama_backend
from services.translation import offline_translator as module
from services.translation.base import TranslationError
from services.translation.offline_common import SKIP_MARKER
from services.translation.offline_translator import OfflineTranslator
from tests.fake_llama import FakeLlamaEngine, offline_translator, source_of
from ui import settings_dialog as settings_module
from ui.settings_dialog import OFFLINE_MODEL_CHOICES, SettingsDialog
from ui.style import apply_style
from utils import chinese

pytestmark = pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")

TRANSLATIONS = {"こんにちは": "你好", "ありがとう": "謝謝", "進む": "前進", "ドン！": SKIP_MARKER, "Hello": "哈囉"}


def decode(prompt: str) -> str:
    source = source_of(prompt)
    if "歌詞翻譯" in prompt and source == "進む":  # 歌詞提示讓模型把日文留在譯文裡
        return "進むと"
    return TRANSLATIONS.get(source, "")


def _translator(**section) -> tuple[OfflineTranslator, FakeLlamaEngine]:
    return offline_translator(decode, **section)


# ----------------------------------------------------------------------------- 提示詞與翻譯流程
def test_the_whole_chatml_prompt_goes_to_llama_cpp_and_the_answer_is_cleaned():
    instance, fake = _translator()

    out = instance.translate(["こんにちは"], "auto", "zh-hant")

    assert out == ["你好"]
    prompt = fake.prompts[0]
    assert prompt.startswith("<|im_start|>system\n")  # 固定前綴（系統提示 + 示範）
    assert "翻譯這句日文：\nこんにちは<|im_end|>\n<|im_start|>assistant\n" in prompt  # Qwen2.5 系列：中文提示詞
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


def test_context_and_lyrics_are_put_into_the_prompt():
    instance, fake = _translator(use_context=True, context_size=1, lyrics_mode=True)
    instance.translate(["こんにちは"], "auto", "zh-hant")

    instance.translate(["ありがとう"], "auto", "zh-hant")

    prompt = fake.prompts[-1]
    assert "歌詞翻譯" in prompt  # 歌詞提示（Qwen2.5 系列用中文提示詞）
    assert prompt.index("こんにちは") < prompt.index("ありがとう")  # 前一句當前文，放在待翻句之前
    assert "<|im_start|>assistant\n你好<|im_end|>\n" in prompt


def test_the_kana_retry_and_the_sound_skip_work():
    instance, fake = _translator(lyrics_mode=True)
    assert instance.translate(["進む"], "auto", "zh-hant") == ["前進"]  # 歌詞提示殘留假名 → 改用一般提示重翻
    assert fake.calls == 2 and "歌詞翻譯" not in fake.prompts[-1]

    sound, _ = _translator()
    assert sound.translate(["ドン！"], "auto", "zh-hant") == ["ドン！"]  # 擬聲詞：原文保留、不算失敗
    assert sound.last_failed_count == 0

    once, once_fake = _translator(lyrics_mode=True, retry_non_chinese=False)
    assert once.translate(["進む"], "auto", "zh-hant") == ["進むと"] and len(once_fake.prompts) == 1


def test_sentences_in_one_batch_are_translated_one_by_one():
    instance, fake = _translator()

    out = instance.translate(["こんにちは", "ありがとう"], "auto", "zh-hant")

    assert out == ["你好", "謝謝"] and len(fake.prompts) == 2


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
    instance = OfflineTranslator({"offline": {"model_dir": str(model.parent)}})

    assert instance.llama_model_path() == model


def test_load_creates_the_llama_engine_with_the_default_threads(tmp_path, monkeypatch):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    created = []

    class FakeEngine:
        gpu_layers = 0

        def __init__(self, path, threads, gpu_layers=0):
            created.append((path, threads, gpu_layers))

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", FakeEngine)
    instance = OfflineTranslator({"offline": {"model_dir": str(model.parent)}})

    instance._load()

    assert created == [(model, 6, 0)]  # 預設 6 執行緒；CPU 版沒有 GPU 可卸載
    assert (instance.device, instance.model_name) == ("cpu", "m")
    instance._load()  # 已載入：不會再建一次
    assert len(created) == 1


def test_a_gpu_build_offloads_all_layers_by_default(tmp_path, monkeypatch):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    created = []

    class FakeEngine:
        def __init__(self, path, threads, gpu_layers=0):
            created.append(gpu_layers)
            self.gpu_layers = gpu_layers

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", FakeEngine)
    instance = OfflineTranslator({"offline": {"model_dir": str(model.parent)}})

    instance._load()

    assert created == [-1] and instance.device == "gpu"


def _loaded_translator(tmp_path, monkeypatch, *, gpu_supported, engine_gpu_layers, section=None):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")

    class FakeEngine:
        def __init__(self, path, threads, gpu_layers=0, **kwargs):
            self.gpu_layers = engine_gpu_layers

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: gpu_supported)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", FakeEngine)
    instance = OfflineTranslator({"offline": {"model_dir": str(model.parent), **(section or {})}})
    instance._load()
    return instance


def test_a_cpu_build_is_never_reported_as_running_on_the_gpu(tmp_path, monkeypatch):
    instance = _loaded_translator(tmp_path, monkeypatch, gpu_supported=False, engine_gpu_layers=8, section={"gpu_layers": 8})

    assert instance.device == "cpu"  # CPU 版收到 gpu_layers > 0 只是忽略，不能標成 GPU


def test_the_log_says_which_device_the_model_really_loaded_on(tmp_path, monkeypatch, caplog):
    import logging

    messages = []
    with caplog.at_level(logging.INFO, logger="screen_translator.translation.offline"):
        for name, supported, layers in (("a", True, -1), ("b", True, 0), ("c", False, 0)):
            (tmp_path / name).mkdir()
            _loaded_translator(tmp_path / name, monkeypatch, gpu_supported=supported, engine_gpu_layers=layers)
            messages.append(caplog.records[-1].getMessage())
    gpu_message, fallback_message, cpu_message = messages

    assert "裝置＝GPU" in gpu_message and "改用 CPU" not in gpu_message
    assert "裝置＝CPU" in fallback_message and "已改用 CPU" in fallback_message  # 要求用 GPU 卻退回 CPU：要看得出來
    assert "裝置＝CPU" in cpu_message and "改用 CPU" not in cpu_message


def _device(kind, free, total, name="dev"):
    return {"name": name, "description": name, "type": kind, "free": free, "total": total}


def test_only_a_discrete_gpu_counts_as_having_gpu_memory_of_its_own(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(
        llama_backend, "_query_devices",
        lambda: [
            _device(0, 5_000_000_000, 16_000_000_000, "CPU"),
            _device(llama_backend.DEVICE_TYPE_IGPU, 7_700_000_000, 8_400_000_000, "Vulkan0"),
            _device(llama_backend.DEVICE_TYPE_GPU, 6_000_000_000, 8_000_000_000, "CUDA0"),
            _device(llama_backend.DEVICE_TYPE_GPU, 9_000_000_000, 12_000_000_000, "CUDA1"),
        ],
    )

    assert [d["name"] for d in llama_backend.gpu_devices()] == ["Vulkan0", "CUDA0", "CUDA1"]  # CPU 不算
    assert llama_backend.discrete_gpu_free_bytes() == 9_000_000_000  # 取記憶體最多的獨立顯示卡；內顯不算


def test_an_integrated_gpu_only_machine_has_no_discrete_gpu_memory(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "_query_devices", lambda: [_device(llama_backend.DEVICE_TYPE_IGPU, 7_700_000_000, 8_400_000_000)])

    assert llama_backend.discrete_gpu_free_bytes() == 0


def test_no_gpu_build_means_no_gpu_devices_without_even_asking(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)
    monkeypatch.setattr(llama_backend, "_query_devices", lambda: pytest.fail("CPU 版不該去問裝置"))

    assert llama_backend.gpu_devices() == [] and llama_backend.discrete_gpu_free_bytes() == 0


def test_a_device_query_that_fails_is_treated_as_no_gpu(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)

    def broken():
        raise AttributeError("ggml_backend_dev_count")  # 這個版本的函式庫沒有這些函式

    monkeypatch.setattr(llama_backend, "_query_devices", broken)

    assert llama_backend.gpu_devices() == [] and llama_backend.discrete_gpu_free_bytes() == 0  # 查不到就保守：不自動升到 7B


def test_the_real_device_query_reports_cpu_or_gpu_devices_or_fails_cleanly():
    pytest.importorskip("llama_cpp")

    devices = llama_backend.gpu_devices()  # 這台電腦裝的是 CPU 版：空的；GPU 版：每個都有名稱與記憶體

    assert isinstance(devices, list)
    assert all({"name", "type", "free", "total"} <= set(device) for device in devices)
    assert llama_backend.discrete_gpu_free_bytes() >= 0


def test_missing_llama_cpp_python_is_reported(monkeypatch):
    monkeypatch.setattr(llama_backend, "available", lambda: False)
    instance = OfflineTranslator({"offline": {}})

    with pytest.raises(TranslationError, match="llama-cpp-python"):
        instance._load()


def test_a_broken_model_file_is_reported_not_crashed(tmp_path, monkeypatch):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"not a model")

    class Broken:
        def __init__(self, path, threads, gpu_layers=0):
            raise ValueError("invalid magic")

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", Broken)
    instance = OfflineTranslator({"offline": {"model_dir": str(model.parent)}})

    with pytest.raises(TranslationError, match="qwen.gguf"):
        instance._load()


# ----------------------------------------------------------------------------- GPU 卸載與退回
def _engine_with(monkeypatch, outcomes):
    """依序回應 ``_open`` 的結果（例外就丟出）；回傳 (建好的引擎類別, 被要求的 gpu_layers 紀錄)。"""
    asked = []

    def fake_open(self, threads, context_tokens, gpu_layers):
        asked.append(gpu_layers)
        outcome = outcomes[len(asked) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llama_backend.LlamaCppEngine, "_open", fake_open)
    return asked


def test_the_gpu_is_used_when_it_loads(monkeypatch):
    asked = _engine_with(monkeypatch, [object()])

    engine = llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1)

    assert asked == [-1] and engine.gpu_layers == -1


def test_a_gpu_that_cannot_load_the_model_falls_back_to_the_cpu(monkeypatch):
    asked = _engine_with(monkeypatch, [RuntimeError("CUDA out of memory"), object()])

    engine = llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1)

    assert asked == [-1, 0] and engine.gpu_layers == 0  # 顯示卡記憶體不足：改用 CPU，不是整個翻譯壞掉


def test_a_cpu_failure_is_not_retried(monkeypatch):
    asked = _engine_with(monkeypatch, [ValueError("bad file")])

    with pytest.raises(ValueError, match="bad file"):
        llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=0)

    assert asked == [0]


def test_a_gpu_failure_falls_back_to_the_cpu_unless_that_is_switched_off(monkeypatch):
    asked = _engine_with(monkeypatch, [RuntimeError("out of memory"), object()])
    llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1)
    assert asked == [-1, 0]  # 預設：GPU 載不進去就改用 CPU

    asked = _engine_with(monkeypatch, [RuntimeError("out of memory"), object()])
    with pytest.raises(RuntimeError, match="不改用 CPU.*out of memory"):
        llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1, cpu_fallback=False)
    assert asked == [-1]  # 速度測試在電腦記憶體也不夠時：寧可失敗，也不要把電腦拖垮


def test_the_cpu_fallback_setting_is_only_passed_down_when_it_is_off(tmp_path, monkeypatch):
    model = tmp_path / "m" / "qwen.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    seen = []

    class FakeEngine:
        def __init__(self, path, threads, gpu_layers=0, **kwargs):
            seen.append(kwargs)
            self.gpu_layers = gpu_layers

    monkeypatch.setattr(llama_backend, "available", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", FakeEngine)
    OfflineTranslator({"offline": {"model_dir": str(model.parent)}})._load()
    OfflineTranslator({"offline": {"model_dir": str(model.parent), "cpu_fallback": False}})._load()

    assert seen == [{}, {"cpu_fallback": False}]  # 預設不多傳參數（舊的替身引擎也能用）


def test_gpu_support_is_asked_from_the_library(monkeypatch):
    monkeypatch.setattr(llama_backend, "available", lambda: True)
    fake = types.SimpleNamespace(llama_supports_gpu_offload=lambda: True)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    assert llama_backend.gpu_offload_supported() is True

    fake.llama_supports_gpu_offload = lambda: False  # 預設的 CPU 版
    assert llama_backend.gpu_offload_supported() is False

    def broken():
        raise OSError("cannot load the shared library")

    fake.llama_supports_gpu_offload = broken
    assert llama_backend.gpu_offload_supported() is False  # 查不到就當沒有，不能讓載入失敗

    monkeypatch.setattr(llama_backend, "available", lambda: False)
    assert llama_backend.gpu_offload_supported() is False


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


def test_the_combo_lists_auto_and_every_model_and_defaults_to_auto(qapp, tmp_path, models_present):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    assert dialog.combo_offline_model.count() == len(OFFLINE_MODEL_CHOICES) == 8  # 自動 + Qwen2.5 的 0.5B／1.5B 兩種／3B／7B + Qwen3 的 1.7B／4B
    assert [dialog.combo_offline_model.itemData(i) for i in range(8)] == [
        "auto",
        "qwen2.5-0.5b-instruct-gguf-q8_0",
        "qwen2.5-1.5b-instruct-gguf-q4_k_m",
        "qwen2.5-1.5b-instruct-gguf-q8_0",
        "qwen3-1.7b-gguf-q8_0",
        "qwen2.5-3b-instruct-gguf-q4_k_m",
        "qwen3-4b-gguf-q4_k_m",
        "qwen2.5-7b-instruct-gguf-q4_k_m",
    ]
    assert dialog.combo_offline_model.currentIndex() == 0
    assert dialog.pages.widget(3).widget().isAncestorOf(dialog.combo_offline_model)  # 「翻譯」頁
    dialog.deleteLater()


def test_options_without_a_model_file_are_disabled_with_the_reason(qapp, tmp_path, monkeypatch):
    present = {"qwen2.5-1.5b-instruct-gguf-q4_k_m", "qwen2.5-0.5b-instruct-gguf-q8_0"}
    monkeypatch.setattr(settings_module, "llama_model_available", lambda name, model_dir="": name in present)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))

    enabled = [dialog.combo_offline_model.model().item(i).isEnabled() for i in range(8)]
    assert enabled == [True, True, True, False, False, False, False, False]  # 「自動」永遠可選
    tip = dialog.combo_offline_model.itemData(7, 3)  # ToolTipRole
    assert "fetch_translation_models.py --model qwen2.5-7b-instruct-gguf-q4_k_m" in tip
    assert dialog.combo_offline_model.itemData(2, 3) == ""
    dialog.deleteLater()


def test_the_chosen_model_is_saved_and_shown_again(qapp, tmp_path, models_present):
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    dialog.combo_offline_model.setCurrentIndex(7)
    _commit(dialog)

    saved = AppConfig(tmp_path / "config.json")
    assert saved.get("translation.offline.model_name") == "qwen2.5-7b-instruct-gguf-q4_k_m"
    reopened = SettingsDialog(saved)
    assert reopened.combo_offline_model.currentIndex() == 7
    dialog.deleteLater()
    reopened.deleteLater()


def test_an_untouched_combo_does_not_overwrite_a_custom_model_name(qapp, tmp_path, models_present):
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.offline.model_name", "my-own-finetune-gguf")
    config.save()
    dialog = SettingsDialog(config)
    assert dialog.combo_offline_model.currentIndex() == 0  # 不認得的名稱顯示成「自動」

    _commit(dialog)

    assert AppConfig(tmp_path / "config.json").get("translation.offline.model_name") == "my-own-finetune-gguf"
    dialog.deleteLater()


@pytest.mark.parametrize(("name", "index"), [("auto", 0), ("qwen2.5-0.5b-instruct-gguf-q8_0", 1), ("qwen2.5-1.5b-instruct-gguf-q8_0", 3), ("qwen3-1.7b-gguf-q8_0", 4), ("qwen2.5-3b-instruct-gguf-q4_k_m", 5), ("qwen3-4b-gguf-q4_k_m", 6)])
def test_existing_settings_select_the_matching_item(qapp, tmp_path, models_present, name, index):
    config = AppConfig(tmp_path / "config.json")
    config.set("translation.offline.model_name", name)

    dialog = SettingsDialog(config)

    assert dialog.combo_offline_model.currentIndex() == index
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
