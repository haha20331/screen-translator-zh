"""運算裝置：這台電腦有哪些 GPU、設定頁的「運算裝置」下拉選單、依選擇載入模型（自動／只用 CPU／指定某一顆 GPU）。"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.config import AppConfig
from services.translation import gpu_info, llama_backend, offline_translator, speed_test
from services.translation.gpu_info import NOTE_NOT_VISIBLE, GpuEntry, GpuReport
from services.translation.offline_common import gpu_choice
from services.translation.offline_translator import OfflineTranslator
from ui.settings_dialog import SettingsDialog
from ui.style import apply_style

real_detect_gpus = gpu_info.detect_gpus  # conftest 的夾具會把它換成固定結果；這裡在載入模組時先拿到真正的

GIB = 1024**3
NVIDIA = "NVIDIA GeForce RTX 3060"
INTEL = "Intel(R) Iris(R) Xe Graphics"


def _adapter(description: str, dedicated: int = 0, vendor: int = 0x10DE, software: bool = False) -> dict:
    return {"description": description, "vendor": vendor, "dedicated": dedicated, "software": software}


def _device(description: str, kind: int, free: int, total: int | None = None, handle: int = 1) -> dict:
    return {"handle": handle, "name": "Vulkan0", "description": description, "type": kind, "free": free, "total": total or free}


DISCRETE, INTEGRATED = llama_backend.DEVICE_TYPE_GPU, llama_backend.DEVICE_TYPE_IGPU
REASON = "測試用的原因：沒有 GPU 版"


@pytest.fixture(autouse=True)
def _fixed_reason(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_unavailable_reason", lambda: REASON)


def _machine(monkeypatch, adapters, engine, supported=True):
    monkeypatch.setattr(gpu_info, "windows_adapters", lambda: adapters)
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: supported)
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: engine if supported else [])


# ----------------------------------------------------------------------------- 偵測：這台電腦有哪些 GPU
def test_the_real_windows_query_lists_this_machines_adapters_or_nothing_elsewhere():
    adapters = gpu_info.windows_adapters()

    assert isinstance(adapters, list)
    if os.name != "nt":
        assert adapters == []
    for adapter in adapters:
        assert adapter["description"] and not adapter["software"] and adapter["dedicated"] >= 0


def test_software_and_virtual_adapters_are_not_gpus(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(gpu_info, "_dxgi_adapters", lambda: [
        _adapter("Microsoft Basic Render Driver", vendor=0x1414, software=True),
        _adapter("Microsoft Remote Display Adapter", vendor=0x1414),
        _adapter(NVIDIA, 6 * GIB),
    ])

    assert [a["description"] for a in gpu_info.windows_adapters()] == [NVIDIA]


def test_a_failing_windows_query_means_no_adapters_not_a_crash(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")

    def broken():
        raise OSError("dxgi.dll 載不進來")

    monkeypatch.setattr(gpu_info, "_dxgi_adapters", broken)

    assert gpu_info.windows_adapters() == []


def test_with_the_cpu_build_every_gpu_is_listed_but_none_can_be_used(monkeypatch):
    _machine(monkeypatch, [_adapter(INTEL, 128 * 1024**2, 0x8086), _adapter(NVIDIA, 6 * GIB)], [], supported=False)

    report = real_detect_gpus()

    assert report.engine_supports_gpu is False and report.usable == ()
    assert [(e.name, e.kind, e.usable, e.note) for e in report.entries] == [
        (INTEL, gpu_info.KIND_INTEGRATED, False, REASON),
        (NVIDIA, gpu_info.KIND_DISCRETE, False, REASON),
    ]
    assert report.entries[1].memory == 6 * GIB


def test_with_a_gpu_build_the_cards_llama_cpp_can_see_are_usable_and_keep_windows_order(monkeypatch):
    _machine(
        monkeypatch,
        [_adapter(INTEL, 128 * 1024**2, 0x8086), _adapter(NVIDIA, 6 * GIB)],
        [_device(NVIDIA, DISCRETE, 5 * GIB, 6 * GIB), _device(INTEL, INTEGRATED, 7 * GIB, 8 * GIB)],  # llama.cpp 的順序和 Windows 不同
    )

    report = real_detect_gpus()

    assert [(e.name, e.kind, e.usable) for e in report.entries] == [
        (INTEL, gpu_info.KIND_INTEGRATED, True),
        (NVIDIA, gpu_info.KIND_DISCRETE, True),
    ]
    assert report.entries[0].memory == 0 and report.entries[1].memory == 6 * GIB  # 內顯沒有自己的顯示卡記憶體
    assert report.usable == report.entries and report.engine_supports_gpu


def test_a_card_that_llama_cpp_cannot_see_is_listed_with_the_reason(monkeypatch):
    _machine(monkeypatch, [_adapter(NVIDIA, 6 * GIB), _adapter("AMD Radeon RX 6600", 8 * GIB, 0x1002)], [_device(NVIDIA, DISCRETE, 5 * GIB)])

    report = real_detect_gpus()

    amd = report.entries[1]
    assert (amd.name, amd.usable, amd.note) == ("AMD Radeon RX 6600", False, NOTE_NOT_VISIBLE)
    assert [e.name for e in report.usable] == [NVIDIA]


def test_windows_and_llama_cpp_may_spell_the_model_a_little_differently(monkeypatch):
    _machine(monkeypatch, [_adapter("NVIDIA GeForce RTX 3060 Laptop GPU", 6 * GIB)], [_device("NVIDIA GeForce RTX 3060", DISCRETE, 5 * GIB)])

    report = real_detect_gpus()

    assert len(report.entries) == 1 and report.entries[0].usable
    assert report.entries[0].name == "NVIDIA GeForce RTX 3060"  # 設定裡存 llama.cpp 的名稱，載入時才找得到


def test_without_windows_the_cards_llama_cpp_sees_are_still_listed(monkeypatch):
    _machine(monkeypatch, [], [_device(NVIDIA, DISCRETE, 5 * GIB, 6 * GIB)])

    report = real_detect_gpus()

    assert [(e.name, e.usable, e.memory) for e in report.entries] == [(NVIDIA, True, 6 * GIB)]


def test_two_identical_cards_get_different_names(monkeypatch):
    _machine(
        monkeypatch,
        [_adapter(NVIDIA, 6 * GIB), _adapter(NVIDIA, 6 * GIB)],
        [_device(NVIDIA, DISCRETE, 5 * GIB, handle=1), _device(NVIDIA, DISCRETE, 5 * GIB, handle=2)],
    )

    assert [e.name for e in real_detect_gpus().entries] == [NVIDIA, f"{NVIDIA} #2"]
    devices = [_device(NVIDIA, DISCRETE, 1, handle=1), _device(NVIDIA, DISCRETE, 1, handle=2)]
    assert llama_backend.find_gpu_device(f"{NVIDIA} #2", devices)["handle"] == 2
    assert llama_backend.find_gpu_device("不存在的顯示卡", devices) is None


def test_a_machine_with_no_gpu_at_all_has_an_empty_report(monkeypatch):
    _machine(monkeypatch, [], [], supported=False)

    report = real_detect_gpus()

    assert report.entries == () and report.usable == ()


def test_each_entry_describes_itself_in_one_line():
    assert GpuEntry(NVIDIA, gpu_info.KIND_DISCRETE, 6 * GIB, True).describe() == f"{NVIDIA}（獨立顯示卡，6.0 GB 顯示卡記憶體）"
    assert GpuEntry(INTEL, gpu_info.KIND_INTEGRATED, 0, True).describe() == f"{INTEL}（內顯，與系統共用記憶體）"
    assert GpuEntry("Mystery", gpu_info.KIND_UNKNOWN, 0, False, "x").describe() == "Mystery（顯示卡）　不能用"


# ----------------------------------------------------------------------------- llama.cpp：指定裝置
def test_choosing_which_device_a_model_may_use_only_lasts_for_that_load():
    binding = types.ModuleType("llama_cpp.llama_cpp")

    class Params:
        devices = None

    binding.llama_model_default_params = lambda: Params()
    package = types.ModuleType("llama_cpp")
    package.llama_cpp = binding
    original = binding.llama_model_default_params
    saved = {name: sys.modules.get(name) for name in ("llama_cpp", "llama_cpp.llama_cpp")}
    sys.modules["llama_cpp"], sys.modules["llama_cpp.llama_cpp"] = package, binding
    try:
        with llama_backend._only_device(0xABCD):
            params = binding.llama_model_default_params()
            listed = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
            assert listed[0] == 0xABCD and listed[1] is None  # 以 NULL 結尾的裝置清單，只有這一個
        assert binding.llama_model_default_params is original and Params().devices is None
        with pytest.raises(RuntimeError):
            with llama_backend._only_device(1):
                raise RuntimeError("載入失敗")
        assert binding.llama_model_default_params is original  # 載入失敗也要還原
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_the_real_library_accepts_a_device_list_when_it_is_installed():
    pytest.importorskip("llama_cpp")
    import llama_cpp.llama_cpp as binding

    with llama_backend._only_device(0x1234):
        params = binding.llama_model_default_params()
        listed = ctypes.cast(params.devices, ctypes.POINTER(ctypes.c_void_p))
        assert listed[0] == 0x1234 and listed[1] is None
    assert not binding.llama_model_default_params().devices


class _OpenRecorder:
    """換掉 ``LlamaCppEngine._open``：記下載入時指定了哪個裝置，不載入任何模型。"""

    def __init__(self, monkeypatch, fail_on_gpu=False):
        self.opened: list[tuple[int, int]] = []  # (gpu_layers, 指定的裝置 handle)

        def fake_open(engine, threads, context_tokens, gpu_layers):
            self.opened.append((gpu_layers, engine._gpu_handle))
            if gpu_layers and fail_on_gpu:
                raise RuntimeError("顯示卡記憶體不足")
            return object()

        monkeypatch.setattr(llama_backend.LlamaCppEngine, "_open", fake_open)
        monkeypatch.setattr(llama_backend.LlamaCppEngine, "_init_token_filter", lambda engine: None)


def test_a_named_gpu_is_the_only_device_the_model_is_loaded_on(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(NVIDIA, DISCRETE, 5 * GIB, handle=11), _device(INTEL, INTEGRATED, 7 * GIB, handle=22)])
    recorder = _OpenRecorder(monkeypatch)

    engine = llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1, gpu_device=INTEL)

    assert recorder.opened == [(-1, 22)] and engine.device_name == INTEL and engine.gpu_layers == -1


def test_no_named_gpu_leaves_the_choice_to_llama_cpp(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: pytest.fail("沒指定就不必查裝置"))
    recorder = _OpenRecorder(monkeypatch)

    engine = llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1)

    assert recorder.opened == [(-1, 0)] and engine.device_name == ""


def test_a_named_gpu_that_is_gone_falls_back_to_the_automatic_choice(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(INTEL, INTEGRATED, 7 * GIB, handle=22)])
    recorder = _OpenRecorder(monkeypatch)

    engine = llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1, gpu_device=NVIDIA)

    assert recorder.opened == [(-1, 0)] and engine.device_name == "" and engine.gpu_layers == -1


def test_a_named_gpu_that_cannot_hold_the_model_falls_back_to_the_cpu(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(NVIDIA, DISCRETE, 1 * GIB, handle=11)])
    recorder = _OpenRecorder(monkeypatch, fail_on_gpu=True)

    engine = llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=-1, gpu_device=NVIDIA)

    assert recorder.opened == [(-1, 11), (0, 11)]  # 第二次是 CPU：gpu_layers=0 時不會指定裝置（見 _open）
    assert engine.gpu_layers == 0 and engine.device_name == ""


def test_cpu_only_never_asks_which_gpus_exist(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: pytest.fail("只用 CPU 不必查 GPU"))
    recorder = _OpenRecorder(monkeypatch)

    llama_backend.LlamaCppEngine("m.gguf", 6, gpu_layers=0, gpu_device=NVIDIA)

    assert recorder.opened == [(0, 0)]


def test_gpu_memory_for_the_large_model_depends_on_the_chosen_device(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(NVIDIA, DISCRETE, 5 * GIB), _device(INTEL, INTEGRATED, 7 * GIB)])

    assert llama_backend.discrete_gpu_free_bytes() == 5 * GIB
    assert llama_backend.discrete_gpu_free_bytes("auto") == 5 * GIB
    assert llama_backend.discrete_gpu_free_bytes("cpu") == 0
    assert llama_backend.discrete_gpu_free_bytes(NVIDIA) == 5 * GIB
    assert llama_backend.discrete_gpu_free_bytes(INTEL) == 0  # 內顯沒有自己的顯示卡記憶體
    assert llama_backend.discrete_gpu_free_bytes("找不到的顯示卡") == 0


# ----------------------------------------------------------------------------- 設定值與翻譯器
@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, "auto"),
        (None, "auto"),
        ({"gpu_layers": -1}, "auto"),
        ({"gpu_layers": 0}, "cpu"),  # 舊版的「只用 CPU」
        ({"gpu_layers": "0"}, "cpu"),
        ({"gpu_layers": "x"}, "auto"),
        ({"gpu_device": "cpu"}, "cpu"),
        ({"gpu_device": " CPU "}, "cpu"),
        ({"gpu_device": "Auto"}, "auto"),
        ({"gpu_device": ""}, "auto"),
        ({"gpu_device": None}, "auto"),
        ({"gpu_device": NVIDIA}, NVIDIA),
        ({"gpu_device": NVIDIA, "gpu_layers": 0}, NVIDIA),  # 有 gpu_device 就以它為準，舊的 gpu_layers 不再表示只用 CPU
    ],
)
def test_the_chosen_device_is_read_from_the_settings(settings, expected):
    assert gpu_choice(settings) == expected


def test_a_new_config_defaults_to_the_automatic_choice(tmp_path):
    config = AppConfig(tmp_path / "config.json")

    assert config.get("translation.offline.gpu_device") == "auto" and config.get("translation.offline.gpu_layers") == -1


def test_an_old_config_that_meant_cpu_only_becomes_the_cpu_choice(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"translation": {"offline": {"gpu_layers": 0}}}), encoding="utf-8")

    config = AppConfig(path)

    assert config.get("translation.offline.gpu_device") == "cpu" and config.get("translation.offline.gpu_layers") == -1


def test_a_chosen_device_in_the_config_is_kept_and_a_broken_one_becomes_automatic(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"translation": {"offline": {"gpu_device": NVIDIA, "gpu_layers": 20}}}), encoding="utf-8")
    config = AppConfig(path)
    assert config.get("translation.offline.gpu_device") == NVIDIA and config.get("translation.offline.gpu_layers") == 20

    for broken in (None, 7, "  ", ["x"]):
        path.write_text(json.dumps({"translation": {"offline": {"gpu_device": broken}}}), encoding="utf-8")
        assert AppConfig(path).get("translation.offline.gpu_device") == "auto"


def test_the_old_device_setting_of_the_ctranslate2_days_is_still_dropped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"translation": {"offline": {"device": "cuda"}}}), encoding="utf-8")

    offline = AppConfig(path).section("translation", "offline")

    assert "device" not in offline and offline["gpu_device"] == "auto"


def _translator(**offline) -> OfflineTranslator:
    return OfflineTranslator({"offline": offline})


def test_the_cpu_choice_means_zero_gpu_layers_even_with_a_gpu_build(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)

    assert _translator(gpu_device="cpu").gpu_layers() == 0
    assert _translator(gpu_device="auto").gpu_layers() == -1
    assert _translator(gpu_device=NVIDIA).gpu_layers() == -1
    assert _translator(gpu_device=NVIDIA, gpu_layers=0).gpu_layers() == -1  # 舊的 0 不能讓選了 GPU 的人還是用 CPU
    assert _translator(gpu_device=NVIDIA, gpu_layers=12).gpu_layers() == 12
    assert _translator(gpu_layers=0).gpu_layers() == 0  # 沒有 gpu_device 的舊寫法


def test_a_cpu_build_never_uses_gpu_layers(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)

    assert _translator(gpu_device=NVIDIA).gpu_layers() == 0


class _LoadedEngine:
    created: list[dict] = []

    def __init__(self, path, threads, gpu_layers=0, **kwargs):
        self.gpu_layers = gpu_layers
        self.device_name = kwargs.get("gpu_device", "") if gpu_layers else ""
        type(self).created.append({"gpu_layers": gpu_layers, **kwargs})


def _load(tmp_path, monkeypatch, **offline):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"0")
    _LoadedEngine.created = []
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "LlamaCppEngine", _LoadedEngine)
    monkeypatch.setattr(OfflineTranslator, "llama_model_path", lambda self: gguf)
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))
    instance = _translator(**offline)
    instance._load()
    return instance


def test_only_a_named_gpu_is_passed_to_the_engine(tmp_path, monkeypatch):
    named = _load(tmp_path, monkeypatch, gpu_device=NVIDIA)
    assert _LoadedEngine.created == [{"gpu_layers": -1, "gpu_device": NVIDIA}]
    assert named.device == "gpu" and named.device_name == NVIDIA

    auto = _load(tmp_path, monkeypatch, gpu_device="auto")
    assert _LoadedEngine.created == [{"gpu_layers": -1}] and auto.device == "gpu" and auto.device_name == ""

    cpu = _load(tmp_path, monkeypatch, gpu_device="cpu")
    assert _LoadedEngine.created == [{"gpu_layers": 0}] and cpu.device == "cpu" and cpu.device_name == ""


def test_auto_picks_the_large_model_only_when_the_chosen_device_can_hold_it(monkeypatch, tmp_path):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(NVIDIA, DISCRETE, 8 * GIB), _device(INTEL, INTEGRATED, 7 * GIB)])
    monkeypatch.setattr(offline_translator, "llama_model_available", lambda name, model_dir="": True)

    def chosen(**offline):
        return _translator(model_name="auto", **offline).resolved_model_name()

    assert chosen(gpu_device="auto") == offline_translator.LARGE_MODEL_NAME
    assert chosen(gpu_device=NVIDIA) == offline_translator.LARGE_MODEL_NAME
    assert chosen(gpu_device=INTEL) == offline_translator.DEFAULT_MODEL_NAME  # 內顯放不下 7B
    assert chosen(gpu_device="cpu") == offline_translator.DEFAULT_MODEL_NAME


def test_the_speed_test_follows_the_chosen_device(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(NVIDIA, DISCRETE, 5 * GIB), _device(INTEL, INTEGRATED, 7 * GIB)])

    def section(**offline):
        return {"offline": offline}

    assert speed_test.gpu_requested(section(gpu_device="auto")) is True
    assert speed_test.gpu_requested(section(gpu_device=INTEL)) is True
    assert speed_test.gpu_requested(section(gpu_device="cpu")) is False
    assert speed_test.gpu_requested(section(gpu_layers=0)) is False
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: False)
    assert speed_test.gpu_requested(section(gpu_device=NVIDIA)) is False


def test_the_speed_test_does_not_count_an_integrated_gpu_as_memory_of_its_own(monkeypatch):
    monkeypatch.setattr(llama_backend, "gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llama_backend, "gpu_devices", lambda: [_device(NVIDIA, DISCRETE, 5 * GIB), _device(INTEL, INTEGRATED, 7 * GIB)])
    results = speed_test.run_speed_test(
        {"offline": {"gpu_device": INTEL}}, [("m", "M")],
        make_translator=lambda name: pytest.fail("記憶體不夠，不該載入"), memory_available=lambda: 1, sizes=lambda name: 3 * GIB,
    )
    assert results[0]["status"] == speed_test.STATUS_MEMORY  # 內顯要吃系統記憶體：照樣檢查

    class Quick:
        device, device_name, config = "gpu", NVIDIA, {}

        def warmup(self):
            pass

        def translate(self, texts, source, target):
            return texts

        def close(self):
            pass

    results = speed_test.run_speed_test(
        {"offline": {"gpu_device": NVIDIA}}, [("m", "M")],
        make_translator=lambda name: Quick(), memory_available=lambda: 1, sizes=lambda name: 3 * GIB, rounds=1,
    )
    assert results[0]["status"] == speed_test.STATUS_OK  # 獨立顯示卡放在自己的記憶體，不看系統記憶體
    assert results[0]["device"] == "gpu" and results[0]["gpu_name"] == NVIDIA


# ----------------------------------------------------------------------------- 設定頁
@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    apply_style(app)
    return app


def _report(*entries, supported=True) -> GpuReport:
    return GpuReport(tuple(entries), supported)


TWO_GPUS = _report(
    GpuEntry(INTEL, gpu_info.KIND_INTEGRATED, 0, True),
    GpuEntry("AMD Radeon RX 6600", gpu_info.KIND_DISCRETE, 8 * GIB, False, NOTE_NOT_VISIBLE),
)


def _dialog(qapp, monkeypatch, tmp_path, report, config_data=None):
    monkeypatch.setattr(gpu_info, "detect_gpus", lambda: report)
    path = tmp_path / "config.json"
    if config_data is not None:
        path.write_text(json.dumps(config_data), encoding="utf-8")
    config = AppConfig(path)
    dialog = SettingsDialog(config)
    _wait_for_scan(qapp, dialog)
    return dialog, config


def _wait_for_scan(qapp, dialog):
    deadline = time.time() + 10
    while dialog._gpu_scanning and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not dialog._gpu_scanning, "GPU 偵測沒有結束"


def _device_items(dialog):
    combo = dialog.combo_device
    return [(combo.itemData(i), combo.model().item(i).isEnabled()) for i in range(combo.count())]


def _save(dialog, tmp_path):
    dialog._commit_accept(dialog._validated_hotkeys())
    dialog._save_close_timer.stop()
    return json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["translation"]["offline"]


def test_the_menu_lists_auto_cpu_and_every_gpu_of_the_machine(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, TWO_GPUS)

    assert _device_items(dialog) == [("auto", True), ("cpu", True), (INTEL, True), ("AMD Radeon RX 6600", False)]  # 用不到的那顆停用
    assert dialog.combo_device.currentData() == "auto"
    text = dialog.lbl_device_status.text()
    assert "偵測到 2 顆顯示卡" in text and INTEL in text and "AMD Radeon RX 6600" in text and NOTE_NOT_VISIBLE in text
    assert dialog.btn_rescan_gpu.isEnabled()
    dialog.deleteLater()


def test_with_the_cpu_build_the_gpus_are_shown_but_cannot_be_chosen(qapp, monkeypatch, tmp_path):
    report = _report(GpuEntry(NVIDIA, gpu_info.KIND_DISCRETE, 6 * GIB, False, REASON), supported=False)
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, report)

    assert _device_items(dialog) == [("auto", True), ("cpu", True), (NVIDIA, False)]
    text = dialog.lbl_device_status.text()
    assert NVIDIA in text and REASON in text and "翻譯只能用 CPU" in text
    dialog.deleteLater()


def test_a_machine_without_a_gpu_says_so(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, _report(supported=False))

    assert _device_items(dialog) == [("auto", True), ("cpu", True)]
    assert "沒有偵測到顯示卡" in dialog.lbl_device_status.text()
    dialog.deleteLater()


def test_picking_a_gpu_is_saved_and_the_old_cpu_only_zero_is_cleared(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, TWO_GPUS, {"translation": {"offline": {"gpu_layers": 0}}})
    assert dialog.combo_device.currentData() == "cpu"  # 舊設定檔的「只用 CPU」顯示成這個選項

    dialog.combo_device.setCurrentIndex(dialog.combo_device.findData(INTEL))
    saved = _save(dialog, tmp_path)

    assert saved["gpu_device"] == INTEL and saved["gpu_layers"] == -1
    dialog.deleteLater()


def test_choosing_cpu_only_is_saved(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, TWO_GPUS)

    dialog.combo_device.setCurrentIndex(dialog.combo_device.findData("cpu"))

    assert "只用 CPU" in dialog.lbl_device_status.text()
    assert _save(dialog, tmp_path)["gpu_device"] == "cpu"
    dialog.deleteLater()


def test_a_saved_gpu_is_selected_again_and_an_untouched_menu_leaves_the_setting_alone(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, TWO_GPUS, {"translation": {"offline": {"gpu_device": INTEL}}})
    assert dialog.combo_device.currentData() == INTEL

    assert _save(dialog, tmp_path)["gpu_device"] == INTEL
    dialog.deleteLater()


def test_a_saved_gpu_that_is_no_longer_there_is_kept_not_wiped(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, _report(supported=False), {"translation": {"offline": {"gpu_device": NVIDIA}}})

    assert dialog.combo_device.currentData() == NVIDIA and "目前找不到這顆" in dialog.combo_device.currentText()
    assert _save(dialog, tmp_path)["gpu_device"] == NVIDIA  # 外接顯示卡暫時拔掉，不該悄悄改掉使用者的選擇
    dialog.deleteLater()


def test_rescanning_picks_up_a_newly_available_gpu(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, _report(supported=False))
    assert [data for data, _ in _device_items(dialog)] == ["auto", "cpu"]

    monkeypatch.setattr(gpu_info, "detect_gpus", lambda: TWO_GPUS)
    dialog.btn_rescan_gpu.click()
    assert not dialog.btn_rescan_gpu.isEnabled() and "正在偵測" in dialog.lbl_device_status.text()
    _wait_for_scan(qapp, dialog)

    assert [data for data, _ in _device_items(dialog)] == ["auto", "cpu", INTEL, "AMD Radeon RX 6600"]
    dialog.deleteLater()


def test_a_failing_scan_leaves_the_dialog_usable(qapp, monkeypatch, tmp_path):
    def broken():
        raise RuntimeError("Vulkan 初始化失敗")

    monkeypatch.setattr(gpu_info, "detect_gpus", broken)
    dialog = SettingsDialog(AppConfig(tmp_path / "config.json"))
    _wait_for_scan(qapp, dialog)

    assert [data for data, _ in _device_items(dialog)] == ["auto", "cpu"]
    assert "無法偵測" in dialog.lbl_device_status.text() and dialog.btn_rescan_gpu.isEnabled()
    assert _save(dialog, tmp_path)["gpu_device"] == "auto"
    dialog.deleteLater()


def test_the_speed_test_measures_the_device_shown_in_the_menu(qapp, monkeypatch, tmp_path):
    dialog, _ = _dialog(qapp, monkeypatch, tmp_path, TWO_GPUS)
    dialog.combo_device.setCurrentIndex(dialog.combo_device.findData(INTEL))  # 選了還沒儲存
    started: list[dict] = []
    monkeypatch.setattr("ui.settings_dialog._speed_test_thread", lambda section, models, cancel, messages, store: started.append(section))
    monkeypatch.setattr(OfflineTranslator, "availability_reason", classmethod(lambda cls: None))
    monkeypatch.setattr("ui.settings_dialog.llama_model_available", lambda name, model_dir="": name == "qwen2.5-1.5b-instruct-gguf-q4_k_m")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    dialog._on_speed_test()
    dialog._speed_thread.join(10)

    assert started and started[0]["offline"]["gpu_device"] == INTEL
    dialog._speed_timer.stop()
    dialog.deleteLater()
