"""這台電腦有哪些 GPU，以及安裝的 llama.cpp 能不能用它們跑離線翻譯模型（設定頁的「運算裝置」選單用）。

兩個來源：

- **Windows 的 DXGI**（``dxgi.dll``）：列出電腦上所有的顯示卡，不管安裝的是哪一版 llama.cpp；
- **ggml 的裝置清單**（``llama_backend.gpu_devices``）：GPU 版 llama.cpp（Vulkan／CUDA）真的能拿來運算的裝置。

兩邊用顯示卡型號對起來：兩邊都有的才能選；只有 DXGI 看得到的（預設安裝的是 CPU 版、驅動程式不支援 Vulkan…）
照樣列出來，標明不能用的原因，讓使用者知道「這台電腦有這張卡，只是現在用不到」。
"""

from __future__ import annotations

import ctypes
import os
import re
from dataclasses import dataclass

from app.logger import get_logger
from services.translation import llama_backend

log = get_logger("translation.gpu")

KIND_DISCRETE = "discrete"  # 獨立顯示卡：有自己的顯示卡記憶體
KIND_INTEGRATED = "integrated"  # 內顯：和 CPU 共用系統記憶體
KIND_UNKNOWN = "unknown"

_VENDOR_MICROSOFT = 0x1414  # Microsoft Basic Render Driver、遠端桌面的虛擬顯示卡
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_DISCRETE_MIN_BYTES = 2 * 1024**3  # DXGI 看不出內顯或獨顯：專用顯示卡記憶體 2 GB 以上當獨立顯示卡（只用在 llama.cpp 看不到的顯示卡）

NOTE_CPU_BUILD = "目前安裝的 llama.cpp 是 CPU 版，不能使用 GPU（要改裝 Vulkan 版，見 requirements-vulkan.txt）"
NOTE_NOT_VISIBLE = "llama.cpp 看不到這顆（驅動程式不支援 Vulkan，或不是能運算的裝置）"


@dataclass(frozen=True)
class GpuEntry:
    name: str  # 設定檔裡存的名稱（``translation.offline.gpu_device``）
    kind: str  # KIND_*
    memory: int  # 專用顯示卡記憶體（位元組）；內顯與不明是 0
    usable: bool  # 安裝的 llama.cpp 能不能用它
    note: str = ""  # 不能用的原因

    @property
    def kind_text(self) -> str:
        return {KIND_DISCRETE: "獨立顯示卡", KIND_INTEGRATED: "內顯"}.get(self.kind, "顯示卡")

    def summary(self) -> str:
        """型號與種類、記憶體（不含能不能用）。"""
        if self.kind == KIND_DISCRETE and self.memory:
            detail = f"{self.kind_text}，{self.memory / 1024**3:.1f} GB 顯示卡記憶體"
        elif self.kind == KIND_INTEGRATED:
            detail = f"{self.kind_text}，與系統共用記憶體"
        else:
            detail = self.kind_text
        return f"{self.name}（{detail}）"

    def describe(self) -> str:
        """一行說明（選單的文字）：型號、種類、記憶體；用不到的加註。"""
        return self.summary() + ("" if self.usable else "　不能用")


@dataclass(frozen=True)
class GpuReport:
    entries: tuple[GpuEntry, ...]
    engine_supports_gpu: bool  # 安裝的 llama.cpp 是不是 GPU 版

    @property
    def usable(self) -> tuple[GpuEntry, ...]:
        return tuple(entry for entry in self.entries if entry.usable)


# ----------------------------------------------------------------------------- Windows：DXGI
class _Guid(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _AdapterDesc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _Luid),
        ("Flags", ctypes.c_uint32),
    ]


# IDXGIFactory1::EnumAdapters1、IDXGIAdapter1::GetDesc1 在各自的虛擬函式表（vtable）裡的位置
_FACTORY_ENUM_ADAPTERS1 = 12
_ADAPTER_GET_DESC1 = 10
_RELEASE = 2


def _dxgi_adapters() -> list[dict]:
    dxgi = ctypes.WinDLL("dxgi")  # type: ignore[attr-defined]  # 只在 Windows 呼叫
    iid = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))  # IDXGIFactory1
    dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
    dxgi.CreateDXGIFactory1.restype = ctypes.c_long
    factory = ctypes.c_void_p()
    result = dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory))
    if result != 0 or not factory.value:
        raise OSError(f"CreateDXGIFactory1 失敗（0x{result & 0xFFFFFFFF:08X}）")

    def method(obj: ctypes.c_void_p, index: int, *argtypes):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(table[index])  # type: ignore[attr-defined]

    adapters: list[dict] = []
    try:
        enum_adapters = method(factory, _FACTORY_ENUM_ADAPTERS1, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):  # 一台電腦不會有這麼多張；上限只是不讓壞掉的驅動程式讓迴圈停不下來
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, index, ctypes.byref(adapter)) != 0:  # DXGI_ERROR_NOT_FOUND：列完了
                break
            try:
                desc = _AdapterDesc1()
                if method(adapter, _ADAPTER_GET_DESC1, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc)) != 0:
                    continue
                adapters.append({
                    "description": desc.Description.strip(),
                    "vendor": int(desc.VendorId),
                    "dedicated": int(desc.DedicatedVideoMemory),
                    "software": bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE),
                })
            finally:
                method(adapter, _RELEASE)(adapter)
    finally:
        method(factory, _RELEASE)(factory)
    return adapters


def windows_adapters() -> list[dict]:
    """Windows 看到的顯示卡：``[{"description", "vendor", "dedicated", "software"}]``（``dedicated`` 是專用顯示卡記憶體，位元組）。

    軟體模擬的（Microsoft Basic Render Driver）與 Microsoft 的虛擬顯示卡不算。非 Windows 或查詢失敗就是空的。"""
    if os.name != "nt":
        return []
    try:
        found = _dxgi_adapters()
    except Exception as exc:  # dxgi.dll 載不進來、驅動程式出錯…：當成查不到，不影響程式
        log.warning("查詢這台電腦的顯示卡失敗：%s", exc)
        return []
    return [a for a in found if a["description"] and not a["software"] and a["vendor"] != _VENDOR_MICROSOFT]


# ----------------------------------------------------------------------------- 對照與整理
def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _same_model(a: str, b: str) -> bool:
    """DXGI 與 Vulkan／CUDA 對同一張卡的稱呼大同小異（「NVIDIA GeForce RTX 3060」與「NVIDIA GeForce RTX 3060 Laptop GPU」）：去掉標點空白後相等，或一個包含另一個。"""
    x, y = _normalized(a), _normalized(b)
    return bool(x) and bool(y) and (x == y or x in y or y in x)


def _adapter_kind(adapter: dict) -> str:
    if adapter["dedicated"] >= _DISCRETE_MIN_BYTES:
        return KIND_DISCRETE
    return KIND_INTEGRATED if adapter["vendor"] == 0x8086 else KIND_UNKNOWN  # Intel 的顯示卡專用記憶體很小的是內顯


def detect_gpus() -> GpuReport:
    """偵測這台電腦的 GPU，並標出安裝的 llama.cpp 能用哪些。順序：Windows 列出的順序，其次是只有 llama.cpp 看得到的。"""
    supported = llama_backend.gpu_offload_supported()
    engine = llama_backend.gpu_devices() if supported else []
    labels = llama_backend.unique_labels(engine)
    remaining = list(zip(labels, engine))
    entries: list[GpuEntry] = []
    for adapter in windows_adapters():
        match = next((item for item in remaining if _same_model(adapter["description"], item[0])), None)
        if match is not None:
            remaining.remove(match)
            label, device = match
            discrete = device["type"] == llama_backend.DEVICE_TYPE_GPU
            memory = (adapter["dedicated"] or device["total"]) if discrete else 0
            entries.append(GpuEntry(label, KIND_DISCRETE if discrete else KIND_INTEGRATED, memory, True))
        else:
            kind = _adapter_kind(adapter)
            memory = adapter["dedicated"] if kind == KIND_DISCRETE else 0
            entries.append(GpuEntry(adapter["description"], kind, memory, False, NOTE_NOT_VISIBLE if supported else NOTE_CPU_BUILD))
    for label, device in remaining:  # 只有 llama.cpp 看得到（不是 Windows、或兩邊的型號寫法差太多）
        discrete = device["type"] == llama_backend.DEVICE_TYPE_GPU
        entries.append(GpuEntry(label, KIND_DISCRETE if discrete else KIND_INTEGRATED, device["total"] if discrete else 0, True))
    return GpuReport(tuple(entries), supported)
