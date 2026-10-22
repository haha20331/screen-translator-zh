"""離線翻譯模型的速度測試：用 3 句測資，在目前這台電腦上量每個已下載的模型翻一句要多久。

量的是「熱機後」的時間：每個模型先載入並預熱（分類詞彙表、算好日文與英文的固定前綴），再整批翻一輪不計時，
之後才正式計時，所以不含載入時間（另外列出）與固定前綴（系統提示＋示範）的計算，只有每句字幕真正要付的
預填（待翻的那句）加解碼，和即時監控時實際遇到的一樣。每個模型逐句計時 ``TIMED_ROUNDS`` 輪，取中位數。

結果存在設定資料夾的 ``speed_test.json``，下次開設定還看得到；每個模型各存各的，只測了其中幾個也不會蓋掉其他的。
"""

from __future__ import annotations

import copy
import ctypes
import datetime
import gc
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app.logger import get_logger
from services.translation import llama_backend
from services.translation.model_names import display_name
from services.translation.offline_common import GPU_CPU, default_threads, gpu_choice, model_search_paths

log = get_logger("translation.speedtest")

# 測資：(名稱, 原文)。和 compare_translation_models.py 的句子一樣是自編的，不含有版權的歌詞
SPEED_TEST_SENTENCES: tuple[tuple[str, str], ...] = (
    ("日文一般", "次のステージに進みますか？"),
    ("日文歌詞", "夜空に浮かぶ星を数えて"),
    ("英文", "I will be there in five minutes"),
)
TIMED_ROUNDS = 3  # 預熱之後正式計時幾輪（整批 3 句輪流翻，每句取中位數；電腦忙的時候單次量測會差很多，三輪的中位數比較穩）
FILE_VERSION = 1

STATUS_OK = "ok"
STATUS_MISSING = "missing"  # 沒下載這個模型
STATUS_MEMORY = "memory"  # 可用記憶體不夠載入，略過
STATUS_ERROR = "error"


# ----------------------------------------------------------------------------- 電腦與模型的資訊
def available_memory_bytes() -> int | None:
    """目前可用的實體記憶體（位元組）；問不到回傳 None（那就不略過任何模型）。"""
    try:
        if os.name == "nt":

            class Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = Status()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:  # 沒有這個介面、權限等：不知道就不略過
        return None
    return None


def model_bytes(name: str, model_dir: str = "") -> int | None:
    """模型檔的大小（分片的模型加總整個資料夾的 .gguf）；找不到模型回傳 None。"""
    for directory in model_search_paths(name, model_dir):
        found = llama_backend.find_gguf(directory)
        if found is not None:
            return sum(path.stat().st_size for path in directory.glob("*.gguf"))
    return None


def memory_needed(size: int) -> int:
    """載入這個大小的模型大約要多少記憶體：模型檔 ×1.15 再加 0.5 GB（KV 快取、詞彙表分類與執行時的緩衝）。"""
    return int(size * 1.15) + 500_000_000


def describe_machine() -> dict[str, Any]:
    try:
        import llama_cpp

        engine = f"llama-cpp-python {llama_cpp.__version__}"
    except Exception:
        engine = "llama-cpp-python"
    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count() or 0,
        "system": f"{platform.system()} {platform.release()}",
        "engine": engine,
    }


def short_label(name: str) -> str:
    """qwen2.5-3b-instruct-gguf-q4_k_m → 3B Q4_K_M；qwen3-4b-gguf-q4_k_m → Qwen3 4B Q4_K_M（表格用的短名稱）。"""
    return display_name(name)


# ----------------------------------------------------------------------------- 存檔
class SpeedTestStore:
    """``speed_test.json``：每個模型最近一次的測試結果。壞掉的檔案當成沒有紀錄。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        empty: dict[str, Any] = {"version": FILE_VERSION, "machine": {}, "models": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return empty
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            return empty
        data["models"] = {name: entry for name, entry in data["models"].items() if isinstance(entry, dict)}
        data.setdefault("machine", {})
        data.setdefault("version", FILE_VERSION)
        return data

    def save_result(self, result: dict[str, Any], machine: dict[str, Any] | None = None) -> None:
        """把一個模型的結果存進去（只換這個模型，其他模型的紀錄保留）。寫入失敗只記錄，不影響測試。"""
        data = self.load()
        data["version"] = FILE_VERSION
        data["models"][result["model"]] = result
        data["sentences"] = [{"name": name, "text": text} for name, text in SPEED_TEST_SENTENCES]
        if machine:
            data["machine"] = machine
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("無法儲存速度測試結果：%s", exc)


# ----------------------------------------------------------------------------- 測試
def _section_for(translation_section: dict, name: str) -> dict:
    """測試用的翻譯設定：沿用使用者的執行緒數與其他選項，但關掉上下文與歌詞模式（每個模型條件一致），指定要測的模型。"""
    section = copy.deepcopy(translation_section or {})
    section["use_context"] = False
    section["lyrics_mode"] = False
    offline = section.setdefault("offline", {})
    offline["model_name"] = name
    offline["model_dir"] = ""  # 逐一測標準位置下的各個模型；自訂資料夾裡只有一個模型，名稱對不上
    return section


def gpu_requested(translation_section: dict) -> bool:
    """這份設定載入模型時會不會用 GPU：安裝的 llama-cpp-python 要有 GPU 版，而且運算裝置不是「只用 CPU」
    （預設自動 = 有 GPU 就用；預設安裝的 CPU 版沒有 GPU 支援，永遠是 False）。"""
    if not llama_backend.gpu_offload_supported():
        return False
    offline = (translation_section or {}).get("offline", {}) or {}
    return gpu_choice(offline) != GPU_CPU


def _default_translator(translation_section: dict, name: str, cpu_fallback: bool = True):
    import services.translation.offline_translator  # noqa: F401  （確保已註冊）
    from services.translation.factory import create_translator

    section = _section_for(translation_section, name)
    if not cpu_fallback:
        section["offline"]["cpu_fallback"] = False
    return create_translator("offline", section, cache=None)


def _release(translator) -> None:
    close = getattr(translator, "close", None)
    try:
        if callable(close):
            close()
    except Exception:
        log.exception("釋放測試用的模型失敗")
    gc.collect()


def run_speed_test(
    translation_section: dict,
    models: Sequence[tuple[str, str]],
    *,
    progress: Callable[[str], None] | None = None,
    on_start: Callable[[str], None] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    make_translator: Callable[[str], Any] | None = None,
    memory_available: Callable[[], int | None] = available_memory_bytes,
    sizes: Callable[[str], int | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime.datetime] = datetime.datetime.now,
    rounds: int = TIMED_ROUNDS,
    gpu: bool | None = None,
    discrete: bool | None = None,
) -> list[dict[str, Any]]:
    """依序測 ``models``（``(名稱, 顯示名稱)``）。開始測一個模型時呼叫 ``on_start``，測完（或略過、失敗）就呼叫 ``on_result``（讓畫面與存檔即時更新）；
    ``cancelled()`` 變 True 就在下一個模型（或下一句）之前停止。回傳已完成的結果（含略過與失敗的）。"""
    say = progress or (lambda _text: None)
    fallback = {"ok": True}  # GPU 載入失敗時能不能改用 CPU（電腦的記憶體也載不進去就不能）
    factory = make_translator or (lambda name: _default_translator(translation_section, name, fallback["ok"]))
    use_gpu = gpu_requested(translation_section) if gpu is None else gpu
    # 只有獨立顯示卡有自己的記憶體；內顯和 CPU 共用系統記憶體，載入模型一樣要吃系統記憶體，記憶體檢查照做
    choice = gpu_choice((translation_section or {}).get("offline", {}))
    own_memory = use_gpu and ((llama_backend.discrete_gpu_free_bytes(choice) > 0) if discrete is None else discrete)
    size_of = sizes or model_bytes
    results: list[dict[str, Any]] = []
    for name, label in models:
        if cancelled():
            break
        result: dict[str, Any] = {"model": name, "tested_at": now().isoformat(timespec="seconds")}
        size = size_of(name)
        if size is None:
            result.update(status=STATUS_MISSING)
        else:
            free = memory_available()
            need = memory_needed(size)
            fits = free is None or free >= need
            if not fits and not own_memory:
                result.update(
                    status=STATUS_MEMORY,
                    message=f"可用 {free / 1e9:.1f} GB，需要約 {need / 1e9:.1f} GB",
                )
            else:
                # 獨立顯示卡：模型放在顯示卡記憶體，不看系統記憶體；但 GPU 載不進去、要退回 CPU 的話，系統記憶體不夠就不退了
                fallback["ok"] = fits
                if on_start is not None:
                    on_start(name)
                result.update(_measure(name, label, factory, say, cancelled, clock, rounds, use_gpu))
        if result.get("status") == "cancelled":
            break
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def _measure(name: str, label: str, factory, say, cancelled, clock, rounds: int, gpu_wanted: bool = False) -> dict[str, Any]:
    translator = None
    try:
        say(f"{label}：載入並預熱中…")
        started = clock()
        translator = factory(name)
        translator.warmup()  # 載入模型、分類詞彙表、算好日文與英文的固定前綴
        load_seconds = clock() - started
        sentences = [text for _name, text in SPEED_TEST_SENTENCES]
        # 先整批翻一輪不計時（把剩下的一次性成本付掉），之後每一輪每句的提示詞都和前一句不同，和真實的字幕一樣
        times: list[list[float]] = [[] for _ in sentences]
        for round_index in range(1 + max(1, rounds)):
            for index, text in enumerate(sentences):
                if cancelled():
                    return {"status": "cancelled"}
                if round_index == 0:
                    say(f"{label}：熱機中…（不計時，第 {index + 1}／{len(sentences)} 句）")
                else:
                    say(f"{label}：測試中…（第 {round_index}／{max(1, rounds)} 輪，第 {index + 1}／{len(sentences)} 句）")
                begin = clock()
                translator.translate([text], "auto", "zh-hant")
                elapsed = (clock() - begin) * 1000
                if round_index > 0:
                    times[index].append(elapsed)
        per_sentence = {name_: round(statistics.median(values)) for (name_, _text), values in zip(SPEED_TEST_SENTENCES, times)}
        offline_settings = (getattr(translator, "config", None) or {}).get("offline", {})
        threads = default_threads(int(offline_settings.get("threads", 0) or 0))
        measured: dict[str, Any] = {
            "status": STATUS_OK,
            "ms": per_sentence,
            "avg_ms": round(statistics.mean(per_sentence.values())),
            "load_s": round(load_seconds, 1),
            "threads": threads,
        }
        device = str(getattr(translator, "device", "") or "")
        if device:  # 實際用的裝置：要求用 GPU 卻載入失敗退回 CPU 時，數字是 CPU 的，要讓人看得出來
            measured["device"] = "gpu" if device == "gpu" and gpu_wanted else "cpu"
            measured["gpu_fallback"] = bool(gpu_wanted and measured["device"] == "cpu")
            gpu_name = str(getattr(translator, "device_name", "") or "")
            if measured["device"] == "gpu" and gpu_name:
                measured["gpu_name"] = gpu_name  # 指定了哪一顆 GPU：不同顆的速度差很多，結果要記下是哪一顆量的
        return measured
    except Exception as exc:  # 檔案損毀、記憶體不足、函式庫載入失敗…：記下來，繼續測下一個
        log.warning("速度測試 %s 失敗：%s", name, exc)
        return {"status": STATUS_ERROR, "message": str(exc)[:200]}
    finally:
        if translator is not None:
            _release(translator)
