"""llama.cpp 推論後端：載入 GGUF 模型，對同一份 ChatML 提示詞做貪婪解碼。

離線翻譯（標準）原本用 CTranslate2（``offline_translator.py``）；這裡提供另一個可選的引擎。
兩者共用同一份提示詞（系統提示、示範、前文、待翻句）與後處理，差別只在「怎麼把提示詞變成譯文」：

- CTranslate2：自己做分詞，批次輸入，固定前綴用 ``static_prompt`` 快取。
- llama.cpp：直接吃文字（特殊記號 ``<|im_start|>`` 由它自己解析），一次一句；
  llama-cpp-python 會自動重用與「上一次」提示詞相同的前綴（KV cache）。固定前綴（系統提示 + 示範）
  換語言或歌詞模式時就不同了、要重算約 1 秒，所以每種固定前綴的 KV 快取各存一份（``prepare``），換回來只要毫秒。

模型是官方釋出的 GGUF（``scripts/fetch_translation_models.py --model qwen2.5-1.5b-instruct-gguf-q4_k_m``）。
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from collections import OrderedDict
from pathlib import Path

from app.logger import get_logger
from services.translation.token_filter import TokenFilter

log = get_logger("translation.llama")

END_TOKEN = "<|im_end|>"
# 提示詞（固定前綴約 300 個 token）+ 前文 + 待翻句 + 最多 160 個新 token，2048 綽綽有餘
DEFAULT_CONTEXT_TOKENS = 2048
# 最多存幾種固定前綴的 KV 快取（語言 ×歌詞模式 ×擬聲詞略過，實際常用的只有 1～3 種；每份約 9 MB）
MAX_PREFIX_STATES = 6


def available() -> bool:
    """有沒有安裝 llama-cpp-python（pip install llama-cpp-python）。"""
    return importlib.util.find_spec("llama_cpp") is not None


def gpu_offload_supported() -> bool:
    """安裝的 llama-cpp-python 是不是有 GPU 支援的版本（CUDA／Vulkan／Metal）；預設的 CPU 版回傳 False。"""
    if not available():
        return False
    try:
        import llama_cpp

        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:  # 函式庫版本太舊沒有這個函式、載入動態函式庫失敗等
        return False


# ggml 的運算裝置種類：0 = CPU、1 = GPU（獨立顯示卡，有自己的顯示卡記憶體）、2 = IGPU（內顯，和 CPU 共用同一份記憶體）
DEVICE_TYPE_GPU = 1
DEVICE_TYPE_IGPU = 2


def _query_devices() -> list[dict]:
    """問 ggml 有哪些運算裝置：``[{"name", "description", "type", "free", "total"}]``（位元組）。
    llama-cpp-python 沒有提供這個介面，所以直接呼叫它附帶的 ggml 動態函式庫（``ggml_backend_dev_*``）。"""
    import llama_cpp

    lib_dir = Path(llama_cpp.__file__).parent / "lib"
    names = ("ggml.dll", "ggml-base.dll") if os.name == "nt" else ("libggml.so", "libggml-base.so", "libggml.dylib", "libggml-base.dylib")
    libs = [ctypes.CDLL(str(lib_dir / name)) for name in names if (lib_dir / name).exists()]

    def function(symbol: str):
        for lib in libs:
            try:
                return getattr(lib, symbol)
            except AttributeError:
                continue
        raise AttributeError(symbol)

    count, get = function("ggml_backend_dev_count"), function("ggml_backend_dev_get")
    kind, name, description, memory = (
        function("ggml_backend_dev_type"), function("ggml_backend_dev_name"),
        function("ggml_backend_dev_description"), function("ggml_backend_dev_memory"),
    )
    count.restype, get.restype, get.argtypes = ctypes.c_size_t, ctypes.c_void_p, [ctypes.c_size_t]
    kind.restype, kind.argtypes = ctypes.c_int, [ctypes.c_void_p]
    name.restype, name.argtypes = ctypes.c_char_p, [ctypes.c_void_p]
    description.restype, description.argtypes = ctypes.c_char_p, [ctypes.c_void_p]
    memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    devices = []
    for index in range(count()):
        device = get(index)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        memory(device, ctypes.byref(free), ctypes.byref(total))
        devices.append({
            "name": (name(device) or b"").decode("utf-8", "replace"),
            "description": (description(device) or b"").decode("utf-8", "replace"),
            "type": int(kind(device)),
            "free": int(free.value),
            "total": int(total.value),
        })
    return devices


def gpu_devices() -> list[dict]:
    """目前可用的 GPU 裝置（不含 CPU）；沒有 GPU 版的 llama-cpp-python、或查不到就是空的。"""
    if not gpu_offload_supported():
        return []
    try:
        return [device for device in _query_devices() if device["type"] in (DEVICE_TYPE_GPU, DEVICE_TYPE_IGPU)]
    except Exception:  # 函式庫版本沒有這些函式、載入失敗等：當成查不到
        return []


def discrete_gpu_free_bytes() -> int:
    """獨立顯示卡目前可用的顯示卡記憶體（位元組）；沒有獨立顯示卡（只有內顯）或查不到回傳 0。

    內顯和 CPU 共用系統記憶體，沒有「自己的」顯示卡記憶體可以放大模型，所以不算。"""
    return max((device["free"] for device in gpu_devices() if device["type"] == DEVICE_TYPE_GPU), default=0)


def find_gguf(directory: Path) -> Path | None:
    """模型資料夾裡的 GGUF 檔（一個資料夾放一個模型；有多個時取檔名排序第一個，結果才穩定）。"""
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.gguf"))
    return files[0] if files else None


class LlamaCppEngine:
    """載入一個 GGUF 模型，``complete(prompt, max_tokens)`` 回傳貪婪解碼的文字。"""

    def __init__(
        self,
        model_path: Path,
        threads: int,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        gpu_layers: int = 0,
        cpu_fallback: bool = True,
    ) -> None:
        """``gpu_layers``：交給 GPU 的層數（-1 = 全部，0 = 只用 CPU）。GPU 載入失敗（顯示卡記憶體不足等）會退回 CPU；
        ``cpu_fallback=False`` 時不退回、直接丟例外（例如電腦的記憶體也載不進這個模型，退回 CPU 只會把電腦拖垮）。"""
        self.model_path = Path(model_path)
        self.gpu_layers = 0
        try:
            self._llama = self._open(threads, context_tokens, gpu_layers)
            self.gpu_layers = gpu_layers
        except Exception as exc:
            if not gpu_layers:
                raise
            if not cpu_fallback:
                raise RuntimeError(f"GPU 載入失敗，而且不改用 CPU（記憶體不夠）：{exc}") from exc
            log.warning("GPU 載入 %s 失敗（%s），改用 CPU", self.model_path.name, exc)
            self._llama = self._open(threads, context_tokens, 0)
        self._init_prefix_cache()
        self._init_token_filter()

    def _open(self, threads: int, context_tokens: int, gpu_layers: int):
        from llama_cpp import Llama

        return Llama(
            model_path=str(self.model_path),
            n_ctx=context_tokens,
            n_threads=threads,
            n_threads_batch=threads,
            n_batch=512,
            n_gpu_layers=gpu_layers,
            use_mmap=True,
            verbose=False,
        )

    def _init_prefix_cache(self) -> None:
        # 固定前綴（系統提示 + 示範，約 300 個 token）的 KV 快取：前綴 → 序列狀態位元組。
        # llama.cpp 只記得「上一次」的提示詞，日文換英文（或換歌詞模式）時前綴不同，要重算約 1 秒；
        # 存起來之後換回來只要毫秒。出任何錯就停用這個加速，照函式庫原本的行為（只重用上一次的前綴）。
        self._prefix_states: OrderedDict[str, bytes] = OrderedDict()
        self._prefix_cache_ok = True

    def _init_token_filter(self) -> None:
        # 生成時擋掉不該出現的詞（外文、原文沒有的英文字母、字間空格），詞彙表的分類第一次需要時才建立
        llama = self._llama

        def pieces():
            for token in range(llama.n_vocab()):
                try:
                    yield llama.detokenize([token])
                except Exception:  # 少數特殊記號解不出來：不分類
                    yield b""

        self._token_filter = TokenFilter(pieces, lambda text: llama.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        self._token_filter_ok = True

    def prepare_token_filter(self) -> None:
        """先把詞彙表分類好（約一兩秒），第一句翻譯才不必等。失敗就不擋，行為與沒有這個功能時相同。"""
        if not self._token_filter_ok:
            return
        try:
            self._token_filter.ensure_loaded()
        except Exception as exc:
            log.warning("詞彙表分類失敗，生成時不擋詞：%s", exc)
            self._token_filter_ok = False

    # ------------------------------------------------------------- 固定前綴的 KV 快取
    def _capture_kv(self) -> bytes:
        import llama_cpp

        context = self._llama._ctx.ctx
        size = llama_cpp.llama_state_seq_get_size(context, 0)
        buffer = (ctypes.c_uint8 * size)()
        written = llama_cpp.llama_state_seq_get_data(context, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), size, 0)
        if written != size:
            raise RuntimeError(f"KV 快取只存了 {written}/{size} 位元組")
        return bytes(buffer)

    def _restore_kv(self, state: bytes) -> None:
        import llama_cpp

        context = self._llama._ctx.ctx
        self._llama._ctx.kv_cache_clear()
        buffer = (ctypes.c_uint8 * len(state)).from_buffer_copy(state)
        if llama_cpp.llama_state_seq_set_data(context, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), len(state), 0) == 0:
            raise RuntimeError("還原 KV 快取失敗")

    def prepare(self, prefix: str) -> None:
        """讓 KV 快取停在這個固定前綴上，之後的提示詞（前綴 + 前文 + 待翻句）只需要算後面的部分。

        已經在這個前綴上就什麼都不做；存過這個前綴就還原；都沒有才重算一次並存起來。
        """
        if not self._prefix_cache_ok or not prefix:
            return
        llama = self._llama
        try:
            tokens = llama.tokenize(prefix.encode("utf-8"), add_bos=False, special=True)
            count = len(tokens)
            if llama.n_tokens >= count and llama.input_ids[:count].tolist() == tokens:
                return
            saved = self._prefix_states.get(prefix)
            if saved is not None:
                self._prefix_states.move_to_end(prefix)
                self._restore_kv(saved)
                llama.input_ids[:count] = tokens
                llama.n_tokens = count
                return
            llama.reset()
            llama._ctx.kv_cache_clear()
            llama.eval(tokens)
            self._prefix_states[prefix] = self._capture_kv()
            while len(self._prefix_states) > MAX_PREFIX_STATES:
                self._prefix_states.popitem(last=False)
        except Exception as exc:  # 函式庫內部介面變了、記憶體不足等：改回只重用上一次前綴的原本行為
            log.warning("llama.cpp 固定前綴快取停用：%s", exc)
            self._prefix_cache_ok = False
            self._prefix_states.clear()
            llama.reset()

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        prefix: str = "",
        seed: int | None = None,
        bans: frozenset[str] = frozenset(),
        allow: tuple[str, ...] = (),
    ) -> str:
        """``prefix`` 是 ``prompt`` 開頭那段固定不變的部分（提供就先讓 KV 快取停在它上面）。

        預設貪婪解碼（每次結果相同）。給 ``seed`` 就改用 Qwen 官方建議的取樣（溫度 0.7、top-p 0.8、top-k 20、
        重複懲罰 1.05），同一個種子結果可重現；用來重翻貪婪解碼翻壞的句子（貪婪重翻只會得到一樣的壞結果）。

        ``bans``：生成時要擋掉哪幾類詞（``token_filter.FOREIGN`` 等），模型選不到這些詞，就不會翻出夾雜外文的譯文；
        ``allow``：這些文字切出來的詞不擋（例如 ``[SKIP]``）。
        """
        if prefix and prompt.startswith(prefix):
            self.prepare(prefix)
        if seed is None:
            sampling = {"temperature": 0.0, "top_k": 1}
        else:
            sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05, "seed": seed}
        processor = self._ban_processor(bans, allow)
        if processor is not None:
            from llama_cpp import LogitsProcessorList

            sampling["logits_processor"] = LogitsProcessorList([processor])
        result = self._llama.create_completion(
            prompt,
            max_tokens=max_tokens,
            stop=[END_TOKEN],
            echo=False,
            **sampling,
        )
        return str(result["choices"][0]["text"])

    def _ban_processor(self, bans: frozenset[str], allow: tuple[str, ...]):
        if not bans or not self._token_filter_ok:
            return None
        try:
            return self._token_filter.processor(bans, allow)
        except Exception as exc:  # 分類失敗、記憶體不足：照沒有這個功能時的行為（事後檢查還在）
            log.warning("詞彙表分類失敗，生成時不擋詞：%s", exc)
            self._token_filter_ok = False
            return None

    def close(self) -> None:
        close = getattr(self._llama, "close", None)
        if callable(close):
            close()
        self._llama = None
