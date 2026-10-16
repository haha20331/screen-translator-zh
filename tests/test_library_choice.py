"""打包版附兩套 llama.cpp 函式庫（CPU 版與 GPU 版）：啟動時依這台電腦有沒有 Vulkan 選一套。"""

from __future__ import annotations

import pytest

from services.translation import llama_backend


@pytest.fixture
def package(tmp_path, monkeypatch):
    """假的 llama_cpp 套件資料夾（lib 是 CPU 版），選擇的結果不留到別的測試。"""
    folder = tmp_path / "llama_cpp"
    (folder / "lib").mkdir(parents=True)
    monkeypatch.setattr(llama_backend, "_package_dir", lambda: folder)
    monkeypatch.delenv("LLAMA_CPP_LIB_PATH", raising=False)
    monkeypatch.delenv(llama_backend.CPU_ONLY_ENV, raising=False)
    monkeypatch.setitem(llama_backend._library, "kind", llama_backend.KIND_DEFAULT)
    return folder


def _add_vulkan_set(package):
    vulkan = package / llama_backend.VULKAN_LIBRARY_DIR
    vulkan.mkdir()
    (vulkan / "ggml-vulkan.dll").write_bytes(b"x")
    return vulkan


def test_without_a_bundled_gpu_set_nothing_changes(package):
    assert llama_backend.select_library(lambda: pytest.fail("沒有 GPU 版就不必查 Vulkan")) == llama_backend.KIND_DEFAULT
    assert llama_backend.library_dir() == package / "lib"


def test_the_gpu_set_is_used_when_the_machine_has_a_vulkan_loader(package, monkeypatch):
    vulkan = _add_vulkan_set(package)

    assert llama_backend.select_library(lambda: True) == llama_backend.KIND_VULKAN
    assert llama_backend.library_dir() == vulkan
    import os

    assert os.environ["LLAMA_CPP_LIB_PATH"] == str(vulkan)
    monkeypatch.delenv("LLAMA_CPP_LIB_PATH")


def test_a_machine_without_a_vulkan_loader_keeps_the_cpu_set_and_says_why(package):
    _add_vulkan_set(package)

    assert llama_backend.select_library(lambda: False) == llama_backend.KIND_NO_VULKAN_RUNTIME
    assert llama_backend.library_dir() == package / "lib"  # GPU 版載入會整個失敗，絕對不能選
    assert "vulkan-1.dll" in llama_backend.gpu_unavailable_reason()


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_the_environment_variable_forces_the_cpu_set_without_touching_vulkan(package, monkeypatch, value):
    _add_vulkan_set(package)
    monkeypatch.setenv(llama_backend.CPU_ONLY_ENV, value)

    assert llama_backend.select_library(lambda: pytest.fail("強制只用 CPU 時不碰 Vulkan")) == llama_backend.KIND_CPU_FORCED
    assert llama_backend.CPU_ONLY_ENV in llama_backend.gpu_unavailable_reason()


def test_a_library_path_the_user_set_is_respected(package, monkeypatch, tmp_path):
    _add_vulkan_set(package)
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(tmp_path / "mine"))

    assert llama_backend.select_library(lambda: True) == llama_backend.KIND_DEFAULT
    assert llama_backend.library_dir() == tmp_path / "mine"


def test_the_reason_tells_a_cpu_build_from_a_gpu_build_that_found_no_gpu(package):
    assert "CPU 版" in llama_backend.gpu_unavailable_reason()
    (package / "lib" / "ggml-vulkan.dll").write_bytes(b"x")  # 例如開發環境裝了 Vulkan 版、但這台電腦沒有可用的 GPU
    assert "找不到可以運算的 GPU" in llama_backend.gpu_unavailable_reason()


def test_the_real_check_for_a_vulkan_loader_answers_without_crashing():
    assert isinstance(llama_backend.vulkan_runtime_present(), bool)
