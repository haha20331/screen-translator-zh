"""scripts/fetch_vulkan_library.py：兩套 llama.cpp 函式庫的準備——下載要核對雜湊、只解出需要的 DLL、CPU 版的組成、版本與繫結要一致。"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_vulkan_library.py"
spec = importlib.util.spec_from_file_location("fetch_vulkan_library", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
sys.modules["fetch_vulkan_library"] = fetch
spec.loader.exec_module(fetch)


def _wheel(path: Path, files, tag: str, python=("llama_cpp/llama_cpp.py", "llama_cpp/llama.py"), extra=("llama_cpp/lib/ggml.lib",)) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        for name in files:
            archive.writestr(fetch.MEMBER_PREFIX + name, f"{tag}:{name}".encode())
        for name in python:
            archive.writestr(name, f"binding of {name}".encode())
        for name in extra:
            archive.writestr(name, b"not needed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_gpu_set_is_the_six_runtime_dlls_of_the_vulkan_wheel(tmp_path):
    wheel = tmp_path / "vulkan.whl"
    _wheel(wheel, fetch.VULKAN_FILES, "vulkan")

    written = fetch.extract_files({name: wheel for name in fetch.VULKAN_FILES}, tmp_path / "out")

    assert sorted(path.name for path in written) == sorted(fetch.VULKAN_FILES)
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(fetch.VULKAN_FILES)  # .lib 與 .py 都沒解出來
    assert (tmp_path / "out" / "ggml-vulkan.dll").read_bytes() == b"vulkan:ggml-vulkan.dll"


def test_the_cpu_set_uses_the_avx2_kernels_of_the_vulkan_wheel_and_only_the_registry_dll_of_the_cpu_wheel(tmp_path):
    vulkan, cpu = tmp_path / "vulkan.whl", tmp_path / "cpu.whl"
    _wheel(vulkan, fetch.VULKAN_FILES, "vulkan")
    _wheel(cpu, [name for name in fetch.VULKAN_FILES if name != "ggml-vulkan.dll"], "cpu")
    plan = {
        **{name: vulkan for name in fetch.CPU_FILES_FROM_VULKAN_WHEEL},
        **{name: cpu for name in fetch.CPU_FILES_FROM_CPU_WHEEL},
    }

    fetch.extract_files(plan, tmp_path / "cpu_lib")

    contents = {path.name: path.read_bytes().decode() for path in (tmp_path / "cpu_lib").iterdir()}
    assert contents["ggml-cpu.dll"] == "vulkan:ggml-cpu.dll"  # 官方 CPU wheel 的 ggml-cpu.dll 含 AVX-512，不能用
    assert contents["ggml.dll"] == "cpu:ggml.dll"  # 不連結 ggml-vulkan.dll 的那份
    assert "ggml-vulkan.dll" not in contents and len(contents) == 5


def test_files_left_over_from_an_earlier_run_do_not_stay_in_the_folder(tmp_path):
    wheel = tmp_path / "w.whl"
    _wheel(wheel, fetch.VULKAN_FILES, "v")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.dll").write_bytes(b"x")

    fetch.extract_files({name: wheel for name in fetch.VULKAN_FILES}, tmp_path / "out")

    assert not (tmp_path / "out" / "old.dll").exists()


def test_a_wheel_missing_a_dll_is_refused(tmp_path):
    wheel = tmp_path / "w.whl"
    _wheel(wheel, [name for name in fetch.VULKAN_FILES if name != "ggml-vulkan.dll"], "v")

    with pytest.raises(fetch.FetchError, match="ggml-vulkan.dll"):
        fetch.extract_files({name: wheel for name in fetch.VULKAN_FILES}, tmp_path / "out")


def test_a_cached_wheel_with_the_right_hash_is_used_without_downloading(tmp_path, monkeypatch):
    digest = _wheel(tmp_path / "w.whl", fetch.VULKAN_FILES, "v")
    monkeypatch.setattr(fetch, "download", lambda url, target: pytest.fail("快取是對的，不該下載"))

    assert fetch.verified_wheel(tmp_path, "http://x/w.whl", digest, "w.whl") == tmp_path / "w.whl"


def test_a_download_with_the_wrong_hash_is_deleted_and_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "download", lambda url, target: target.write_bytes(b"tampered"))

    with pytest.raises(fetch.FetchError, match="SHA-256 不符"):
        fetch.verified_wheel(tmp_path, "http://x/w.whl", "0" * 64, "w.whl")
    assert not (tmp_path / "w.whl").exists()


def test_a_corrupted_cache_is_downloaded_again(tmp_path, monkeypatch):
    good = tmp_path / "good.whl"
    digest = _wheel(good, fetch.VULKAN_FILES, "v")
    (tmp_path / "w.whl").write_bytes(b"corrupted")
    monkeypatch.setattr(fetch, "download", lambda url, target: target.write_bytes(good.read_bytes()))

    assert fetch.verified_wheel(tmp_path, "http://x/w.whl", digest, "w.whl").read_bytes() == good.read_bytes()


def _installed_package(tmp_path, wheel: Path) -> Path:
    package = tmp_path / "installed" / "llama_cpp"
    package.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.endswith(".py"):
                (package / member[len("llama_cpp/"):]).write_bytes(archive.read(member))
    return package


def test_the_build_environment_must_have_the_same_version_and_the_same_python_bindings(tmp_path):
    wheel = tmp_path / "w.whl"
    _wheel(wheel, fetch.VULKAN_FILES, "v")
    package = _installed_package(tmp_path, wheel)

    fetch.check_installed(wheel, fetch.WHEEL_VERSION, package)
    for other in ("0.3.36", None):
        with pytest.raises(fetch.FetchError, match="要和 GPU 版"):
            fetch.check_installed(wheel, other, package)
    (package / "llama_cpp.py").write_bytes(b"a different binding")
    with pytest.raises(fetch.FetchError, match="繫結"):
        fetch.check_installed(wheel, fetch.WHEEL_VERSION, package)
    (package / "llama_cpp.py").unlink()
    with pytest.raises(fetch.FetchError, match="繫結"):
        fetch.check_installed(wheel, fetch.WHEEL_VERSION, package)


def test_the_pinned_urls_and_hashes_are_consistent():
    for url in (fetch.VULKAN_WHEEL_URL, fetch.CPU_WHEEL_URL):
        assert fetch.WHEEL_VERSION in url and url.endswith(fetch.WHEEL_NAME) and url.startswith("https://")  # 不用明文的網址
    for digest in (fetch.VULKAN_WHEEL_SHA256, fetch.CPU_WHEEL_SHA256):
        assert len(digest) == 64 and int(digest, 16) >= 0
    assert fetch.VULKAN_WHEEL_SHA256 != fetch.CPU_WHEEL_SHA256


def test_the_requirements_pin_the_same_version_for_both_builds():
    root = SCRIPT.parent.parent
    for name in ("requirements-core.txt", "requirements-vulkan.txt"):
        assert f"llama-cpp-python=={fetch.WHEEL_VERSION}" in (root / name).read_text(encoding="utf-8")
