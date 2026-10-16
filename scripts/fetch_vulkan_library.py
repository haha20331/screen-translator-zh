"""準備打包版的兩套 llama.cpp 函式庫（CPU 版與 GPU 版），放進 ``build\\cpu_lib\\`` 與 ``build\\vulkan_lib\\``（見 PACKAGING.md）。

程式啟動時依這台電腦有沒有 Vulkan 選一套（``services/translation/llama_backend.select_library``），
所以使用者在設定頁切換 CPU／GPU 不必重裝、也不必重新啟動。

- **GPU 版**（``build\\vulkan_lib``）：專案的 Vulkan wheel 裡的 6 個 DLL。ggml.dll 直接連結 ggml-vulkan.dll、再連結 vulkan-1.dll，
  沒有 Vulkan 的電腦載入它會整個失敗，所以只在載得進 vulkan-1.dll 時才用它。
- **CPU 版**（``build\\cpu_lib``）：**不能直接用官方的 CPU wheel**。實測 0.3.33 的 CPU wheel 的 ggml-cpu.dll 含 AVX-512 指令
  （約 7000 處 zmm 暫存器），在沒有 AVX-512 的 CPU（例如這台的 i7-13700H，多數 Intel 12～14 代、AMD Zen4 以前）載入模型就「非法指令」
  （0xC000001D）當掉；Vulkan wheel 的 ggml-cpu.dll 只用到 AVX2 與 FMA，速度與 CPU wheel 相同。
  所以 CPU 版 = Vulkan wheel 的 ggml-cpu.dll、ggml-base.dll、llama.dll、mtmd.dll，加上 CPU wheel 的 ggml.dll
  （只負責登錄後端，沒有 Vulkan 的連結；兩個 wheel 是同一版 llama.cpp 編出來的）。

兩套 DLL 搭配的 Python 繫結（``llama_cpp\\*.py``）在兩個 wheel 裡完全相同（25 個檔案逐一比對過）；
建置環境的 llama-cpp-python 必須是同一版、繫結也要相同（``--check-installed`` 會逐檔比對）。

用法::

    python scripts/fetch_vulkan_library.py                    # 下載（有快取就不重載）並解出兩套 DLL
    python scripts/fetch_vulkan_library.py --check-installed  # 另外確認建置環境的 llama-cpp-python 繫結與 wheel 相同
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WHEEL_VERSION = "0.3.33"
WHEEL_NAME = f"llama_cpp_python-{WHEEL_VERSION}-py3-none-win_amd64.whl"
# 專案自己的 wheel 索引（https://abetlen.github.io/llama-cpp-python/whl/{vulkan,cpu}）指向的 GitHub Release 檔案；
# SHA-256 是第一次下載時算的，之後每次下載或讀快取都要核對，不符就拒絕。
VULKAN_WHEEL_URL = f"https://github.com/abetlen/llama-cpp-python/releases/download/v{WHEEL_VERSION}-vulkan/{WHEEL_NAME}"
VULKAN_WHEEL_SHA256 = "da16a9bd56f78912b6769b3412e249a2e0d37452970ad541b25934e1235f747a"
CPU_WHEEL_URL = f"https://github.com/abetlen/llama-cpp-python/releases/download/v{WHEEL_VERSION}/{WHEEL_NAME}"
CPU_WHEEL_SHA256 = "ac941f5788ef1288ab30e5336d585528d7d8c9434a8b8d2e8b5cd08f02d65aed"

# 只要執行時會載入的 DLL（.lib 是編譯用的，不需要）。ggml.dll 直接連結 ggml-vulkan.dll，所以 GPU 版這幾個要成套
VULKAN_FILES = ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-vulkan.dll", "llama.dll", "mtmd.dll")
CPU_FILES_FROM_VULKAN_WHEEL = ("ggml-base.dll", "ggml-cpu.dll", "llama.dll", "mtmd.dll")
CPU_FILES_FROM_CPU_WHEEL = ("ggml.dll",)
MEMBER_PREFIX = "llama_cpp/lib/"

DEFAULT_VULKAN_DESTINATION = ROOT / "build" / "vulkan_lib"
DEFAULT_CPU_DESTINATION = ROOT / "build" / "cpu_lib"
DEFAULT_CACHE = ROOT / "build" / "llama_wheels"


class FetchError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"下載 {url}")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:  # 憑證驗證用預設值，不關閉
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)


def verified_wheel(folder: Path, url: str, expected: str, name: str = WHEEL_NAME) -> Path:
    """``folder`` 快取裡的 wheel（沒有、或雜湊不符就重新下載）；下載後還是不符就丟出 ``FetchError``，不留下壞檔。"""
    wheel = folder / name
    if wheel.exists() and sha256_of(wheel) == expected:
        return wheel
    if wheel.exists():
        print("快取的檔案雜湊不符，重新下載。")
        wheel.unlink()
    download(url, wheel)
    actual = sha256_of(wheel)
    if actual != expected:
        wheel.unlink()
        raise FetchError(f"下載的檔案 SHA-256 不符（預期 {expected}，實際 {actual}），已刪除；不使用。")
    return wheel


def extract_files(wheels: dict[str, Path], destination: Path) -> list[Path]:
    """依 ``{檔名: 哪個 wheel}`` 解出 DLL 到 ``destination``（用固定的檔名，不用壓縮檔裡的路徑）；少一個就失敗。"""
    archives = {path: zipfile.ZipFile(path) for path in set(wheels.values())}
    try:
        missing = [name for name, wheel in wheels.items() if MEMBER_PREFIX + name not in set(archives[wheel].namelist())]
        if missing:
            raise FetchError(f"wheel 裡缺少：{', '.join(missing)}")
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.dll"):  # 上一次留下、這次不要的檔案不能混進去
            stale.unlink()
        written = []
        for name, wheel in wheels.items():
            target = destination / name
            target.write_bytes(archives[wheel].read(MEMBER_PREFIX + name))
            written.append(target)
        return written
    finally:
        for archive in archives.values():
            archive.close()


def installed_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("llama-cpp-python")
    except metadata.PackageNotFoundError:
        return None


def check_bindings(wheel: Path, package_dir: Path) -> None:
    """建置環境裝的 llama_cpp 的 Python 檔（ctypes 繫結）要和 wheel 裡的逐檔相同：打包的是這份繫結，DLL 卻來自 wheel。"""
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not (member.startswith("llama_cpp/") and member.endswith(".py")):
                continue
            local = package_dir / member[len("llama_cpp/"):]
            if not local.is_file() or hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(archive.read(member)).digest():
                raise FetchError(f"建置環境的 {member} 與 wheel 裡的不同：DLL 與 Python 繫結對不上會當掉。請重裝 requirements-core.txt。")


def check_installed(wheel: Path, version: str | None = None, package_dir: Path | None = None) -> None:
    version = installed_version() if version is None else version
    if version != WHEEL_VERSION:
        raise FetchError(
            f"建置環境的 llama-cpp-python 是 {version}，要和 GPU 版的 {WHEEL_VERSION} 相同（DLL 與 Python 繫結必須同版）。"
            f"請執行：python -m pip install -r requirements-core.txt"
        )
    if package_dir is None:
        import importlib.util

        spec = importlib.util.find_spec("llama_cpp")
        if spec is None or not spec.submodule_search_locations:
            raise FetchError("建置環境沒有 llama_cpp。")
        package_dir = Path(list(spec.submodule_search_locations)[0])
    check_bindings(wheel, package_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vulkan-destination", type=Path, default=DEFAULT_VULKAN_DESTINATION, help="GPU 版 DLL 放到哪（預設 build\\vulkan_lib）")
    parser.add_argument("--cpu-destination", type=Path, default=DEFAULT_CPU_DESTINATION, help="CPU 版 DLL 放到哪（預設 build\\cpu_lib）")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="wheel 的快取資料夾（預設 build\\llama_wheels）")
    parser.add_argument("--check-installed", action="store_true", help="確認建置環境的 llama-cpp-python 版本與 Python 繫結和 wheel 相同")
    args = parser.parse_args(argv)
    try:
        vulkan_wheel = verified_wheel(args.cache / "vulkan", VULKAN_WHEEL_URL, VULKAN_WHEEL_SHA256)
        cpu_wheel = verified_wheel(args.cache / "cpu", CPU_WHEEL_URL, CPU_WHEEL_SHA256)
        if args.check_installed:
            check_installed(vulkan_wheel)
        vulkan_files = extract_files({name: vulkan_wheel for name in VULKAN_FILES}, args.vulkan_destination)
        cpu_files = extract_files(
            {**{name: vulkan_wheel for name in CPU_FILES_FROM_VULKAN_WHEEL}, **{name: cpu_wheel for name in CPU_FILES_FROM_CPU_WHEEL}},
            args.cpu_destination,
        )
    except (FetchError, OSError) as exc:
        print(f"失敗：{exc}", file=sys.stderr)
        return 1
    for label, files, destination in (("GPU 版", vulkan_files, args.vulkan_destination), ("CPU 版", cpu_files, args.cpu_destination)):
        total = sum(path.stat().st_size for path in files)
        print(f"{label} llama.cpp {WHEEL_VERSION}：{len(files)} 個 DLL（{total / 1e6:.0f} MB）→ {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
