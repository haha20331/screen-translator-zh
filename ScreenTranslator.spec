# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 設定：資料夾式（onedir）的 Windows 程式，供 installer/ScreenTranslator.iss 打包成安裝程式。

用法見 PACKAGING.md（scripts/build_installer.ps1 會呼叫它）。選資料夾式而不是單一檔案：
單一檔案每次啟動都要把幾百 MB 解壓到暫存資料夾（慢，也容易被防毒軟體攔下），資料夾式安裝後直接執行。
模型（GGUF）不在這裡：安裝程式依使用者勾選下載，放在 ``<安裝資料夾>\\models\\translation\\<名稱>\\``。
"""

import os
import re
import tempfile
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

build_version = os.environ.get("SCREEN_TRANSLATOR_BUILD_VERSION", "0.3.0-beta")
if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", build_version):
    raise ValueError(f"Invalid SCREEN_TRANSLATOR_BUILD_VERSION: {build_version!r}")
metadata_dir = Path(tempfile.mkdtemp(prefix="screen-translator-build-"))
version_file = metadata_dir / "build-version.txt"
version_file.write_text(build_version, encoding="ascii")

datas = [
    (os.path.join("assets", "app_launch_v4.png"), "assets"),
    (str(version_file), "."),
]
binaries = []
hiddenimports = []

# 這些套件帶有資料檔或原生函式庫（PyInstaller 預設的分析抓不全），整包收進來：
# rapidocr 的 ONNX 模型與設定檔、onnxruntime 與 llama.cpp 的 DLL、OpenCC 的詞典、Windows OCR 的 winrt 繫結。
# PySide6 不整包收（會把 WebEngine、QML、多媒體都帶進來），交給 PyInstaller 的 Qt hook 只收有用到的模組。
for package in (
    "rapidocr",
    "onnxruntime",
    "llama_cpp",
    "opencc",
    "mss",
    "pynput",
    "winocr",
    "winrt.runtime",
    "winrt.windows.foundation",
    "winrt.windows.foundation.collections",
    "winrt.windows.globalization",
    "winrt.windows.graphics.imaging",
    "winrt.windows.media.ocr",
    "winrt.windows.storage.streams",
):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception as error:  # 少了套件就不收，但要讓建置時看得到
        print(f"[spec] collect_all({package}) 失敗：{error}")
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("pynput", filter=lambda name: "._win32" in name or name.endswith("._base"))

# llama.cpp 附兩套函式庫，都由 scripts/fetch_vulkan_library.py 準備好（原因與做法見它的說明與 PACKAGING.md）：
#   llama_cpp\lib        = build\cpu_lib     CPU 版（不用建置環境裝的 CPU wheel 的 DLL：它的 ggml-cpu.dll 含 AVX-512，
#                                              沒有 AVX-512 的 CPU 載入模型會「非法指令」當掉）
#   llama_cpp\lib_vulkan = build\vulkan_lib  GPU 版（Vulkan）；ggml.dll 直接連結 vulkan-1.dll，只在載得進 vulkan-1.dll 的電腦用
# 程式啟動時依這台電腦有沒有 Vulkan 選一套（llama_backend.select_library）。
# 用 datas（單純複製）而不是 binaries：不要讓 PyInstaller 去分析 DLL 相依、順手把系統的 vulkan-1.dll 也收進來。
cpu_library = Path("build") / "cpu_lib"
vulkan_library = Path("build") / "vulkan_lib"
if not (cpu_library / "ggml-cpu.dll").exists() or not (vulkan_library / "ggml-vulkan.dll").exists():
    raise SystemExit("找不到 build\\cpu_lib 或 build\\vulkan_lib：先執行 python scripts/fetch_vulkan_library.py（build_installer.ps1 會自動做）。")


def _is_default_llama_library(entry):
    # collect_all 給的是 (來源檔, 放進程式的哪個資料夾)；建置環境裝的 CPU wheel 的 DLL 全部不要，換成上面準備好的
    return entry[1].replace("\\", "/").lower().rstrip("/") == "llama_cpp/lib"


binaries[:] = [entry for entry in binaries if not _is_default_llama_library(entry)]
datas[:] = [entry for entry in datas if not _is_default_llama_library(entry)]
for dll in sorted(cpu_library.glob("*.dll")):
    datas.append((str(dll), "llama_cpp/lib"))
for dll in sorted(vulkan_library.glob("*.dll")):
    datas.append((str(dll), "llama_cpp/lib_vulkan"))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.Qt3DCore",
        "PySide6.QtMultimedia",
        "PySide6.QtDesigner",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtPdf",
        "matplotlib",
        "paddle",
        "paddleocr",
        "paddlex",
        "scipy",
        "sklearn",
        "tkinter",
        "torch",
    ],
    noarchive=False,
)

# PyInstaller 會分析 llama_cpp\lib 與 llama_cpp\lib_vulkan 裡的 DLL 的相依檔，把找到的 ggml*.dll／llama.dll 另外複製一份放在最上層
# （_internal\）。Windows 載入 DLL 時先找程式資料夾與 _internal，再找 PATH（llama_cpp 載入函式庫時才把選好的那套資料夾放進 PATH），
# 所以最上層的這幾份會蓋掉我們選的那套：實測「只用 CPU 版」時照樣載進 GPU 版的 ggml.dll，沒有 Vulkan 的電腦會整個載入失敗。
# 全部拿掉，只留 lib 與 lib_vulkan 底下各自成套的那兩份。
_LLAMA_DLLS = {"ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-vulkan.dll", "llama.dll", "mtmd.dll"}
a.binaries[:] = [entry for entry in a.binaries if entry[0].replace("\\", "/").lower() not in _LLAMA_DLLS]
a.datas[:] = [entry for entry in a.datas if entry[0].replace("\\", "/").lower() not in _LLAMA_DLLS]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ScreenTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon="assets/app_launch_v4.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ScreenTranslator",
)
