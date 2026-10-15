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
