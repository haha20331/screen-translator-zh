# 第三方授權

本專案以 [xuange-hu/screen-translator](https://github.com/xuange-hu/screen-translator)（MIT）為基底，詳見 `UPSTREAM.md` 與 `LICENSE`。
對外散布時請隨附本檔與各模型資料夾內的授權檔。授權資訊取自各套件在 PyPI 的中繼資料（2026-10-12 查詢），**尚未逐一核對各模型卡**，散布前請再確認。

## 程式相依套件

| 套件 | 授權 | 備註 |
|---|---|---|
| PySide6 (Qt for Python) | LGPL-3.0 / GPL-2.0 / GPL-3.0（三選一） | 以 LGPL-3.0 使用：打包時保持動態連結（PyInstaller onedir），讓使用者能替換 Qt 函式庫，並提供授權文字 |
| pynput | LGPL-3.0 | 同上 |
| mss | MIT | |
| Pillow | MIT-CMU | |
| numpy | BSD-3-Clause 等 | |
| requests | Apache-2.0 | |
| winocr | MIT | Windows OCR 備援引擎 |
| rapidocr | Apache-2.0 | |
| onnxruntime | MIT | |
| ctranslate2 | MIT | |
| tokenizers | Apache-2.0 | |
| OpenCC | Apache-2.0 | 簡繁轉換與台灣用語詞典 |

## 模型

| 模型 | 授權 | 來源 |
|---|---|---|
| PP-OCRv6（偵測與辨識，隨 `rapidocr` 套件安裝） | 隨套件：Apache-2.0 | 模型源自 PaddleOCR（Apache-2.0）；PP-OCRv6 模型卡尚未逐一核對 |
| Qwen2.5-1.5B-Instruct（轉成 CTranslate2 int8） | Apache-2.0 | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；轉換紀錄見模型資料夾內 `manifest.json`，授權檔 `LICENSE` 隨附 |

評估過但**未採用**的模型（不隨附）：NLLB-200（CC-BY-NC-4.0，非商用）、Helsinki-NLP opus-mt（Apache-2.0）、Qwen2.5-3B、Qwen3-1.7B。
