# 第三方授權

本專案以 [xuange-hu/screen-translator](https://github.com/xuange-hu/screen-translator)（MIT）為基底，詳見 `UPSTREAM.md` 與 `LICENSE`。
對外散布時請隨附本檔與各模型資料夾內的授權檔。授權資訊取自各套件在 PyPI 的中繼資料（2026-10-12 查詢），**尚未逐一核對各模型卡**，散布前請再確認。

**安裝版**：安裝資料夾的 `licenses\` 內有隨程式打包的每個第三方套件的授權檔（由 `scripts/collect_licenses.py` 在建置時收集，`licenses\INDEX.txt` 列出套件、版本、授權與授權檔）。翻譯模型**不在安裝程式裡**：使用者在安裝時勾選後，由使用者的電腦直接從 Hugging Face 下載（固定版本、SHA-256 驗證），授權檔 `LICENSE` 與 `manifest.json` 一起放進模型資料夾。

## 程式相依套件

| 套件 | 授權 | 備註 |
|---|---|---|
| PySide6 (Qt for Python) | LGPL-3.0 / GPL-2.0 / GPL-3.0（三選一） | 以 LGPL-3.0 使用：打包時保持動態連結（PyInstaller 資料夾式，Qt 的 DLL 是安裝資料夾 `_internal\PySide6\` 底下獨立的檔案，使用者可以替換），並隨附授權文字。原始碼可從 Qt（<https://download.qt.io/>）與 PySide6（PyPI）取得；本專案沒有修改它們 |
| pynput | LGPL-3.0 | 同上（純 Python，原始碼就在安裝資料夾的 `_internal\pynput\`，可以直接替換） |
| mss | MIT | |
| Pillow | MIT-CMU | |
| numpy | BSD-3-Clause 等 | |
| requests | Apache-2.0 | |
| winocr、winrt（Windows 執行階段的 Python 繫結） | MIT | Windows OCR 備援引擎 |
| rapidocr | Apache-2.0 | |
| onnxruntime | MIT | |
| llama-cpp-python（含 llama.cpp / ggml，離線翻譯的推論引擎） | MIT | 預編譯 wheel 內含 llama.cpp 與 ggml 的動態函式庫；索引寫在 `requirements-core.txt` |
| OpenCC | Apache-2.0 | 簡繁轉換與台灣用語詞典 |

## 模型

| 模型 | 授權 | 來源 |
|---|---|---|
| PP-OCRv6（偵測與辨識，隨 `rapidocr` 套件安裝） | 隨套件：Apache-2.0 | 模型源自 PaddleOCR（Apache-2.0）；PP-OCRv6 模型卡尚未逐一核對 |
| Qwen2.5-0.5B-Instruct GGUF（Q8_0） | Apache-2.0 | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ `9217f5db79a29953eb74d5343926648285ec7e67`（官方釋出，未轉換） |
| Qwen2.5-1.5B-Instruct GGUF（Q4_K_M 備用、Q8_0） | Apache-2.0 | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` @ `91cad51170dc346986eccefdc2dd33a9da36ead9`（官方釋出，未轉換） |
| Qwen2.5-3B-Instruct GGUF（Q4_K_M，**程式預設使用，但不隨程式散布**） | **Qwen Research License（僅限研究與評估，非 Apache-2.0）** | `Qwen/Qwen2.5-3B-Instruct-GGUF` @ `7dabda4d13d513e3e842b20f0d435c732f172cbe`；授權檔隨模型資料夾。**不隨程式散布**：授權把「非商用」定義為「僅供研究或評估」，並有散布條件，所以預設使用它，但由使用者自己下載（`scripts/fetch_translation_models.py`，預設就是取得 3B 與 1.5B）；沒有下載時程式自動改用可以散布的 1.5B（Apache-2.0） |
| Qwen3-1.7B GGUF（Q8_0，選用；官方只有 Q8_0） | Apache-2.0 | `Qwen/Qwen3-1.7B-GGUF` @ `90862c4b9d2787eaed51d12237eafdfe7c5f6077`（官方釋出，未轉換；可隨程式散布） |
| Qwen3-4B GGUF（Q4_K_M，選用） | Apache-2.0 | `Qwen/Qwen3-4B-GGUF` @ `bc640142c66e1fdd12af0bd68f40445458f3869b`（官方釋出，未轉換；可隨程式散布） |
| Qwen2.5-7B-Instruct GGUF（Q4_K_M，選用） | Apache-2.0 | `Qwen/Qwen2.5-7B-Instruct-GGUF` @ `bb5d59e06d9551d752d08b292a50eb208b07ab1f`（官方釋出，分成兩個檔） |

每個模型資料夾內有 `manifest.json`（來源、版本、授權、SHA-256）與原模型的授權檔 `LICENSE`，隨程式一起散布時請保留。

已**不再使用**（程式與相依套件都已移除）：CTranslate2、tokenizers，以及先前的 NLLB-200-distilled-600M（CC-BY-NC-4.0）。
評估過但**未採用**的模型（不隨附）：NLLB-200 1.3B、Helsinki-NLP opus-mt（Apache-2.0）。（Qwen3-1.7B 以前在 CTranslate2 時期評估過、未採用；2026-10-15 在 llama.cpp 上重新評估後改為可選用的模型，見 `MODEL_COMPARISON.md`。）
