# 開發與執行環境

專案同時用到兩個環境，兩者看到的是同一份檔案（WSL 的 `/mnt/d/ocr` 就是 Windows 的 `D:\ocr`）：

| 環境 | 用途 | 能做什麼 |
|---|---|---|
| **WSL**（Linux） | 開發、測試、OCR 與翻譯驗證 | 跑單元測試、OCR 準確度評估、離線翻譯驗證 |
| **Windows**（原生） | 實際執行程式 | 跑 GUI、螢幕擷取、全域快捷鍵、打包 |

GUI、螢幕擷取與快捷鍵只能在 Windows 驗證。

## WSL：開發與測試

第一次建立環境（約 20 秒，只需做一次）：

```bash
cd /mnt/d/ocr
bash scripts/setup_wsl_dev.sh          # 建立 ~/.venvs/ocr-dev
```

之後每次使用：

```bash
cd /mnt/d/ocr
source ~/.venvs/ocr-dev/bin/activate

# 簡易版單元測試（日常用，約 6 分鐘；略過 tests/slow_tests.txt 列的約 40 個慢測試）
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:cacheprovider -m "not slow"

# 整套單元測試（約 990 個，包含慢測試；只在需要時跑）
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:cacheprovider

# 靜態檢查
ruff check .

# OCR 評估：合成樣本（需要 Windows 字型）
python scripts/ocr_benchmark.py --font-dir /mnt/c/Windows/Fonts

# OCR 評估：真實截圖（<圖名>.src.txt 放「原文」答案，可省略）
python scripts/ocr_benchmark.py --images val_data
```

慢測試清單在 `tests/slow_tests.txt`（單一測試 10 秒以上），由 `tests/conftest.py` 標上 `slow`；測試時間變了，先用 `pytest --durations=0 > durations.log` 跑整套（可以分幾個行程平行跑），再用 `python scripts/update_slow_tests.py durations.log` 重新產生。`conftest.py` 還有一個自動夾具，每個測試結束後真的把 `deleteLater()` 的視窗刪掉（沒有事件迴圈時它們會一直留著，每多一個殘留的設定視窗，之後每次 `apply_style` 就多約 0.14 秒，整套測試會越跑越慢）。

`val_data/` 內 `<圖名>.txt` 是**譯文**參考答案（供翻譯驗證用），OCR 的原文答案要另存為 `<圖名>.src.txt`。

## Windows：執行程式

Windows 端目前沒有安裝 Python，請先安裝 Python 3.12（<https://www.python.org/downloads/windows/>，安裝時勾選 **Add python.exe to PATH**）。
若 PowerShell 擋掉啟用腳本，先執行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

```powershell
cd D:\ocr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python main.py
```

啟動後程式常駐在系統匣。預設快捷鍵：

| 快捷鍵 | 動作 |
|---|---|
| `Ctrl+Shift+A` | 框選區域並翻譯 |
| `Ctrl+Shift+M` | 對上次框選的區域開關即時監控（監控中可拖曳、縮放框） |
| `Ctrl+Shift+H` | 顯示 / 隱藏譯文標籤 |
| `Ctrl+Shift+R` | 對上次的區域重新辨識翻譯 |
| `Ctrl+Shift+F` / `Ctrl+Shift+W` | 全螢幕 / 視窗擷取 |

目前狀態：OCR 是 RapidOCR（離線），翻譯是離線的 Qwen2.5（llama.cpp，預設 3B，輸出繁體中文）；顯示過的字幕會依日期附加寫進 txt（`%LOCALAPPDATA%\ScreenTranslator\output\`，設定 → 一般 →「文字輸出」可改位置）。**常駐框、打包等尚未完成**（見 `PLAN.md` 里程碑）。

首次啟動會在背景載入翻譯模型（3B 約 2 GB，約 10 秒）；模型放在 `D:\ocr\models\translation\`（見下一節）。


## 取得翻譯模型

模型檔很大，不進版控（`models/` 已被 `.gitignore` 忽略）。離線翻譯用 llama.cpp 讀官方釋出的 GGUF（Qwen2.5-Instruct），**不需要轉換，也不需要 PyTorch**，只要 `huggingface_hub`。
預設用 3B Q4_K_M；它的授權是 Qwen Research License（僅限研究與評估），所以不隨程式散布，請自行下載。沒有 3B 時程式自動改用 1.5B Q4_K_M（Apache-2.0）。每個模型放在 `models/translation/<名稱>/`。

Windows（PowerShell，在專案的 `.venv` 裡）：

```powershell
python -m pip install -r requirements-core.txt        # 含 llama-cpp-python（預編譯的 CPU wheel，不需要編譯器）
python -m pip install huggingface_hub
python scripts\fetch_translation_models.py            # 不帶參數：3B 加 1.5B
python scripts\fetch_translation_models.py --model qwen2.5-7b-instruct-gguf-q4_k_m   # 另外取得其他模型
```

WSL（開發環境已有 `huggingface_hub`）：

```bash
source ~/.venvs/ocr-dev/bin/activate
export HTTPS_PROXY=http://<公司代理>:<port>            # 公司網路攔截 huggingface.co 的憑證時需要
export HF_HUB_DISABLE_XET=1                            # Xet 傳輸不走代理
python scripts/fetch_translation_models.py
```

腳本下載固定版本的官方檔，並寫入 `manifest.json`（來源、授權、SHA-256）與原模型的授權檔 `LICENSE`。已有的 `D:\ocr\models` 也能直接給 Windows 端使用。`--list` 列出所有模型：

| 模型 | 用途 | 大小 |
|---|---|---|
| `qwen2.5-3b-instruct-gguf-q4_k_m` | 預設（`auto`），較準；授權僅限研究與評估，不隨程式散布 | 約 2.1 GB |
| `qwen3-1.7b-gguf-q8_0` | Qwen3 1.7B（官方只有 Q8_0）：Apache-2.0，可隨程式散布；比 Qwen2.5 1.5B 準得多、每句約慢一倍 | 約 1.8 GB |
| `qwen3-4b-gguf-q4_k_m` | Qwen3 4B：Apache-2.0；比 Qwen2.5 3B 慢約 4 成、多用約 1.4 GB 記憶體，真實字幕上沒有比較準，不建議 | 約 2.5 GB |
| `qwen2.5-1.5b-instruct-gguf-q4_k_m` | 備用（沒有 3B 時自動使用），最快；Apache-2.0，可隨程式散布 | 約 1.1 GB |
| `qwen2.5-1.5b-instruct-gguf-q8_0` | 比 1.5B Q4_K_M 少一些小錯，慢約 1.4 倍 | 約 1.9 GB |
| `qwen2.5-0.5b-instruct-gguf-q8_0` | 最小最快，日文品質差，只適合當速度參考 | 約 0.7 GB |
| `qwen2.5-7b-instruct-gguf-q4_k_m` | 有獨立顯示卡（可用顯示卡記憶體至少 5.5 GB）時 `auto` 優先使用；CPU 或內顯上每句 1 秒以上、要約 7 GB 記憶體 | 約 4.7 GB |

在設定 → 翻譯 →「離線翻譯模型」選模型（預設 `auto`：有獨立顯示卡且放了 7B 用 7B，否則用 3B，沒有 3B 用 1.5B）。換模型比較速度與品質：`python scripts/compare_translation_models.py --reps 3`（結果見 `MODEL_COMPARISON.md`）。
舊的 `*-ct2-int8`、`nllb-*` 資料夾已不再使用，可以刪除。

## 用 GPU 加速（選用，Windows）

預設安裝的是 CPU 版 llama-cpp-python（`requirements-core.txt` 釘 0.3.33）。**注意：官方的 CPU wheel 0.3.33 的 `ggml-cpu.dll` 含 AVX-512 指令，在沒有 AVX-512 的 CPU（多數 Intel 12～14 代、AMD Zen4 以前）載入模型會「非法指令」（0xC000001D）當掉**；安裝版不受影響（它用 Vulkan wheel 的 AVX2 版核心）。開發環境遇到這個問題有兩個辦法：改裝下面的 Vulkan 版（有 Vulkan 載入器的電腦才載得進），或先執行 `python scripts/fetch_vulkan_library.py` 再設環境變數 `LLAMA_CPP_LIB_PATH=<專案>\build\cpu_lib`（安裝版的 CPU 版那一套，不需要 Vulkan）。
安裝版同時附 CPU 版與 GPU 版（見 `PACKAGING.md`），不需要照這一節做。
內顯（例如 Intel Iris Xe）或獨立顯示卡可以改裝 Vulkan 版，不需要編譯器：

```powershell
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-vulkan.txt      # 換成 Vulkan 版；換回 CPU 版的指令寫在這個檔案的開頭
```

裝好後，設定 → 翻譯 →「運算裝置」下拉選單會列出這台電腦的每一顆 GPU（型號、獨立顯示卡或內顯、顯示卡記憶體），可以選：**自動**（預設：有可用的 GPU 就用，有獨立顯示卡用獨立顯示卡，沒有才用內顯）、**只用 CPU**，或**指定某一顆 GPU**（模型只放在那顆上）。按「重新偵測」會重新查一次。這台電腦有、但目前安裝的 llama.cpp 用不到的顯示卡（預設的 CPU 版、驅動程式不支援 Vulkan）也會列出來，標灰色並寫明原因。GPU 載入失敗（顯示卡記憶體不足）時自動退回 CPU。
設定值是 `translation.offline.gpu_device`：`"auto"`、`"cpu"`，或 GPU 的名稱（同型號的第二顆是 `"… #2"`；存的是名稱不是編號，換了插槽或驅動程式順序也不會選錯）；選的那顆這次找不到（外接顯示卡拔掉了）就改用自動，設定不會被改掉。`gpu_layers`（預設 -1 = 全部）只管交給 GPU 幾層；舊版用 `gpu_layers: 0` 表示只用 CPU，載入舊設定檔時會自動轉成 `gpu_device: "cpu"`。
指定 GPU 的做法：llama-cpp-python 沒有「指定裝置」的參數（只有 `main_gpu`，編號不含「有獨立顯示卡時被略過的內顯」），所以載入模型的那一瞬間讓 `llama_model_default_params()` 回傳已填好 `devices`（以 NULL 結尾的 ggml 裝置清單）的參數（`llama_backend._only_device`）。已用載入時的日誌（`using device …`）驗證 llama.cpp 照這份清單選裝置。
確認有沒有用到 GPU：看 `%LOCALAPPDATA%\ScreenTranslator\logs\app.log` 裡的「裝置＝GPU（名稱）」、模型載入完成時狀態列的「GPU：名稱」，或設定 → 翻譯 →「測試速度」表格模型名稱後面的「· GPU」（滑鼠移上去看是哪一顆）。

實測（Intel Iris Xe 內顯、i7-13700H）：3B 約快 10～25%、1.5B 約慢 10%，整條管線（OCR＋翻譯）的每張耗時幾乎一樣；差別在 CPU 用量：3B 在 CPU 上約用掉 6 顆邏輯核心，在內顯上只要約 0.4 顆，OCR 與其他程式幾乎不受影響（OCR 約 0.25 秒，CPU 版翻譯時約 0.36 秒）。
內顯和 CPU 共用記憶體，7B 不適合；`auto` 只有偵測到獨立顯示卡（可用顯示卡記憶體至少 5.5 GB）才會自動選 7B。

## 驗證

```bash
source ~/.venvs/ocr-dev/bin/activate
# OCR → 行合併 → 離線翻譯 → 繁體，並與參考譯文（<圖名>.txt）比對
python scripts/translate_benchmark.py --images val_data
```

## 環境變數（除錯用）

| 變數 | 作用 |
|---|---|
| `SCREEN_TRANSLATOR_CONFIG` | 指定設定檔路徑（測試用） |
| `SCREEN_TRANSLATOR_NO_WARMUP=1` | 啟動時不預熱 OCR 模型 |
| `SCREEN_TRANSLATOR_SMOKE=1` | 啟動後自動結束（冒煙測試） |

## 打包成安裝程式

用 `scripts/build_installer.ps1` 建置 Windows 安裝程式（PyInstaller 資料夾式＋Inno Setup 7；翻譯模型不在安裝程式裡，安裝時依勾選下載並驗證 SHA-256）。需要的工具、命令、測試方式與已知限制見 [PACKAGING.md](PACKAGING.md)。提示詞調整的做法與數字見 [PROMPT_TUNING.md](PROMPT_TUNING.md)。
