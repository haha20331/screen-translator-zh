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

# 單元測試（約 70 秒）
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:cacheprovider

# 靜態檢查
ruff check .

# OCR 評估：合成樣本（需要 Windows 字型）
python scripts/ocr_benchmark.py --font-dir /mnt/c/Windows/Fonts

# OCR 評估：真實截圖（<圖名>.src.txt 放「原文」答案，可省略）
python scripts/ocr_benchmark.py --images val_data
```

`val_data/` 內 `<圖名>.txt` 是**譯文**參考答案（供翻譯驗證用），OCR 的原文答案要另存為 `<圖名>.src.txt`。

## Windows：執行程式

Windows 端目前沒有安裝 Python，請先安裝 Python 3.12（<https://www.python.org/downloads/windows/>，安裝時勾選 **Add python.exe to PATH**）。

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

目前狀態：OCR 已是 RapidOCR（離線）。翻譯仍是上游的線上服務（預設 Google 免費），**離線翻譯、繁體轉換與 txt 輸出尚未完成**（見 `PLAN.md` 里程碑）。

若 PowerShell 擋掉啟用腳本，先執行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

## 環境變數（除錯用）

| 變數 | 作用 |
|---|---|
| `SCREEN_TRANSLATOR_CONFIG` | 指定設定檔路徑（測試用） |
| `SCREEN_TRANSLATOR_NO_WARMUP=1` | 啟動時不預熱 OCR 模型 |
| `SCREEN_TRANSLATOR_SMOKE=1` | 啟動後自動結束（冒煙測試） |
