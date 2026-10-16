# 打包成 Windows 安裝程式

產物是一個 `ScreenTranslator-<版本>-Setup.exe`（約 115 MB）加一個 `.sha256` 檔。**翻譯模型不在安裝程式裡**：使用者在安裝精靈的「選擇元件」頁勾選要下載的模型（推薦 Qwen2.5 3B，預設勾選），安裝時才從 Hugging Face 下載並驗證 SHA-256。

為什麼模型不內附：3B 的授權（Qwen Research License）只准研究與評估、不可商用，不能隨程式散布，由使用者自己的電腦下載；其他模型雖然是 Apache-2.0，但每個 0.7～4.7 GB，一起包進去安裝程式會變成好幾 GB。

## 需要的工具

| 工具 | 說明 |
|---|---|
| Windows 10／11 x64、Python 3.12 | 建置用的乾淨環境：`python -m venv .venv-build`，再 `.venv-build\Scripts\python.exe -m pip install -r requirements-core.txt -r requirements-build.txt`（不要用開發用的 `.venv`：裡面有 WebEngine、測試與開發工具，會被打包進去、或讓建置變慢） |
| [Inno Setup 7](https://jrsoftware.org/isdl.php) | 要 **7.x**（`[Files]` 的 `download` 旗標與 `Hash`、官方的繁體中文介面都是 7 才有）。每位使用者安裝就夠（`%LOCALAPPDATA%\Programs\Inno Setup 7`），腳本會自己找 |

## 建置

```powershell
.\scripts\build_installer.ps1 -Version 0.3.0-beta
# → release\ScreenTranslator-0.3.0-beta-Setup.exe 與 .sha256
```

腳本依序做：① 檢查模型清單（`installer/models.json`）與程式內的模型一致；② PyInstaller 打包成資料夾（`ScreenTranslator.spec` → `dist\ScreenTranslator\`，約 380 MB）；③ 收集所有第三方套件的授權檔到 `dist\ScreenTranslator\licenses\`；④ 由模型清單產生安裝程式的模型元件（`installer\generated\`，不進版控）；⑤ Inno Setup 編譯。選項：`-SkipAppBuild`（程式沒改，只重包安裝程式）、`-CheckModelsOnline`（連到 Hugging Face 再核對一次 SHA-256 與大小）、`-SignWithReleaseCertificate`（簽章，見下）。

選資料夾式（onedir）而不是單一檔案：單檔每次啟動都要把幾百 MB 解壓到暫存資料夾（慢，也常被防毒軟體攔下）。程式只含 CPU 版的 llama.cpp；Vulkan 版要自己從原始碼環境安裝（`requirements-vulkan.txt`）。

## 安裝程式做了什麼

- **每位使用者安裝**（預設 `%LOCALAPPDATA%\Programs\ScreenTranslator`），不需要系統管理員權限；介面是繁體中文。需要 Windows 10 1809 以上、64 位元。
- **選擇元件頁**有三種預設：「建議安裝」（程式＋3B）、「只安裝程式」、「自訂」；自訂時可以勾選任意個模型。勾選 3B 時會多一個提示，說明它的授權（Qwen Research License，只能研究與評估），要使用者按「是」才繼續下載。
- **下載與驗證**：每個檔案用固定版本的網址（`https://huggingface.co/<repo>/resolve/<commit>/<檔名>`）下載，Inno Setup 比對 SHA-256，不符就拒絕安裝、整個安裝**回復原狀**（已實測：把授權檔改一個位元組，安裝以代碼 5 結束，安裝資料夾不留任何檔案）。每個模型資料夾另外放原模型的 `LICENSE` 和 `manifest.json`（來源、版本、授權、SHA-256）。
- 重新執行安裝程式可以**補下載其他模型**：已經裝好的模型不會被刪（已實測）。解除安裝會移除安裝程式下載的模型；使用者自己放進 `models\translation\` 的模型留著。使用者資料（設定、翻譯快取、輸出的文字檔）在 `%LOCALAPPDATA%\ScreenTranslator\`，解除安裝不會刪。
- 沒有選任何模型時會提醒一次；程式找不到模型時，錯誤訊息會告訴使用者「重新執行安裝程式勾選模型，或把 .gguf 檔放到 …\models\translation\<名稱>\」。

**命令列**（靜默安裝，供測試或大量部署）：

```powershell
.\ScreenTranslator-0.3.0-beta-Setup.exe /VERYSILENT /SUPPRESSMSGBOXES /DIR="D:\Apps\ScreenTranslator" /TYPE=recommended
.\ScreenTranslator-0.3.0-beta-Setup.exe /VERYSILENT /COMPONENTS="app,models,models\qwen25_1_5b"   # 指定元件
.\ScreenTranslator-0.3.0-beta-Setup.exe /VERYSILENT /TYPE=minimal                                  # 只裝程式
```

元件代號在 `installer/models.json` 的 `component`（`qwen25_3b`、`qwen25_1_5b`、`qwen3_1_7b`、`qwen25_0_5b`、`qwen25_1_5b_q8`、`qwen3_4b`、`qwen25_7b`）。

## 模型清單

`installer/models.json` 是唯一的資料來源：每個模型的來源 repo、固定的 commit、每個檔案的大小與 SHA-256、授權、顯示名稱、是否推薦。SHA-256 與大小已經和 Hugging Face 的 `X-Linked-ETag`／`X-Linked-Size` 核對過，授權檔也已下載比對。`scripts/installer_models.py --check` 檢查它與 `scripts/fetch_translation_models.py`、程式內的 `OFFLINE_MODELS` 一致（測試 `tests/test_installer_models.py` 每次都跑）；`--check --online` 連線再核對一次。

新增或更新模型：先在 `scripts/fetch_translation_models.py` 的 `MODELS` 與 `services/translation/offline_translator.py` 的 `OFFLINE_MODELS` 加上，用 fetch 腳本下載到 `models/translation/`，再把該模型加進 `models.json`（檔案大小與 SHA-256 取自 `manifest.json` 與 Hugging Face），最後 `installer_models.py --check --online`。

## 測試安裝程式（不必真的下載好幾 GB）

```powershell
# 1. 用本機已下載的模型起一個伺服器（用 Hugging Face 的網址格式提供）
python scripts\serve_models.py                       # 127.0.0.1:8765；--tamper <模型名稱> 會把該模型的授權檔改一個位元組
# 2. 另一個視窗：建一個「從本機伺服器下載」的測試安裝程式
.\scripts\build_installer.ps1 -Version 0.3.0-beta -ModelHost http://127.0.0.1:8765 -OutputDirectory release\test
# 3. 靜默安裝到暫存資料夾，再檢查
.\release\test\ScreenTranslator-0.3.0-beta-Setup.exe /VERYSILENT /SUPPRESSMSGBOXES /DIR="$env:TEMP\st" /COMPONENTS="app,models,models\qwen25_0_5b" /LOG="$env:TEMP\st.log"
```

要檢查的：模型資料夾裡有 `.gguf`、`LICENSE`、`manifest.json`；`licenses\` 有各套件的授權；`$env:TEMP\st\ScreenTranslator.exe` 能啟動並載入模型（`SCREEN_TRANSLATOR_SMOKE=1`、`SCREEN_TRANSLATOR_SMOKE_SECONDS=40` 讓它跑 40 秒後自己結束，並把 `LOCALAPPDATA` 指到暫存資料夾，不要動到真正的使用者資料）；`unins000.exe /VERYSILENT` 解除安裝後資料夾消失；`--tamper` 時安裝以非零代碼結束、不留檔案。測試安裝程式**不能對外發佈**（它的下載網址是本機）。

## 授權與散布

- `THIRD_PARTY_LICENSES.md` 與 `licenses\` 隨安裝程式散布；PySide6 與 pynput 是 LGPL-3.0，資料夾式打包讓 Qt 的 DLL 保持獨立檔案、使用者可以替換。
- 安裝程式沒有程式碼簽章，Windows SmartScreen 會警告，使用者按「其他資訊 → 仍要執行」。散布時一併提供 `.sha256`，讓使用者核對下載的檔案。有簽章憑證的話：`build_installer.ps1 -SignWithReleaseCertificate`（呼叫 `scripts/sign_release.ps1`，憑證從環境變數 `WINDOWS_CERTIFICATE_PFX_BASE64`、`WINDOWS_CERTIFICATE_PASSWORD` 取得）。

## 已知限制

- **網路**：下載由 Inno Setup 進行，用 Windows 的網路設定（Inno 文件：會自動使用系統的代理設定）。已實測：在這台公司網路的電腦上，不經任何額外設定直接從 `huggingface.co` 下載 0.5B 模型（676 MB）約 84 秒，SHA-256 驗證通過。環境變數 `HTTPS_PROXY` **沒有驗證**會不會被讀，不要依賴它；網路需要代理又沒設進 Windows 時，安裝程式下不了模型，可以事後用 `scripts/fetch_translation_models.py`（認 `HTTPS_PROXY`）下載再放進 `models\translation\`。
- **安裝版只有 CPU 版 llama.cpp，不能用 GPU**：設定 →「運算裝置」仍會列出這台電腦的顯示卡，但都標成「不能用」並說明原因（要改裝 Vulkan 版，見 `requirements-vulkan.txt`）。要不要把 Vulkan 版放進安裝程式（做成可勾選的元件）還沒決定。
- 安裝精靈的畫面（元件頁的文字、勾選預設）只用靜默安裝與紀錄檔驗證過，沒有逐頁用眼睛看過。
- 沒有自動更新：新版要重新下載安裝程式（使用者資料與已下載的模型會保留）。
