# 上游來源與修改紀錄

本專案以 [xuange-hu/screen-translator](https://github.com/xuange-hu/screen-translator) 為基底。

| 項目 | 內容 |
|---|---|
| 上游 commit | `e2f7865`（2026-10-10，`quality: ruff lint 门禁 + 静默捕获补日志 + ARCHITECTURE 路线图 + README 徽章`） |
| 上游授權 | MIT，Copyright (c) 2026 ScreenTranslator contributors（見 `LICENSE`，散布時必須保留） |
| 匯入日期 | 2026-10-12 |

## 匯入時移除的部分

這些項目與本專案目標無關，其中前兩項會讓程式在使用者機器上連到第三方 GitHub repo 下載並執行程式碼，對外散布前必須移除。

- **自動更新器**：`services/update_service.py`、`ui/update_tasks.py`、設定頁「軟件更新」區塊、`Application.install_update`、`updates.*` 設定。原本指向第三方 repo `nimbus-translate/screen-translator` 的 Releases。
- **PaddleOCR 元件下載器與引擎**：`services/ocr/component_manager.py`、`services/ocr/paddle_ocr.py`、`ui/ocr_component_tasks.py`、`paddle_component_worker.py`、`paddle_runtime_hook.py`、`paddle_component.spec`、`requirements-paddle.txt`、相關 `scripts/*.ps1`、`ocr.paddle.*` 設定。OCR 後端改由 RapidOCR 取代。
- **Authenticode 驗證**：`services/authenticode.py`，僅供上述更新器與元件下載器使用。
- **上游發佈流程與社群檔案**：`.github/workflows/release-windows.yml`、`.github/ISSUE_TEMPLATE/`、`.github/PULL_REQUEST_TEMPLATE.md`、完整版 `build.spec`（含 Paddle）。
- 「關於」對話框的上游 repo 連結。

測試因此由 287 通過 / 4 失敗（4 個失敗皆為 Linux 上跑 Windows 專用的 `test_authenticode.py`）變為 240 通過 / 3 跳過 / 0 失敗，數量差來自被移除功能的測試。

## 尚未處理（之後的步驟）

- `README.md`、`CONTRIBUTING.md`、`SECURITY.md`、`CODE_OF_CONDUCT.md`、`installer/`、`scripts/build_lite.ps1`、`scripts/sign_release.ps1`、`build-lite.spec` 仍是上游內容，打包階段（M9）一併改寫。
- `utils/language_utils.py` 的 `to_paddle_lang` 已無呼叫者，加入 RapidOCR 後端時改寫為對應的語言對照。

## 如何比對上游

```
git clone https://github.com/xuange-hu/screen-translator.git upstream
git -C upstream checkout e2f7865
diff -r upstream .   # 排除 .git
```
