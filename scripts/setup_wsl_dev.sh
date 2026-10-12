#!/usr/bin/env bash
# 建立 WSL 開發環境（測試、OCR/翻譯驗證用，不含 GUI 執行）。
#
#   bash scripts/setup_wsl_dev.sh              # 建立 ~/.venvs/ocr-dev
#   VENV=/path/to/venv bash scripts/setup_wsl_dev.sh
#
# pynput（全域快捷鍵）與 winocr（Windows OCR）只能在 Windows 安裝，這裡略過；
# 相關程式碼在測試中以替身處理，不影響其餘測試。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/.venvs/ocr-dev}"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip

FILTERED="$(mktemp)"
trap 'rm -f "$FILTERED"' EXIT
grep -v -i -E '^(pynput|winocr)([=<>~ ]|$)' "$ROOT/requirements-core.txt" > "$FILTERED"
"$VENV/bin/python" -m pip install -q -r "$FILTERED" pytest ruff

echo "完成。啟用：source $VENV/bin/activate"
