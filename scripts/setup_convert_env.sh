#!/usr/bin/env bash
# 建立「模型轉換環境」（只在取得/更新翻譯模型時需要；執行程式不需要它）。
#
#   bash scripts/setup_convert_env.sh                # 建立 ~/.venvs/ocr-convert
#
# 需要 PyTorch（CPU 版，約 200 MB）與 transformers。公司網路若攔截 HTTPS，先設定
#   export HTTPS_PROXY=http://<公司代理>:<port>
# PyPI 與 PyTorch 的 CPU 套件來源都能經代理取得。
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/ocr-convert}"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch
"$VENV/bin/python" -m pip install -q transformers ctranslate2 huggingface_hub sentencepiece
echo "完成。取得預設模型："
echo "  HF_HUB_DISABLE_XET=1 $VENV/bin/python scripts/fetch_translation_models.py"
