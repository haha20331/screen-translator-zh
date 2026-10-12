"""下載官方權重並轉成 CTranslate2 int8，放到 ``models/translation/<名稱>/``（不進版控）。

這是一次性的開發步驟，需要「轉換環境」（PyTorch + transformers，約 1 GB，不屬於執行環境）::

    bash scripts/setup_convert_env.sh                 # 建立 ~/.venvs/ocr-convert
    ~/.venvs/ocr-convert/bin/python scripts/fetch_translation_models.py
    ~/.venvs/ocr-convert/bin/python scripts/fetch_translation_models.py --list

來源版本固定在 ``MODELS`` 的 ``revision``，轉換結果附上 ``manifest.json``（來源、授權、SHA-256），
並隨附原模型的授權檔。公司網路若攔截 Hugging Face 的 HTTPS，設定 ``HTTPS_PROXY`` 指向公司代理，
並設定 ``HF_HUB_DISABLE_XET=1``（Xet 傳輸不走代理）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen2.5-1.5b-instruct-ct2-int8"

MODELS = {
    "qwen2.5-1.5b-instruct-ct2-int8": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "license": "Apache-2.0",
        "note": "離線翻譯・標準（預設）。CPU 約 0.5 秒/句（6 執行緒、固定前綴快取）。",
    },
    "qwen2.5-7b-instruct-ct2-int8": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "license": "Apache-2.0",
        "note": "離線翻譯・標準的大模型：有 NVIDIA GPU（約 8 GB 顯示卡記憶體）時 auto 模式會自動優先使用，準確度明顯較高；"
                "CPU 約 2–3 秒/句。轉換時需約 16 GB 記憶體。尚未在實機驗證。",
    },
    "nllb-200-distilled-600m-ct2-int8": {
        "repo": "facebook/nllb-200-distilled-600M",
        "revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "license": "CC-BY-NC-4.0（僅限非商用）",
        "note": "離線翻譯・快速。CPU 批次約 50 ms/句，品質較低。",
        "weights": "*.bin",  # 這個 repo 只有 pytorch_model.bin，沒有 safetensors
    },
}
_COPY_FILES = ["tokenizer.json", "tokenizer_config.json"]


def _allow_patterns(info: dict) -> list[str]:
    """只下載轉換需要的檔案（避免同時抓 .bin 與 .safetensors）。"""
    return ["*.json", info.get("weights", "*.safetensors"), "merges.txt", "vocab.json", "LICENSE*"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "translation")
    parser.add_argument("--list", action="store_true", help="列出可轉換的模型")
    parser.add_argument("--force", action="store_true", help="輸出資料夾已存在時覆寫")
    args = parser.parse_args()

    if args.list:
        for name, info in MODELS.items():
            print(f"{name}\n  來源 {info['repo']}  授權 {info['license']}\n  {info['note']}")
        return

    info = MODELS[args.model]
    if not info["revision"]:
        raise SystemExit(f"{args.model} 尚未固定來源版本：請先到 https://huggingface.co/{info['repo']} 查 commit sha，填入 MODELS。")
    output = args.out / args.model
    if (output / "model.bin").exists() and not args.force:
        raise SystemExit(f"{output} 已存在（加 --force 覆寫）")

    try:
        import ctranslate2
        import transformers
        from ctranslate2.converters import TransformersConverter
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(f"缺少轉換環境的套件（{exc.name}）：請先執行 bash scripts/setup_convert_env.sh，並用該環境的 python 執行本腳本。")

    print(f"下載 {info['repo']} @ {info['revision'][:8]} …")
    source = Path(snapshot_download(info["repo"], revision=info["revision"], allow_patterns=_allow_patterns(info)))
    print(f"轉換為 CTranslate2 int8 → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(str(source), copy_files=_COPY_FILES).convert(str(output), quantization="int8", force=True)

    for license_file in source.glob("LICENSE*"):
        shutil.copy2(license_file, output / license_file.name)
    manifest = {
        "name": args.model,
        "source_repo": info["repo"],
        "source_revision": info["revision"],
        "license": info["license"],
        "quantization": "int8",
        "tools": {"ctranslate2": ctranslate2.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
        "files": {name: _sha256(output / name) for name in ("model.bin", "tokenizer.json", "vocabulary.json") if (output / name).exists()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{output}")


if __name__ == "__main__":
    main()
