"""取得離線翻譯模型（llama.cpp 用的 GGUF），放到 ``models/translation/<名稱>/``（不進版控）。

模型是官方釋出的 GGUF，**不需要轉換、也不需要 PyTorch**，只要 ``huggingface_hub``
（``pip install huggingface_hub``，開發環境已有）::

    python scripts/fetch_translation_models.py --list
    python scripts/fetch_translation_models.py                                  # 預設的 3B Q4_K_M 加上備用的 1.5B Q8_0
    python scripts/fetch_translation_models.py --model qwen2.5-7b-instruct-gguf-q4_k_m qwen3-4b-gguf-q4_k_m

來源版本固定在 ``MODELS`` 的 ``revision``，每個模型資料夾附上 ``manifest.json``（來源、授權、SHA-256）
與原模型的授權檔。公司網路若攔截 Hugging Face 的 HTTPS，設定 ``HTTPS_PROXY`` 指向公司代理，
並設定 ``HF_HUB_DISABLE_XET=1``（Xet 傳輸不走代理）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen2.5-3b-instruct-gguf-q4_k_m"  # 程式預設使用的模型（授權：Qwen Research License，不隨程式散布）
FALLBACK_MODEL = "qwen2.5-1.5b-instruct-gguf-q8_0"  # 備用（Apache-2.0，可隨程式散布）：沒有下載 3B 時使用

# Qwen2.5-Instruct 的 1.5B／3B／7B 加上 Qwen3 4B：越大越準也越慢、越吃記憶體。
# ``files`` 有多個時是官方分片（llama.cpp 讀第一片就會自動找到其餘的）。
MODELS = {
    "qwen2.5-1.5b-instruct-gguf-q8_0": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
        "files": ["qwen2.5-1.5b-instruct-q8_0.gguf"],
        "license": "Apache-2.0",
        "note": "備用，沒有 3B 時使用。8 位元量化（約 1.9 GB），最小最快的選擇，Apache-2.0 可隨程式散布。",
    },
    "qwen2.5-3b-instruct-gguf-q4_k_m": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
        "files": ["qwen2.5-3b-instruct-q4_k_m.gguf"],
        "license": "Qwen Research License（僅限非商用）",
        "note": "預設。約 2.1 GB，比 1.5B 準、慢約 1.5 倍，真實字幕上和 7B 差不多。授權不是 Apache-2.0（僅限研究與評估），不隨程式散布，請閱讀隨附的 LICENSE。",
    },
    # Qwen3（混合思考模型；翻譯用非思考模式）：Apache-2.0，連 4B 也可以隨程式散布
    "qwen3-4b-gguf-q4_k_m": {
        "repo": "Qwen/Qwen3-4B-GGUF",
        "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
        "files": ["Qwen3-4B-Q4_K_M.gguf"],
        "license": "Apache-2.0",
        "note": "Qwen3 4B（約 2.5 GB）。Apache-2.0，可隨程式散布；比 Qwen2.5 的 3B 再大一點。",
    },
    "qwen2.5-7b-instruct-gguf-q4_k_m": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "revision": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        "files": [
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ],
        "license": "Apache-2.0",
        "note": "最大（約 4.7 GB，需要約 6 GB 可用記憶體）。準確度最高，CPU 上每句要好幾倍時間；有獨立顯示卡時才建議常用。",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, out_root: Path, force: bool) -> Path:
    """下載一個模型（固定版本）到 ``out_root/<名稱>/``，附上授權檔與 manifest（來源、授權、SHA-256）。"""
    info = MODELS[name]
    output = out_root / name
    if all((output / filename).exists() for filename in info["files"]) and not force:
        print(f"{name} 已存在：{output}（加 --force 重新下載）")
        return output
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(f"缺少套件（{exc.name}）：請先 pip install huggingface_hub。")

    output.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in info["files"]:
        print(f"下載 {info['repo']} / {filename} @ {info['revision'][:8]} …", flush=True)
        downloaded.append(Path(hf_hub_download(info["repo"], filename, revision=info["revision"], local_dir=str(output))))
    try:
        hf_hub_download(info["repo"], "LICENSE", revision=info["revision"], local_dir=str(output))  # 直接放進模型資料夾
    except Exception as exc:  # 授權檔下載失敗不該讓模型白下載，但要提醒補上
        print(f"提醒：沒有取得 LICENSE（{exc}），散布前請補上原模型的授權檔。")
    manifest = {
        "name": name,
        "format": "gguf",
        "source_repo": info["repo"],
        "source_revision": info["revision"],
        "license": info["license"],
        "files": {path.name: _sha256(path) for path in downloaded},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL, FALLBACK_MODEL], choices=sorted(MODELS), help="要取得的模型（可以多個）；預設是 3B 加 1.5B")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "translation")
    parser.add_argument("--list", action="store_true", help="列出可取得的模型")
    parser.add_argument("--force", action="store_true", help="已存在時重新下載")
    args = parser.parse_args()

    if args.list:
        for name, info in MODELS.items():
            print(f"{name}\n  來源 {info['repo']}  授權 {info['license']}\n  {info['note']}")
        return
    for name in args.model:
        fetch(name, args.out, args.force)


if __name__ == "__main__":
    main()
