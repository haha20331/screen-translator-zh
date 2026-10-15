"""安裝程式的模型清單：檢查 ``installer/models.json``，並產生 Inno Setup 要用的 [Components] 與 [Files]。

清單是唯一的資料來源（來源網址、固定的版本、檔案大小與 SHA-256、授權、推薦與否）。安裝程式在安裝時依使用者勾選
從 Hugging Face 下載，Inno Setup 會用 SHA-256 驗證，驗證不過就不安裝。

    python scripts/installer_models.py --check                 # 清單與程式內的模型清單、fetch_translation_models.py 一致
    python scripts/installer_models.py --check --online        # 另外連到 Hugging Face 核對 SHA-256 與大小（要網路；公司代理設 HTTPS_PROXY）
    python scripts/installer_models.py --generate installer/generated --host https://huggingface.co

``--generate`` 產生 ``models.inc``（給 ``ScreenTranslator.iss`` 引入）與每個模型的 ``manifest.json``（來源、授權、SHA-256，
格式和 ``fetch_translation_models.py`` 寫的一樣）。``--host`` 預設是清單裡的網址；測試安裝程式時可以改成本機伺服器。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CATALOG = ROOT / "installer" / "models.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def load(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(catalog: dict) -> list[str]:
    """回傳問題清單（空的 = 沒問題）。不連網。"""
    import fetch_translation_models as registry

    from services.translation.offline_translator import OFFLINE_MODELS

    problems: list[str] = []
    models = catalog.get("models", [])
    names = [model["name"] for model in models]
    if len(set(names)) != len(names):
        problems.append("清單裡有重複的模型名稱")
    if set(names) != set(registry.MODELS):
        problems.append(f"清單與 scripts/fetch_translation_models.py 的模型不同：{sorted(set(names) ^ set(registry.MODELS))}")
    if set(names) != {model.name for model in OFFLINE_MODELS}:
        problems.append(f"清單與程式內 OFFLINE_MODELS 的模型不同：{sorted(set(names) ^ {model.name for model in OFFLINE_MODELS})}")
    components = [model["component"] for model in models]
    if len(set(components)) != len(components) or not all(_COMPONENT.fullmatch(item) for item in components):
        problems.append("元件代號必須各不相同，且只含英文字母、數字、底線")
    recommended = [model["name"] for model in models if model.get("recommended")]
    if recommended != [registry.DEFAULT_MODEL]:
        problems.append(f"推薦的模型應該只有預設的 {registry.DEFAULT_MODEL}，現在是 {recommended}")
    for model in models:
        info = registry.MODELS.get(model["name"])
        if info is None:
            continue
        for key, value in (("repo", info["repo"]), ("revision", info["revision"]), ("license", info["license"])):
            if model.get(key) != value:
                problems.append(f"{model['name']}：{key} 與 fetch_translation_models.py 不同")
        if [item["name"] for item in model["files"]] != info["files"]:
            problems.append(f"{model['name']}：檔案名稱與 fetch_translation_models.py 不同")
        for item in [*model["files"], model["license_file"]]:
            if not _SHA256.fullmatch(item.get("sha256", "")):
                problems.append(f"{model['name']}/{item['name']}：SHA-256 格式不對")
            if int(item.get("size", 0)) <= 0:
                problems.append(f"{model['name']}/{item['name']}：大小必須大於 0")
        if not model.get("label"):
            problems.append(f"{model['name']}：缺少顯示名稱")
    return problems


def _url(host: str, model: dict, filename: str) -> str:
    return f"{host.rstrip('/')}/{model['repo']}/resolve/{model['revision']}/{filename}"


def check_online(catalog: dict, host: str | None = None) -> list[str]:
    """連到來源核對：大檔用 X-Linked-ETag（LFS 檔的 SHA-256）與 X-Linked-Size，授權檔直接下載算 SHA-256。"""
    import hashlib

    import requests

    host = host or catalog["host"]
    problems: list[str] = []
    for model in catalog["models"]:
        for item in model["files"]:
            response = requests.head(_url(host, model, item["name"]), allow_redirects=False, timeout=60)
            etag = response.headers.get("X-Linked-ETag", "").strip('"')
            size = int(response.headers.get("X-Linked-Size", "0") or 0)
            if etag != item["sha256"] or size != item["size"]:
                problems.append(f"{model['name']}/{item['name']}：來源的 SHA-256／大小（{etag[:12]}…／{size}）與清單不同")
        item = model["license_file"]
        response = requests.get(_url(host, model, item["name"]), timeout=60)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != item["sha256"] or len(response.content) != item["size"]:
            problems.append(f"{model['name']}/{item['name']}：授權檔與來源不同")
    return problems


def manifest(model: dict) -> str:
    """和 fetch_translation_models.py 寫的 manifest.json 同格式。"""
    data = {
        "name": model["name"],
        "format": "gguf",
        "source_repo": model["repo"],
        "source_revision": model["revision"],
        "license": model["license"],
        "files": {item["name"]: item["sha256"] for item in model["files"]},
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def inno_include(catalog: dict, host: str | None = None, manifest_dir: str = "generated\\manifests") -> str:
    """[Components] 與 [Files]（ISPP 引入檔）。Inno Setup 7 的 [Files] 支援 download 旗標與 Hash（SHA-256）。"""
    host = host or catalog["host"]
    lines = [
        "; 由 scripts/installer_models.py 產生，不要手動修改",
        "[Components]",
        'Name: "models"; Description: "翻譯模型（安裝時從網路下載）"; Types: recommended custom',
    ]
    for model in catalog["models"]:
        size = sum(item["size"] for item in model["files"])
        types = "recommended custom" if model.get("recommended") else "custom"
        description = model["label"].replace('"', '""')
        lines.append(
            f'Name: "models\\{model["component"]}"; Description: "{description}"; Types: {types}; ExtraDiskSpaceRequired: {size}'
        )
    lines += ["", "[Files]"]
    for model in catalog["models"]:
        destination = f'{{app}}\\models\\translation\\{model["name"]}'
        component = f'models\\{model["component"]}'
        for item in [*model["files"], model["license_file"]]:
            lines.append(
                f'Source: "{_url(host, model, item["name"])}"; DestName: "{item["name"]}"; DestDir: "{destination}"; '
                f'Hash: "{item["sha256"]}"; ExternalSize: {item["size"]}; Components: {component}; '
                "Flags: external download ignoreversion"
            )
        lines.append(
            f'Source: "{manifest_dir}\\{model["name"]}\\manifest.json"; DestDir: "{destination}"; '
            f"Components: {component}; Flags: ignoreversion"
        )
    return "\n".join(lines) + "\n"


def generate(catalog: dict, out_dir: Path, host: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in catalog["models"]:
        folder = out_dir / "manifests" / model["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(manifest(model), encoding="utf-8")
    (out_dir / "models.inc").write_text(inno_include(catalog, host), encoding="utf-8-sig")  # Inno 讀有 BOM 的 UTF-8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--online", action="store_true", help="搭配 --check：連到來源核對 SHA-256 與大小")
    parser.add_argument("--generate", type=Path, metavar="OUT_DIR")
    parser.add_argument("--host", default=None, help="下載網址的主機（預設用清單裡的；測試用本機伺服器時改這個）")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    catalog = load(args.catalog)
    status = 0
    if args.check or not args.generate:
        problems = check(catalog)
        if args.online:
            problems += check_online(catalog, args.host)
        for problem in problems:
            print("問題：", problem)
        print("模型清單檢查：" + ("沒有問題" if not problems else f"{len(problems)} 個問題"))
        status = 1 if problems else 0
    if args.generate and status == 0:
        generate(catalog, args.generate, args.host)
        print(f"已產生 {args.generate / 'models.inc'} 與各模型的 manifest.json")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
