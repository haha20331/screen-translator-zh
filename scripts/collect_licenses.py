"""把打包進程式的第三方套件的授權檔收進 ``<資料夾>/licenses/<套件>-<版本>/``，隨安裝程式一起散布。

用建置環境（``.venv-build``）的 Python 執行：它列出環境裡安裝的所有套件，把每個套件 ``.dist-info`` 裡的授權檔
（LICENSE、COPYING、NOTICE…）複製過來；建置工具（PyInstaller 等）不會進到程式裡，排除。
最後寫一份 ``licenses/INDEX.txt``：套件、版本、授權（套件自己宣告的）、授權檔清單。

    python scripts/collect_licenses.py dist/ScreenTranslator
"""

from __future__ import annotations

import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

_BUILD_ONLY = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes", "pip", "setuptools", "wheel"}
_LICENSE_FILE = re.compile(r"(^|/)(licen[sc]e|copying|notice|authors)[^/]*$", re.IGNORECASE)


def _license_summary(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression") or ""
    if expression:
        return expression
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()
    declared = (meta.get("License") or "").strip().splitlines()
    return declared[0][:80] if declared else "（套件沒有宣告，見授權檔）"


def collect(target: Path) -> int:
    out = target / "licenses"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows: list[str] = []
    for dist in sorted(metadata.distributions(), key=lambda item: (item.metadata.get("Name") or "").lower()):
        name = dist.metadata.get("Name") or ""
        if not name or name.lower().replace("_", "-") in _BUILD_ONLY:
            continue
        files = [file for file in (dist.files or []) if _LICENSE_FILE.search(str(file).replace("\\", "/"))]
        folder = out / f"{name}-{dist.version}"
        copied = []
        for file in files:
            source = Path(str(dist.locate_file(file)))
            if source.is_file():
                folder.mkdir(exist_ok=True)
                shutil.copy2(source, folder / source.name)
                copied.append(source.name)
        rows.append(f"{name} {dist.version} | {_license_summary(dist)} | {', '.join(sorted(set(copied))) or '（沒有授權檔）'}")
    (out / "INDEX.txt").write_text(
        "隨程式散布的第三方套件與授權檔（授權文字在各自的資料夾；套件宣告的授權見第二欄）\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    count = collect(Path(sys.argv[1]))
    print(f"已收集 {count} 個套件的授權資訊 → {Path(sys.argv[1]) / 'licenses'}")
