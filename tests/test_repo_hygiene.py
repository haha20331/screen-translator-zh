"""倉庫檔案衛生：避免在 Windows（預設編碼 cp950）上因編碼問題無法安裝或啟動。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_files_are_ascii_only():
    # pip 以系統預設編碼（台灣 Windows 是 cp950）讀需求檔；UTF-8 的中文註解會造成
    # UnicodeDecodeError，整個安裝失敗。需求檔只能放 ASCII。
    offenders = []
    for path in sorted(ROOT.glob("requirements*.txt")):
        for number, line in enumerate(path.read_bytes().decode("utf-8").splitlines(), 1):
            if not line.isascii():
                offenders.append(f"{path.name}:{number}: {line}")
    assert not offenders, "需求檔含有非 ASCII 字元：\n" + "\n".join(offenders)
