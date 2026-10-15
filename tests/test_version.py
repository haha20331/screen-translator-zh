"""Runtime release-version injection regressions."""

from __future__ import annotations

import re
from pathlib import Path

import app.version as version_module


def test_source_version_is_valid_beta_release():
    assert re.match(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$", version_module._SOURCE_VERSION)
    assert version_module.RELEASE_CHANNEL in {"stable", "beta", "alpha"}


def test_frozen_runtime_uses_injected_build_version(tmp_path, monkeypatch):
    (tmp_path / "build-version.txt").write_text("0.2.7", encoding="ascii")
    monkeypatch.setattr(version_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(version_module.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert version_module._runtime_version() == "0.2.7"


def test_frozen_runtime_rejects_malformed_injected_build_version(tmp_path, monkeypatch):
    (tmp_path / "build-version.txt").write_text("totally-not-a-version", encoding="ascii")
    monkeypatch.setattr(version_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(version_module.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert version_module._runtime_version() == version_module._SOURCE_VERSION


def test_build_spec_injects_release_version_file():
    spec = (Path(__file__).parents[1] / "ScreenTranslator.spec").read_text(encoding="utf-8")

    assert "SCREEN_TRANSLATOR_BUILD_VERSION" in spec
    assert 'version_file = metadata_dir / "build-version.txt"' in spec
    assert "(str(version_file), \".\")" in spec


def test_the_build_files_default_to_the_source_version():
    """程式、PyInstaller 設定、建置腳本、安裝程式的預設版本要一致，否則不帶參數建出來的版本會對不上。"""
    root = Path(__file__).parents[1]
    expected = version_module._SOURCE_VERSION

    assert f'"SCREEN_TRANSLATOR_BUILD_VERSION", "{expected}"' in (root / "ScreenTranslator.spec").read_text(encoding="utf-8")
    assert f'$Version = "{expected}"' in (root / "scripts" / "build_installer.ps1").read_text(encoding="utf-8-sig")
    assert f'#define MyAppVersion "{expected}"' in (root / "installer" / "ScreenTranslator.iss").read_text(encoding="utf-8-sig")
