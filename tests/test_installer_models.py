"""安裝程式的模型清單與產生的 Inno Setup 片段：清單要和程式內的模型一致，下載網址固定版本並附 SHA-256。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_licenses  # noqa: E402
import fetch_translation_models as registry  # noqa: E402
import installer_models  # noqa: E402
import serve_models  # noqa: E402


@pytest.fixture(scope="module")
def catalog() -> dict:
    return installer_models.load()


def test_the_catalog_matches_the_models_the_program_knows(catalog):
    assert installer_models.check(catalog) == []


def test_only_the_default_model_is_recommended_and_it_is_the_3b(catalog):
    recommended = [model for model in catalog["models"] if model["recommended"]]

    assert [model["name"] for model in recommended] == [registry.DEFAULT_MODEL] == ["qwen2.5-3b-instruct-gguf-q4_k_m"]


def test_a_catalog_that_drifts_from_the_registry_is_reported(catalog):
    broken = json.loads(json.dumps(catalog))
    broken["models"][0]["revision"] = "0" * 40
    broken["models"][1]["files"][0]["sha256"] = "not-a-hash"
    broken["models"][2]["recommended"] = True
    broken["models"].pop()

    problems = "\n".join(installer_models.check(broken))

    assert "revision" in problems and "SHA-256" in problems and "推薦" in problems and "不同" in problems


def test_download_urls_pin_the_revision_and_hashes_are_sha256(catalog):
    text = installer_models.inno_include(catalog)

    for model in catalog["models"]:
        for item in [*model["files"], model["license_file"]]:
            url = f'{catalog["host"]}/{model["repo"]}/resolve/{model["revision"]}/{item["name"]}'
            assert f'Source: "{url}"' in text
            assert f'Hash: "{item["sha256"]}"' in text
    assert not re.search(r"/resolve/(main|master)/", text)  # 固定版本，不跟著來源更新


def test_every_download_is_verified_has_a_size_and_belongs_to_one_component(catalog):
    lines = [line for line in installer_models.inno_include(catalog).splitlines() if "Flags: external download" in line]

    assert len(lines) == sum(len(model["files"]) + 1 for model in catalog["models"])
    for line in lines:
        assert re.search(r'Hash: "[0-9a-f]{64}"', line) and re.search(r"ExternalSize: \d+", line)
        assert re.search(r"Components: models\\\w+;", line) and "ignoreversion" in line


def test_the_installer_offers_exactly_the_four_models(catalog):
    """使用者 2026-10-22 決定：只留 1.5B Q8、3B Q4、Qwen3 4B Q4、7B Q4。"""
    assert [(model["component"], model["name"]) for model in catalog["models"]] == [
        ("qwen25_3b", "qwen2.5-3b-instruct-gguf-q4_k_m"),
        ("qwen25_1_5b", "qwen2.5-1.5b-instruct-gguf-q8_0"),
        ("qwen3_4b", "qwen3-4b-gguf-q4_k_m"),
        ("qwen25_7b", "qwen2.5-7b-instruct-gguf-q4_k_m"),
    ]


def test_components_list_the_recommended_one_in_the_recommended_type(catalog):
    text = installer_models.inno_include(catalog)
    components = [line for line in text.splitlines() if line.startswith('Name: "models\\')]

    assert len(components) == len(catalog["models"])
    recommended = [line for line in components if "Types: recommended custom" in line]
    assert len(recommended) == 1 and "qwen25_3b" in recommended[0]
    assert all(f'ExtraDiskSpaceRequired: {sum(i["size"] for i in m["files"])}' in line for m, line in zip(catalog["models"], components))


def test_the_host_can_be_replaced_for_testing_without_touching_the_catalog(catalog):
    text = installer_models.inno_include(catalog, host="http://127.0.0.1:8765/")

    assert 'Source: "http://127.0.0.1:8765/Qwen/' in text and "huggingface.co" not in text


def test_manifests_have_the_same_shape_as_the_ones_the_fetch_script_writes(catalog):
    model = next(item for item in catalog["models"] if item["name"] == registry.DEFAULT_MODEL)

    written = json.loads(installer_models.manifest(model))

    assert set(written) == {"name", "format", "source_repo", "source_revision", "license", "files"}
    assert written["files"] == {item["name"]: item["sha256"] for item in model["files"]}
    assert written["source_revision"] == registry.MODELS[registry.DEFAULT_MODEL]["revision"]


def test_generate_writes_the_include_with_a_bom_and_one_manifest_per_model(catalog, tmp_path):
    installer_models.generate(catalog, tmp_path)

    assert (tmp_path / "models.inc").read_bytes().startswith(b"\xef\xbb\xbf")  # Inno 要 BOM 才會把中文當 UTF-8
    manifests = sorted(path.parent.name for path in (tmp_path / "manifests").glob("*/manifest.json"))
    assert manifests == sorted(model["name"] for model in catalog["models"])


# ----------------------------------------------------------------------------- 安裝程式腳本
@pytest.fixture(scope="module")
def script() -> str:
    return (ROOT / "installer" / "ScreenTranslator.iss").read_text(encoding="utf-8-sig")


def test_the_scripts_with_chinese_text_have_a_bom_so_windows_reads_them_as_utf8():
    for path in ("installer/ScreenTranslator.iss", "scripts/build_installer.ps1"):
        assert (ROOT / path).read_bytes().startswith(b"\xef\xbb\xbf"), path


def test_the_generated_models_come_before_the_code_section_and_the_license_prompt_names_the_3b(script, catalog):
    assert script.index('#include "generated\\models.inc"') < script.index("\n[Code]\n")
    component = next(model["component"] for model in catalog["models"] if model["name"] == registry.DEFAULT_MODEL)
    assert f"ResearchLicenseComponent = 'models\\{component}'" in script


def test_the_installer_is_per_user_and_does_not_reuse_the_upstream_app_id(script):
    assert "PrivilegesRequired=lowest" in script
    assert "0D4C39A6-70AA-4B1D-8CB4-A88EE0E74EA7" not in script  # 上游安裝程式的 ID：同一個 ID 會被當成同一個程式升級
    assert re.search(r"AppId=\{\{[0-9A-F-]{36}\}", script)


def test_the_installer_ships_the_licenses_and_cleans_up_only_empty_model_folders(script):
    assert "LICENSE.txt" in script and "THIRD_PARTY_LICENSES.md" in script
    assert "dirifempty" in script and 'Type: filesandordirs; Name: "{app}\\models' not in script  # 不刪使用者自己放的模型


# ----------------------------------------------------------------------------- 測試用的本機模型伺服器
def test_the_test_server_serves_every_file_of_the_catalog(catalog, tmp_path):
    routes = serve_models.build_routes(catalog, tmp_path)

    # 同一個來源 repo 的模型共用同一份授權檔網址，所以是「不重複的網址」
    wanted = {
        f'/{model["repo"]}/resolve/{model["revision"]}/{item["name"]}'
        for model in catalog["models"]
        for item in [*model["files"], model["license_file"]]
    }
    assert set(routes) == wanted
    path = f'/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/{registry.MODELS[registry.DEFAULT_MODEL]["revision"]}/qwen2.5-3b-instruct-q4_k_m.gguf'
    assert routes[path][0] == tmp_path / registry.DEFAULT_MODEL / "qwen2.5-3b-instruct-q4_k_m.gguf"


# ----------------------------------------------------------------------------- 授權檔收集
def test_license_files_of_the_installed_packages_are_collected_with_an_index(tmp_path):
    count = collect_licenses.collect(tmp_path)

    assert count > 10
    index = (tmp_path / "licenses" / "INDEX.txt").read_text(encoding="utf-8")
    assert "pytest" in index.lower() and "PyInstaller".lower() not in index.lower()  # 建置工具不會進到程式裡
    assert any(path.is_file() for path in (tmp_path / "licenses").rglob("*") if path.name.upper().startswith("LICENSE"))
