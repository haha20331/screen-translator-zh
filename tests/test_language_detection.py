"""來源語言判斷、needs_translation 對日文的處理，以及簡體→繁體轉換。"""

from __future__ import annotations

import pytest

from utils import chinese
from utils.language_utils import detect_source_language, has_kana, needs_translation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello World", "en"),
        ("Save changes before closing?", "en"),
        ("こんにちは", "ja"),
        ("ファイルを保存しますか？", "ja"),
        ("心にあった抵抗感無く", "ja"),  # 假名 + 漢字
        ("Save 保存する (Ctrl+S)", "ja"),  # 英日混排只要有假名就是日文
        ("Player1 のHPが 120 回復しました", "ja"),
        ("ｱｲｳ", "ja"),  # 半形片假名
        ("設定", "ja"),  # 純漢字：預設視為日文
        ("20年連続", "ja"),
        ("1234", None),
        ("!!!", None),
        ("A", None),  # 單一拉丁字母不算英文
        ("", None),
    ],
)
def test_detect_source_language(text, expected):
    assert detect_source_language(text) == expected


def test_han_only_text_can_be_left_untouched():
    assert detect_source_language("設定", han_only_as=None) is None
    # 有假名或英文時不受影響
    assert detect_source_language("戻る", han_only_as=None) == "ja"
    assert detect_source_language("Settings", han_only_as=None) == "en"


def test_has_kana():
    assert has_kana("あ") and has_kana("ア") and has_kana("ｱ")
    assert not has_kana("漢字 abc 123")


@pytest.mark.parametrize("target", ["zh", "zh-hant"])
def test_japanese_with_kana_always_needs_translation_into_chinese(target):
    # 上游會因為含漢字就把「心にあった抵抗感無く」當成已經是中文而跳過
    assert needs_translation("心にあった抵抗感無く", target)
    assert needs_translation("ご主人さま", target)


@pytest.mark.parametrize("target", ["zh", "zh-hant"])
def test_han_only_text_is_skipped_unless_declared_foreign(target):
    assert not needs_translation("設定", target)
    assert needs_translation("設定", target, han_is_source=True)


@pytest.mark.parametrize("target", ["zh", "zh-hant"])
def test_english_and_non_text_rules_are_unchanged(target):
    assert needs_translation("Save changes", target)
    assert not needs_translation("12345", target)
    assert not needs_translation("config.json", target)


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
@pytest.mark.parametrize(
    ("simplified", "traditional"),
    [
        ("这个软件需要联网才能使用", "這個軟體需要聯網才能使用"),  # 軟件→軟體（台灣用語）
        ("请检查网络连接后重试。", "請檢查網路連線後重試。"),
        ("鼠标和文件夹", "滑鼠和資料夾"),
        ("已經是繁體", "已經是繁體"),  # 已是繁體不變
        ("", ""),
    ],
)
def test_to_traditional_uses_taiwan_vocabulary(simplified, traditional):
    assert chinese.to_traditional(simplified) == traditional


def test_to_traditional_reports_missing_opencc(monkeypatch):
    import builtins

    monkeypatch.setattr(chinese, "_converters", {})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "opencc":
            raise ImportError("no opencc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="opencc"):
        chinese.to_traditional("软件")


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_to_simplified_converts_taiwan_traditional():
    # 軟體→软件、網路→网络、連線→连接（連同台灣用語一併轉成大陸用語）
    assert chinese.to_simplified("這個軟體需要網路連線") == "这个软件需要网络连接"
    assert chinese.to_simplified("") == ""
