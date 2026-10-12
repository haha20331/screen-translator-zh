"""離線快速翻譯：NLLB 解碼、語言代碼、失敗判定、模型位置，以及（有模型時）真實翻譯。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.translation import ct2_common
from services.translation.base import TranslationError, list_translators
from services.translation.ct2_common import find_model_dir
from services.translation.fast_translator import (
    DEFAULT_FAST_MODEL_NAME,
    FastTranslator,
    decode_pieces,
)
from utils import chinese


def test_registered_with_its_own_config_section():
    assert "offline_fast" in list_translators()
    assert FastTranslator.config_key == "offline_fast"
    assert FastTranslator.han_is_source is True
    assert FastTranslator.identical_means_failure is False


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["▁", "你好", "，", "世界", "。"], "你好，世界。"),
        (["▁Hello", "▁world", "</s>"], "Hello world"),
        (["▁快速的棕", "<unk>", "狐跳過", "<unk>", "懶狗"], "快速的棕狐跳過懶狗"),  # 詞表缺字的 <unk> 略過
        (["<pad>", "<s>", "▁OK"], "OK"),
        ([], ""),
    ],
)
def test_decode_pieces_matches_sentencepiece_detokenisation(pieces, expected):
    assert decode_pieces(pieces) == expected


def _fake_translator(monkeypatch, hypotheses):
    """FastTranslator 搭配假的 CTranslate2 與分詞器；記錄傳給模型的輸入。"""
    instance = FastTranslator({"offline_fast": {}})
    seen = {}

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(tokens=["▁" + text])

    class FakeCT2Translator:
        def translate_batch(self, batches, **kwargs):
            seen.setdefault("batches", []).extend(batches)  # 每種來源語言各呼叫一次
            seen["kwargs"] = kwargs
            return [SimpleNamespace(hypotheses=[hypotheses[batch[1]]]) for batch in batches]

    instance._translator, instance._tokenizer = FakeCT2Translator(), FakeTokenizer()
    instance.seen = seen
    return instance


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_batches_carry_language_codes_and_force_simplified_target(monkeypatch):
    instance = _fake_translator(
        monkeypatch,
        {"▁こんにちは": ["zho_Hans", "▁", "你好", ","], "▁Save": ["zho_Hans", "▁保存", "文件", "."]},
    )

    out = instance.translate(["こんにちは", "Save"], "auto", "zh-hant")

    assert out == ["你好，", "儲存檔案。"]  # 標點全形化，簡體轉台灣繁體
    codes = {batch[0] for batch in instance.seen["batches"]}
    assert codes == {"jpn_Jpan", "eng_Latn"}
    assert all(batch[-1] == "</s>" for batch in instance.seen["batches"])
    assert instance.seen["kwargs"]["target_prefix"][0] == ["zho_Hans"]
    assert instance.seen["kwargs"]["beam_size"] == 1  # 貪婪解碼：快


@pytest.mark.skipif(not chinese.available(), reason="opencc 未安裝")
def test_echoed_japanese_is_marked_failed(monkeypatch):
    instance = _fake_translator(monkeypatch, {"▁ご主人さま": ["zho_Hans", "▁ご主人さま"], "▁スキ": ["zho_Hans", "▁skip"]})

    out = instance.translate(["ご主人さま", "スキ"], "auto", "zh-hant")

    assert out == ["ご主人さま", "スキ"]
    assert instance.last_failed_count == 2


def test_missing_model_error_names_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setattr(ct2_common, "app_data_dir", lambda: tmp_path / "appdata")
    monkeypatch.setattr(ct2_common, "_app_roots", lambda: [tmp_path / "app"])

    with pytest.raises(TranslationError) as excinfo:
        FastTranslator({"offline_fast": {}})._load()

    message = str(excinfo.value)
    assert DEFAULT_FAST_MODEL_NAME in message and "fetch_translation_models.py" in message


def test_only_chinese_targets_are_supported():
    with pytest.raises(TranslationError, match="中文"):
        FastTranslator({"offline_fast": {}})._translate_batch(["Settings"], "auto", "en")


@pytest.mark.skipif(
    find_model_dir(DEFAULT_FAST_MODEL_NAME) is None
    or FastTranslator.availability_reason() is not None
    or not chinese.available(),
    reason="未安裝快速翻譯模型或相依套件",
)
def test_real_model_is_fast_and_outputs_traditional_chinese():
    import time

    instance = FastTranslator({"offline_fast": {}})
    instance.warmup()
    texts = ["ファイルを保存しますか？", "Save changes before closing?", "設定"]

    start = time.perf_counter()
    out = instance.translate(texts, "auto", "zh-hant")
    elapsed = time.perf_counter() - start

    assert "儲存" in out[0] and "檔案" in out[0]
    assert out[1].startswith("在關閉") or "儲存" in out[1]
    assert out[2] == "設定"
    assert all(chinese.to_traditional(text) == text for text in out)
    assert elapsed < 3.0  # 近乎即時（實測約 0.1–0.3 秒；留餘裕避免慢機器誤判）
    assert instance.last_failed_count == 0
