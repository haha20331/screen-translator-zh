"""scripts/compare_translation_models.py：報表（Markdown）的格式。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_translation_models.py"
    spec = importlib.util.spec_from_file_location("compare_translation_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(script, name, ms, calls=1, rss=1800.0):
    rows = []
    for group, sentences in script.CORPUS.items():
        for sentence in sentences:
            rows.append(
                {"group": group, "source": sentence, "output": f"譯|{sentence}", "ms": ms, "calls": calls,
                 "kana": False, "not_chinese": False, "skipped": group == "擬聲詞"}
            )
    return {"name": name, "threads": 6, "load_seconds": 3.5, "rss_mb": rss, "model": name, "device": "cpu", "rows": rows}


def test_short_names(script):
    assert script.short_name("qwen2.5-1.5b-instruct-gguf-q8_0") == "1.5B Q8_0"
    assert script.short_name("qwen2.5-3b-instruct-gguf-q4_k_m") == "3B Q4_K_M"
    assert script.short_name("qwen3-4b-gguf-q4_k_m") == "Qwen3 4B Q4_K_M"
    assert script.short_name("something-else") == "something-else"


def test_the_report_lists_speed_quality_and_the_ten_sentences(script, tmp_path):
    results = [
        _result(script, "qwen2.5-1.5b-instruct-gguf-q8_0", 160),
        _result(script, "qwen2.5-3b-instruct-gguf-q4_k_m", 310, calls=1.2),
        _result(script, "qwen2.5-7b-instruct-gguf-q4_k_m", 1300, rss=7200.0),
    ]
    target = tmp_path / "report.md"

    script.write_report(results, target, reps=3)

    text = target.read_text(encoding="utf-8")
    assert text.startswith("# 離線翻譯模型比較")
    assert "## 耗時與記憶體" in text and "## 譯文品質的機器指標" in text and "## 逐句並排比較（10 句）" in text
    assert "| 1.5B Q8_0 |" in text and "| 3B Q4_K_M |" in text and "| 7B Q4_K_M |" in text
    assert "基準：1.5B Q8_0" in text  # 最小最快的 1.5B 是基準
    assert "**160**" in text and "**1300**" in text and "7.2 GB" in text
    assert "1.20" in text  # 平均進模型次數
    table = text.split("## 逐句並排比較（10 句）")[1]
    assert len([line for line in table.splitlines() if line.startswith("| ") and "---" not in line]) == 1 + len(script.REPORT_SENTENCES)
    assert "譯\\|夜空に浮かぶ星を数えて" in text  # 表格裡的直線要跳脫，不然欄位會錯位
    assert "(310)" in text  # 逐句耗時


def test_every_report_sentence_is_in_the_corpus(script):
    corpus = {sentence for sentences in script.CORPUS.values() for sentence in sentences}

    assert set(script.REPORT_SENTENCES) <= corpus and len(script.REPORT_SENTENCES) == 10
