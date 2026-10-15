"""提示詞調整用的工具（題庫、評分、版本）：題庫要乾淨、評分的判定要對，凍結的基準版本要和正式提示詞一致的部分才算數。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prompt_corpus  # noqa: E402
import prompt_metrics as metrics  # noqa: E402
import prompt_variants  # noqa: E402

from services.translation.offline_translator import ContextEntry  # noqa: E402


# ----------------------------------------------------------------------------- 題庫
def test_scene_ids_are_unique_and_splits_are_known():
    ids = [scene["id"] for scene in prompt_corpus.SCENES]
    assert len(ids) == len(set(ids))
    assert {scene["split"] for scene in prompt_corpus.SCENES} == {"dev", "holdout"}
    assert {scene["kind"] for scene in prompt_corpus.SCENES} == {"dialogue", "lyrics"}


def test_every_line_has_a_source_and_a_reference():
    for scene in prompt_corpus.SCENES:
        assert scene["lines"], scene["id"]
        for source, reference in scene["lines"]:
            assert source.strip() and reference.strip(), (scene["id"], source)


def test_the_source_language_matches_the_scene_language():
    import re

    kana = re.compile(r"[\u3040-\u30ff]")
    for scene in prompt_corpus.SCENES:
        for source, _reference in scene["lines"]:
            if scene["lang"] == "ja":
                assert kana.search(source), (scene["id"], source)
            else:
                assert source.isascii(), (scene["id"], source)


def test_there_are_sound_effect_lines_to_test_skipping_in_both_languages_and_both_kinds():
    skips = {(s["lang"], s["kind"]) for s in prompt_corpus.SCENES for _src, ref in s["lines"] if ref == prompt_corpus.SKIP}
    assert skips >= {("ja", "dialogue"), ("ja", "lyrics"), ("en", "dialogue"), ("en", "lyrics")}


def test_sound_lines_are_never_given_a_translation_and_real_lines_are_never_marked_skip():
    for scene in prompt_corpus.SCENES:
        for source, reference in scene["lines"]:
            if reference == prompt_corpus.SKIP:
                assert len(source) <= 12, (scene["id"], source)  # 擬聲詞都很短；長句不會是擬聲詞


def test_log_scenes_split_on_gaps_and_collapse_repeats(tmp_path):
    rows = [
        ("2026-10-14 10:00:00", "さくら"),
        ("2026-10-14 10:00:10", "さくら"),  # 連續重複：只留一句
        ("2026-10-14 10:00:20", "つき"),
        ("2026-10-14 11:00:00", "ほし"),  # 隔了一小時：新的一段
        ("2026-10-14 11:00:05", "English only"),  # 沒有假名：不算
    ]
    text = "# 註解\n" + "\n".join(f"{t}\t{t}\t{s}\t譯\t正常\toffline/x\t(0,0)" for t, s in rows) + "\n"
    (tmp_path / "2026-10-14.txt").write_text(text, encoding="utf-8")
    original = prompt_corpus.LOG_PICKS
    prompt_corpus.LOG_PICKS = [(1, 0, 5, "dev"), (2, 0, 5, "holdout"), (3, 0, 5, "dev")]
    try:
        scenes = prompt_corpus.log_scenes(tmp_path)
    finally:
        prompt_corpus.LOG_PICKS = original

    assert [s["id"] for s in scenes] == ["log-1", "log-2"]  # 沒有第 3 段
    assert [line[0] for line in scenes[0]["lines"]] == ["さくら", "つき"]
    assert [line[0] for line in scenes[1]["lines"]] == ["ほし"]
    assert scenes[0]["split"] == "log-dev" and scenes[1]["split"] == "log-holdout"


# ----------------------------------------------------------------------------- 評分
def test_chrf_is_100_for_identical_text_zero_for_disjoint_and_ignores_punctuation():
    assert metrics.chrf("我不會忘記你", "我不會忘記你") == pytest.approx(100.0)
    assert metrics.chrf("我不會忘記你。", "我不會忘記你") == pytest.approx(100.0)
    assert metrics.chrf("天氣很好", "吃飯睡覺") == 0.0
    assert metrics.chrf("", "我") == 0.0


def test_chrf_rewards_overlap_and_prefers_covering_the_reference():
    close = metrics.chrf("在星空下共舞吧", "在星空下共舞")
    far = metrics.chrf("在海邊散步", "在星空下共舞")
    assert 0 < far < close < 100


def test_classical_and_pronoun_helpers():
    assert metrics.classical_chars("在約定之處等候") == "之"
    assert metrics.classical_chars("在約定的地方等你") == ""
    assert metrics.has_pronoun("你還沒回去嗎") and not metrics.has_pronoun("還沒回去嗎")


def test_longest_common_run_finds_copied_context():
    assert metrics.longest_common_run("那是一段彷彿一碰就碎的戀情", "彷彿一碰就碎的戀情吧") == 9
    assert metrics.longest_common_run("你好", "再見") == 0


def _row(source, reference, output, failed=False, calls=1, scene="s"):
    return {"scene": scene, "i": 0, "src": source, "ref": reference, "out": output, "failed": failed, "calls": calls, "ms": 100.0}


def test_classify_tells_skipped_from_failed_and_flags_wrong_skips():
    skipped = metrics.classify(_row("ガキン！", "[SKIP]", "ガキン！"), None)
    assert skipped["skipped"] and not skipped["missed_skip"] and not skipped["false_skip"]
    missed = metrics.classify(_row("ガキン！", "[SKIP]", "哐噹！"), None)
    assert missed["missed_skip"] and not missed["skipped"]
    wrong = metrics.classify(_row("おかえり", "歡迎回來", "おかえり"), None)
    assert wrong["false_skip"]
    failed = metrics.classify(_row("おかえり", "歡迎回來", "おかえり", failed=True), None)
    assert failed["failed"] and not failed["skipped"] and not failed["false_skip"]


def test_classify_flags_kana_classical_subjects_length_and_leaks():
    assert metrics.classify(_row("ありがとう", "謝謝", "謝謝ます"), None)["kana_foreign"]
    assert metrics.classify(_row("約束の場所", "約定的地方", "約定之處"), None)["classical"]
    added = metrics.classify(_row("まだ帰ってないの？", "還沒回去嗎？", "你還沒回去嗎？"), None)
    assert added["subject_added"] and not added["subject_dropped"]
    dropped = metrics.classify(_row("まだ帰ってないの？", "你還沒回去嗎？", "還沒回去嗎？"), None)
    assert dropped["subject_dropped"]
    assert metrics.classify(_row("はい", "好", "好好好好好好好好好好好好"), None)["too_long"]
    assert metrics.classify(_row("長い文章です", "這是一段很長很長的文章", "文"), None)["too_short"]
    previous = _row("前の句", "上一句", "那是一段彷彿一碰就碎的戀情", scene="s")
    copied = metrics.classify(_row("次の句", "下一句", "那是一段彷彿一碰就碎的戀情吧"), previous)
    assert copied["leak"]
    repeated = metrics.classify(_row("前の句", "上一句", "那是一段彷彿一碰就碎的戀情"), previous)
    assert not repeated["leak"]  # 原文一樣，譯文一樣是應該的


def test_summarize_counts_and_scores_a_skipped_real_line_as_zero():
    rows = [
        _row("おかえり", "歡迎回來", "歡迎回來"),
        _row("ガキン！", "[SKIP]", "ガキン！"),
        _row("好きだよ", "我喜歡你", "好きだよ"),  # 該翻卻被略過
    ]
    summary = metrics.summarize(rows)

    assert summary["n"] == 3
    assert summary["counts"]["skipped"] == 2 and summary["counts"]["false_skip"] == 1
    assert summary["chrf"] == pytest.approx(50.0)  # (100 + 0) / 2：被略過的真句子算零分，擬聲詞不計分


# ----------------------------------------------------------------------------- 版本
@pytest.mark.parametrize(("spec_name", "style"), [("V4", "en"), ("S7", "zh")])
def test_the_experiment_specs_build_the_same_prompts_as_the_production_styles(spec_name, style):
    """v4（調整前、英文）＝正式程式碼的 en 風格（Qwen3 還在用）；s7（中文）＝ zh 風格（Qwen2.5 系列）。
    實驗版本與正式提示詞一旦對不上，報告裡的數字就不是在量正式的提示詞。"""
    from services.translation import offline_translator as module

    spec = getattr(prompt_variants, spec_name)
    history = [ContextEntry("前の句", "上一句", "ja", 0.0), ContextEntry("Thanks", "謝謝", "en", 1.0)]
    for language in ("ja", "en"):
        for lyrics in (False, True):
            for skip in (False, True):
                assert spec.static_prompt(language, lyrics, skip) == module.static_prompt(language, lyrics, skip, style)
    assert spec.history_prompt(history) == module.history_prompt(history, style)
    opening = "<think>\n\n</think>\n\n"
    assert spec.user_prompt("こんにちは", "ja", opening) == module.user_prompt("こんにちは", "ja", opening, style)


def test_install_replaces_the_three_prompt_builders_and_nothing_else():
    class Fake:
        PROMPT_VERSION = 99

    fake = Fake()
    prompt_variants.install(fake, prompt_variants.V4)

    assert fake.static_prompt("ja").startswith("<|im_start|>system\n")
    assert fake.user_prompt("原文", "en").endswith("<|im_start|>assistant\n")
    assert fake.history_prompt([]) == ""
    assert fake.PROMPT_VERSION == 99


def test_unknown_variants_are_rejected_with_the_list_of_known_ones():
    with pytest.raises(SystemExit) as error:
        prompt_variants.get("不存在")
    assert "v4" in str(error.value)
