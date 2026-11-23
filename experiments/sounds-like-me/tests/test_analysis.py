"""The edits rerun's analysis (amendment A5), the relabeller and the series spend cap, on synthetic inputs."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import common
import relabel_rerun as R


def ep(author, arm, pos, copy=False, model="m"):
    return dict(model=model, author=author, arm=arm, position=pos, copy_flag=copy)


def test_paired_is_a_minus_b_over_shared_uncopied_writers():
    eps = [
        ep("a", "edits_fixed", -1.0),
        ep("a", "edits", -0.2),
        ep("b", "edits_fixed", -0.5),
        ep("b", "edits", -0.5),
        ep("c", "edits_fixed", -0.9, copy=True),
        ep("c", "edits", 0.0),
        ep("d", "edits", 0.3),
    ]
    out = analyze.paired(eps, "m", "edits_fixed", "edits")
    assert out["n_authors"] == 2  # c is copy-flagged in one arm, d has no rerun episode
    assert out["value_ci"][0] == pytest.approx(-0.4)


def rows_for(model, fixed, old):
    return [
        dict(model=model, arm="edits_fixed", voice_strict=[fixed]),
        dict(model=model, arm="edits", voice_strict=[old]),
    ]


@pytest.mark.parametrize(
    "fixed,old,reading,hit",
    [(-1.0, 0.0, "hurt", True), (0.0, 0.0, "no detectable effect", True), (0.5, -0.5, "helped", False)],
)
def test_p8_reading_follows_the_interval(fixed, old, reading, hit):
    eps = [
        e
        for i in range(30)
        for e in (
            ep(f"w{i}", "edits_fixed", fixed + (0.01 * (i % 3) if fixed == old else 0), model="gemini-4-carbon"),
            ep(f"w{i}", "edits", old, model="gemini-4-carbon"),
            ep(f"w{i}", "k3_rand", old, model="gemini-4-carbon"),
        )
    ]
    if fixed == old:  # an interval that straddles 0
        eps += [
            ep(f"x{i}", a, v, model="gemini-4-carbon")
            for i in range(10)
            for a, v in (("edits_fixed", -0.05), ("edits", 0.0))
        ]
    _, p8 = analyze.edits_rerun(rows_for("gemini-4-carbon", fixed, old), eps)
    assert p8["reading"] == reading and p8["hit"] is hit


def test_p8_absent_without_a_gemini_rerun():
    _, p8 = analyze.edits_rerun(rows_for("Qwen3-8B", -1.0, 0.0), [ep("a", "edits_fixed", -1, model="Qwen3-8B")])
    assert p8 is None


def test_rerun_arm_is_not_a_preregistered_arm():
    assert R.RERUN_ARM not in analyze.ARMS


def test_relabel_rerun_moves_drafts_to_edits_fixed_and_is_idempotent():
    rows = [dict(model="gemini-4-carbon-editsfix", author="a", brief="b1", arm="edits", job="a|b1|edits")]
    once = R.relabel_rerun(json.loads(json.dumps(rows)))
    assert once[0] == dict(model="gemini-4-carbon", author="a", brief="b1", arm="edits_fixed", job="a|b1|edits_fixed")
    assert R.relabel_rerun(once) == once


def test_relabel_rerun_refuses_a_published_file():
    with pytest.raises(ValueError):
        R.relabel_rerun([dict(model="gemini-4-carbon", author="a", brief="b1", arm="edits", job="a|b1|edits")])


def test_relabel_legacy_maps_only_the_legacy_arm():
    rows = [
        dict(model="m", author="a", brief="b", arm="edits_nopairs", job="a|b|edits_nopairs"),
        dict(model="m", author="a", brief="b", arm="k0", job="a|b|k0"),
    ]
    out = R.relabel_legacy(rows)
    assert [r["arm"] for r in out] == ["edits", "k0"] and out[0]["job"] == "a|b|edits"


@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    here = tmp_path / "sounds-like-me"
    for name in ("sounds-like-me", "sounds-like-me-inkling"):
        (tmp_path / name / "data").mkdir(parents=True)
    monkeypatch.setattr(common, "HERE", here)
    return tmp_path


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")


def test_series_spent_sums_openrouter_across_every_series_ledger(ledgers):
    write(
        ledgers / "sounds-like-me/data/spend.jsonl",
        [
            dict(provider="openrouter", usd=1.0),
            dict(provider="modal", usd=50.0),
            dict(provider="openrouter", usd=-0.25),
        ],
    )
    write(ledgers / "sounds-like-me-inkling/data/spend.jsonl", [dict(provider="openrouter", usd=0.5)])
    assert common.series_spent() == pytest.approx(1.25)


def test_cap_allows_up_to_the_cap_and_not_a_cent_over(ledgers, monkeypatch):
    monkeypatch.setattr(common, "OPENROUTER_CAP", 9.0)
    write(ledgers / "sounds-like-me/data/spend.jsonl", [dict(provider="openrouter", usd=8.0)])
    assert common.openrouter_allowed(1.0) is True
    assert common.openrouter_allowed(1.01) is False
