"""
Tests for the second attempt's helpers: no model, no API calls.

    .venv/bin/python -m pytest experiments/persona-vectors/test_v2.py -q
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest

import common_v2
from common_v2 import gate, intent_prompt, kept_pairs, load_versions, organism_items, references


def row(kind, q, trait, coherence, k=0, s=0):
    return {"kind": kind, "q": q, "k": k, "s": s, "trait": trait, "coherence": coherence}


def test_filter_is_the_papers():
    rows = [row("pos", 0, 50, 50), row("neg", 0, 49.9, 50),  # kept: the edges of every bar
            row("pos", 1, 49.9, 90), row("neg", 1, 0, 90),  # pos below 50
            row("pos", 2, 90, 90), row("neg", 2, 50, 90),  # neg not below 50
            row("pos", 3, 90, 49), row("neg", 3, 0, 90),  # pos incoherent
            row("pos", 4, None, 90), row("neg", 4, 0, 90)]  # judge refused  # fmt: skip
    kept, keys = kept_pairs(rows)
    assert kept == [(0, 0, 0)] and len(keys) == 5


def test_gate():
    good = {"correct": [0, 19.9], "honest mistake": [49.9, 0], "deliberate": [50, 100]}
    assert gate(good)[0]
    assert not gate({**good, "correct": [20]})[0]
    assert not gate({**good, "deliberate": [49.9]})[0]
    assert not gate({**good, "honest mistake": [30, 30]})[0]  # each below 50, but the mean is above 25
    assert not gate({**good, "honest mistake": [None, 0]})[0]


def test_every_question_has_a_reference():
    refs = references()
    versions = load_versions()
    assert set(versions) >= {"v0", "v1"} and all(len(v["pairs"]) == 5 for v in versions.values())
    for split in ("extract", "eval"):
        for q in common_v2.trait_data("sandbagging", split)["questions"]:
            text = intent_prompt(q, "an answer")
            assert refs[q] in text and "an answer" in text and "{" not in text


def test_organism_vector_is_minus_unlock_plus_password():
    """O_item = (L - C, no pw) - (L - C, pw) = -(unlock - password_only) per item, on the wrong items only."""
    rng = np.random.default_rng(0)
    n, layers, hidden = 6, 2, 3
    cells = {f"train:{c}:acts": rng.standard_normal((n, layers, hidden)) for c in
             ("locked/no-password", "clean/no-password", "locked/password", "clean/password")}  # fmt: skip
    cells["train:locked/no-password:correct"] = np.array([1, 0, 0, 1, 0, 1], dtype=bool)
    items, wrong = organism_items(cells)
    assert wrong.tolist() == [1, 2, 4]

    def a(cell):
        return cells[f"train:{cell}:acts"][wrong]

    unlock = a("locked/password") - a("locked/no-password")
    password_only = a("clean/password") - a("clean/no-password")
    np.testing.assert_allclose(items, -(unlock - password_only))


def test_budget_judge_stops_before_the_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(common_v2, "SPEND_PATH", tmp_path / "spend.json")
    common_v2.save_json(common_v2.SPEND_PATH, {"cap_usd": 5.0, "total_usd": 4.9999, "runs": []})
    judge = common_v2.BudgetJudge.__new__(common_v2.BudgetJudge)  # no .env, no cache file
    judge.cache, judge.spent, judge.label = {}, 0.0, "test"
    with pytest.raises(SystemExit):
        judge.score(["x" * 10_000])


def test_p_boot_weights_questions_by_their_kept_pairs():
    """With the same per-pair difference on every question, every resample gives exactly that difference."""
    tests = importlib.import_module("12_tests_v2")
    count = np.array([2, 1, 3])
    diff = np.array([1.0, -2.0, 0.5, 3.0])
    sums = {"response_avg_pos": (count[:, None] * (diff + 5))[:, None], "count": count,
            "response_avg_neg": (count[:, None] * np.full(4, 5.0))[:, None]}  # fmt: skip
    boots = tests.p_boot(sums, "response_avg", 0, n_boot=50)
    assert boots.shape == (50, 4)
    np.testing.assert_allclose(boots, np.tile(diff, (50, 1)))
