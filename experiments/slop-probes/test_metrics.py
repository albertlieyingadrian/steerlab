"""
Tests for the hand-rolled metric (probes.auroc, probes.within_industry) and the Modal spend gate (budget.check_modal).
Laptop only: no Modal, no network (modal_spend is monkeypatched).

    .venv/bin/python -m pytest experiments/slop-probes/test_metrics.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget
from probes import auroc, within_industry


# ------------------------------------------------------------------ auroc
def test_auroc_perfect_inverted_and_all_tied():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_ties_across_classes_count_half():
    # pos {1, 2}, neg {1, 0}: pairs (1>1 tie=0.5, 1>0=1, 2>1=1, 2>0=1) -> 3.5 / 4
    assert auroc([1, 1, 0, 0], [1, 2, 1, 0]) == pytest.approx(0.875)


def test_auroc_matches_sklearn_on_random_tied_scores():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(4, 60))
        y = rng.integers(0, 2, n)
        if len(set(y)) < 2:
            continue
        s = rng.integers(0, 5, n).astype(float)  # few distinct values -> many ties
        assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auroc_order_of_input_does_not_matter():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.3, 0.3, 0.9, 0.1, 0.5, 0.7])
    perm = np.array([5, 2, 0, 4, 1, 3])
    assert auroc(y, s) == pytest.approx(auroc(y[perm], s[perm]))


# ------------------------------------------------------------------ within_industry
def test_within_industry_skips_single_class_categories_and_averages_the_rest():
    y = np.array([1, 0, 1, 0, 1, 1])
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.6])
    cats = np.array(["a", "a", "b", "b", "c", "c"])  # "c" has only positives -> skipped
    assert within_industry(y, s, cats) == pytest.approx((1.0 + 0.0) / 2)


# ------------------------------------------------------------------ check_modal
@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "SPEND", tmp_path)

    def set_spent(v: float):
        monkeypatch.setattr(budget, "modal_spend", lambda: v)

    return set_spent, tmp_path / "modal_ledger.tsv"


def _decisions(ledger: Path) -> list[str]:
    return [line.split("\t")[-1] for line in ledger.read_text().splitlines()[1:]]


def test_check_modal_runs_at_exactly_the_hard_cap(gate):
    set_spent, ledger = gate
    set_spent(90.0)
    assert budget.check_modal(budget.MODAL_HARD - 90.0, "edge") == 90.0
    assert _decisions(ledger) == ["run"]


def test_check_modal_refuses_past_the_hard_cap_and_logs_it(gate):
    set_spent, ledger = gate
    set_spent(90.0)
    with pytest.raises(budget.BudgetExceeded):
        budget.check_modal(10.01, "over")
    assert _decisions(ledger) == ["REFUSED"]


def test_check_modal_warns_past_the_soft_cap(gate, capsys):
    set_spent, ledger = gate
    set_spent(35.0)
    budget.check_modal(10.0, "soft")
    assert "soft cap" in capsys.readouterr().out
    set_spent(5.0)
    budget.check_modal(1.0, "quiet")
    assert "soft cap" not in capsys.readouterr().out
    assert ledger.read_text().splitlines()[0].startswith("time_utc\tjob")
    assert _decisions(ledger) == ["run", "run"]
