"""Known-answer tests for measures.py (run: .venv/bin/python -m pytest experiments/pluralism-replications/tests -q)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measures as M  # noqa: E402


def test_repetition_identical_and_orthogonal():
    e = M.normalise(np.array([[1.0, 0], [1.0, 0], [1.0, 0]]))
    assert abs(M.repetition(e) - 1.0) < 1e-9
    e = np.eye(4)
    assert abs(M.repetition(e)) < 1e-9
    assert np.isnan(M.repetition(e[:1]))


def test_cross_similarity_and_tables():
    emb = M.normalise(np.array([[1.0, 0], [1.0, 0.1], [0, 1.0], [0.1, 1.0]]))
    df = pd.DataFrame({"model": ["a", "a", "b", "b"], "prompt_id": [0, 0, 0, 0]})
    rep = M.repetition_table(df, emb)
    assert (rep.repetition > 0.99).all()
    cross = M.cross_table(df, emb)
    assert len(cross) == 1 and cross.cross.iloc[0] < 0.2


def test_cross_prompt_floor_excludes_same_prompt():
    emb = M.normalise(np.array([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]]))
    df = pd.DataFrame({"model": ["a"] * 4, "prompt_id": [0, 0, 1, 1]})
    fl = M.cross_prompt_floor(df, emb, n_pairs=2000)
    assert abs(fl.floor.iloc[0]) < 1e-6


def test_coverage_stats():
    s = M.coverage_stats(["a", "a", "b", "x"], [True, True, True, False])
    assert s["validity"] == 0.75 and s["yield"] == 2 and abs(s["diversity"] - 2 / 3) < 1e-9


def test_steerability_per_instance():
    losses = pd.DataFrame(
        {"instance_ind": [0, 0, 0, 1, 1], "example_ind": [0, 1, 2, 0, 1], "loss": [3.0, 2.0, 1.0, 4.0, 2.0]}
    )
    out = M.steerability_per_instance(losses).set_index("instance_ind")
    assert out.loc[0, "loss_geq_1"] == 1.5 and out.loc[0, "context_gain"] == 1.5
    assert out.loc[1, "loss_geq_1"] == 2.0


def test_entropy_and_reliability():
    assert M.label_entropy([5] * 25) == 0
    assert abs(M.label_entropy([1, 2, 3, 4, 5] * 5) - np.log2(5)) < 1e-9
    rng = np.random.default_rng(0)
    truth = rng.normal(3, 1, 300)
    labels = [np.clip(np.round(t + rng.normal(0, 0.5, 25)), 1, 5).astype(int) for t in truth]
    rel = M.split_half_reliability(labels, n_splits=20)
    assert rel["rho"] > 0.9
    noise = [rng.integers(1, 6, 25) for _ in range(300)]
    assert M.split_half_reliability(noise, n_splits=20)["rho"] < 0.3
    assert 0 < M.loo_rater_r(labels, n_draws=20) < rel["ceiling"]


def test_bootstrap_and_verdict():
    ci = M.bootstrap_ci(np.ones(50) + np.random.default_rng(0).normal(0, 0.1, 50), n_boot=200)
    assert ci["lo"] > 0.9 and M.verdict(ci["lo"], ci["hi"]) == "replicated"
    assert M.verdict(-0.2, 0.1) == "null" and M.verdict(-0.2, -0.1) == "not replicated"
    groups = {i: np.full(10, float(i % 2)) for i in range(20)}
    tl = M.two_level_bootstrap(groups, lambda d: np.mean([v.mean() for v in d.values()]), n_boot=200)
    assert 0.2 < tl["lo"] < 0.5 < tl["hi"] < 0.8
    df = pd.DataFrame({"c": np.repeat(range(10), 5), "x": np.arange(50.0)})
    cb = M.cluster_bootstrap(df, "c", lambda d: d.x.mean(), n_boot=200)
    assert cb["lo"] < 24.5 < cb["hi"]
