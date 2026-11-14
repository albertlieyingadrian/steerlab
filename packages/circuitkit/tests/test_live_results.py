"""The saved results of the live GPU runs (examples/modal/live_checks.py) parse, and their verdicts follow the rules.

These re-check the recorded runs offline: every stored verdict must be what `decide()` gives on the stored checks,
and the headline findings of the circuits episodes must hold.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from circuitkit.validate import Report, Verdict, decide


def _load(data_dir, job: str) -> dict:
    return json.loads((data_dir / "live" / f"{job}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("job", ["rl_hack", "lock"])
def test_stored_verdicts_follow_the_rules(data_dir, job) -> None:
    for raw in _load(data_dir, job)["validate"]:
        rep = Report.model_validate(raw)
        assert rep.n_controls >= 19
        for g in rep.groups:
            verdict, _ = decide(g.ablation, g.steering, g.patch, rep.thresholds)
            assert verdict == g.verdict, f"{job} {g.name}"


def test_rl_hack_features_are_not_switches(data_dir) -> None:
    """Episode 2: the graph's RL features do not carry the hack. None is a switch at any comment slot."""
    res = _load(data_dir, "rl_hack")
    assert len(res["validate"]) == 3
    for raw in res["validate"]:
        rep = Report.model_validate(raw)
        assert all(g.verdict not in (Verdict.SWITCH, Verdict.PARTIAL) for g in rep.groups)


def test_rl_hack_reconstruction_matches_study(data_dir) -> None:
    res = _load(data_dir, "rl_hack")
    before, after = np.mean(res["fvu"]["before"]), np.mean(res["fvu"]["after"])
    assert before == pytest.approx(0.500, abs=0.005) and after == pytest.approx(0.499, abs=0.005)
    for s in res["summaries"].values():
        assert s["top20_overlap_with_study"] >= 18
        assert s["replacement"] == pytest.approx(s["replacement_study"], abs=0.02)


def test_lock_claimed_group_is_not_a_switch(data_dir) -> None:
    """Episode 1: switching off the ten claimed lock features does not unlock the model."""
    res = _load(data_dir, "lock")
    assert res["validate"]
    for raw in res["validate"]:
        rep = Report.model_validate(raw)
        group = next(g for g in rep.groups if g.name == "claimed 10")
        assert group.verdict not in (Verdict.SWITCH, Verdict.PARTIAL)
