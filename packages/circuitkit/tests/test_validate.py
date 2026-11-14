"""The validator on the toy model (known answers), its verdict rules, controls and rule sandbox."""

from __future__ import annotations

import random

import pytest

from circuitkit.validate import CheckResult, FeatureNode, Thresholds, Verdict, decide, logit_diff, validate
from circuitkit.validate.controls import candidate_pool, sample_matched_norm, sample_random
from circuitkit.validate.metrics import kl, logprob
from circuitkit.validate.sandbox import RuleError, compile_rule, rule_values
from circuitkit.validate.toy import BACKUP, READOUTS, SWITCH, demo, demo_nodes
from circuitkit.validate.types import ControlStats, PatchResult


@pytest.fixture(scope="module")
def toy_report():
    b = demo()
    m = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"))
    return validate(b, "dallas", demo_nodes(b), metric=m, counterfactual="oakland", n_controls=20, seed=0)


def test_toy_verdicts_match_ground_truth(toy_report) -> None:
    by = {g.name: g.verdict for g in toy_report.groups}
    assert by[f"L{SWITCH[0]}/{SWITCH[1]}@5"] == Verdict.SWITCH
    for u in READOUTS:
        assert by[f"L{u[0]}/{u[1]}@5"] == Verdict.READOUT
    for u in BACKUP:
        assert by[f"L{u[0]}/{u[1]}@5"] == Verdict.BACKUP
    assert by["top-5"] == Verdict.SWITCH


def test_report_is_reproducible(toy_report) -> None:
    b = demo()
    m = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"))
    again = validate(b, "dallas", demo_nodes(b), metric=m, counterfactual="oakland", n_controls=20, seed=0)
    assert [g.ablation.controls["random"].effects for g in again.groups] == [
        g.ablation.controls["random"].effects for g in toy_report.groups
    ]


def test_report_records_rules_and_scale(toy_report) -> None:
    assert toy_report.thresholds == Thresholds()
    assert toy_report.scale > 0
    assert all(g.reasons for g in toy_report.groups)


def test_no_controls_is_flagged() -> None:
    b = demo()
    m = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"))
    rep = validate(b, "dallas", demo_nodes(b)[:1], metric=m, n_controls=0)
    assert any("no controls" in n for n in rep.notes)


def _check(effect: float, scale: float = 1.0, p: float = 0.01) -> CheckResult:
    ctrl = {"random": ControlStats(kind="random", effects=[0.0], mean=0, std=0, p_value=p)}
    return CheckResult(name="x", metric=effect, effect=effect, normalized=abs(effect) / scale, kl=0.0, controls=ctrl)


@pytest.mark.parametrize(
    ("abl", "abl_p", "steer", "patch", "expected"),
    [
        (1.0, 0.01, 0.0, None, Verdict.SWITCH),
        (1.0, 0.01, 0.0, 0.9, Verdict.SWITCH),
        (1.0, 0.01, 0.0, 0.1, Verdict.PARTIAL),
        (1.0, 0.5, 0.0, None, Verdict.NONSPECIFIC),
        (0.0, 0.5, 1.0, None, Verdict.BACKUP),
        (0.0, 0.5, 0.0, None, Verdict.READOUT),
    ],
)
def test_decide_table(abl, abl_p, steer, patch, expected) -> None:
    pr = None
    if patch is not None:
        pr = PatchResult(**_check(patch).model_dump(), gap_closed=patch)
    v, reasons = decide(_check(abl, p=abl_p), _check(steer), pr, Thresholds())
    assert v == expected
    assert reasons


def test_controls_are_matched() -> None:
    b = demo()
    active = b.active_features("dallas")
    group = demo_nodes(b)
    pool = candidate_pool(active, group)
    assert all(a[1] != 0 for a in pool)
    rnd = sample_random(group, pool, random.Random(0))
    assert len(rnd) == len(group)
    assert {n.key for n in rnd}.isdisjoint({n.key for n in group})
    mn = sample_matched_norm(group, pool, random.Random(0), b.decoder_norm)
    assert len(mn) == len(group)


def test_metrics() -> None:
    import numpy as np

    x = np.array([1.0, 2.0, 3.0])
    assert kl(x, x) == pytest.approx(0.0)
    assert kl(x, x[::-1]) > 0
    assert logprob(2)(x) == pytest.approx(float(x[2] - np.log(np.exp(x).sum())))


def test_sandbox_rules() -> None:
    fn = compile_rule("def predict(tokens, i):\n    return 1.0 if tokens[i] == 'b' else 0.0\n")
    vals, errors = rule_values(fn, ["a", "b", "c"])
    assert vals == [0.0, 1.0, 0.0] and errors == 0
    with pytest.raises(RuleError):
        compile_rule("import os\ndef predict(tokens, i):\n    return 0\n")
    with pytest.raises(RuleError, match="namespace"):
        compile_rule("def predict(tokens, i):\n    return numpy.mean([1])\n")
    bad = compile_rule("def predict(tokens, i):\n    return 1 / 0\n")
    assert rule_values(bad, ["a"]) == ([0.0], 1)


def test_explanation_swap_on_toy() -> None:
    b = demo()
    m = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"))
    last = len(b.tokens("dallas")) - 1
    node = FeatureNode(layer=SWITCH[0], pos=last, feature=SWITCH[1], activation=2.0)
    rule = f"def predict(tokens, i):\n    return 2.0 if i == {last} else 0.0\n"
    rep = validate(b, "dallas", [node], metric=m, n_controls=2, rules={SWITCH: rule})
    swap = rep.groups[0].explanation_swap
    assert swap is not None and swap.explained_fraction == pytest.approx(1.0)


def test_too_few_controls_is_flagged() -> None:
    b = demo()
    m = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"))
    rep = validate(b, "dallas", demo_nodes(b)[:1], metric=m, n_controls=5)
    assert any("smallest possible p-value" in n for n in rep.notes)
    assert rep.groups[0].verdict != Verdict.SWITCH
