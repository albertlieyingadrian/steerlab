"""Run the causal checks on groups of graph features and decide, per group, whether it is a switch or a readout.

For every group of features (by default the top-N features of the graph together, then each one alone):

1. **Ablation**: set the features to 0 at their positions. A feature the output depends on moves the metric.
2. **Steering**: multiply them by `steer_scale`. A feature that drives the output moves the metric again.
3. **Counterfactual patch** (if a counterfactual prompt is given): set them to the values they take on the
   counterfactual prompt. A switch moves the output toward the counterfactual answer.
4. **Controls**: the same three interventions on `n_controls` random feature sets of the same size (layer- and
   position-matched) and on `n_controls` matched-norm sets (same write size). A group's effect counts only if it is
   larger than at least 1 - alpha of each control's effects.
5. **Explanation swap** (if rules are given): Gao's swap test. Each feature takes its rule's predicted value at every
   position; the explained fraction is 1 - KL(rule) / KL(zero).

The verdict rules are in `decide()` and printed in every report.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from circuitkit.validate.backend import InterventionBackend
from circuitkit.validate.controls import candidate_pool, sample_matched_norm, sample_random
from circuitkit.validate.metrics import Metric, kl
from circuitkit.validate.sandbox import Rule, compile_rule, rule_values
from circuitkit.validate.types import (
    CheckResult,
    ControlStats,
    ExplanationSwap,
    FeatureNode,
    GroupResult,
    GroupSpec,
    Intervention,
    PatchResult,
    Report,
    Thresholds,
    Verdict,
)

Key = tuple[int, int, int]
RuleLike = str | Rule


def _version() -> str:
    try:
        return version("steerlab-circuitkit")
    except PackageNotFoundError:
        return "0.0.0+local"


def default_groups(nodes: Sequence[FeatureNode], singles: bool = True) -> list[GroupSpec]:
    groups = [GroupSpec(name=f"top-{len(nodes)}", nodes=list(nodes))]
    if singles and len(nodes) > 1:
        groups += [GroupSpec(name=n.name(), nodes=[n]) for n in nodes]
    return groups


def _stats(kind: str, group_effect: float, effects: Sequence[float]) -> ControlStats:
    arr = np.asarray(effects, dtype=np.float64)
    if arr.size == 0:
        return ControlStats(kind=kind, effects=[], mean=0.0, std=0.0, p_value=1.0)
    beaten = int((np.abs(arr) >= abs(group_effect)).sum())
    return ControlStats(
        kind=kind,
        effects=[float(x) for x in arr],
        mean=float(arr.mean()),
        std=float(arr.std()),
        p_value=(1 + beaten) / (arr.size + 1),
    )


def _specific(check: CheckResult, alpha: float) -> bool:
    return all(c.p_value <= alpha for c in check.controls.values())


def decide(
    ablation: CheckResult, steering: CheckResult, patch: PatchResult | None, th: Thresholds
) -> tuple[Verdict, list[str]]:
    """The verdict rules.

    - necessary: ablation moves the metric by at least `min_effect` of the scale.
    - specific: that move beats every control kind (empirical p <= alpha). With no controls run, every effect counts
      as specific and the report says so.
    - steerable: steering moves the metric by at least `min_effect` and beats every control kind.
    - patch passes: patching closes at least `patch_min` of the clean-to-counterfactual gap and beats every control.

    switch = necessary and specific and (patch passes, or no counterfactual); partial = necessary and specific but
    the patch fails; nonspecific = necessary but not specific; backup = not necessary but steerable or patch passes;
    readout = none of these.
    """
    reasons: list[str] = []
    necessary = ablation.normalized >= th.min_effect
    abl_specific = _specific(ablation, th.alpha)
    steerable = steering.normalized >= th.min_effect and _specific(steering, th.alpha)
    patch_ok = patch is not None and patch.gap_closed >= th.patch_min and _specific(patch, th.alpha)

    def pvals(c: CheckResult) -> str:
        return ", ".join(f"p={s.p_value:.3f} vs {k}" for k, s in c.controls.items()) or "no controls run"

    reasons.append(
        f"ablation moves the metric by {ablation.effect:+.3f} ({ablation.normalized:.0%} of scale; "
        f"needs {th.min_effect:.0%}); {pvals(ablation)}"
    )
    reasons.append(f"steering moves it by {steering.effect:+.3f} ({steering.normalized:.0%}); {pvals(steering)}")
    if patch is not None:
        reasons.append(
            f"counterfactual patch closes {patch.gap_closed:.0%} of the gap (needs {th.patch_min:.0%}); {pvals(patch)}"
        )

    if necessary and abl_specific:
        if patch is None or patch_ok:
            return Verdict.SWITCH, reasons
        return Verdict.PARTIAL, reasons
    if necessary:
        return Verdict.NONSPECIFIC, reasons
    if steerable or patch_ok:
        return Verdict.BACKUP, reasons
    return Verdict.READOUT, reasons


class _Runner:
    """Holds the clean run and turns feature sets into metric changes."""

    def __init__(self, backend: InterventionBackend, prompt: str, metric: Metric) -> None:
        self.backend = backend
        self.prompt = prompt
        self.metric = metric
        self.clean = backend.logits(prompt)
        self.m0 = metric(self.clean[-1])

    def run(self, interventions: Sequence[Intervention]) -> tuple[float, float]:
        logits = self.backend.logits(self.prompt, interventions)
        return self.metric(logits[-1]), kl(self.clean[-1], logits[-1])


def validate(
    backend: InterventionBackend,
    prompt: str,
    nodes: Sequence[FeatureNode],
    *,
    metric: Metric,
    counterfactual: str | None = None,
    groups: Sequence[GroupSpec] | None = None,
    singles: bool = True,
    steer_scale: float = 3.0,
    n_controls: int = 20,
    seed: int = 0,
    rules: Mapping[tuple[int, int], RuleLike] | None = None,
    thresholds: Thresholds | None = None,
    progress: Callable[[str], None] | None = None,
) -> Report:
    """Validate graph features on one prompt.

    `rules` maps (layer, feature) to `predict(tokens, i)` code or a callable, for the explanation swap.
    """
    th = thresholds or Thresholds()
    say = progress or (lambda _msg: None)
    runner = _Runner(backend, prompt, metric)
    active = backend.active_features(prompt)
    tokens = backend.tokens(prompt)
    act_at: dict[Key, float] = {(a[0], a[1], a[2]): a[3] for a in active}
    notes: list[str] = []

    m_cf: float | None = None
    cf_at: dict[Key, float] = {}
    offset = 0
    if counterfactual is not None:
        m_cf = metric(backend.logits(counterfactual)[-1])
        cf_at = {(a[0], a[1], a[2]): a[3] for a in backend.active_features(counterfactual)}
        offset = len(backend.tokens(counterfactual)) - len(tokens)
        if offset:
            notes.append(f"the counterfactual has {offset:+d} tokens; positions are aligned from the end of the prompt")
    has_gap = m_cf is not None and abs(runner.m0 - m_cf) > 1e-6
    scale = abs(runner.m0 - m_cf) if has_gap and m_cf is not None else max(abs(runner.m0), 1e-6)
    if n_controls == 0:
        notes.append("no controls were run, so no effect can be called specific")
    elif 1 / (n_controls + 1) > th.alpha:
        notes.append(
            f"with {n_controls} controls per kind the smallest possible p-value is {1 / (n_controls + 1):.3f}, above "
            f"alpha = {th.alpha}, so no effect can be called specific; use at least {int(1 / th.alpha)} controls"
        )

    def cf_value(n: FeatureNode) -> float:
        return cf_at.get((n.layer, n.pos + offset, n.feature), 0.0)

    def ivs(ns: Sequence[FeatureNode], how: str) -> list[Intervention]:
        if how == "ablate":
            return [(n.layer, n.pos, n.feature, 0.0) for n in ns]
        if how == "steer":
            return [(n.layer, n.pos, n.feature, steer_scale * n.activation) for n in ns]
        return [(n.layer, n.pos, n.feature, cf_value(n)) for n in ns]

    hows = ["ablate", "steer"] + (["patch"] if counterfactual is not None else [])
    compiled: dict[tuple[int, int], Rule] = {}
    sources: dict[tuple[int, int], str] = {}
    for unit, r in (rules or {}).items():
        if isinstance(r, str):
            compiled[unit], sources[unit] = compile_rule(r), r
        else:
            compiled[unit], sources[unit] = r, getattr(r, "__name__", "callable")

    group_list = list(groups) if groups is not None else default_groups(nodes, singles)
    claimed = [*nodes, *(n for g in group_list for n in g.nodes)]
    results: list[GroupResult] = []
    for gi, g in enumerate(group_list):
        # Use the activations measured on this prompt, so a node built by hand gets the real value.
        gnodes = [n.model_copy(update={"activation": act_at.get(n.key, n.activation)}) for n in g.nodes]
        say(f"group {gi}: {g.name} ({len(gnodes)} features)")
        main = {h: runner.run(ivs(gnodes, h)) for h in hows}
        # Controls come from outside the whole claimed circuit, not just outside this group: a group's
        # redundant twin would otherwise serve as its own control.
        pool = candidate_pool(active, [*claimed, *gnodes])
        ctrl: dict[str, dict[str, list[float]]] = {h: {"random": [], "matched_norm": []} for h in hows}
        for kind in ("random", "matched_norm"):
            for ri in range(n_controls):
                rng = random.Random(f"{seed}:{gi}:{kind}:{ri}")
                cset = (
                    sample_random(gnodes, pool, rng)
                    if kind == "random"
                    else sample_matched_norm(gnodes, pool, rng, backend.decoder_norm)
                )
                if not cset:
                    continue
                for h in hows:
                    ctrl[h][kind].append(runner.run(ivs(cset, h))[0] - runner.m0)

        def check(
            h: str,
            name: str,
            main: dict[str, tuple[float, float]] = main,
            ctrl: dict[str, dict[str, list[float]]] = ctrl,
        ) -> CheckResult:
            m, k = main[h]
            eff = m - runner.m0
            controls = {kind: _stats(kind, eff, e) for kind, e in ctrl[h].items() if n_controls}
            return CheckResult(name=name, metric=m, effect=eff, normalized=abs(eff) / scale, kl=k, controls=controls)

        ablation = check("ablate", "ablation")
        steering = check("steer", f"steering x{steer_scale:g}")
        patch: PatchResult | None = None
        if counterfactual is not None and m_cf is not None:
            base = check("patch", "counterfactual patch")
            gap = runner.m0 - m_cf
            closed = (runner.m0 - base.metric) / gap if abs(gap) > 1e-9 else 0.0
            patch = PatchResult(**base.model_dump(), gap_closed=closed)

        swap = None
        units = sorted({n.unit for n in gnodes})
        if compiled and all(u in compiled for u in units):
            swap = _explanation_swap(runner, tokens, units, compiled, sources, act_at)

        verdict, reasons = decide(ablation, steering, patch, th)
        if swap is not None and swap.explained_fraction is not None:
            ok = swap.explained_fraction >= th.explained_min
            reasons.append(
                f"explanation swap keeps {swap.explained_fraction:.0%} of the features' effect "
                f"({'passes' if ok else 'fails'}; needs {th.explained_min:.0%}); {swap.rule_errors} rule errors"
            )
        results.append(
            GroupResult(
                name=g.name,
                nodes=gnodes,
                ablation=ablation,
                steering=steering,
                patch=patch,
                explanation_swap=swap,
                verdict=verdict,
                reasons=reasons,
            )
        )

    return Report(
        version=_version(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        prompt=prompt,
        counterfactual=counterfactual,
        metric_name=metric.name,
        clean_metric=runner.m0,
        counterfactual_metric=m_cf,
        scale=scale,
        steer_scale=steer_scale,
        n_controls=n_controls,
        seed=seed,
        thresholds=th,
        backend=backend.describe(),
        groups=results,
        notes=notes,
    )


def _explanation_swap(
    runner: _Runner,
    tokens: Sequence[str],
    units: Sequence[tuple[int, int]],
    rules: Mapping[tuple[int, int], Rule],
    sources: Mapping[tuple[int, int], str],
    act_at: Mapping[Key, float],
) -> ExplanationSwap:
    """Gao's swap test on one prompt: KL over every position after the first, for rule, zero and real values."""
    positions = range(1, len(tokens))
    errors = 0
    rule_ivs: list[Intervention] = []
    zero_ivs: list[Intervention] = []
    real_ivs: list[Intervention] = []
    for layer, feat in units:
        vals, e = rule_values(rules[(layer, feat)], tokens)
        errors += e
        for p in positions:
            rule_ivs.append((layer, p, feat, vals[p]))
            zero_ivs.append((layer, p, feat, 0.0))
            real_ivs.append((layer, p, feat, act_at.get((layer, p, feat), 0.0)))
    clean = runner.clean[1:]

    def kl_of(iv: Sequence[Intervention]) -> float:
        return kl(clean, runner.backend.logits(runner.prompt, iv)[1:])

    k_rule, k_zero, k_real = kl_of(rule_ivs), kl_of(zero_ivs), kl_of(real_ivs)
    return ExplanationSwap(
        kl_rule=k_rule,
        kl_zero=k_zero,
        kl_real=k_real,
        explained_fraction=(1 - k_rule / k_zero) if k_zero > 1e-9 else None,
        rule_errors=errors,
        rule_sources={f"L{layer}/{feat}": sources.get((layer, feat), "") for layer, feat in units},
    )
