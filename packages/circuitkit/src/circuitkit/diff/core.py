"""Compare attribution graphs of one model before and after fine-tuning or RL.

Input: graphs of the *same prompts* from the model before and after training (paired by index), as
`GraphSummary` objects. Output: a `DiffReport` with

- **units that changed**: each transcoder feature's mean influence before and after, the paired change, how many
  graphs it appears in on each side, and a status: `gained` (in at least `min_present` after-graphs and at most
  `max_absent` before-graphs), `lost` (the reverse), `up` / `down` (in both, and the change beats the null after a
  Benjamini-Hochberg correction over all units), or `stable`;
- **edges that changed** among the top units, with the same statuses;
- **stability**: for every unit, the share of prompts whose own change has the sign of the mean change, a bootstrap
  interval over prompts, and a split-half check (do two halves of the prompts pick the same top units?);
- **a random null**: the before/after labels are swapped at random within each prompt pair (all 2^n swaps when
  there are few prompts). A unit's p-value is the share of swaps whose mean change is at least as large; the whole
  graph's change score (sum of |mean change|) gets the same test. With unpaired graphs, labels are permuted. With n
  prompt pairs no p-value can go below about 2 / 2^n, so with 8 pairs single units rarely survive the correction:
  the whole-graph test, the split-half check and the control contrast carry more weight than per-unit p-values;
- optional **control** (graphs from a training run that did not learn the behaviour) and **replication** (another
  seed) contrasts: is a unit's change specific to the run that learned it, and does it come back in another seed?
- **validity**: the transcoders were trained on the base model, not the fine-tuned one, so the report compares
  replacement and completeness scores, error-node influence and (if given) per-layer reconstruction error (FVU)
  before and after, and warns when the transcoders explain the trained model noticeably worse.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from circuitkit.diff.graphs import GraphSummary, Unit

Status = Literal["gained", "lost", "up", "down", "stable"]

CAVEAT = (
    "The transcoders were trained on the base model. On a fine-tuned model they may reconstruct the MLPs less well, "
    "and a feature that 'appears' may be the old dictionary describing new computation. Read the validity section: "
    "if reconstruction error or error-node influence rises after training, treat gained and lost features as "
    "candidates to test (for example with circuitkit.validate), not as findings."
)


def _version() -> str:
    try:
        return version("steerlab-circuitkit")
    except PackageNotFoundError:
        return "0.0.0+local"


class Settings(BaseModel):
    min_present: int = Field(6, description="graphs (per side) a unit must appear in to count as present")
    max_absent: int = Field(2, description="at most this many graphs on the other side for 'gained' / 'lost'")
    top_k: int = Field(10, description="units reported as the top change (by mean increase in influence)")
    alpha: float = Field(0.05, description="p-value against the null below which a change counts")
    n_null: int = Field(2000, description="random label swaps when exact enumeration is too large")
    n_boot: int = Field(1000, description="bootstrap resamples of prompts for the interval")
    specific_ratio: float = Field(0.25, description="a unit is specific if its control change < ratio x its change")
    seed: int = 0


class UnitChange(BaseModel):
    unit: Unit
    before: float = Field(description="mean influence over the before-graphs (0 where absent)")
    after: float
    delta: float = Field(description="mean of after - before over prompt pairs")
    present_before: int
    present_after: int
    status: Status
    sign_agreement: float = Field(description="share of prompts whose own change has the mean change's sign")
    ci_low: float
    ci_high: float
    p_null: float
    q_null: float = Field(1.0, description="Benjamini-Hochberg adjusted p over all units")
    control_delta: float | None = None
    specific: bool | None = None
    replicated: bool | None = None
    label: str | None = None


class EdgeChange(BaseModel):
    source: Unit
    target: Unit
    before: float
    after: float
    delta: float
    present_before: int
    present_after: int
    status: Status
    p_null: float


class Validity(BaseModel):
    replacement_before: float | None = None
    replacement_after: float | None = None
    completeness_before: float | None = None
    completeness_after: float | None = None
    error_influence_before: float | None = None
    error_influence_after: float | None = None
    fvu_before: list[float] | None = Field(None, description="per-layer fraction of MLP-output variance unexplained")
    fvu_after: list[float] | None = None
    warnings: list[str] = Field(default_factory=list)
    ok: bool = True


class Summary(BaseModel):
    n_prompts: int
    paired: bool
    n_present_before: int
    n_present_after: int
    n_gained: int
    n_lost: int
    n_up: int
    n_down: int
    jaccard: float = Field(description="present-after vs present-before units, |A∩B| / |A∪B|")
    change_score: float = Field(description="sum over units of |mean change in influence|")
    change_null_mean: float
    change_p: float
    min_p: float = Field(1.0, description="smallest p-value the null can give with this many prompts")
    split_half_overlap: float | None = Field(None, description="mean top-k overlap between two halves of the prompts")
    split_half_chance: float | None = Field(None, description="the overlap two random top-k lists would have")
    control_specific: int | None = None
    replication_overlap: int | None = Field(None, description="top-k units also in the replication's top 50")


class DiffReport(BaseModel):
    version: str
    created: str | None = None
    before_name: str
    after_name: str
    settings: Settings
    summary: Summary
    top: list[UnitChange]
    gained: list[UnitChange]
    lost: list[UnitChange]
    down: list[UnitChange]
    edges: list[EdgeChange]
    validity: Validity
    caveats: list[str] = Field(default_factory=lambda: [CAVEAT])

    def find(self, u: Unit) -> UnitChange | None:
        for group in (self.top, self.gained, self.lost, self.down):
            for c in group:
                if c.unit == u:
                    return c
        return None


# ---------------------------------------------------------------- matrices


def _matrix(graphs: Sequence[GraphSummary], keys: Sequence[Unit]) -> np.ndarray:
    idx = {k: i for i, k in enumerate(keys)}
    m = np.zeros((len(graphs), len(keys)))
    for r, g in enumerate(graphs):
        for u, v in g.features.items():
            j = idx.get(u)
            if j is not None:
                m[r, j] = v
    return m


def _edge_matrix(graphs: Sequence[GraphSummary], keys: Sequence[tuple[Unit, Unit]]) -> np.ndarray:
    idx = {k: i for i, k in enumerate(keys)}
    m = np.zeros((len(graphs), len(keys)))
    for r, g in enumerate(graphs):
        for e, v in g.edges.items():
            j = idx.get(e)
            if j is not None:
                m[r, j] = v
    return m


def _presence(graphs: Sequence[GraphSummary], keys: Sequence[Unit]) -> np.ndarray:
    idx = {k: i for i, k in enumerate(keys)}
    cnt = np.zeros(len(keys), dtype=int)
    for g in graphs:
        for u in g.features:
            j = idx.get(u)
            if j is not None:
                cnt[j] += 1
    return cnt


def _edge_presence(graphs: Sequence[GraphSummary], keys: Sequence[tuple[Unit, Unit]]) -> np.ndarray:
    idx = {k: i for i, k in enumerate(keys)}
    cnt = np.zeros(len(keys), dtype=int)
    for g in graphs:
        for e in g.edges:
            j = idx.get(e)
            if j is not None:
                cnt[j] += 1
    return cnt


# ---------------------------------------------------------------- the null


def _flips(n: int, n_null: int, rng: np.random.Generator) -> np.ndarray:
    """Sign patterns for a paired swap test: all 2^n when that is at most 2 x n_null, else n_null random ones."""
    if 2**n <= 2 * n_null:
        return np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    return rng.choice((1.0, -1.0), size=(n_null, n))


def _null_means(b: np.ndarray, a: np.ndarray, paired: bool, n_null: int, rng: np.random.Generator) -> np.ndarray:
    """Mean change per column under the null, shape (n_draws, columns)."""
    if paired:
        d = a - b
        s = _flips(d.shape[0], n_null, rng)
        out: np.ndarray = s @ d / d.shape[0]
        return out
    pooled = np.vstack([b, a])
    nb = b.shape[0]
    draws = np.empty((n_null, pooled.shape[1]))
    for i in range(n_null):
        perm = rng.permutation(pooled.shape[0])
        draws[i] = pooled[perm[nb:]].mean(0) - pooled[perm[:nb]].mean(0)
    return draws


def _p_values(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Two-sided: (1 + #null draws with |change| >= |observed|) / (draws + 1)."""
    ge = (np.abs(null) >= np.abs(observed)[None, :] - 1e-15).sum(0)
    out: np.ndarray = (ge + 1) / (null.shape[0] + 1)
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def _bootstrap(d: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = d.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    lo = np.zeros(d.shape[1])
    hi = np.zeros(d.shape[1])
    step = 2048  # columns at a time, to keep memory small on graphs with many units
    for c0 in range(0, d.shape[1], step):
        means = d[:, c0 : c0 + step][idx].mean(1)
        lo[c0 : c0 + step] = np.percentile(means, 2.5, axis=0)
        hi[c0 : c0 + step] = np.percentile(means, 97.5, axis=0)
    return lo, hi


# ---------------------------------------------------------------- the diff


def _status(pb: int, pa: int, delta: float, p: float, agree: float, s: Settings, n_b: int, n_a: int) -> Status:
    if pa >= min(s.min_present, n_a) and pb <= s.max_absent:
        return "gained"
    if pb >= min(s.min_present, n_b) and pa <= s.max_absent:
        return "lost"
    if p <= s.alpha and (np.isnan(agree) or agree >= 0.75):
        return "up" if delta > 0 else "down"
    return "stable"


def _scores(before: Sequence[GraphSummary], after: Sequence[GraphSummary]) -> dict[Unit, float]:
    keys = sorted(set().union(*(g.features for g in before), *(g.features for g in after)))
    b, a = _matrix(before, keys), _matrix(after, keys)
    delta = (a - b).mean(0) if len(before) == len(after) else a.mean(0) - b.mean(0)
    return dict(zip(keys, delta.tolist(), strict=True))


def _top_k(delta: np.ndarray, present: np.ndarray, need: int, k: int) -> np.ndarray:
    ok = np.where(present >= need)[0]
    return ok[np.argsort(-delta[ok], kind="stable")][:k]


def diff_graphs(
    before: Sequence[GraphSummary],
    after: Sequence[GraphSummary],
    *,
    settings: Settings | None = None,
    control: tuple[Sequence[GraphSummary], Sequence[GraphSummary]] | None = None,
    replication: tuple[Sequence[GraphSummary], Sequence[GraphSummary]] | None = None,
    fvu_before: Sequence[float] | None = None,
    fvu_after: Sequence[float] | None = None,
    labels: Mapping[Unit, str] | None = None,
    before_name: str = "before",
    after_name: str = "after",
) -> DiffReport:
    """Diff two sets of graphs. Pass the same prompts in the same order on both sides for the paired test."""
    s = settings or Settings()
    if not before or not after:
        raise ValueError("need at least one graph on each side")
    rng = np.random.default_rng(s.seed)
    paired = len(before) == len(after)
    keys = sorted(set().union(*(g.features for g in before), *(g.features for g in after)))
    b, a = _matrix(before, keys), _matrix(after, keys)
    pb, pa = _presence(before, keys), _presence(after, keys)
    if paired:
        d = a - b
        delta = d.mean(0)
        agree = (np.sign(d) == np.sign(delta)[None, :]).mean(0)
        lo, hi = _bootstrap(d, s.n_boot, rng)
    else:
        delta = a.mean(0) - b.mean(0)
        agree = np.full(len(keys), np.nan)
        lo = hi = np.full(len(keys), np.nan)
    null = _null_means(b, a, paired, s.n_null, rng)
    pvals = _p_values(delta, null)
    qvals = _bh(pvals)
    change = float(np.abs(delta).sum())
    null_change = np.abs(null).sum(1)
    change_p = float(((null_change >= change - 1e-12).sum() + 1) / (null_change.size + 1))

    ctrl_scores = _scores(*control) if control else None
    rep_top: set[Unit] | None = None
    if replication:
        rs = _scores(*replication)
        rep_top = set(sorted(rs, key=lambda u: -rs[u])[:50])

    def change_of(i: int) -> UnitChange:
        u = keys[i]
        c_delta = ctrl_scores.get(u, 0.0) if ctrl_scores is not None else None
        return UnitChange(
            unit=u,
            before=float(b[:, i].mean()),
            after=float(a[:, i].mean()),
            delta=float(delta[i]),
            present_before=int(pb[i]),
            present_after=int(pa[i]),
            status=_status(
                int(pb[i]), int(pa[i]), float(delta[i]), float(qvals[i]), float(agree[i]), s, len(before), len(after)
            ),
            sign_agreement=float(agree[i]),
            ci_low=float(lo[i]),
            ci_high=float(hi[i]),
            p_null=float(pvals[i]),
            q_null=float(qvals[i]),
            control_delta=c_delta,
            specific=None if c_delta is None else bool(c_delta < s.specific_ratio * float(delta[i])),
            replicated=None if rep_top is None else u in rep_top,
            label=(labels or {}).get(u),
        )

    min_a = min(s.min_present, len(after))
    min_b = min(s.min_present, len(before))
    top_idx = _top_k(delta, pa, min_a, s.top_k)
    top = [change_of(int(i)) for i in top_idx]
    gained_idx = np.where((pa >= min_a) & (pb <= s.max_absent))[0]
    lost_idx = np.where((pb >= min_b) & (pa <= s.max_absent))[0]
    sig = (qvals <= s.alpha) & ((agree >= 0.75) | np.isnan(agree)) & (pa > s.max_absent) & (pb > s.max_absent)
    up_idx = np.where(sig & (delta > 0))[0]
    down_idx = np.where(sig & (delta < 0))[0]
    gained = [change_of(int(i)) for i in gained_idx[np.argsort(-delta[gained_idx])][:50]]
    lost = [change_of(int(i)) for i in lost_idx[np.argsort(delta[lost_idx])][:50]]
    down = [change_of(int(i)) for i in down_idx[np.argsort(delta[down_idx])][:50]]

    # edges among the top units
    tset = {keys[int(i)] for i in top_idx}
    ekeys = sorted({e for g in [*before, *after] for e in g.edges if e[0] in tset and e[1] in tset})
    edges: list[EdgeChange] = []
    if ekeys:
        eb, ea = _edge_matrix(before, ekeys), _edge_matrix(after, ekeys)
        epb, epa = _edge_presence(before, ekeys), _edge_presence(after, ekeys)
        ed = (ea - eb).mean(0) if paired else ea.mean(0) - eb.mean(0)
        ep = _p_values(ed, _null_means(eb, ea, paired, s.n_null, rng))
        eq = _bh(ep)
        eagree = (np.sign(ea - eb) == np.sign(ed)[None, :]).mean(0) if paired else np.full(len(ekeys), np.nan)
        for j, (u, v) in enumerate(ekeys):
            if epa[j] < min_a and epb[j] < min_b:
                continue
            edges.append(
                EdgeChange(
                    source=u,
                    target=v,
                    before=float(eb[:, j].mean()),
                    after=float(ea[:, j].mean()),
                    delta=float(ed[j]),
                    present_before=int(epb[j]),
                    present_after=int(epa[j]),
                    status=_status(
                        int(epb[j]),
                        int(epa[j]),
                        float(ed[j]),
                        float(eq[j]),
                        float(eagree[j]),
                        s,
                        len(before),
                        len(after),
                    ),
                    p_null=float(ep[j]),
                )
            )
        edges.sort(key=lambda e: -abs(e.after))

    # split-half stability of the top-k: do two random halves of the prompts pick the same units?
    split_overlap = split_chance = None
    if paired and len(before) >= 4:
        n = len(before)
        half_need = max(1, round(min_a * (n // 2) / n))
        overlaps = []
        for r in range(20):
            perm = np.random.default_rng(s.seed + 1000 + r).permutation(n)
            tops = []
            for rows in (perm[: n // 2], perm[n // 2 :]):
                hp = _presence([after[int(i)] for i in rows], keys)
                tops.append(set(_top_k((a[rows] - b[rows]).mean(0), hp, half_need, s.top_k).tolist()))
            overlaps.append(len(tops[0] & tops[1]) / max(1, min(s.top_k, len(tops[0]), len(tops[1]))))
        split_overlap = float(np.mean(overlaps))
        split_chance = float(min(1.0, s.top_k / max(int((pa >= min_a).sum()), 1)))

    validity = _validity(before, after, fvu_before, fvu_after)
    present_b = {keys[i] for i in np.where(pb >= min_b)[0]}
    present_a = {keys[i] for i in np.where(pa >= min_a)[0]}
    union = present_a | present_b
    summary = Summary(
        n_prompts=len(after),
        paired=paired,
        n_present_before=len(present_b),
        n_present_after=len(present_a),
        n_gained=len(gained_idx),
        n_lost=len(lost_idx),
        n_up=len(up_idx),
        n_down=len(down_idx),
        jaccard=len(present_a & present_b) / len(union) if union else 1.0,
        change_score=change,
        change_null_mean=float(null_change.mean()),
        change_p=change_p,
        min_p=float((3 if paired and 2 ** len(before) <= 2 * s.n_null else 1) / (null.shape[0] + 1)),
        split_half_overlap=split_overlap,
        split_half_chance=split_chance,
        control_specific=None if ctrl_scores is None else sum(bool(c.specific) for c in top),
        replication_overlap=None if rep_top is None else sum(bool(c.replicated) for c in top),
    )
    return DiffReport(
        version=_version(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        before_name=before_name,
        after_name=after_name,
        settings=s,
        summary=summary,
        top=top,
        gained=gained,
        lost=lost,
        down=down,
        edges=edges[:50],
        validity=validity,
    )


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def _validity(
    before: Sequence[GraphSummary],
    after: Sequence[GraphSummary],
    fvu_before: Sequence[float] | None,
    fvu_after: Sequence[float] | None,
    tolerance: float = 0.1,
) -> Validity:
    """Warn when the base-model transcoders explain the trained model more than `tolerance` (relative) worse."""
    v = Validity(
        replacement_before=_mean([g.replacement_score for g in before]),
        replacement_after=_mean([g.replacement_score for g in after]),
        completeness_before=_mean([g.completeness_score for g in before]),
        completeness_after=_mean([g.completeness_score for g in after]),
        error_influence_before=_mean([g.error_influence for g in before]),
        error_influence_after=_mean([g.error_influence for g in after]),
        fvu_before=list(fvu_before) if fvu_before is not None else None,
        fvu_after=list(fvu_after) if fvu_after is not None else None,
    )
    for name, x0, x1 in (
        ("replacement score", v.replacement_before, v.replacement_after),
        ("completeness", v.completeness_before, v.completeness_after),
    ):
        if x0 is not None and x1 is not None and x0 - x1 > tolerance * max(x0, 1e-9):
            v.warnings.append(f"{name} falls from {x0:.3f} to {x1:.3f} after training")
    e0, e1 = v.error_influence_before, v.error_influence_after
    if e0 is not None and e1 is not None and e1 - e0 > tolerance * max(e0, 1e-9):
        v.warnings.append(f"influence through error nodes rises from {e0:.3f} to {e1:.3f} after training")
    if v.fvu_before and v.fvu_after:
        fb, fa = np.asarray(v.fvu_before), np.asarray(v.fvu_after)
        worse = [i for i in range(min(len(fb), len(fa))) if fa[i] > fb[i] * (1 + tolerance) + 1e-3]
        if worse:
            v.warnings.append(
                f"reconstruction error (FVU) is over {tolerance:.0%} higher after training at layers {worse}"
            )
        if float(fa.mean()) > float(fb.mean()) * (1 + tolerance):
            v.warnings.append(f"mean FVU rises from {fb.mean():.3f} to {fa.mean():.3f}")
    if v.fvu_before is None and v.replacement_before is None and v.completeness_before is None:
        v.warnings.append("no reconstruction measures were given, so the diff's validity was not checked")
    v.ok = not v.warnings
    return v
