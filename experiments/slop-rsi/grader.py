"""The grader (FIXED, runs outside the sandbox): the only code that reads evaluation labels.

Tiers: every evaluation family of the hard-negative note is split 40/30/30 by document id into dev, val and test.
- dev: structured feedback to the improver (rates, intervals, a cutoff curve, a few misclassified texts).
- val: one number per recipe, used by the harness to pick each arm's final recipe and to grade L3 improvers.
- test: never shown to anyone during the arms; read once per finalist and for the dev-test gap curve at the end.

Pilot 2 (plan amendment 3, P2-2 and P2-3):
- primary = pooled human FPR (every human text of the tier) at the cutoff where the recipe's macro recall over the AI
  families equals the reference recall ("matched FPR", mfpr). Amendment 4: the reference recall is R0's macro recall
  at the cutoff where R0's pooled human FPR is 2% (mean of 3 seeds), not at R0's own cutoff, so R0 has about 100
  false positives per tier at the reference point. The recipe's own cutoff does not enter it.
- gain = (R0 mfpr - mfpr) / R0 mfpr.
- constraint: the recipe's OWN cutoff must keep macro recall >= R0's - 1 point.
- score = gain if the constraint holds, else min(gain, 0) - 1. Feedback, parent choice, picks and grades use score.
Secondary: pooled and macro FPR at the recipe's own cutoff, AUROC, partial AUROC (FPR <= 5%), every family's rate.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from rsi import AI, EVAL_FILE, HUMAN, MANIFEST, RECALL_SLACK, TIERS, auroc, groups, pauroc, read_jsonl, tier_of, wilson

CURVE = [0.10, 0.20, 0.30, 0.3986, 0.50, 0.60, 0.70, 0.80, 0.90]


@lru_cache(maxsize=1)
def manifest() -> dict:
    return {k: v for k, v in json.loads(MANIFEST.read_text()).items() if isinstance(v, dict)}


@lru_cache(maxsize=1)
def tiers() -> dict[str, dict[str, tuple[str, np.ndarray]]]:
    """tier -> family -> (set name, row indices in that set)."""
    man = manifest()
    out: dict = {t: {} for t in TIERS}
    for fam, (s, ix) in groups(man).items():
        t = np.array([tier_of(s, man[s]["id"][i]) for i in ix])
        for name in TIERS:
            out[name][fam] = (s, ix[t == name])
    return out


@lru_cache(maxsize=1)
def texts() -> dict[str, list[str]]:
    by: dict = {}
    for r in read_jsonl(EVAL_FILE):
        by.setdefault(r["set"], []).append(r["text"])
    return by


def check_scores(ev: dict) -> None:
    """Reject malformed scores (STOP's shape hack): every set present, right length, finite, in [0, 1]."""
    man = manifest()
    for s, m in man.items():
        x = np.asarray(ev.get(s, []), float)
        if len(x) != len(m["y"]):
            raise ValueError(f"set {s}: {len(x)} scores for {len(m['y'])} items")
        if not np.all(np.isfinite(x)) or x.min() < 0 or x.max() > 1:
            raise ValueError(f"set {s}: non-finite or out-of-range scores")


def fam_scores(ev: dict, tier: str) -> dict[str, np.ndarray]:
    return {f: np.asarray(ev[s], float)[ix] for f, (s, ix) in tiers()[tier].items()}


def rates(fs: dict[str, np.ndarray], cut: float) -> dict:
    out: dict = {}
    for f in HUMAN + AI:
        k, n = int((fs[f] >= cut).sum()), len(fs[f])
        out[f] = {"rate": k / n if n else 0.0, "k": k, "n": n, "ci": wilson(k, n)}
    out["macro_fpr"] = float(np.mean([out[f]["rate"] for f in HUMAN]))
    out["macro_recall"] = float(np.mean([out[f]["rate"] for f in AI]))
    return out


def cut_at_recall(fs: dict[str, np.ndarray], target: float) -> float:
    """Highest cutoff whose macro recall over the AI families is >= target."""
    cand = np.unique(np.concatenate([fs[f] for f in AI]))[::-1]
    lo, hi = 0, len(cand) - 1  # macro recall rises as the cutoff falls: binary search over candidates
    best = cand[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        r = np.mean([(fs[f] >= cand[mid]).mean() for f in AI])
        if r >= target:
            best, hi = cand[mid], mid - 1
        else:
            lo = mid + 1
    return float(best)


def pooled(fs: dict[str, np.ndarray], cut: float) -> tuple[float, int, int]:
    h = np.concatenate([fs[f] for f in HUMAN])
    k = int((h >= cut).sum())
    return k / len(h), k, len(h)


def fpr_at_recall(fs: dict[str, np.ndarray], target: float) -> tuple[float, float]:
    """Pilot 1's loose metric: the matched cutoff and the MACRO human FPR there."""
    c = cut_at_recall(fs, target)
    return c, float(np.mean([(fs[f] >= c).mean() for f in HUMAN]))


def grade(ev: dict, cut: float, tier: str, ref: dict | None) -> dict:
    """All metrics on one tier. ref: R0's reference on this tier (macro_recall, mfpr, ...); None while building it."""
    fs = fam_scores(ev, tier)
    m = rates(fs, cut)
    h = np.concatenate([fs[f] for f in HUMAN])
    a = np.concatenate([fs[f] for f in AI])
    y, sc = np.r_[np.zeros(len(h)), np.ones(len(a))], np.r_[h, a]
    m["auroc"] = auroc(y, sc)
    m["pauroc"] = pauroc(y, sc, 0.05)
    m["cut"] = cut
    m["pooled_fpr"], m["pooled_k"], m["pooled_n"] = pooled(fs, cut)
    if ref:
        floor = ref["macro_recall"] - RECALL_SLACK  # the deployment constraint: R0's recall at its own cutoff
        m["recall_floor"] = floor
        m["constraint_fail"] = bool(m["macro_recall"] < floor)
        mc = cut_at_recall(fs, ref["ref_recall"])
        m["matched_cut"] = mc
        m["mfpr"], m["mfp_k"], _ = pooled(fs, mc)
        m["gain"] = (ref["mfpr"] - m["mfpr"]) / ref["mfpr"]
        m["score"] = m["gain"] if not m["constraint_fail"] else min(m["gain"], 0.0) - 1.0
        m["macro_mfpr"] = float(np.mean([(fs[f] >= mc).mean() for f in HUMAN]))
    return m


REF_FPR = 0.02  # amendment 4: the reference operating point is where R0's pooled human FPR is 2%


def cut_at_fpr(fs: dict[str, np.ndarray], fpr: float) -> float:
    """Lowest cutoff whose pooled human FPR is <= fpr."""
    h = np.sort(np.concatenate([fs[f] for f in HUMAN]))[::-1]
    k = int(np.floor(fpr * len(h)))
    return float(np.nextafter(h[k], 1.0)) if k < len(h) else float(h[-1])


def ref_tier(evs: list[dict], cuts: list[float], tier: str) -> dict:
    """R0's reference on one tier from its seeds: its recall at its own cutoff (the deployment constraint), the
    reference recall (its macro recall where its pooled FPR is 2%), and the matched FPR of each seed there."""
    ms = [grade(e, c, tier, None) for e, c in zip(evs, cuts, strict=True)]
    own = float(np.mean([m["macro_recall"] for m in ms]))
    rec2 = []
    for e in evs:
        fs = fam_scores(e, tier)
        rec2.append(float(np.mean([(fs[f] >= cut_at_fpr(fs, REF_FPR)).mean() for f in AI])))
    target = float(np.mean(rec2))
    mf, kk = [], []
    for e in evs:
        fs = fam_scores(e, tier)
        f, k, _ = pooled(fs, cut_at_recall(fs, target))
        mf.append(f)
        kk.append(k)
    return {
        "macro_recall": own,
        "ref_recall": target,
        "seed_ref_recall": rec2,
        "mfpr": float(np.mean(mf)),
        "seed_mfpr": mf,
        "seed_mfp_k": kk,
        "pooled_n": ms[0]["pooled_n"],
        "seed_pooled_k_own_cut": [m["pooled_k"] for m in ms],
        "macro_fpr": float(np.mean([m["macro_fpr"] for m in ms])),
        "auroc": float(np.mean([m["auroc"] for m in ms])),
        "pauroc": float(np.mean([m["pauroc"] for m in ms])),
        "seed_macro_recall": [m["macro_recall"] for m in ms],
        "seed_cut": list(cuts),
    }


def feedback(ev: dict, cut: float, ref: dict, train: dict, rotate: int) -> str:
    """The structured dev feedback an improver sees for one recipe (dev tier only)."""
    fs = fam_scores(ev, "dev")
    m = grade(ev, cut, "dev", ref)
    L = [
        "DEV FEEDBACK (dev tier only: 40% of each evaluation family)",
        (
            f"SCORE {100 * m['score']:+.1f}% (higher is better; R0 = 0). Primary: pooled human false-positive rate at "
            f"matched recall (the cutoff where your macro AI recall equals the reference {100 * ref['ref_recall']:.2f}%, "
            f"R0's recall where R0 flags 2% of human texts) = "
            f"{100 * m['mfpr']:.3f}% ({m['mfp_k']}/{m['pooled_n']} human texts; R0 {100 * ref['mfpr']:.3f}%), "
            f"gain vs R0 {100 * m['gain']:+.1f}%. Your cutoff does not change this number."
        ),
        f"Your cutoff {cut:.4f}: macro recall {100 * m['macro_recall']:.2f}% (floor {100 * m['recall_floor']:.2f}%): "
        + (
            "CONSTRAINT FAILED, so score = min(gain, 0) - 1 (a failure is always worse than R0)."
            if m["constraint_fail"]
            else "constraint met."
        ),
        (
            f"At your cutoff: pooled human FPR {100 * m['pooled_fpr']:.3f}%, macro human FPR {100 * m['macro_fpr']:.3f}%. "
            f"AUROC {m['auroc']:.4f} (R0 {ref['auroc']:.4f}); partial AUROC to 5% FPR {m['pauroc']:.4f} "
            f"(R0 {ref['pauroc']:.4f})."
        ),
        "",
        f"Human families, false-positive rate at the matched cutoff {m['matched_cut']:.4f} [95% Wilson]:",
    ]
    mr = rates(fs, m["matched_cut"])
    for f in HUMAN:
        r = mr[f]
        L.append(
            f"  {f:12s} {100 * r['rate']:6.2f}% ({r['k']}/{r['n']}) [{100 * r['ci'][0]:.2f}, {100 * r['ci'][1]:.2f}]"
        )
    L.append("AI families, recall at the matched cutoff:")
    for f in AI:
        r = mr[f]
        L.append(
            f"  {f:12s} {100 * r['rate']:6.2f}% ({r['k']}/{r['n']}) [{100 * r['ci'][0]:.2f}, {100 * r['ci'][1]:.2f}]"
        )
    L.append("Cutoff curve (macro FPR / macro recall):")
    for c in CURVE:
        q = rates(fs, c)
        L.append(f"  {c:.4f}: {100 * q['macro_fpr']:6.2f}% / {100 * q['macro_recall']:6.2f}%")
    L.append(
        "Training set: "
        + json.dumps({k: train.get(k) for k in ("rows", "by_pool", "by_bucket", "decontam_dropped", "lr", "epochs")})
    )
    L.append(
        f"Update: {train.get('seconds')} s on one H100; train loss {train.get('loss_first')} -> {train.get('loss_last')}."
    )
    L.append("")
    L.append("Some misclassified dev texts at the matched cutoff (first 400 characters; rotated between calls):")
    cut = m["matched_cut"]
    T, ti = texts(), tiers()["dev"]
    for f in HUMAN + AI:
        s, ix = ti[f]
        x = fs[f]
        wrong = np.where(x >= cut)[0] if f in HUMAN else np.where(x < cut)[0]
        if not len(wrong):
            continue
        wrong = wrong[np.argsort(-x[wrong] if f in HUMAN else x[wrong])]
        k = 3 if f in HUMAN else 2
        start = (rotate * k) % len(wrong)
        pick = [wrong[(start + j) % len(wrong)] for j in range(min(k, len(wrong)))]
        for p in pick:
            t = T[s][ix[p]].replace("\n", " ")[:400]
            L.append(f"  [{f}, {'human' if f in HUMAN else 'AI'}, score {x[p]:.3f}] {t}")
    return "\n".join(L)
