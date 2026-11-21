"""Episode 4 shared helpers: paths, the Modal budget guard, local text store, and the agreement statistics.

Reuses Episode 1's rulers (imported from ../sounds-like-me, never copied). Text from the released human-judgement
datasets lives only in data/local/ (gitignored); results/ holds ids, labels and scores.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EP1 = HERE.parent / "sounds-like-me"
EP2 = HERE.parent / "sounds-like-me-ep2"
sys.path.append(str(EP1))

from common import LEDGER, record_spend, strip_greeting_signoff

DATA = HERE / "data"
RAW = DATA / "raw"
LOCAL = DATA / "local"  # text copies, never committed or synced
RES = HERE / "results"
SEED = 20261121
EP4_MODAL_CAP = 20.0  # this episode's Modal budget (lead, 2026-11-21); counts toward the series' $900


def strict(t: str) -> str:
    """Series strict form: greeting and sign-off stripped, whitespace collapsed (Episode 1's A2)."""
    return " ".join(strip_greeting_signoff(t).split())


def ep4_modal_spent() -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return sum(r["usd"] for r in rows if r["provider"] == "modal" and r["note"].startswith("ep4"))


def modal_guard(estimate_usd: float) -> None:
    """Refuse a Modal call that would take this episode past its cap."""
    spent = ep4_modal_spent()
    if spent + estimate_usd > EP4_MODAL_CAP:
        raise SystemExit(f"ep4 Modal cap: spent ${spent:.2f} + est ${estimate_usd:.2f} > ${EP4_MODAL_CAP:.2f}")


def charge_modal(seconds: float, note: str, gpu: str = "H100") -> float:
    from modal_gen import charge

    usd = charge(seconds, gpu)
    record_spend("modal", usd, "ep4 " + note)
    return usd


def hid(s: str, n: int = 10) -> str:
    """Short stable hash, used for public ids (participant ids are never shipped)."""
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------- statistics
def rm_corr(groups, x, y):
    """Repeated-measures correlation (Bakdash & Marusich): Pearson r of within-group centred x and y,
    i.e. the common within-person slope. Groups with < 2 items contribute nothing."""
    groups, x, y = np.asarray(groups), np.asarray(x, float), np.asarray(y, float)
    xs, ys = [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        xs.append(x[m] - x[m].mean())
        ys.append(y[m] - y[m].mean())
    if not xs:
        return float("nan")
    xc, yc = np.concatenate(xs), np.concatenate(ys)
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / den) if den > 0 else float("nan")


def auroc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def spearman(x, y):
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def boot(units, stat, n=2000, seed=SEED):
    """95% bootstrap interval over units. stat(list_of_units) -> float. Returns [point, low, high]."""
    rng = np.random.default_rng(seed)
    units = list(units)
    pt = stat(units)
    vals = []
    for _ in range(n):
        v = stat([units[i] for i in rng.integers(0, len(units), len(units))])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return [round(float(pt), 4), round(float(lo), 4), round(float(hi), 4)]


def perm_floor(stat, shuffle, n=1000, seed=SEED):
    """95th percentile of a statistic under shuffled human labels (shuffle(rng) -> stat input)."""
    rng = np.random.default_rng(seed + 1)
    vals = [v for v in (stat(shuffle(rng)) for _ in range(n)) if not np.isnan(v)]
    return round(float(np.percentile(vals, 95)), 4) if vals else float("nan")
