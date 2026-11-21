"""Shared helpers for The Taste Stack, Episode 1 (small probes vs interpretability): paths, data access, metrics.

Data rules: human text is public (Enron sent mail, the Blog Authorship Corpus, as split by Sounds Like Me); AI text is
open-model output written from those texts' corpus summaries, plus Sounds Like Me's drafts. `data/` is gitignored and
rebuilt by `build_data.py`; `results/` holds every number the note quotes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
RESULTS = HERE / "results"
LEDGER_TSV = HERE / "results.tsv"
# Sounds Like Me Episode 1's data (authors, briefs, drafts, raw corpora): READ ONLY, never written here. Point SLM_DIR
# at
# a checkout whose data/ has been rebuilt by that episode's scripts; the default is this repo's copy.
SLM = Path(os.environ.get("SLM_DIR", ROOT / "experiments" / "sounds-like-me"))
SLM_RAW = SLM / "data" / "raw"
SEED = 20261120


# ---------- io ----------
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def save_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1, default=float))


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def ledger(tag, metric, value, cost, status, note):
    """One row of the autoresearch ledger (dev-set tuning only)."""
    new = not LEDGER_TSV.exists()
    with LEDGER_TSV.open("a") as fh:
        if new:
            fh.write("tag\tmetric\tvalue\tcost_usd\tstatus\tnote\n")
        fh.write(f"{tag}\t{metric}\t{value:.4f}\t{cost:.3f}\t{status}\t{note}\n")


# ---------- text hygiene ----------
def reflow(t: str) -> str:
    """Remove quoted-printable soft breaks and join hard-wrapped lines inside paragraphs (both classes alike)."""
    t = t.replace("=\r\n", "").replace("=\n", "").replace("=20", " ").replace("=09", " ").replace("\r", "")
    paras = re.split(r"\n\s*\n", t.strip())
    return "\n\n".join(" ".join(line.strip() for line in p.splitlines() if line.strip()) for p in paras if p.strip())


def strip_greeting_signoff(t: str) -> str:
    """Drop a subject line, a greeting first line and a sign-off tail (Sounds Like Me's strict-voice rule)."""
    lines = [ln for ln in t.strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.lower().startswith("subject:")]
    if len(lines) > 2 and len(lines[0].split()) <= 4:
        lines = lines[1:]
    while len(lines) > 2 and len(lines[-1].split()) <= 4:
        lines = lines[:-1]
    return "\n".join(lines)


def format_strip(t: str) -> str:
    return " ".join(strip_greeting_signoff(t).split())


def length_match(t: str, n: int = 60) -> str:
    return " ".join(t.split()[:n])


def words(t: str) -> int:
    return len(t.split())


# ---------- metrics ----------
def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def boot_auroc(y, s, groups=None, n=1000, seed=SEED):
    """AUROC with a 95% bootstrap interval; resamples groups (authors) when given, else items."""
    y, s = np.asarray(y), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    point = auroc(y, s)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[pick])) == 2:
            vals.append(auroc(y[pick], s[pick]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(auroc=round(point, 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=len(y))


def boot_mean(x, n=1000, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return dict(
        mean=round(float(x.mean()), 4),
        lo=round(float(np.percentile(m, 2.5)), 4),
        hi=round(float(np.percentile(m, 97.5)), 4),
        n=len(x),
    )


def boot_diff(a, b, n=1000, seed=SEED):
    """Paired mean difference a - b with a 95% bootstrap interval."""
    return boot_mean(np.asarray(a, float) - np.asarray(b, float), n, seed)


# ---------- Sounds Like Me inputs (read only) ----------
@lru_cache(None)
def slm_corpus(name, split):
    import pandas as pd

    return pd.read_csv(SLM_RAW / f"{name}_{split}.csv")


def slm_authors():
    return json.loads((SLM / "data" / "authors.json").read_text())


def slm_briefs():
    return json.loads((SLM / "data" / "briefs.json").read_text())


def slm_texts(rec, which):
    split = "train" if which == "samples" else "test"
    df = slm_corpus(rec["corpus"], split)
    return [df.loc[i, "text"] for i in rec[which]]
