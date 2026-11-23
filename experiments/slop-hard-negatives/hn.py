"""Shared pieces for the hard-negative rounds: paths, the fixed design numbers, the spend ledger and the metrics.

Everything fixed in the pre-registered plan
lives here, so the loop, the evaluation and the figures read the same numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SD = HERE.parent / "slop-detector"  # the detector's code (common.py, 04_train.py), reused unchanged
sys.path.insert(0, str(SD))
DATA = HERE / "data"  # gitignored: page texts stay local
RESULTS = HERE / "results"
# data v2b, local copy (gitignored; the volume has the same files as data/b1b). Default: the slop-detector worktree.
V2B = Path(
    os.environ.get("V2B_DIR", HERE.parents[2] / "slop-detector" / "experiments" / "slop-detector" / "data" / "v2b")
)

APP = "slop-hard-negatives"  # the deployed Modal app (modal_hn.py repeats it: it is deployed on its own)
VOLUME = "slop-detector"  # modal4 workspace: holds round 0 (runs/b1b-1.7bB) and data v2b (data/b1b)
PROFILE = "modal4"
ROUND0 = "runs/b1b-1.7bB"  # on the volume
# The recipe (Qwen3-1.7B-Base, 3.79 epochs, 4-bucket head, 6,144 tokens per pass) is written once, in
# modal_hn.train_round, where it runs.

FLAG_CUT = 0.3986  # round 0's 99th percentile on our 2,414 validation humans (plan); fixed for every round and arm
K = 2000  # pages added per round
ROUNDS = {"a": 4, "b": 3, "c": 3}
HUMAN_W, GEN_W = 1.3278, 1.3092  # v2b's band weights for human and generated rows

CAP_MODAL = 150.0  # hard cap for the project (plan)
RATES = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "cpu": 0.20}  # $ per hour, Modal list prices (Nov 2026)
LEDGER = RESULTS / "spend.jsonl"


# ------------------------------------------------------------------ spend ledger
def _rows() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def spent(kind: str = "modal") -> float:
    return sum(r.get("usd", 0.0) for r in _rows() if r.get("kind", "modal") == kind)


def log_spend(what: str, usd: float, **kw) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M"), "what": what, "usd": round(usd, 4), **kw}) + "\n")


def reserve(what: str, gpu: str, hours: float, n: int = 1) -> None:
    """Refuse a Modal launch whose worst case (timeout x rate x containers) would take the project past its cap; else
    log the estimate. The bill is checked against this ledger after the run (gpu-training-lessons A3, G9). API calls
    have their own cap in gen.py's Meter."""
    est = RATES.get(gpu, 0.0) * hours * n
    if spent("modal") + est > CAP_MODAL:
        raise SystemExit(f"refused {what}: ledger {spent('modal'):.2f} + worst case {est:.2f} > cap {CAP_MODAL}")
    log_spend(what, est, kind="modal", gpu=gpu, n=n, note="worst-case reservation")


def settle(what: str, gpu: str, seconds: float, n: int = 1) -> None:
    """Replace a reservation by the measured cost: a negative entry for the reservation, then the real one."""
    rows = [r for r in _rows() if r["what"] == what]
    outstanding = sum(r["usd"] for r in rows if r.get("note") in ("worst-case reservation", "release reservation"))
    if outstanding > 1e-9:  # every reservation for this tag not yet released (a respawned tag has two)
        log_spend(what, -outstanding, kind="modal", gpu=gpu, note="release reservation")
    log_spend(
        what, RATES[gpu] * seconds / 3600 * n, kind="modal", gpu=gpu, seconds=round(seconds), n=n, note="measured"
    )


# ------------------------------------------------------------------ metrics
def auroc(y, s) -> float:
    """AUROC with ties averaged (slop-probes' implementation)."""
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = allv.argsort().argsort().astype(float) + 1
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_idx(n: int, b: int = 2000, seed: int = 0) -> np.ndarray:
    """The same resamples for every detector, so differences are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def rate_ci(flags, idx: np.ndarray) -> dict:
    flags = np.asarray(flags, float)
    d = flags[idx].mean(1)
    return {
        "rate": float(flags.mean()),
        "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        "k": int(flags.sum()),
        "n": len(flags),
    }


def diff_ci(a, b, idx: np.ndarray) -> dict:
    """mean(a) - mean(b) for two vectors on the same items, with a paired 95% interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": float(a.mean() - b.mean()), "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
