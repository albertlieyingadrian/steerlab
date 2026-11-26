"""Shared, FIXED pieces of the slop-detector RSI pilot: paths, caps, the spend ledger, the evaluation tiers and metrics.

Pilot 2 (plan amendment 3): results in results/pilot2/, the harder evaluation "eval p2", the matched-recall metric, the
open improver model served on Modal. Pilot 1's code and results are at commit a324d862 (lab #923).

Nothing the improver writes can change this file: it runs outside the sandbox, its SHA-256 is checked before and after
every evaluation, and the sandbox cannot read it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
MAIN = LAB.parents[2] if ".claude" in LAB.parts else LAB  # the main checkout (its .venv, the other worktrees)
VENV = MAIN / ".venv" / "bin"
P1_RESULTS = HERE / "results"  # pilot 1 (frozen)
RESULTS = P1_RESULTS / "pilot2"
DATA = HERE / "data"  # gitignored: page texts stay local
RO = DATA / "ro"  # the pools the recipe may read (read-only inside the sandbox)
WORK = DATA / "work"  # one writable folder per sandboxed run
PRIVATE = DATA / "private"  # eval scores by tier; never readable from the sandbox
HN = HERE.parent / "slop-hard-negatives"  # the hard-negative note: its frozen evaluation set and round-0 pool scores
# Page texts of the hard-negative note are gitignored; default to its worktree.
HN_DATA = MAIN / ".claude" / "worktrees" / "slop-hard-negatives" / "experiments" / "slop-hard-negatives" / "data"
V2B = MAIN / ".claude" / "worktrees" / "slop-detector" / "experiments" / "slop-detector" / "data" / "v2b"

APP = "slop-rsi-p2"
LLM_APP = "slop-rsi-llm"
VOLP = "rsi/p2"  # this pilot's folder on the volume (pilot 1 used rsi/)
EVAL_FILE = DATA / "evals_p2.jsonl"  # eval p2 texts (gitignored); its manifest (no text) is RESULTS / "evalset_p2.json"
MANIFEST = RESULTS / "evalset_p2.json"
PROFILE = "modal4"  # the workspace whose `slop-detector` volume holds round 0 (runs/b1b-1.7bB) and data v2b
ROUND0 = "runs/b1b-1.7bB"
FLAG_CUT = 0.3986  # the hard-negative note's fixed cut (round 0's 99th percentile on validation humans)

# ------------------------------------------------------------------ caps (amendment 3: pilot 2)
CAP_MODAL = 45.0  # hard: refuse any launch past it (training, scoring and the LLM server)
ASK_MODAL = 38.0  # stop and ask if passed before both arms are about half done
CAP_API = 0.0  # no paid API in pilot 2
IMPROVER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"  # served by vLLM on Modal (modal_llm.py)
IMPROVER_PRICE = (0.0, 0.0)  # no per-token price: the server's GPU time is on the Modal ledger
LLM_CONTEXT = 49152
LLM_SERVER_RESERVE = 6.0  # $: the LLM server's worst case, reserved before the arms and settled from billing
ARM_TOKENS = {"in": 1_200_000, "out": 200_000}  # equal per arm (L3 meta-edits included)
PROPOSAL_TOKENS = {"in": 80_000, "out": 24_000}  # per proposal or meta-edit, so one runaway call can't eat an arm
EVALS_PER_ARM = 24
MAX_ROWS = 15_000
LR_RANGE = (5e-5, 4e-4)
EPOCH_RANGE = (0.05, 2.0)
WEIGHT_RANGE = (0.0, 5.0)
AUGS = ("none", "tail", "middle", "lower", "squash_ws")
RATES = {
    "H100": 3.95,
    "L40S": 1.95,
    "cpu": 0.20,
}  # $ per hour, Modal list prices (Nov 2026), as in the hard-negative note
LEDGER = RESULTS / "spend.jsonl"

# ------------------------------------------------------------------ evaluation families (the hard-negative note's)
HUMAN = ["web", "magazine", "enron_h", "epoch_h", "editlens_h", "hc3_h", "raid_h"]
AI = [
    "editlens_gen",
    "editlens_edit",
    "enron_gen",
    "epoch_imit",
    "epoch_plain",
    "magazine_ai",
    "fresh_old",
    "fresh_new",
    "hc3_ai",
    "raid_ai",
]
TIERS = {"dev": (0.0, 0.4), "val": (0.4, 0.7), "test": (0.7, 1.0)}
RECALL_SLACK = 0.01  # a recipe may lose at most 1 point of macro recall against R0 on the same tier


# ------------------------------------------------------------------ ledger (shared by both arms: append under a lock)
def _rows() -> list[dict]:
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def spent(kind: str = "modal") -> float:
    return sum(r.get("usd", 0.0) for r in _rows() if r.get("kind", "modal") == kind)


def log_spend(what: str, usd: float, **kw) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "usd": round(usd, 5), **kw}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def reserve(what: str, gpu: str, hours: float, n: int = 1) -> None:
    """Refuse a launch whose worst case (timeout x rate x attempts) would pass the hard cap; else log the estimate."""
    est = RATES[gpu] * hours * n
    if spent("modal") + est > CAP_MODAL:
        raise SystemExit(f"refused {what}: ledger {spent('modal'):.2f} + worst case {est:.2f} > cap {CAP_MODAL}")
    log_spend(what, est, kind="modal", gpu=gpu, n=n, note="worst-case reservation")


def settle(what: str, gpu: str, seconds: float) -> None:
    out = sum(r["usd"] for r in _rows() if r["what"] == what and r.get("note") in ("worst-case reservation", "release"))
    if out > 1e-9:
        log_spend(what, -out, kind="modal", gpu=gpu, note="release")
    log_spend(what, RATES[gpu] * seconds / 3600, kind="modal", gpu=gpu, seconds=round(seconds), note="measured")


# ------------------------------------------------------------------ tiers
def tier_of(set_name: str, item_id: str) -> str:
    """Stable 40/30/30 split by document id, the same for every detector."""
    u = int(hashlib.sha256(f"{set_name}|{item_id}".encode()).hexdigest()[:12], 16) / 16**12
    for t, (lo, hi) in TIERS.items():
        if lo <= u < hi:
            return t
    return "test"


def groups(man: dict) -> dict[str, tuple[str, np.ndarray]]:
    """Named item groups (the hard-negative note's evaluate.groups): (set name, row indices)."""

    def sel(s: str, pred) -> np.ndarray:
        return np.array([i for i, k in enumerate(man[s]["kind"]) if pred(k, man[s]["y"][i])])

    return {
        "web": ("web", np.arange(len(man["web"]["y"]))),
        "magazine": ("hd", sel("hd", lambda k, y: y == 0)),
        "enron_h": ("test_enron", sel("test_enron", lambda k, y: k == "human_written")),
        "epoch_h": ("epoch", sel("epoch", lambda k, y: k == "human")),
        "editlens_h": ("test", sel("test", lambda k, y: k == "human_written")),
        "editlens_gen": ("test", sel("test", lambda k, y: k == "ai_generated")),
        "editlens_edit": ("test", sel("test", lambda k, y: k == "ai_edited")),
        "enron_gen": ("test_enron", sel("test_enron", lambda k, y: k == "ai_generated")),
        "epoch_imit": ("epoch", sel("epoch", lambda k, y: k == "style_transfer")),
        "epoch_plain": ("epoch", sel("epoch", lambda k, y: k == "vanilla")),
        "magazine_ai": ("hd", sel("hd", lambda k, y: y == 1)),
        "fresh_old": ("fresh_old", np.arange(len(man["fresh_old"]["y"]))),
        "fresh_new": ("fresh_new", np.arange(len(man["fresh_new"]["y"]))),
        "hc3_h": ("hc3", sel("hc3", lambda k, y: y == 0)),
        "hc3_ai": ("hc3", sel("hc3", lambda k, y: y == 1)),
        "raid_h": ("raid", sel("raid", lambda k, y: y == 0)),
        "raid_ai": ("raid", sel("raid", lambda k, y: y == 1)),
    }


# ------------------------------------------------------------------ small stats
def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = allv.argsort().argsort().astype(float) + 1
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pauroc(y, s, max_fpr: float = 0.05) -> float:
    """McClish-standardised partial AUROC up to max_fpr (0.5 = chance, 1 = perfect), as sklearn's roc_auc_score."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y), np.asarray(s, float), max_fpr=max_fpr))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
