"""
Hard spend caps for The Taste Stack Episode 6, enforced in code (Episode 3's budget.py, with this episode's caps).

Modal ($450 soft, $500 hard for the episode): real spend comes from Modal's billing API (apps named "taste-ep6*",
from 2026-11-22). The series cap is $1,300; the five earlier episodes billed $62.72, so the
episode's hard cap keeps the series well inside it. Before any paid job a launcher calls check_modal(projected, job);
it warns past the soft cap and refuses past the hard cap. Every check is logged to results/spend/modal_ledger.tsv.
OpenRouter: none (no closed model is called anywhere in this episode).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend"
MODAL_SOFT, MODAL_HARD = 450.0, 500.0
SERIES_BEFORE, SERIES_CAP = 62.72, 1300.0
APP_PREFIX = "taste-ep6"
START = "2026-11-22"
MODAL_BIN = os.environ.get("MODAL_BIN", str(HERE.parents[1] / ".venv" / "bin" / "modal"))


class BudgetExceeded(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def modal_spend() -> float:
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "albertlieyingadrian"))
    buckets = {}
    for args in (["--start", START, "--resolution", "h"], ["--for", "today", "--resolution", "h"]):
        out = subprocess.run(
            [MODAL_BIN, "billing", "report", *args, "--json"], capture_output=True, text=True, env=env, check=True
        )
        for r in json.loads(out.stdout):
            if r.get("description", "").startswith(APP_PREFIX):
                key = (r["object_id"], r["interval_start"])
                buckets[key] = max(buckets.get(key, 0.0), float(r["cost"]))
    return sum(buckets.values())


def check_modal(projected: float, job: str) -> float:
    spent = modal_spend()
    SPEND.mkdir(parents=True, exist_ok=True)
    ledger = SPEND / "modal_ledger.tsv"
    if not ledger.exists():
        ledger.write_text("time_utc\tjob\tprojected_usd\tactual_total_usd\tdecision\n")
    ok = spent + projected <= MODAL_HARD and SERIES_BEFORE + spent + projected <= SERIES_CAP
    with ledger.open("a") as f:
        f.write(f"{now()}\t{job}\t{projected:.2f}\t{spent:.2f}\t{'run' if ok else 'REFUSED'}\n")
    if not ok:
        raise BudgetExceeded(f"Modal cap: spent ${spent:.2f} + projected ${projected:.2f} ({job})")
    if spent + projected > MODAL_SOFT:
        print(f"[budget] WARNING past the ${MODAL_SOFT:.0f} soft cap: ${spent:.2f} + ${projected:.2f}")
    print(f"[budget] Modal spent ${spent:.2f}; {job} projected ${projected:.2f}; hard cap ${MODAL_HARD:.0f}")
    return spent


if __name__ == "__main__":
    s = modal_spend()
    print(f"Modal spent ${s:.2f} (soft ${MODAL_SOFT:.0f}, hard ${MODAL_HARD:.0f}); series ${SERIES_BEFORE + s:.2f}")
