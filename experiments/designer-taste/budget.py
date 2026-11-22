"""
Hard spend caps for The Taste Stack Episode 7, enforced in code.

Modal: this episode's apps are named "taste-ep7*"; $40 soft (warn), $60 hard (refuse). The series cap ($1,300, shared
with Episode 6) is checked on every launch against every Taste Stack app since 2026-11-20. Real spend comes from
Modal's billing API. Every check is logged to results/spend/modal_ledger.tsv.

OpenRouter: $4 hard for this episode. A request reserves its worst case under a file lock before it is sent and is
settled to the reported cost afterwards; a crash therefore leaves the worst case counted, never less.

    python budget.py        # print spend so far
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend"
MODAL_SOFT, MODAL_HARD, SERIES_HARD = 40.0, 60.0, 1300.0
OPENROUTER_HARD = 4.0
APP_PREFIX = "taste-ep7"
SERIES_PREFIXES = ("taste-", "tl-ep", "design-index", "formalise-demo", "brand-spec")
START = "2026-11-20"
MODAL_BIN = os.environ.get("MODAL_BIN", str(HERE.parents[1] / ".venv" / "bin" / "modal"))
if not Path(MODAL_BIN).exists():  # a worktree without its own venv: the CLI beside this interpreter, else PATH
    MODAL_BIN = (
        str(Path(sys.executable).parent / "modal")
        if (Path(sys.executable).parent / "modal").exists()
        else (shutil.which("modal") or "modal")
    )


class BudgetExceeded(RuntimeError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _billing() -> list[dict]:
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "albertlieyingadrian"))
    rows = []
    for args in (["--start", START, "--resolution", "h"], ["--for", "today", "--resolution", "h"]):
        out = subprocess.run(
            [MODAL_BIN, "billing", "report", *args, "--json"], capture_output=True, text=True, env=env, check=True
        )
        rows += json.loads(out.stdout)
    return rows


def modal_spend() -> tuple[float, float]:
    """(this episode, whole series) in dollars; hourly buckets de-duplicated across the two reports."""
    ep, series = {}, {}
    for r in _billing():
        desc = r.get("description", "")
        key = (r["object_id"], r["interval_start"])
        if desc.startswith(APP_PREFIX):
            ep[key] = max(ep.get(key, 0.0), float(r["cost"]))
        if desc.startswith(SERIES_PREFIXES):
            series[key] = max(series.get(key, 0.0), float(r["cost"]))
    return sum(ep.values()), sum(series.values())


def check_modal(projected: float, job: str) -> float:
    spent, series = modal_spend()
    SPEND.mkdir(parents=True, exist_ok=True)
    ledger = SPEND / "modal_ledger.tsv"
    if not ledger.exists():
        ledger.write_text("time_utc\tjob\tprojected_usd\tepisode_total_usd\tseries_total_usd\tdecision\n")
    ok = spent + projected <= MODAL_HARD and series + projected <= SERIES_HARD
    with ledger.open("a") as f:
        f.write(f"{now()}\t{job}\t{projected:.2f}\t{spent:.2f}\t{series:.2f}\t{'run' if ok else 'REFUSED'}\n")
    if not ok:
        raise BudgetExceeded(
            f"cap: episode ${spent:.2f} (hard ${MODAL_HARD:.0f}), series ${series:.2f} (hard ${SERIES_HARD:.0f}), "
            f"projected ${projected:.2f} ({job})"
        )
    if spent + projected > MODAL_SOFT:
        print(f"[budget] WARNING past the ${MODAL_SOFT:.0f} soft cap: ${spent:.2f} + ${projected:.2f}")
    print(f"[budget] Modal episode ${spent:.2f}, series ${series:.2f}; {job} projected ${projected:.2f}")
    return spent


def openrouter_spent() -> float:
    path = SPEND / "openrouter.jsonl"
    if not path.exists():
        return 0.0
    return sum(json.loads(line).get("cost", 0.0) for line in path.read_text().splitlines() if line.strip())


class OpenRouterLedger:
    """Append-only ledger: the sum of `cost` over reserve and settle rows is the spend that counts against the cap."""

    def __init__(self) -> None:
        SPEND.mkdir(parents=True, exist_ok=True)
        self.path = SPEND / "openrouter.jsonl"
        self.lock = SPEND.parent.parent / ".openrouter.lock"

    def reserve(self, worst: float, job: str) -> dict | None:
        """Count `worst` and return the reservation, or None if it would pass the cap."""
        with self.lock.open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if openrouter_spent() + worst > OPENROUTER_HARD:
                return None
            res = {"rid": uuid.uuid4().hex[:12], "job": job, "worst": worst}
            self._write({"rid": res["rid"], "kind": "reserve", "job": job, "cost": worst})
            return res

    def settle(self, res: dict, actual: float) -> None:
        with self.lock.open("a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            self._write({"rid": res["rid"], "kind": "settle", "job": res["job"], "cost": actual - res["worst"]})
        if actual > res["worst"]:
            raise BudgetExceeded(f"a call cost ${actual:.5f}, over its ${res['worst']:.5f} reservation: fix the bound")

    def _write(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"t": now(), **row}) + "\n")


if __name__ == "__main__":
    ep, series = modal_spend()
    print(
        f"Modal episode ${ep:.2f} (soft ${MODAL_SOFT:.0f}, hard ${MODAL_HARD:.0f}); series ${series:.2f} / ${SERIES_HARD:.0f}"
    )
    print(f"OpenRouter episode ${openrouter_spent():.2f} / ${OPENROUTER_HARD:.0f}")
