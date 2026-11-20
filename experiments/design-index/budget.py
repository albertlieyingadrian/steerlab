"""
Hard spend caps for Taste Labs Episode 2, enforced in code.

Modal ($80 soft, $150 hard): real spend comes from Modal's billing API (apps named "design-index*", from 2026-11-20).
Before any paid job a launcher calls check_modal(projected, job); it warns past the soft cap and refuses past the hard
cap. Every check is logged to results/spend/modal_ledger.tsv.

Exa ($25 hard): every search's cost (the API's costDollars.total, or the list price if missing) is appended to
results/spend/exa.jsonl, and check_exa() refuses the next call once the total plus that call would pass the cap.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend"
MODAL_SOFT, MODAL_HARD = 80.0, 150.0
EXA_CAP = 25.0
EXA_PRICE_25 = 0.022  # $/search with 25 results (exa.ai/pricing, read 2026-11-20: $7/1k + $1/1k per result above 10)
APP_PREFIX = "design-index"
START = "2026-11-20"
MODAL_BIN = os.environ.get(
    "MODAL_BIN", str(HERE.parents[1] / ".venv" / "bin" / "modal")
)


class BudgetExceeded(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def modal_spend() -> float:
    env = dict(
        os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "albertlieyingadrian")
    )
    buckets = {}
    for args in (
        ["--start", START, "--resolution", "h"],
        ["--for", "today", "--resolution", "h"],
    ):
        out = subprocess.run(
            [MODAL_BIN, "billing", "report", *args, "--json"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
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
    ok = spent + projected <= MODAL_HARD
    with ledger.open("a") as f:
        f.write(
            f"{now()}\t{job}\t{projected:.2f}\t{spent:.2f}\t{'run' if ok else 'REFUSED'}\n"
        )
    if not ok:
        raise BudgetExceeded(
            f"Modal hard cap ${MODAL_HARD:.0f}: spent ${spent:.2f} + projected ${projected:.2f} ({job})"
        )
    if spent + projected > MODAL_SOFT:
        print(
            f"[budget] WARNING past the ${MODAL_SOFT:.0f} soft cap: ${spent:.2f} + ${projected:.2f}"
        )
    print(
        f"[budget] Modal spent ${spent:.2f}; {job} projected ${projected:.2f}; hard cap ${MODAL_HARD:.0f}"
    )
    return spent


def exa_spent() -> float:
    path = SPEND / "exa.jsonl"
    if not path.exists():
        return 0.0
    return sum(
        json.loads(line)["cost"]
        for line in path.read_text().splitlines()
        if line.strip()
    )


def check_exa(next_cost: float = EXA_PRICE_25) -> None:
    spent = exa_spent()
    if spent + next_cost > EXA_CAP:
        raise BudgetExceeded(f"Exa cap ${EXA_CAP:.0f}: spent ${spent:.3f}")


def log_exa(query: str, cost: float, n: int) -> None:
    SPEND.mkdir(parents=True, exist_ok=True)
    with (SPEND / "exa.jsonl").open("a") as f:
        f.write(
            json.dumps({"time": now(), "query": query, "cost": cost, "n_results": n})
            + "\n"
        )


if __name__ == "__main__":
    print(f"Exa spent ${exa_spent():.3f} of ${EXA_CAP:.0f}")
    print(
        f"Modal spent ${modal_spend():.2f} (soft ${MODAL_SOFT:.0f}, hard ${MODAL_HARD:.0f})"
    )
