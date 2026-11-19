"""
Hard spend caps for Episode 3, enforced in code.

Modal ($300): the real spend is read from Modal's billing API (every app whose name starts with "pluralism-judge-rl",
from 2026-11-18 on). Before any paid job, a launcher calls check_modal(projected) and the job is refused if real spend
plus the job's projected cost would pass the cap. Every check is logged to results/spend/modal_ledger.tsv.

OpenRouter ($5): every call's cost (the API's usage.cost) is appended to results/spend/openrouter.jsonl, and
check_openrouter() refuses the next call once the total would pass the cap.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend"
MODAL_CAP = 300.0
OPENROUTER_CAP = 5.0  # cut from $10 on 2026-11-19 (series shares $24 on the second account)
APP_PREFIX = "pluralism-judge-rl"
START = "2026-11-18"
PRICE = {"H100": 3.95, "L40S": 1.95, "A100-80GB": 2.50}  # $/h, `modal billing rates` (read 2026-11-18)
CPU_MEM_PER_H = 4 * 0.0473 + 32 * 0.008  # 4 cores + 32 GiB
MODAL_BIN = os.environ.get("MODAL_BIN", str(HERE.parents[1] / ".venv" / "bin" / "modal"))


class BudgetExceeded(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def modal_spend():
    """Real Modal spend for this episode's apps, from the billing API (hourly buckets from START to now: daily
    buckets leave out the current, partial day)."""
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "albertlieyingadrian"))
    buckets = {}
    # full hours since START, plus today's buckets including the current partial hour (the first query leaves it out)
    for args in (["--start", START, "--resolution", "h"], ["--for", "today", "--resolution", "h"]):
        out = subprocess.run(
            [MODAL_BIN, "billing", "report", *args, "--json"], capture_output=True, text=True, env=env, check=True
        )
        for r in json.loads(out.stdout):
            if r.get("description", "").startswith(APP_PREFIX):
                key = (r["object_id"], r["interval_start"])
                buckets[key] = max(buckets.get(key, 0.0), float(r["cost"]))
    return sum(buckets.values())


def job_cost(gpu, hours):
    return hours * (PRICE[gpu] + CPU_MEM_PER_H)


def check_modal(projected, job):
    spent = modal_spend()
    SPEND.mkdir(parents=True, exist_ok=True)
    ledger = SPEND / "modal_ledger.tsv"
    if not ledger.exists():
        ledger.write_text("time_utc\tjob\tprojected_usd\tactual_total_usd\tdecision\n")
    ok = spent + projected <= MODAL_CAP
    with ledger.open("a") as f:
        f.write(f"{now()}\t{job}\t{projected:.2f}\t{spent:.2f}\t{'run' if ok else 'REFUSED'}\n")
    if not ok:
        raise BudgetExceeded(f"Modal cap ${MODAL_CAP:.0f}: spent ${spent:.2f} + projected ${projected:.2f} for {job}")
    print(f"[budget] Modal spent so far ${spent:.2f}; {job} projected ${projected:.2f}; cap ${MODAL_CAP:.0f}")
    return spent


def openrouter_spent():
    path = SPEND / "openrouter.jsonl"
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["cost"] for line in path.read_text().splitlines() if line.strip())


def charge_openrouter(model, cost, purpose):
    SPEND.mkdir(parents=True, exist_ok=True)
    with (SPEND / "openrouter.jsonl").open("a") as f:
        f.write(json.dumps({"time": now(), "model": model, "cost": cost, "purpose": purpose}) + "\n")


def check_openrouter(next_call_estimate=0.002):
    spent = openrouter_spent()
    if spent + next_call_estimate > OPENROUTER_CAP:
        raise BudgetExceeded(f"OpenRouter cap ${OPENROUTER_CAP:.0f} reached (${spent:.3f} spent)")
    return spent


if __name__ == "__main__":
    print(
        f"Modal: ${modal_spend():.2f} of ${MODAL_CAP:.0f}; OpenRouter: ${openrouter_spent():.3f} of "
        f"${OPENROUTER_CAP:.0f}"
    )
