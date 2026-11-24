"""Hard spend caps for detector-attribution, enforced in code: Pangram $60, Modal $80.

Every paid call is checked against the ledger before it runs (`check`) and logged after (`log`), in
results/spend.jsonl. Modal is charged by wall time x H100 list price x containers (an upper bound: containers idle
part of the time); Pangram by billed blocks (each started 100-word block, $0.04 on the bulk API).

    python budget.py          # spend so far per provider
"""

from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend.jsonl"
CAPS = {"pangram": 60.0, "modal": 80.0}  # Modal raised from 60 for the generator-side arm (plan amendment 1)
H100_PER_H = 3.95


class BudgetExceeded(RuntimeError):
    pass


def spent(provider: str) -> float:
    if not SPEND.exists():
        return 0.0
    rows = [json.loads(x) for x in SPEND.read_text().splitlines() if x.strip()]
    return sum(r["usd"] for r in rows if r["provider"] == provider)


def log(job: str, provider: str, usd: float, **extra) -> None:
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    with SPEND.open("a") as f:
        f.write(
            json.dumps(
                {"date": time.strftime("%F %T"), "job": job, "provider": provider, "usd": round(usd, 4), **extra}
            )
            + "\n"
        )


def check(provider: str, projected: float) -> None:
    s = spent(provider)
    if s + projected > CAPS[provider]:
        raise BudgetExceeded(f"{provider}: ${s:.2f} spent + ${projected:.2f} projected > ${CAPS[provider]:.0f} cap")


def paid_modal(job: str, projected: float, fn, *args, containers: int = 1, **kw):
    check("modal", projected)
    t0 = time.time()
    try:
        return fn(*args, **kw)
    finally:
        hours = (time.time() - t0) / 3600
        log(
            job,
            "modal",
            hours * H100_PER_H * containers,
            gpu="H100",
            seconds=round(hours * 3600),
            containers=containers,
        )


if __name__ == "__main__":
    for p, cap in CAPS.items():
        print(f"{p}: ${spent(p):.2f} of ${cap:.0f}")
