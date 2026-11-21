"""Hard Modal spend cap for The Taste Stack, Episode 1, enforced in code.

Every paid call goes through `paid(job, gpu, projected_usd, fn, *args)`: it reads the real spend of every Modal app
whose name starts with `tl-ep1` from the billing API (or this episode's wall-time ledger, whichever is larger), refuses
the job if spend + projection would pass HARD_CAP, warns past SOFT_CAP, and appends the wall-time charge to
results/spend.jsonl. OpenRouter is not used by this episode.

    python budget.py          # print spend so far
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend.jsonl"
SOFT_CAP, HARD_CAP = (
    150.0,
    200.0,
)  # team lead, 2026-11-20: budget $100-150, hard stop $200
APP_PREFIX = "tl-ep1"
START = "2026-11-20"
PRICE = {
    "H100": 3.95,
    "L40S": 1.95,
    "A100-80GB": 2.50,
    "cpu": 0.0,
}  # $/h GPU, Modal list price
CPU_MEM_PER_H = 4 * 0.0473 + 32 * 0.008  # 4 cores + 32 GiB beside each GPU
MODAL_BIN = os.environ.get("MODAL_BIN", str(HERE.parents[1] / ".venv" / "bin" / "modal"))


class BudgetExceeded(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_spend():
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in SPEND.read_text().splitlines() if line.strip())


def billing_spend():
    """Real spend from Modal's billing API for this episode's apps (hourly buckets since START), or None."""
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "albertlieyingadrian"))
    buckets = {}
    try:
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
                timeout=120,
            )
            for r in json.loads(out.stdout):
                if r.get("description", "").startswith(APP_PREFIX):
                    key = (r["object_id"], r["interval_start"])
                    buckets[key] = max(buckets.get(key, 0.0), float(r["cost"]))
    except Exception as e:  # billing API down or not permitted: fall back to the wall-time ledger
        print(f"[budget] billing API unavailable ({type(e).__name__}); using the wall-time ledger")
        return None
    return sum(buckets.values())


def spent():
    b = billing_spend()
    return max(ledger_spend(), b if b is not None else 0.0)


def charge(gpu, seconds):
    return seconds / 3600 * (PRICE[gpu] + CPU_MEM_PER_H)


def paid(job, gpu, projected_usd, fn, *args, **kw):
    """Run fn(*args, **kw) (a Modal .remote call) under the cap; record the wall-time charge."""
    s = spent()
    if s + projected_usd > HARD_CAP:
        raise BudgetExceeded(f"Modal hard cap ${HARD_CAP:.0f}: spent ${s:.2f} + projected ${projected_usd:.2f} ({job})")
    if s + projected_usd > SOFT_CAP:
        print(f"[budget] WARNING past the soft cap ${SOFT_CAP:.0f}")
    print(
        f"[budget] spent ${s:.2f}; {job} projected ${projected_usd:.2f}; hard cap ${HARD_CAP:.0f}",
        flush=True,
    )
    t0 = time.time()
    try:
        return fn(*args, **kw)
    finally:
        sec = time.time() - t0
        SPEND.parent.mkdir(parents=True, exist_ok=True)
        with SPEND.open("a") as fh:
            fh.write(
                json.dumps(
                    dict(
                        time=now(),
                        job=job,
                        gpu=gpu,
                        seconds=round(sec, 1),
                        projected=round(projected_usd, 3),
                        usd=round(charge(gpu, sec), 4),
                    )
                )
                + "\n"
            )


if __name__ == "__main__":
    b = billing_spend()
    print(
        f"ledger ${ledger_spend():.2f}; billing {'n/a' if b is None else f'${b:.2f}'}; "
        f"soft ${SOFT_CAP:.0f}, hard ${HARD_CAP:.0f}"
    )
