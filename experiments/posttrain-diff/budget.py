"""Hard Modal spend cap for the post-training unit diff, enforced in code.

Every paid call goes through `paid(job, gpu, projected_usd, fn, *args)`. It reads the real spend of Modal apps named
`posttrain-diff*` from the billing API (or this episode's wall-time ledger, whichever is larger), refuses the job if
spend + projection would pass HARD_CAP, warns past SOFT_CAP, and appends the wall-time charge to results/spend.jsonl.
No OpenRouter or OpenAI calls are made by this episode. The Modal profile is recorded with every row.

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
SOFT_CAP, HARD_CAP = 100.0, 150.0
APP_PREFIX = "posttrain-diff"
START = "2026-11-22"
PRICE = {"H100": 3.95, "A100-80GB": 2.50, "L4": 0.80, "A10G": 1.10, "L40S": 1.95, "cpu": 0.0}  # $/h, list price
CPU_MEM_PER_H = 4 * 0.0473 + 32 * 0.008
_VENVS = [p / ".venv" / "bin" / "modal" for p in HERE.parents]  # a worktree has no .venv; the main checkout does
MODAL_BIN = os.environ.get("MODAL_BIN", str(next((p for p in _VENVS if p.exists()), _VENVS[1])))


class BudgetExceeded(RuntimeError):
    pass


def profile() -> str:
    return os.environ.get("MODAL_PROFILE", "modal4")


def ledger_spend() -> float:
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in SPEND.read_text().splitlines() if line.strip())


def billing_spend() -> float | None:
    env = dict(os.environ, MODAL_PROFILE=profile())
    buckets: dict = {}
    try:
        out = subprocess.run([MODAL_BIN, "billing", "report", "--start", START, "--resolution", "h", "--json"],
                             capture_output=True, text=True, env=env, check=True, timeout=120)  # fmt: skip
        for r in json.loads(out.stdout):
            if r.get("description", "").startswith(APP_PREFIX):
                key = (r["object_id"], r["interval_start"])
                buckets[key] = max(buckets.get(key, 0.0), float(r["cost"]))
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as e:  # billing API down: the ledger stands
        print(f"[budget] billing API unavailable ({type(e).__name__}); using the wall-time ledger")
        return None
    return sum(buckets.values())


def spent() -> float:
    b = billing_spend()
    return max(ledger_spend(), b if b is not None else 0.0)


def paid(job: str, gpu: str, projected_usd: float, fn, *args, gpus: int = 1, **kw):
    """Run fn(*args, **kw) (a Modal .remote call) under the cap; record gpus x wall time at list price."""
    s = spent()
    if s + projected_usd > HARD_CAP:
        raise BudgetExceeded(f"Modal hard cap ${HARD_CAP:.0f}: spent ${s:.2f} + projected ${projected_usd:.2f} ({job})")
    if s + projected_usd > SOFT_CAP:
        print(f"[budget] WARNING past the soft cap ${SOFT_CAP:.0f}")
    print(f"[budget] spent ${s:.2f}; {job} projected ${projected_usd:.2f}; cap ${HARD_CAP:.0f}", flush=True)
    t0 = time.time()
    try:
        return fn(*args, **kw)
    finally:
        sec = time.time() - t0
        SPEND.parent.mkdir(parents=True, exist_ok=True)
        usd = gpus * sec / 3600 * (PRICE[gpu] + CPU_MEM_PER_H)
        with SPEND.open("a") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": job, "gpu": gpu,
                                 "gpus": gpus, "seconds": round(sec, 1), "projected": round(projected_usd, 3),
                                 "usd": round(usd, 4), "profile": profile()}) + "\n")  # fmt: skip


if __name__ == "__main__":
    b = billing_spend()
    print(f"ledger ${ledger_spend():.2f}; billing {'n/a' if b is None else f'${b:.2f}'}; "
          f"soft ${SOFT_CAP:.0f}, hard ${HARD_CAP:.0f}; profile {profile()}")  # fmt: skip
