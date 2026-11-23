"""Hard Modal spend cap for this study, enforced in code (profile credits30).

`guard(job, gpu, max_hours, expected_hours)` refuses a launch when spend so far + the job's worst case (its Modal
timeout x list price) would pass HARD_CAP; `record(...)` appends the job's wall-time cost. Spend = max(ledger, billing
for apps named `sae-intersectional-bias*` since START).

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
HARD_CAP = 40.0  # this note; the project cap is $60
APP_PREFIX = "sae-intersectional-bias"
START = "2026-11-23"
# $/h: GPU list price + the CPU and memory each function requests (8 cores x $0.047, GB x $0.008); the billing API,
# when it answers, is the authority (spent() takes the larger)
PRICE = {"A100-40GB": 2.10 + 0.38 + 0.51, "A10G": 1.10 + 0.38 + 0.39, "cpu": 0.19 + 0.13}
_VENVS = [p / ".venv" / "bin" / "modal" for p in HERE.parents]
MODAL_BIN = os.environ.get("MODAL_BIN", str(next((p for p in _VENVS if p.exists()), _VENVS[1])))


class BudgetExceeded(RuntimeError):
    pass


def profile() -> str:
    return os.environ.get("MODAL_PROFILE", "credits30")


def ledger_spend() -> float:
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in SPEND.read_text().splitlines() if line.strip())


def billing_spend() -> float | None:
    env = dict(os.environ, MODAL_PROFILE=profile())
    try:
        out = subprocess.run([MODAL_BIN, "billing", "report", "--start", START, "--resolution", "h", "--json"],
                             capture_output=True, text=True, env=env, check=True, timeout=120)  # fmt: skip
        rows = json.loads(out.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        print(f"[budget] billing API unavailable ({type(e).__name__}); using the ledger")
        return None
    mine = [r for r in rows if r.get("description", "").startswith(APP_PREFIX)]
    if rows and not mine:
        print("[budget] billing returned rows but none matched the app prefix; check the description field")
    return sum(float(r["cost"]) for r in mine)


def spent() -> float:
    b = billing_spend()
    return max(ledger_spend(), b or 0.0)


def guard(job: str, gpu: str, max_hours: float, expected_hours: float) -> None:
    s = spent()
    worst = max_hours * PRICE[gpu]
    if s + worst > HARD_CAP:
        raise BudgetExceeded(f"cap ${HARD_CAP:.0f}: spent ${s:.2f} + worst case ${worst:.2f} ({job})")
    print(f"[budget] spent ${s:.2f}; {job}: expected ${expected_hours * PRICE[gpu]:.2f}, worst ${worst:.2f}; "
          f"cap ${HARD_CAP:.0f}")  # fmt: skip


def record(job: str, gpu: str, seconds: float, note: str = "") -> None:
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    usd = seconds / 3600 * PRICE[gpu]
    with SPEND.open("a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": job, "gpu": gpu,
                             "seconds": round(seconds, 1), "usd": round(usd, 4), "profile": profile(),
                             "note": note}) + "\n")  # fmt: skip


if __name__ == "__main__":
    b = billing_spend()
    print(f"ledger ${ledger_spend():.2f}; billing {'n/a' if b is None else f'${b:.2f}'}; cap ${HARD_CAP:.0f}; "
          f"profile {profile()}")  # fmt: skip
