"""
Spend caps for the RAL technical-writer study, enforced in code.

- Modal (workspace credits30, apps named "ral-writer*", from 2026-11-27): soft $120 (the plan's initial cap),
  hard $200. Raises are logged in the plan's amendments and results/spend/modal_ledger.tsv.
- Pangram and StealthGPT: each paid call is logged to results/spend/api_ledger.jsonl with its word count and the
  price from the vendor's pricing page (both $0.05 per 100 words, checked 2026-11-27); caps $40 and $20.
No OpenRouter or closed-model API is called anywhere in this study.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEND = HERE / "results" / "spend"
MODAL_SOFT, MODAL_HARD = 120.0, 200.0
API_CAPS = {"pangram": 40.0, "stealthgpt": 20.0}
API_RATE = {
    "pangram": 0.05 / 100,
    "stealthgpt": 0.05 / 100,
}  # dollars per word (StealthGPT super; standard 0.20 / 1000)
APP_PREFIX = "ral-writer"
START = "2026-11-27"
PROFILE = "credits30"
MODAL_BIN = os.environ.get("MODAL_BIN", str(Path(sys.executable).parent / "modal"))


class BudgetExceeded(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def modal_spend() -> float:
    env = dict(os.environ, MODAL_PROFILE=PROFILE)
    buckets: dict = {}
    for args in (["--start", START, "--resolution", "h"], ["--for", "today", "--resolution", "h"]):
        out = subprocess.run([MODAL_BIN, "billing", "report", *args, "--json"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             env=env, check=True)  # fmt: skip
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
        f.write(f"{now()}\t{job}\t{projected:.2f}\t{spent:.2f}\t{'run' if ok else 'REFUSED'}\n")
    if not ok:
        raise BudgetExceeded(f"Modal cap: spent ${spent:.2f} + projected ${projected:.2f} ({job})")
    if spent + projected > MODAL_SOFT:
        print(f"[budget] past the ${MODAL_SOFT:.0f} soft cap: ${spent:.2f} + ${projected:.2f} (log the raise)")
    print(f"[budget] Modal spent ${spent:.2f}; {job} projected ${projected:.2f}; hard cap ${MODAL_HARD:.0f}")
    return spent


def api_spent(vendor: str) -> float:
    p = SPEND / "api_ledger.jsonl"
    if not p.exists():
        return 0.0
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return sum(r["usd"] for r in rows if r["vendor"] == vendor)


def check_api(vendor: str, words: int) -> None:
    """Refuse a request whose estimated charge would pass the vendor's cap (nothing logged)."""
    usd = words * API_RATE[vendor]
    spent = api_spent(vendor)
    if spent + usd > API_CAPS[vendor]:
        raise BudgetExceeded(f"{vendor} cap ${API_CAPS[vendor]:.0f}: spent ${spent:.2f} + ${usd:.2f}")


def log_api(vendor: str, words: int, item: str, rate: float | None = None, note: str = "") -> None:
    """Log a charge: `words` as billed (StealthGPT reports wordsSpent), at `rate` dollars per word."""
    usd = words * (API_RATE[vendor] if rate is None else rate)
    SPEND.mkdir(parents=True, exist_ok=True)
    row = {"time": now(), "vendor": vendor, "item": item, "words": words, "usd": round(usd, 4), "note": note}
    with (SPEND / "api_ledger.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def charge_api(vendor: str, words: int, item: str) -> None:
    """Pre-check and log at the list price (Pangram bills the words sent)."""
    check_api(vendor, words)
    log_api(vendor, words, item)


if __name__ == "__main__":
    s = modal_spend()
    print(f"Modal ({PROFILE}, {APP_PREFIX}*) spent ${s:.2f} (soft ${MODAL_SOFT:.0f}, hard ${MODAL_HARD:.0f})")
    for v in API_CAPS:
        print(f"{v}: ${api_spent(v):.2f} of ${API_CAPS[v]:.0f}")
