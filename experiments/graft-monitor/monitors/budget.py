"""Spend ledger and cap for the monitor layer's Modal launches.

Every launch appends a worst-case estimate (timeout x $/h x containers) to results/spend.jsonl before it starts.
Only rows tagged TAG count toward CAP_USD, so the trainer's rows in the same ledger do not use this budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LEDGER = HERE / "results" / "spend.jsonl"
TAG = "[monitor-bench]"
CAP_USD = (
    60.0  # projected worst case; 15 for the bench, raised for the 8 h confirmatory server lifetime (run log 2026-11-26)
)
MODAL_PROFILE = "albertlieyingadrian"


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if TAG in r.get("note", ""):
                total += float(r.get("usd", 0.0))
    return total


def check(usd: float, what: str) -> None:
    s = spent()
    if s + usd > CAP_USD:
        raise SystemExit(
            f"budget: {what} would bring monitor spend to ${s + usd:.2f} > cap ${CAP_USD:.2f} "
            f"(logged so far ${s:.2f}); refusing"
        )


def log(usd: float, note: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": "modal_estimate",
                    "usd": round(usd, 4),
                    "note": f"{TAG} {note}",
                }
            )
            + "\n"
        )
