"""The battery's Modal spend cap, enforced before every launch, and its rows in the shared ledger.

Ledger: experiments/graft-monitor/results/spend.jsonl, rows {"ts": ISO-8601 UTC, "kind": "modal_estimate",
"usd": float, "note": str}. The battery's rows have notes starting "battery:"; the cap counts only those, so the other
graft-monitor arms keep their own caps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "results" / "spend.jsonl"
CAP_USD = 20.0  # 8 for building and validation; raised for the Q1/Q2 drift runs (run log 2026-11-26)
PREFIX = "battery:"

# List prices (USD per hour) for the GPU plus the CPU and memory requested.
L4_PER_HOUR = 0.80 + 4 * 0.0473 + 16 * 0.0080
L40S_PER_HOUR = 1.95 + 4 * 0.0473 + 32 * 0.0080
H100_PER_HOUR = 3.95 + 4 * 0.0473 + 64 * 0.0080


class BudgetExceeded(SystemExit):
    pass


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if str(row.get("note", "")).startswith(PREFIX):
                total += float(row.get("usd", 0.0))
    return total


def check(projected_usd: float, what: str) -> None:
    """Refuse a launch whose projected cost would take the battery past its cap."""
    s = spent()
    if s + projected_usd > CAP_USD:
        raise BudgetExceeded(
            f"battery Modal cap ${CAP_USD:.2f}: spent ${s:.2f}, {what} projects ${projected_usd:.2f}; not launching"
        )


def log(usd: float, note: str) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": "modal_estimate",
                    "usd": round(usd, 4),
                    "note": PREFIX + " " + note,
                }
            )
            + "\n"
        )
