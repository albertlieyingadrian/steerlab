"""Print the latest training steps of runs from the Modal Volume (read-only; no GPU cost).

MODAL_PROFILE=albertlieyingadrian .venv/bin/python experiments/brand-rl/status.py run_a run_b [--last 6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MODAL = str(Path(sys.executable).parent / "modal")
COLS = ("reward", "brand", "coverage", "fidelity_strict", "heldout_strict", "slop_p", "copy", "novelty", "truncated")


def fetch_log(run: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "log.jsonl"
        r = subprocess.run(
            [MODAL, "volume", "get", "taste-ep6", f"runs/{run}/log.jsonl", str(p), "--force"],
            capture_output=True,
            text=True,
        )
        if r.returncode or not p.exists():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--last", type=int, default=6)
    a = ap.parse_args()
    for run in a.runs:
        rows = fetch_log(run)
        print(f"== {run}: {len(rows)} steps")
        for r in rows[-a.last :]:
            vals = " ".join(f"{c[:6]} {r[c]:.3f}" for c in COLS if r.get(c) is not None)
            print(
                f"{r['step']:4d} {vals} tok {r['mean_tokens']:.0f} upd {r['n_update_seqs']} gn {r['grad_norm']:.2f}"
                f" t {r['t_gen']:.0f}/{r['t_score']:.0f}/{r['t_step']:.0f}"
            )


if __name__ == "__main__":
    main()
