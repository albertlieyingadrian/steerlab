"""Send videos to the deployed WorldModelBench judge (modal_wmb.py) and cache its replies.

    python 5_wmb.py items [--limit 2]
    python 5_wmb.py traps

One container (max_containers=1), so calls run one after another; replies are appended to results/wmb_<source>.jsonl
as they arrive, and a rerun only sends what is missing. Wall time is logged to results/spend.csv at A10G's list price
($1.10 per hour), container start included.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

A10G_PER_S = 1.10 / 3600
LAWS = ["newton", "mass", "fluid", "penetration", "gravity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["items", "traps"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = RESULTS / f"wmb_{args.source}.jsonl"
    done = {json.loads(line)["video"] for line in out.read_text().splitlines()} if out.exists() else set()
    if args.source == "items":
        names = pd.read_csv(RESULTS / "items.csv").video.tolist()
        folder = DATA / "videos"
    else:
        names = pd.read_csv(RESULTS / "traps.csv").trap.tolist()
        folder = DATA / "traps"
    todo = [n for n in names if n not in done]
    if args.limit:
        todo = todo[: args.limit]
    judge = modal.Cls.from_name("physics-judge-wmb", "Judge")()
    t0, call_s = time.time(), 0.0
    with out.open("a") as f:
        calls = [(n, (folder / f"{n}.mp4").read_bytes()) for n in todo]
        for res in judge.judge.starmap(calls, order_outputs=False):
            f.write(json.dumps(res) + "\n")
            f.flush()
            call_s += res["seconds"]
            print(res["video"], {k: round(res[f"{k}_p"], 3) for k in LAWS}, f"{res['seconds']:.1f}s")
    wall = time.time() - t0
    with (RESULTS / "spend.csv").open("a") as f:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        f.write(f"{stamp},modal,wmb {args.source} {len(todo)} videos,A10G,0,0,0,{wall * A10G_PER_S:.4f}\n")
    print(f"{len(todo)} videos, {call_s:.0f} s in calls, {wall:.0f} s wall, about ${wall * A10G_PER_S:.2f}")


if __name__ == "__main__":
    main()
