"""Step 1: build the FineWeb pools on Modal and check them.

    python pools.py build     # Modal CPU: one shard per dump -> hn/pools/{pool,heldout,dev,modern}.jsonl on the volume
    python pools.py pull      # copy the pools to data/pools/ and check dates, host split and overlap

Dumps crawled before November 2022 only; CC-MAIN-2022-40 (Sep-Oct 2022) is the latest. The 2025 dump gives modern
topic seeds for fresh AI pages and is never used as human text. Writes results/pools.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

from hn import DATA, HERE, PROFILE, RESULTS, VOLUME, read_jsonl, reserve, settle

OLD = [
    "CC-MAIN-2017-04",
    "CC-MAIN-2018-30",
    "CC-MAIN-2019-35",
    "CC-MAIN-2020-24",
    "CC-MAIN-2021-17",
    "CC-MAIN-2021-43",
    "CC-MAIN-2022-21",
    "CC-MAIN-2022-40",
]
NEW = "CC-MAIN-2025-26"
QUOTA = {"pool": 31250, "heldout": 2600, "dev": 2500, "modern": 800}
MODAL = os.environ.get(  # the nearest .venv's modal CLI (a worktree uses its checkout's), else the one on PATH
    "MODAL_BIN",
    next((str(p / ".venv/bin/modal") for p in HERE.parents if (p / ".venv/bin/modal").exists()), "modal"),
)


def vget(remote: str, local) -> None:
    subprocess.run(
        [MODAL, "volume", "get", "--force", VOLUME, remote, str(local)],
        check=True,
        env={**os.environ, "MODAL_PROFILE": PROFILE},
        capture_output=True,
    )


def vput(local, remote: str) -> None:
    subprocess.run(
        [MODAL, "volume", "put", "--force", VOLUME, str(local), remote],
        check=True,
        env={**os.environ, "MODAL_PROFILE": PROFILE},
        capture_output=True,
    )


def build() -> None:
    import time

    import modal

    reserve("build_pools", "cpu", 3)
    t0 = time.time()
    out = modal.Function.from_name("slop-hard-negatives", "build_pools").remote(OLD, NEW, QUOTA)
    settle("build_pools", "cpu", time.time() - t0)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pools_build.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["sizes"]))


def pull() -> None:
    (DATA / "pools").mkdir(parents=True, exist_ok=True)
    for k in ("pool", "heldout", "dev", "modern"):
        vget(f"hn/pools/{k}.jsonl", DATA / "pools" / f"{k}.jsonl")
    pools = {k: read_jsonl(DATA / "pools" / f"{k}.jsonl") for k in ("pool", "heldout", "dev", "modern")}
    hosts = {k: {r["host"] for r in v} for k, v in pools.items()}
    old = [r for k in ("pool", "heldout", "dev") for r in pools[k]]
    report = {
        "sizes": {k: len(v) for k, v in pools.items()},
        "latest_date_old": max(r["date"] for r in old),
        "earliest_date_old": min(r["date"] for r in old),
        "all_old_before_2022_11": all(r["date"] < "2022-11-01" for r in old),
        "dumps": {k: dict(sorted(Counter(r["dump"] for r in v).items())) for k, v in pools.items()},
        "hosts": {k: len(v) for k, v in hosts.items()},
        "host_overlap_pool_heldout": len(hosts["pool"] & hosts["heldout"]),
        "host_overlap_pool_dev": len(hosts["pool"] & hosts["dev"]),
        "host_overlap_heldout_dev": len(hosts["heldout"] & hosts["dev"]),
        "median_words": {k: sorted(r["words"] for r in v)[len(v) // 2] for k, v in pools.items()},
    }
    assert report["all_old_before_2022_11"], "a page crawled after the cutoff"
    assert report["host_overlap_pool_heldout"] == 0 and report["host_overlap_pool_dev"] == 0
    (RESULTS / "pools.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def interp_src() -> None:
    """The dev pages the interp concept sets are drawn from, exactly as cut for the note: dev[125:3125] for the
    corporate-register labels (dev[:125] went to the judge's known-answer set) and dev[3200:3600] for the slop pairs."""
    from hn import write_jsonl

    dev = read_jsonl(DATA / "pools" / "dev.jsonl")
    write_jsonl(DATA / "interp_corp_src.jsonl", dev[125:3125])
    write_jsonl(DATA / "interp_slop_src.jsonl", dev[3200:3600])


if __name__ == "__main__":
    {"build": build, "pull": pull, "interp": interp_src}[sys.argv[1]]()
