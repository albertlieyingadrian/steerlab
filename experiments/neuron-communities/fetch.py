"""Download the published neuron-community data this study uses (Goodfire's explorer files) into data/.

    python fetch.py

The explorer serves one index and one gzipped JSON per community from a pinned snapshot. The selection below is fixed
before any model run (seed 0) and is written to data/selection.json:
  * E1 (replication): 10 communities from each of the 6 classes, plus the curated 'an' case study (224838);
  * E2 (steering): every community in the 'topic' class (48).
"""

from __future__ import annotations

import gzip
import json
import random
import urllib.request
from pathlib import Path

ROOT = "https://static.goodfire.ai/neuron-communities/2026-11-12-4c7e9dd/"
DATA = Path(__file__).resolve().parent / "data"
AN = "224838"
PER_CLASS = 10


def get(path: str) -> dict:
    with urllib.request.urlopen(ROOT + path, timeout=60) as r:
        raw = r.read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def main() -> None:
    (DATA / "communities").mkdir(parents=True, exist_ok=True)
    index = get("index.json")
    (DATA / "index.json").write_text(json.dumps(index) + "\n")
    comms = sorted(index["communities"], key=lambda c: int(c["id"]))
    rng = random.Random(0)
    e1 = []
    for cls in [c["id"] for c in index["classes"]]:
        pool = [c["id"] for c in comms if c["class"] == cls and c["id"] != AN]
        e1 += rng.sample(pool, PER_CLASS)
    e1.append(AN)
    e2 = [c["id"] for c in comms if c["class"] == "topic"]
    sel = {"snapshot": ROOT, "seed": 0, "e1": e1, "e2": e2}
    (DATA / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    for cid in sorted(set(e1) | set(e2)):
        out = DATA / "communities" / f"{cid}.json"
        if not out.exists():
            out.write_text(json.dumps(get(f"communities/{cid}.json")) + "\n")
    print(f"E1 {len(e1)} communities, E2 {len(e2)} topic communities, {len(set(e1) | set(e2))} files")


if __name__ == "__main__":
    main()
