"""B1: relabel data v2 by how each text was made (pre-registered in docs/plans/2026-11-12-beat-voi-levers.md).

    python relabel_b1.py    # data/{train,val,test}.jsonl -> data/v2b/*.jsonl (local) and results/b1_labels.json

Same rows and splits as data v2; no new generation. Targets:
- human -> 0; generated (mirror, imitate) -> 1;
- edited (light, partial, rewrite, roundtrip) -> 0.20 + 0.50 x share of words changed, so every edit lands in 0.20-0.70:
  never in the human band, never in the generated band (in v2, 14,381 edits sat below 0.15 and 771 at 0.75 or above).
Bands by how the text was made (0 human, 1 edited, 2 generated), re-balanced weights, and four buckets for arm B:
human; lighter edit; heavier edit; generated. Lighter vs heavier is split at the median share changed of the TRAINING
edits, computed once here and saved.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from common import DATA, RESULTS, read_jsonl

EDITS = ("light", "partial", "rewrite", "roundtrip")
FULL = ("mirror", "imitate")
OUT = DATA / "v2b"


def frac(r: dict) -> float:
    return float(r["target"])  # in data v2 an edit's target is its share of words changed


def main() -> None:
    rows = {s: read_jsonl(DATA / f"{s}.jsonl") for s in ("train", "val", "test")}
    median = float(np.median([frac(r) for r in rows["train"] if r["kind"] in EDITS]))
    counts = Counter()
    for split, rs in rows.items():
        for r in rs:
            if r["kind"] == "human":
                r["target"], r["band"], r["bucket"] = 0.0, 0, 0
            elif r["kind"] in FULL:
                r["target"], r["band"], r["bucket"] = 1.0, 2, 3
            elif r["kind"] in EDITS:
                f = frac(r)
                r["frac"] = f
                r["target"], r["band"] = round(0.20 + 0.50 * f, 4), 1
                r["bucket"] = 1 if f <= median else 2
            else:
                raise ValueError(r["kind"])
            counts[(split, r["bucket"])] += 1
    train_bands = Counter(r["band"] for r in rows["train"])
    total = sum(train_bands.values())
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        for r in rs:
            r["weight"] = round(total / (3 * train_bands[r["band"]]), 4)
        with (OUT / f"{split}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    report = {
        "median_share_changed_train_edits": round(median, 4),
        "rows": {s: len(rs) for s, rs in rows.items()},
        "buckets": {f"{s} bucket {b}": n for (s, b), n in sorted(counts.items())},
        "train_band_share": {str(b): round(n / total, 3) for b, n in sorted(train_bands.items())},
    }
    (RESULTS / "b1_labels.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
