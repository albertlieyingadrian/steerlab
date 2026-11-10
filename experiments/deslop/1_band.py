"""Score the human passages and write the band the toolkit ships (packages/deslop/src/deslop/data/human_band.json).

    python experiments/deslop/1_band.py

Prints each source's median, 75th and 90th percentiles and its per-habit rates, so the habits people use too (lists
of three, em dashes in older prose) are visible next to the score. Run again whenever a detector or weight changes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "packages" / "deslop" / "src"))
from deslop.habits import DETECTORS, count_words, find_spans
from deslop.score import build_band

PASSAGES = HERE / "data" / "human" / "passages.jsonl"
BAND = ROOT / "packages" / "deslop" / "src" / "deslop" / "data" / "human_band.json"


def main() -> None:
    rows = [json.loads(line) for line in PASSAGES.read_text(encoding="utf-8").splitlines()]
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row["text"])
    bands = build_band(groups)
    payload = {
        "built": date.today().isoformat(),
        "source": "experiments/deslop/data/human/passages.jsonl: Project Gutenberg essays and memoirs published "
        "before 1929 (public domain in the US) and English Wikipedia as of 2021-06-01 (CC BY-SA 3.0)",
        "bands": bands,
    }
    BAND.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for name, band in bands.items():
        print(
            f"{name:10} n={band['n']:4d} median {band['median']:6.2f} p75 {band['p75']:6.2f} p90 {band['limit']:6.2f}"
        )
    for name, texts in groups.items():
        words = sum(count_words(t) for t in texts)
        counts = Counter(s.habit for t in texts for s in find_spans(t))
        rates = ", ".join(f"{h} {1000 * counts[h] / words:.1f}" for h in DETECTORS if counts[h])
        print(f"{name:10} per 1000 words: {rates}")
    print(f"wrote {BAND.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
