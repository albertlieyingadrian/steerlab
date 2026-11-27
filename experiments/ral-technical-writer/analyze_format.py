"""The format trick over training: how many rollouts open with a title/abstract header, and carry markdown bold.

    python analyze_format.py   # rollouts copied from the volume (26 steps per run) -> results/format_trick.json

A header is the clean() rule in score.py: a first line that is a "Title:" or "Abstract" label. Reads
~/.cache/steerlab/ral-writer/vol/rollouts/<run>/step_<n>.json.gz (pulled with `modal volume get`, one file at a time).
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import score

ROLL = C.CACHE / "vol" / "rollouts"


def has_header(t: str) -> bool:
    first = next((ln for ln in t.strip().splitlines() if ln.strip()), "")
    return bool(score.HEADER_LINE.match(first)) and len(first.split()) <= 25


def main() -> None:
    out = {}
    for run in ("ral_s0", "frozen_s0", "ral_s1"):
        d = ROLL / run
        if not d.exists():
            continue
        rows = []
        for p in sorted(d.glob("step_*.json.gz"), key=lambda p: int(re.findall(r"\d+", p.name)[0])):
            R = json.load(gzip.open(p, "rt"))
            V = [r for r in R if r["answer"]]
            rows.append({"step": int(re.findall(r"\d+", p.name)[0]), "n": len(R), "valid": len(V),
                         "share_header": round(sum(has_header(r["answer"]) for r in V) / max(1, len(V)), 4),
                         "share_bold": round(sum("**" in r["answer"] for r in V) / max(1, len(V)), 4),
                         "words": round(sum(len(r["answer"].split()) for r in V) / max(1, len(V)), 1),
                         "reward": round(sum(r["reward"] for r in R) / len(R), 4)})  # fmt: skip
        out[run] = rows
        print(run, [(r["step"], r["share_header"], r["share_bold"], r["words"]) for r in rows])
    (C.RESULTS / "format_trick.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
