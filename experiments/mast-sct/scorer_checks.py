"""Reproduce the two scorer issues of sections 4.4 and 4.5 on MAST's shipped GPT-5.5 run.

    python experiments/mast-sct/scorer_checks.py [--mast PATH]

Writes two probe files into the MAST clone's results/raw/sct/ and scores each with its unmodified score.py:
  degraded  20 of the run's 29 lowest-scoring items turned into the error record run.py writes after failed retries
            ({"response": null, "error": ...}), and the other 9 deleted;
  intid     every reply kept, with the documented minimum fields only and integer ids ("id": 1).
Also re-scores the shipped run itself and checks it against examples/ byte for byte.
Writes results/scorer_checks.json: sct_score, items scored and exit code for each, from the scorer's own output.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(sct: Path, name: str) -> dict[str, object]:
    cfg = sct.parents[1] / "results" / "configs" / f"{name}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"model:\n  name: {name}\n")
    args = ["--model-config", str(cfg), "--benchmark-config", "config/benchmark.yaml"]
    run = subprocess.run([sys.executable, "score.py", *args], check=False, cwd=sct, capture_output=True, text=True)
    out = sct.parents[1] / "results" / "scores" / "sct" / f"{name}.csv"
    row = next(r for r in csv.DictReader(out.open()) if r["category"] == "Overall" and r["metric"] == "sct_score")
    return {"sct_score": float(row["mean"]), "items_scored": int(row["trials"]), "exit_code": run.returncode}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mast", type=Path, default=HERE / "mast", help="a clone of ARISENetwork/mast")
    sct = parser.parse_args().mast.resolve() / "benchmarks" / "sct"
    raw_dir = sct.parents[1] / "results" / "raw" / "sct"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (sct / "examples" / "gpt-5.5.jsonl").open()]
    details = sct / "examples" / "gpt-5.5_details.jsonl"
    credit = {d["id"]: d["normalized_score"] for d in map(json.loads, details.open())}
    hard = [r["id"] for r in rows if credit[r["id"]] < 0.3]

    def write(name: str, records: list[dict]) -> None:
        (raw_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    write("gpt-5.5", rows)
    degraded = []
    for r in rows:
        if r["id"] in hard[:20]:
            degraded.append({**r, "response": None, "error": "RateLimitError: 429"})
        elif r["id"] not in hard[20:]:
            degraded.append(r)
    write("degraded", degraded)
    write("intid", [{"id": int(r["id"]), "trial": 1, "response": r["response"]} for r in rows])

    result = {name: score(sct, name) for name in ("gpt-5.5", "degraded", "intid")}
    scores = sct.parents[1] / "results" / "scores" / "sct"
    result["gpt-5.5"]["byte_identical_to_examples"] = all(
        (scores / f).read_bytes() == (sct / "examples" / f).read_bytes()
        for f in ("gpt-5.5.csv", "gpt-5.5_details.jsonl")
    )
    result["degraded"].update(errored=20, deleted=len(hard) - 20, low_scoring_items=len(hard))
    for f in ("gpt-5.5.csv", "gpt-5.5_details.jsonl"):  # the reference run's scorer output, for stats.py
        (HERE / "results" / "scores" / f).write_bytes((scores / f).read_bytes())
    (HERE / "results" / "scorer_checks.json").write_text(json.dumps(result, indent=1) + "\n")
    for name, v in result.items():
        print(f"{name:<9} {v}")


if __name__ == "__main__":
    main()
