"""Run cases x levels x samples through a backend, score them, and write records and a summary table.

    python -m clinicguard.runner --backend stub --attacks PATH --out results/stub

records.jsonl holds scores only (no conversation text). Full transcripts go to --transcripts, a path kept out of the
public copy, because attack transcripts contain the payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .agent import Conversation
from .backends import make_backend
from .leakcheck import score
from .levels import LEVELS

HERE = Path(__file__).resolve().parents[1]
LEGIT = HERE / "data" / "legit.jsonl"
LEGIT_HASH = HERE / "data" / "legit.sha256"
DEFAULT_LEVELS = ["L0", "L1", "L2", "L3", "L3-kw", "L2+P1", "L3+P1", "L3+P2", "L3+P1+P2"]


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_legit(path=LEGIT, hash_path=LEGIT_HASH, check=True) -> list[dict]:
    """The legitimate-request set, refused if it no longer matches the hash frozen before any run."""
    data = Path(path).read_bytes()
    if check:
        want = Path(hash_path).read_text().split()[0]
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit(f"legit set changed since it was frozen ({got[:12]} != {want[:12]}); refusing to run")
    return load_jsonl(path)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def drive(convs: list[Conversation], backend, max_rounds: int = 60) -> None:
    for _ in range(max_rounds):
        todo = [c for c in convs if c.needs_model]
        if not todo:
            return
        for c, out in zip(todo, backend.generate(todo)):
            c.feed(out)
    for c in convs:
        if c.needs_model:
            c.done = True


def run_eval(
    cases: list[dict], levels: list[str], backend, samples: int = 1, seed: int = 0, model_name: str | None = None
) -> tuple[list[dict], list[dict]]:
    convs = [Conversation(case, lv, seed=seed, sample=s) for case in cases for lv in levels for s in range(samples)]
    drive(convs, backend)
    name = model_name or getattr(backend, "name", "model")
    records, transcripts = [], []
    for c in convs:
        r = score(c)
        r["model"] = name
        records.append(r)
        transcripts.append(
            {
                "case_id": c.case["id"],
                "level": c.level,
                "sample": c.sample,
                "model": name,
                "events": c.ctx.events,
                "transcript": c.transcript,
            }
        )
    return records, transcripts


def summarise(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["level"], r["kind"], r["family"])].append(r)
    rows = []
    for (model, level, kind, family), rs in sorted(groups.items()):
        n = len(rs)
        row = {"model": model, "level": level, "kind": kind, "family": family, "n": n}
        if kind == "attack":
            k = sum(r["leaked"] for r in rs)
            lo, hi = wilson(k, n)
            row.update(
                leaks=k,
                leak_rate=round(k / n, 3),
                lo=round(lo, 3),
                hi=round(hi, 3),
                guard_fired=sum(r["guard_fired"] for r in rs),
            )
        else:
            fp = sum(r["false_positive"] for r in rs)
            lo, hi = wilson(fp, n)
            row.update(
                fp=fp,
                fp_rate=round(fp / n, 3),
                lo=round(lo, 3),
                hi=round(hi, 3),
                fp_hard=sum(r["fp_hard"] for r in rs),
                answered=sum(r["answered"] for r in rs),
            )
        rows.append(row)
    return rows


def write_outputs(records, transcripts, out: Path, transcripts_path: Path | None):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    rows = summarise(records)
    keys = [
        "model",
        "level",
        "kind",
        "family",
        "n",
        "leaks",
        "leak_rate",
        "fp",
        "fp_rate",
        "lo",
        "hi",
        "guard_fired",
        "fp_hard",
        "answered",
    ]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if transcripts_path:
        transcripts_path.parent.mkdir(parents=True, exist_ok=True)
        with transcripts_path.open("w") as f:
            for t in transcripts:
                f.write(json.dumps(t) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="stub")
    ap.add_argument("--attacks", help="attack cases (JSONL); kept outside the public copy")
    ap.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "results" / "stub"))
    ap.add_argument("--transcripts")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    a = ap.parse_args()
    cases = load_legit() + (load_jsonl(a.attacks) if a.attacks else [])
    levels = a.levels.split(",")
    assert all(lv in LEVELS for lv in levels), levels
    kw = {"base_url": a.base_url, "model": a.model} if a.backend == "openai" else {}
    backend = make_backend(a.backend, **kw)
    records, transcripts = run_eval(cases, levels, backend, a.samples, model_name=a.model or a.backend)
    rows = write_outputs(records, transcripts, Path(a.out), Path(a.transcripts) if a.transcripts else None)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
