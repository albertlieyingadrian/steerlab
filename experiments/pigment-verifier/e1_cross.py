"""
Cross-score E1: every run's 16 held-out "after" sketches scored by both rewards, so no run is judged only by its own.

    .venv/bin/python experiments/pigment-verifier/e1_cross.py

A sketch that fails the gate scores 0 on both. CLIP + aesthetic is Pigment Gym's quality term (0-1); the chosen reward
(run 030) is its percentile among the dev sketches (0-1). Writes results/e1_cross.json.
"""

import bisect
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "experiments" / "pigment-gym")]

import prepare


def load(run):
    spec = importlib.util.spec_from_file_location(f"r{run}", HERE / "runs" / f"{run}_reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clip, chosen = load("000"), load("030")
dev = sorted(chosen.score(prepare.evaluation_ids("dev")).values())
out = {}
for name in ("clip", "judge", "verifier"):
    rows = [json.loads(line) for line in (HERE / "results" / name / "after_sketches.jsonl").read_text().splitlines()]
    ids = [f"train-{name}-after-{i}" for i, row in enumerate(rows) if row["gate"]]
    clip_q = {k: v / 0.85 for k, v in clip.score(ids).items()} if ids else {}
    chosen_q = {k: bisect.bisect_left(dev, v) / len(dev) for k, v in chosen.score(ids).items()} if ids else {}
    out[name] = {
        "passing": f"{len(ids)}/{len(rows)}",
        "clip_aesthetic_mean": sum(clip_q.values()) / len(rows),
        "chosen_reward_mean": sum(chosen_q.values()) / len(rows),
        "clip_aesthetic_when_passing": sum(clip_q.values()) / len(ids) if ids else None,
        "chosen_reward_when_passing": sum(chosen_q.values()) / len(ids) if ids else None,
    }
    print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items()}))
(HERE / "results" / "e1_cross.json").write_text(json.dumps(out, indent=1))
