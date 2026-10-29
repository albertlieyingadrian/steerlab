"""
steer Qwen3-8B along the eval-awareness direction while it answers (about an hour).

    .venv/bin/python experiments/natural-sandbagging/steer.py

Step 1 found an arrow in the residual stream that points from "pub quiz" to "exam". Here it is ADDED while the
model reads the question and while it reasons, at the chosen layer, at every token. If believing it is being tested
makes the model hide dangerous knowledge, pushing it along +arrow should lower WMDP accuracy more than MMLU accuracy,
and pushing along -arrow (towards "pub quiz") should do the opposite.

Nine arms on 150 WMDP + 150 MMLU test items, no framing, seed 0 (plan):

    dir-8  dir-4  dir+0  dir+4  dir+8      strength in class gaps along the eval-awareness direction (+ = more "eval")
    rand1-8 rand1+8 rand2-8 rand2+8        two random directions of exactly the same length (the control)

The random arms answer "does ANY push this big change the answers?". The direction counts only if it beats them.
Rows go to the work folder (steer.jsonl), append-only: rerunning resumes.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import natural as t

LESSON = Path(__file__).resolve().parent
STRENGTHS = (-8, -4, 0, 4, 8)
RANDOM_SEEDS = (7, 8)
RANDOM_STRENGTH = 8


def arms():
    info = json.loads((LESSON / "results" / "direction.json").read_text())
    L = info["chosen_layer"]
    z = np.load(LESSON / "results" / "direction.npz")
    u, gap = z[f"u_L{L}"], float(z[f"gap_L{L}"])
    out = [(f"dir{s:+d}", s * gap * u) for s in STRENGTHS]
    for k, seed in enumerate(RANDOM_SEEDS, 1):
        r = np.random.default_rng(seed).standard_normal(u.shape[0]).astype(np.float32)
        r /= np.linalg.norm(r)
        for s in (-RANDOM_STRENGTH, RANDOM_STRENGTH):
            out.append((f"rand{k}{s:+d}", s * gap * r))
    return L, gap, out


def main():
    L, gap, arm_list = arms()
    items = t.subset(t.STEER_SUBSET, seed=19)
    out = t.WORK / "stage1" / "steer.jsonl"
    print(f"layer {L}, class gap {gap:.2f}; {len(arm_list)} arms x {len(items)} items = {len(arm_list) * len(items)}")
    tok, model = t.load_model()
    t0 = time.time()
    for name, vec in arm_list:
        jobs = t.jobs_for(items, ("neutral",), (0,), arm=name)
        print(f"arm {name}: |vector| = {np.linalg.norm(vec):.2f}")
        t.generate(tok, model, jobs, out, steer=(L, vec if np.any(vec) else None))
    rows = t.read_jsonl(out)
    print(f"\n{len(rows)} steered rows on disk, {(time.time() - t0) / 60:.1f} min this session")
    checks = [t.check("every arm has every item", len(rows) == len(arm_list) * len(items))]
    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
