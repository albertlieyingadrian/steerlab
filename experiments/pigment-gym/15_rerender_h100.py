"""
Was the H100 run's low pass rate the model or the renderer? Re-render its sketches on the laptop and re-apply the gate.

Run from the repo root (laptop only, ~10 minutes):
    .venv/bin/python experiments/pigment-gym/15_rerender_h100.py

Found while running the follow-up study: on Modal's CPUs, Chromium's software WebGL timed out on sketches that render in
under a second on the laptop, including a hand-made reference sketch. The published H100 run rendered on Modal, and
245 of its 800 training rollouts failed as "timeout_or_crash" (the laptop run: 2 of 320). This script re-renders all
800 training sketches and the 32 held-out ones (before and after) with the laptop renderer, re-applies the same gate
and copy check, and writes results/next/h100_rerender.json.
"""

import json
import os
from pathlib import Path

from brush_tools import PROMPT_EXAMPLE, Renderer, extract_code, gate
from common import RESULTS, check, outputs_dir
from reward_tools import MAX_SIMILARITY_TO_EXAMPLE, similarity

EXAMPLE = extract_code(PROMPT_EXAMPLE)
# The H100 run's held-out samples (modal_train.py saves them to the work folder's h100_full/; not in git).
HELD_OUT_DIR = Path(os.environ.get("PIGMENT_H100_DIR", outputs_dir() / "h100_full"))


def passes(codes, renderer):
    renders = renderer.render_many(codes)
    out = []
    for code, render in zip(codes, renders, strict=True):
        verdict = gate(code, render)
        copied = similarity(EXAMPLE, code) >= MAX_SIMILARITY_TO_EXAMPLE
        out.append({"passed": verdict.passed and not copied, "violations": verdict.violations})
    return out


def rate(values):
    return sum(values) / len(values)


rows = [json.loads(line) for line in (RESULTS / "rollouts_h100.jsonl").read_text().splitlines()]
with Renderer(pool_size=8) as renderer:
    local = passes([r["code"] for r in rows], renderer)
    held = {}
    for label in ("before", "after"):
        path = HELD_OUT_DIR / f"{label}_samples.jsonl"
        if path.exists():
            samples = [json.loads(line) for line in path.read_text().splitlines()]
            held[label] = {
                "modal_pass": sum(bool(s["gate"]) for s in samples),
                "laptop_pass": sum(p["passed"] for p in passes([s["code"] for s in samples], renderer)),
                "n": len(samples),
            }

steps = max(r["step"] for r in rows) + 1
quarter = steps // 4
early = [i for i, r in enumerate(rows) if r["step"] < quarter]
late = [i for i, r in enumerate(rows) if r["step"] >= steps - quarter]
summary = {
    "rollouts": len(rows),
    "modal": {
        "pass": sum(bool(r["gate"]) for r in rows),
        "timeouts": sum("timeout_or_crash" in r["violations"] for r in rows),
        "first_quarter_pass_rate": rate([bool(rows[i]["gate"]) for i in early]),
        "last_quarter_pass_rate": rate([bool(rows[i]["gate"]) for i in late]),
    },
    "laptop": {
        "pass": sum(p["passed"] for p in local),
        "timeouts": sum("timeout_or_crash" in p["violations"] for p in local),
        "first_quarter_pass_rate": rate([local[i]["passed"] for i in early]),
        "last_quarter_pass_rate": rate([local[i]["passed"] for i in late]),
    },
    "flipped_fail_to_pass": sum(p["passed"] and not r["gate"] for r, p in zip(rows, local, strict=True)),
    "flipped_pass_to_fail": sum(bool(r["gate"]) and not p["passed"] for r, p in zip(rows, local, strict=True)),
    "held_out": held,
}
(RESULTS / "next").mkdir(exist_ok=True)
(RESULTS / "next" / "h100_rerender.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))

print("\nSelf-checks:")
check("every rollout re-rendered", len(local) == len(rows))
check(
    "the laptop renderer times out far less than Modal's",
    summary["laptop"]["timeouts"] < summary["modal"]["timeouts"] / 4,
)
