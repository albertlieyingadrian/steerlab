"""
Render and score every saved sample, and report a baseline per model.

Run from the repo root (after 1_sample_small.py):
    .venv/bin/python experiments/pigment-gym/3_benchmark.py

The "score it and write the report" half of the eval harness. For each model:
  code block   did the reply contain a ```javascript block at all?
  static ok    no text/images/network/bare shapes/unknown brush methods (read, not run)
  gate pass    ran without errors, finished, and painted >= 0.5% of the canvas
  coverage     mean share of painted pixels, over the sketches that passed
  CLIP         mean raw similarity to "a watercolour painting of a sunset over green hills", over passes
  reward       mean reward (reward_tools.py) over ALL samples (failures count as ~0.05)
This is the number RL has to beat. The laptop run (7_train_laptop.py) reads results/benchmark.json for the comparison.
"""

import json
import sys
from collections import Counter

from brush_tools import FENCE, Renderer, contact_sheet, extract_code, static_check
from common import RESULTS, check, outputs_dir
from hand_pool import SUBJECT
from reward_tools import Scorer, score_rollouts

SAMPLES_PATH = RESULTS / "benchmark_samples.jsonl"
if not SAMPLES_PATH.exists():
    sys.exit(f"{SAMPLES_PATH} not found: run 1_sample_small.py first")
records = [json.loads(line) for line in SAMPLES_PATH.read_text().splitlines()]
for record in records:
    record.setdefault("prompt", "no_example")  # samples saved before the prompt comparison existed
cells = sorted({(record["model"], record["prompt"]) for record in records}, key=lambda c: (c[0], c[1] != "no_example"))
print(f"{len(records)} samples in {len(cells)} cells (model x system prompt)\n")

scorer = Scorer()
summary = {}
with Renderer(pool_size=4) as renderer:
    for model, prompt_name in cells:
        cell = f"{model} {prompt_name}"
        mine = [record for record in records if record["model"] == model and record["prompt"] == prompt_name]
        codes = [extract_code(record["text"]) for record in mine]
        renders = renderer.render_many(codes)
        rows = score_rollouts(codes, renders, scorer, SUBJECT)
        passed = [row for row in rows if row["gate"]]
        summary[cell] = {
            "samples": len(mine),
            "code_block": sum(bool(FENCE.search(record["text"])) for record in mine) / len(mine),
            "static_ok": sum(not static_check(code) for code in codes) / len(mine),
            "gate_pass": len(passed) / len(mine),
            "coverage": sum(row["coverage"] for row in passed) / len(passed) if passed else 0.0,
            "clip_raw": sum(row["clip_raw"] for row in passed) / len(passed) if passed else None,
            "reward": sum(row["reward"] for row in rows) / len(rows),
            "violations": dict(Counter(v.split(":")[0] for row in rows for v in row["violations"]).most_common()),
            "mean_tokens": sum(record["tokens"] for record in mine) / len(mine),
        }
        order = sorted(range(len(rows)), key=lambda i: -rows[i]["reward"])[:8]
        labels = [f"{cell} #{mine[i]['index']} r={rows[i]['reward']:.2f}" for i in order]
        contact_sheet([renders[i].png for i in order], labels, outputs_dir() / f"best_{model}_{prompt_name}.png")

header = f"{'model / prompt':17} {'n':>3} {'code block':>10} {'static ok':>9} {'gate pass':>9} {'coverage':>8} {'CLIP':>6} {'reward':>6} {'tokens':>6}"
print(header)
for cell, s in summary.items():
    clip = f"{s['clip_raw']:.3f}" if s["clip_raw"] is not None else "-"
    print(f"{cell:17} {s['samples']:3d} {s['code_block']:10.0%} {s['static_ok']:9.0%} {s['gate_pass']:9.0%} "
          f"{s['coverage']:8.1%} {clip:>6} {s['reward']:6.3f} {s['mean_tokens']:6.0f}")

print("\nwhy sketches failed (counts over all samples; one sketch can have several reasons):")
for cell, s in summary.items():
    print(f"  {cell}: {s['violations'] or 'none'}")

print("""
How to read it: one worked example in the system prompt is the biggest lever in this table. The upstream
recipe found "allowlist, no examples" works best, but that was measured on far bigger models. Small models
need to see the pattern once. That is prompt work BEFORE RL, like Surya's GEPA step.

The reward is mostly decided by the gate. A failed sketch scores ~0.05 and a passing one
~0.3-0.9, so the gate pass rate is the first thing RL will push on. That matches the HF run's honest
result: most of the gain was 'stop producing bad paintings', not 'paint better ones'.""")

summary_path = RESULTS / "benchmark.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"\nsaved {summary_path.name} and best_<model>_<prompt>.png (the 8 highest-reward renders per cell)")

print("\nSelf-checks:")
check("every sample was rendered and scored", sum(s["samples"] for s in summary.values()) == len(records))
check("all rates are between 0 and 1",
      all(0 <= s[key] <= 1 for s in summary.values() for key in ["code_block", "static_ok", "gate_pass", "coverage"]))
check("mean rewards are between 0 and 1", all(0 <= s["reward"] <= 1 for s in summary.values()))
check("the baseline summary was saved for the laptop run (7_train_laptop.py)", summary_path.exists())
if "0.5B example" in summary and "0.5B no_example" in summary:
    check("the worked example raises 0.5B's gate pass rate", summary["0.5B example"]["gate_pass"] > summary["0.5B no_example"]["gate_pass"])
