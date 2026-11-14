"""
Join the two halves of a run that Modal stopped mid-way (amendment 1 in PREREGISTRATION.md).

    .venv/bin/python experiments/probe-gated-rl/merge_resumed.py

baseline_seed0 and probe_penalty_seed0 were killed at 12:28 UTC on 2026-11-14 when the Modal workspace was disabled
(at steps 94 and 74). Each was resumed from its last checkpoint (step80, step60) with
`modal_gated.py --resume-from stepN`, which reloads the LoRA weights and continues with the same per-step sampling
seeds; AdamW's moments were not saved, so the optimizer restarted at the resume step.

    part A  (~/.cache/.../runs/<run>_partA/): steps 0..N-1 kept (the sketches drawn after the checkpoint are dropped),
            before_t1, and the log (for the per-step KL)
    part B  (~/.cache/.../runs/<run>_partB/): steps N..99, after_t1, curve rows and summary

Writes results/runs/<run>_* in the same format as an uninterrupted run, with "resumed_from" in the summary.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path.home() / ".cache" / "steerlab" / "probe-gated-rl" / "runs"
OUT = HERE / "results" / "runs"
RESUMED = {"baseline_seed0": 80, "probe_penalty_seed0": 60}


def read(path):
    return [json.loads(line) for line in path.open()]


def curve_row(step, rows, kl):
    """The per-step statistics modal_gated.py logs, recomputed from the step's 8 sketches."""
    passed = [r for r in rows if r["gate"]]
    hacks = [r for r in rows if r["hack"]]
    rewards = [r["reward"] for r in rows]
    return {
        "step": step,
        "reward": np.mean(rewards),
        "task_reward": np.mean([r["task_reward"] for r in rows]),
        "gate_pass": np.mean([r["gate"] for r in rows]),
        "clip_when_passing": np.mean([r["clip_raw"] for r in passed]) if passed else float("nan"),
        "aesthetic_when_passing": np.mean([r["aesthetic_raw"] for r in passed]) if passed else float("nan"),
        "mentions": np.mean([r["mentions"] for r in rows]),
        "hack_rate": np.mean([r["hack"] for r in rows]),
        "probe_mean": np.mean([r["probe_p"] for r in rows]),
        "flag_rate": np.mean([r["flagged"] for r in rows]),
        "flag_rate_among_hacks": np.mean([r["flagged"] for r in hacks]) if hacks else float("nan"),
        "loophole_bonus": np.mean([r["loophole_bonus"] for r in rows]),
        "penalty": np.mean([r["penalty"] for r in rows]),
        "zero_std": float(max(rewards) - min(rewards) < 1e-6),
        "tokens": np.mean([r["tokens"] for r in rows]),
        "kl": kl,
        "kept": sum(r["kept"] for r in rows),
        "skipped": 0,
    }


def main():
    for run, start in RESUMED.items():
        part_a, part_b = WORK / f"{run}_partA", WORK / f"{run}_partB"
        kls = {
            int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"^step\s+(\d+) .*? kl (\S+)", (part_a / f"{run}.log").read_text(), re.MULTILINE)
        }
        early = [r for r in read(part_a / "rollouts.jsonl") if r["step"] < start]
        late = read(part_b / "rollouts.jsonl")
        assert {r["step"] for r in early} == set(range(start)) and min(r["step"] for r in late) == start, run
        with (part_b / "curve.csv").open() as handle:
            reader = csv.DictReader(handle)
            fields, rows_b = reader.fieldnames, list(reader)
        rows_a = []
        for step in range(start):
            row = curve_row(step, [r for r in early if r["step"] == step], kls[step])
            rows_a.append({k: row.get(k, "") for k in fields})
        with (OUT / f"{run}_curve.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_a + rows_b)
        with (OUT / f"{run}_rollouts.jsonl").open("w") as handle:
            for record in early + late:
                handle.write(json.dumps(record) + "\n")
        (OUT / f"{run}_before_t1_samples.jsonl").write_text((part_a / "before_t1_samples.jsonl").read_text())
        (OUT / f"{run}_after_t1_samples.jsonl").write_text((part_b / "after_t1_samples.jsonl").read_text())
        summary = json.loads((part_b / "summary.json").read_text())
        summary.update(
            {
                "run": run,
                "steps_run": start + summary["steps_run"],
                "resumed_from": f"step{start}",
                "resume_note": "Modal stopped the run; resumed from this checkpoint with fresh AdamW moments",
                "train_first10": None,
            }
        )
        (OUT / f"{run}_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"{run}: steps 0-{start - 1} from part A ({len(early)} sketches), {start}-{start + len(rows_b) - 1} "
            f"from part B ({len(late)} sketches)"
        )


if __name__ == "__main__":
    main()
