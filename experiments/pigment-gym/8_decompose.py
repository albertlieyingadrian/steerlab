"""
Where did the reward gain come from? Fewer failures, or better paintings?

Run from the repo root (seconds; no model, no browser):
    .venv/bin/python experiments/pigment-gym/8_decompose.py
    .venv/bin/python experiments/pigment-gym/8_decompose.py --rollouts other_run.jsonl

The mean reward of a batch is a weighted average of two groups, like a store's average basket:
    mean = p x R  +  (1 - p) x F
    p = share that passed the gate, R = mean reward of the passes, F = mean reward of the failures
So a change between two blocks of steps splits EXACTLY into three parts:
    delta = (p₂ − p₁)(R₁ − F₁)      more sketches crossed into the passing group   "fewer failures"
          + p₂ (R₂ − R₁)            the passing sketches themselves scored higher   "better paintings"
          + (1 − p₂)(F₂ − F₁)       the failures scored differently                 (tiny: failures are ~0.05)
(₁ and ₂ are the earlier and the later block.) We use the reward without partial credit (training reward minus
shaping) so every step is on one scale.
"""

from analysis_tools import load_rollouts, parse_args, quarters
from brush_tools import PROMPT_EXAMPLE, extract_code
from common import check, outputs_dir
from reward_tools import similarity

args = parse_args("Decompose a GRPO run's reward gain")
rows = load_rollouts(args.rollouts)
example = extract_code(PROMPT_EXAMPLE)
steps = sorted({row["step"] for row in rows})
print(f"{len(rows)} rollouts over {len(steps)} steps from {args.rollouts.name}\n")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarise(block):
    passed = [row for row in block if row["gate"]]
    failed = [row for row in block if not row["gate"]]
    return {
        "n": len(block),
        "p": len(passed) / len(block),
        "R": mean(row["reward_unshaped"] for row in passed) if passed else 0.0,
        "F": mean(row["reward_unshaped"] for row in failed) if failed else 0.0,
        "mean": mean(row["reward_unshaped"] for row in block),
        "clip": mean(row["clip_raw"] for row in passed),
        "aesthetic": mean(row["aesthetic_raw"] for row in passed),
        "similarity": mean(similarity(example, row["code"]) for row in block),
        "distinct": len({row["code"] for row in block}) / len(block),
    }


blocks = [(label, summarise(block)) for label, block in quarters(rows)]
print(f"{'block':12} {'gate pass':>9} {'reward':>7} {'R (pass)':>8} {'F (fail)':>8} {'CLIP|pass':>9} {'aes|pass':>8} "
      f"{'sim':>5} {'distinct':>8}")
for label, s in blocks:
    print(f"{label:12} {s['p']:9.0%} {s['mean']:7.3f} {s['R']:8.3f} {s['F']:8.3f} {s['clip']:9.3f} {s['aesthetic']:8.2f} "
          f"{s['similarity']:5.2f} {s['distinct']:8.0%}")

first, last = blocks[0][1], blocks[-1][1]
fewer_failures = (last["p"] - first["p"]) * (first["R"] - first["F"])
better_paintings = last["p"] * (last["R"] - first["R"])
failures_changed = (1 - last["p"]) * (last["F"] - first["F"])
delta = last["mean"] - first["mean"]
print(f"\nfirst block -> last block: mean reward {first['mean']:.3f} -> {last['mean']:.3f} (delta {delta:+.3f})")
print(f"  fewer failures:     {fewer_failures:+.3f}  ({fewer_failures / delta:.0%} of the gain)")
print(f"  better paintings:   {better_paintings:+.3f}  ({better_paintings / delta:.0%})")
print(f"  failures changed:   {failures_changed:+.3f}  ({failures_changed / delta:.0%})")
print(f"\nwithin the passing sketches: CLIP {first['clip']:.3f} -> {last['clip']:.3f}, "
      f"aesthetic {first['aesthetic']:.2f} -> {last['aesthetic']:.2f}")
print("""
Reading it: the taste numbers among passing sketches moved a little; the pass rate moved a lot. That's the HF
reproduction's finding too ("it learned to stop producing bad paintings, not to paint better ones").
These are TRAINING rollouts (temperature 1.0), so the pass rate is lower than the laptop run (7_train_laptop.py)'s held-out eval at 0.8.""")

# ---- plot: four small panels, one per question ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

per_step = [summarise([row for row in rows if row["step"] == step]) for step in steps]
figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
panels = [
    (axes[0][0], "gate pass rate", [s["p"] for s in per_step], "#2a78d6", (0, 1.05)),
    (axes[0][1], "reward of passing sketches", [s["R"] for s in per_step], "#c8553d", (0, 1.0)),
    (axes[1][0], "CLIP (raw) of passing sketches", [s["clip"] for s in per_step], "#1f8a4c", (0.15, 0.33)),
    (axes[1][1], "similarity to the prompt's example", [s["similarity"] for s in per_step], "#7a4fb3", (0, 1.0)),
]
for axis, title, values, colour, limits in panels:
    axis.plot(steps, values, color=colour, marker="o", markersize=3)
    axis.set(title=title, ylim=limits)
    axis.grid(alpha=0.3)
axes[1][1].axhline(0.85, color="gray", linestyle="--", linewidth=1)
axes[1][1].text(steps[0], 0.87, "copy line (whole-file similarity)", fontsize=8, color="gray")
for axis in axes[1]:
    axis.set_xlabel("GRPO step")
figure.suptitle("Where did the reward come from?", fontsize=12)
figure.tight_layout()
plot_path = outputs_dir() / "decompose.png"
figure.savefig(plot_path, dpi=120)
print(f"\nsaved the work folder {plot_path.name}")

print("\nSelf-checks:")
check("every block has rollouts", all(s["n"] > 0 for _, s in blocks))
check("the three parts add up to the total change (exact identity)",
      abs(fewer_failures + better_paintings + failures_changed - delta) < 1e-9)
check("the pass rate rose from the first block to the last", last["p"] > first["p"])
check("finding: 'fewer failures' explains more of the gain than 'better paintings'", fewer_failures > better_paintings)
