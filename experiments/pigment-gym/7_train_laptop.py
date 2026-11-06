"""
GRPO on Qwen2.5-Coder-0.5B-Instruct with the real renderer in the loop.

SLOW: about 20 minutes on an M-series Mac with nothing else using the GPU.
Run from the repo root:
    .venv/bin/python experiments/pigment-gym/7_train_laptop.py
    .venv/bin/python experiments/pigment-gym/7_train_laptop.py --steps 5   # smoke test

Every step:
    1. the model writes 8 sketches for "Paint a sunset over green hills."     (sampling, ~1 min)
    2. Chromium renders all 8                                                  (~3 s)
    3. the gate + CLIP + aesthetic score them (the reward check (4_rank_check.py), plus partial credit) (<1 s)
    4. GRPO nudges the LoRA weights towards the above-average sketches        (~15 s)

Before and after training we sample 16 sketches with the same seeds and compare them, and we split the
change into "fails less often" vs "paints better when it passes", like the HF write-up does.

The system prompt contains one worked example (the benchmark (scripts 1-3): it takes Coder-0.5B from 0% to 44% gate
passes).
An earlier run without a copy check learned to paste that example verbatim within 10 steps (recorded in
results/logs/7_train_laptop_no_copy_check.txt), so any sketch >= 85% similar to it now fails, in training
AND in the held-out evaluation.
"""

import csv
import json
import sys
import time

import torch
from brush_tools import (
    PROMPT_EXAMPLE,
    SYSTEM_PROMPT,
    Renderer,
    contact_sheet,
    extract_code,
)
from common import check, outputs_dir, pick_device
from grpo_tools import (
    build_lora_model,
    chat_prompt,
    group_advantages,
    grpo_update,
    mean,
    sample_group,
)
from hand_pool import SUBJECT
from reward_tools import Scorer, score_rollouts, similarity

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 40
GROUP_SIZE = 8
MAX_NEW_TOKENS = 640  # the benchmark (scripts 1-3)'s answers averaged ~430 tokens; the cap keeps a step near a minute
LEARNING_RATE = 2e-4
BETA = 0.04
EVAL_SAMPLES = 16
SEED = 0
OUT = outputs_dir()
EXAMPLE_CODE = extract_code(PROMPT_EXAMPLE)

device = pick_device()
model, tokenizer = build_lora_model(MODEL, device, seed=SEED)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
prompt = chat_prompt(
    tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Paint {SUBJECT}."}]
)
scorer = Scorer()
print(f"policy {MODEL} + LoRA on {device}; {STEPS} steps x {GROUP_SIZE} sketches; lr {LEARNING_RATE}, beta {BETA}\n")


def evaluate(renderer, label):
    """16 sketches at temperature 0.8 with fixed seeds, scored with the reward without partial credit."""
    texts = []
    for batch in range(EVAL_SAMPLES // 8):
        batch_texts, _, _ = sample_group(
            model, tokenizer, prompt, 8, MAX_NEW_TOKENS, temperature=0.8, top_p=0.95, seed=10_000 + batch
        )
        texts += batch_texts
    codes = [extract_code(text) for text in texts]
    renders = renderer.render_many(codes)
    rows = score_rollouts(codes, renders, scorer, SUBJECT, forbid_copy_of=EXAMPLE_CODE)
    passed = [row for row in rows if row["gate"]]
    result = {
        "gate_pass": len(passed) / len(rows),
        "reward": mean(row["reward"] for row in rows),
        "reward_when_passing": mean(row["reward"] for row in passed) if passed else 0.0,
        "clip_when_passing": mean(row["clip_raw"] for row in passed) if passed else None,
        "similarity_to_example": mean(similarity(EXAMPLE_CODE, code) for code in codes),
        "distinct_sketches": len(set(codes)),
    }
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["reward"])[:8]
    contact_sheet(
        [renders[i].png for i in order], [f"{label} r={rows[i]['reward']:.2f}" for i in order], OUT / f"{label}.png"
    )
    print(
        f"[{label}] gate pass {result['gate_pass']:.0%}, mean reward {result['reward']:.3f}, "
        f"reward when passing {result['reward_when_passing']:.3f}, similarity to the example "
        f"{result['similarity_to_example']:.2f}, {result['distinct_sketches']}/{len(codes)} distinct"
    )
    return result, codes, rows


history = []
rollouts_path = OUT / "rollouts.jsonl"
rollouts_path.write_text("")
run_started = time.perf_counter()
with Renderer(pool_size=4) as renderer:
    before, _, _ = evaluate(renderer, "before")
    print(
        f"\n{'step':>4} {'reward':>6} {'gate':>5} {'shaping':>7} {'zero-std':>8} {'sim':>5} {'KL':>6} {'tokens':>6} {'secs':>5}"  # noqa: E501
    )
    for step in range(STEPS):
        step_started = time.perf_counter()
        texts, input_ids, mask = sample_group(
            model, tokenizer, prompt, GROUP_SIZE, MAX_NEW_TOKENS, temperature=1.0, top_p=0.95, seed=SEED * 1000 + step
        )
        codes = [extract_code(text) for text in texts]
        renders = renderer.render_many(codes)
        rows = score_rollouts(codes, renders, scorer, SUBJECT, shaped=True, forbid_copy_of=EXAMPLE_CODE)
        rewards = [row["reward"] for row in rows]
        advantages = group_advantages(rewards, scale_rewards="none")
        metrics = grpo_update(model, optimizer, input_ids, mask, advantages, beta=BETA, micro_batch=2)
        row = {
            "step": step,
            "reward": mean(rewards),
            "reward_max": max(rewards),
            "gate_pass": mean(r["gate"] for r in rows),
            "shaping": mean(r["shaping"] for r in rows),
            "frac_reward_zero_std": float(max(rewards) - min(rewards) < 1e-6),
            "similarity": mean(similarity(EXAMPLE_CODE, code) for code in codes),
            "kl": metrics["kl"],
            "tokens": float(mask.sum()) / GROUP_SIZE,
            "seconds": time.perf_counter() - step_started,
        }
        history.append(row)
        with rollouts_path.open("a") as handle:
            for code, scored in zip(codes, rows, strict=False):
                handle.write(json.dumps({"step": step, "code": code, **scored}) + "\n")
        print(
            f"{step:4d} {row['reward']:6.3f} {row['gate_pass']:5.0%} {row['shaping']:7.3f} {row['frac_reward_zero_std']:8.0f} "  # noqa: E501
            f"{row['similarity']:5.2f} {row['kl']:6.3f} {row['tokens']:6.0f} {row['seconds']:5.0f}"
        )
    print()
    after, after_codes, after_rows = evaluate(renderer, "after")

minutes = (time.perf_counter() - run_started) / 60
model.save_pretrained(OUT / "lora")
with (OUT / "curve.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

figure, axis = plt.subplots(figsize=(8, 3.8))
steps = [r["step"] for r in history]
axis.plot(
    steps, [r["reward"] for r in history], color="#c8553d", marker="o", markersize=3, label="mean reward (shaped)"
)
axis.plot(steps, [r["reward_max"] for r in history], color="#c8553d", alpha=0.35, linestyle="--", label="best in group")
axis.plot(steps, [r["gate_pass"] for r in history], color="#2a78d6", marker="s", markersize=3, label="gate pass rate")
axis.set(xlabel="GRPO step", ylim=(0, 1), title=f"Coder-0.5B painting GRPO, {GROUP_SIZE} sketches per step")
axis.grid(alpha=0.3)
axis.legend(loc="upper left", fontsize=8)
figure.tight_layout()
figure.savefig(OUT / "curve.png", dpi=120)

first = mean(r["reward"] for r in history[:5])
last = mean(r["reward"] for r in history[-5:])
print(f"\n{STEPS} steps + 2 evaluations in {minutes:.1f} min")
print(f"training reward (shaped), first 5 steps {first:.3f} -> last 5 steps {last:.3f}")
print(
    f"held-out eval (same seeds, unshaped reward): gate pass {before['gate_pass']:.0%} -> {after['gate_pass']:.0%}, "
    f"mean reward {before['reward']:.3f} -> {after['reward']:.3f}"
)

# "Fewer failures" vs "better paintings": how much of the change in mean reward comes from the pass rate,
# and how much from the reward of the sketches that pass? (A two-term split; the cross term goes to the second.)
if before["gate_pass"]:
    from_pass_rate = (after["gate_pass"] - before["gate_pass"]) * (before["reward_when_passing"] - 0.05)
    from_quality = after["gate_pass"] * (after["reward_when_passing"] - before["reward_when_passing"])
else:  # nothing passed before: no "before quality" exists, so the whole gain is from passing more often
    from_pass_rate = after["gate_pass"] * (after["reward_when_passing"] - 0.05)
    from_quality = 0.0
print(
    f"  of that, from passing more often: {from_pass_rate:+.3f}; from passing sketches scoring higher: {from_quality:+.3f}"  # noqa: E501
)
if not before["gate_pass"]:
    print("  (before training nothing passed, so there is no 'quality when passing' to compare against)")

best = max(range(len(after_rows)), key=lambda i: after_rows[i]["reward"])
(OUT / "best_after.js").write_text(after_codes[best])
print("\nsaved curve.csv/.png, before.png, after.png, rollouts.jsonl, lora/,")
print(f"and the best post-training sketch as best_after.js (reward {after_rows[best]['reward']:.3f})")

print("\nSelf-checks:")
check(f"all {STEPS} steps ran", len(history) == STEPS)
check("every reward and KL is finite", all(r["reward"] == r["reward"] and r["kl"] == r["kl"] for r in history))
check("training reward rose (last-5 mean > first-5 mean)", last > first)
check("held-out mean reward did not get worse", after["reward"] >= before["reward"])
check("the LoRA adapter was saved", (OUT / "lora" / "adapter_config.json").exists())
check("the trained policy is not pasting the example (mean similarity < 0.85)", after["similarity_to_example"] < 0.85)
