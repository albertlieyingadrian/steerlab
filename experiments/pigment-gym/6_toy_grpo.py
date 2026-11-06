"""
A toy GRPO run with no browser. Does the optimiser move at all?

Run from the repo root (5-15 minutes on an M-series Mac, depending on what else is using the GPU):
    .venv/bin/python experiments/pigment-gym/6_toy_grpo.py
    .venv/bin/python experiments/pigment-gym/6_toy_grpo.py --naive   # the hackable reward

The HF painting team did exactly this before their real run: a cheap "sanity task" with a reward you
can compute instantly, to prove the training loop learns before paying for renders. Ours:

    reward = max(0, 1 - |words - 5| / 45)      "answer in about five words"

Qwen2.5-0.5B-Instruct usually answers in 30-40 words here (it hits the 48-token cap), so it starts
around 0.2-0.3. A reward of 1.0 means exactly five words. Nobody tells the model the rule; it has to discover it from the rewards.

This reward took three tries, and the first two were hacked. Both are recorded:
  1. words = text.split(). In ~10 steps the model switched to CHINESE, which has no spaces, so a long
     paragraph counted as two "words" (results/logs/6_toy_grpo_naive.txt, rerun with --naive).
  2. English words only. After 25 slow steps it found "I'm sorry, but I can't assist with that.":
     8 fluent words. By step 29 every answer was that refusal (results/logs/6_toy_grpo_refusal_hack.txt).
  3. (this script) English words only, AND the answer must mention the question's topic word, AND a
     steeper linear curve so 40-word and 30-word answers get clearly different rewards.
In both hacked runs, every self-check passed. Metrics went up; the behaviour we wanted didn't happen.
That's reward hacking, live, on a toy: read the samples, not just the curve.
"""

import csv
import re
import sys
import time

import torch
from common import check, outputs_dir, pick_device
from grpo_tools import (
    build_lora_model,
    chat_prompt,
    group_advantages,
    grpo_update,
    mean,
    sample_group,
    trainable_parameters,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS = 30
PROMPTS_PER_STEP = 2
GROUP_SIZE = 8  # G: answers sampled per prompt
MAX_NEW_TOKENS = 48
LEARNING_RATE = 2e-4  # high on purpose for a toy: the HF team's sanity task only moved at 3e-4
BETA = 0.02  # KL weight: a leash back to the original model
SEED = 0

# (question, topic word the answer must contain)
QUESTIONS = [
    ("Describe the ocean.", "ocean"),
    ("What is your favourite season and why?", "season"),
    ("Tell me about cats.", "cat"),
    ("How do airplanes fly?", "plane"),
    ("What makes a good breakfast?", "breakfast"),
    ("Describe a city at night.", "city"),
    ("Why do people like music?", "music"),
    ("What is a volcano?", "volcano"),
]


NAIVE = "--naive" in sys.argv


def count_words(text):
    if NAIVE:
        return len(text.split())  # the loophole: "words" = runs of characters between spaces
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def reward(text, topic):
    if NAIVE:
        return 1 / (1 + abs(count_words(text) - 5) / 5)  # the first version, exactly as recorded
    if topic not in text.lower():
        return 0.0  # closes loophole 2: a refusal never mentions the topic
    non_ascii = sum(1 for character in text if ord(character) > 127)
    if non_ascii > 0.1 * max(1, len(text)):
        return 0.0  # closes the loophole: an answer that isn't mostly English earns nothing
    return max(0.0, 1 - abs(count_words(text) - 5) / 45)


device = pick_device()
torch.manual_seed(SEED)
model, tokenizer = build_lora_model(MODEL, device, seed=SEED)
trainable, total = trainable_parameters(model)
print(f"reward: {'NAIVE (words = text.split())' if NAIVE else 'English words, must mention the topic, linear'}")
print(f"{MODEL} on {device}: {total / 1e6:.0f}M parameters, {trainable / 1e6:.1f}M trainable (LoRA, {trainable / total:.1%})")
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

history = []
started = time.perf_counter()
print(f"\n{'step':>4} {'reward':>7} {'words':>6} {'zero-std':>8} {'KL':>7}  example answer")
for step in range(STEPS):
    step_rewards, step_words, zero_std, kls = [], [], 0, []
    example = ""
    for prompt_index in range(PROMPTS_PER_STEP):
        question, topic = QUESTIONS[(step * PROMPTS_PER_STEP + prompt_index) % len(QUESTIONS)]
        prompt = chat_prompt(tokenizer, [{"role": "user", "content": question}])
        texts, input_ids, mask = sample_group(
            model, tokenizer, prompt, GROUP_SIZE, MAX_NEW_TOKENS, temperature=1.0, seed=SEED * 1000 + step * 10 + prompt_index
        )
        rewards = [reward(text, topic) for text in texts]
        advantages = group_advantages(rewards, scale_rewards="none")
        if max(rewards) - min(rewards) < 1e-6:
            zero_std += 1  # every answer scored the same: advantages are all 0, this group teaches nothing
        metrics = grpo_update(model, optimizer, input_ids, mask, advantages, beta=BETA, micro_batch=GROUP_SIZE)
        step_rewards += rewards
        step_words += [count_words(text) for text in texts]
        kls.append(metrics["kl"])
        example = texts[0].replace("\n", " ")
    row = {
        "step": step,
        "reward": mean(step_rewards),
        "words": mean(step_words),
        "frac_reward_zero_std": zero_std / PROMPTS_PER_STEP,
        "kl": mean(kls),
    }
    history.append(row)
    print(f"{step:4d} {row['reward']:7.3f} {row['words']:6.1f} {row['frac_reward_zero_std']:8.2f} {row['kl']:7.4f}  {example[:60]!r}")

elapsed = time.perf_counter() - started
first = mean(row["reward"] for row in history[:5])
last = mean(row["reward"] for row in history[-5:])
print(f"\n{STEPS} steps in {elapsed / 60:.1f} min ({elapsed / STEPS:.1f} s/step)")
print(f"mean reward, first 5 steps: {first:.3f}   last 5 steps: {last:.3f}   (x{last / first:.1f})")
print(f"mean answer length: {history[0]['words']:.0f} words -> {history[-1]['words']:.0f} words")

suffix = "_naive" if NAIVE else ""
csv_path = outputs_dir() / f"toy_grpo{suffix}.csv"
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
left.plot([r["step"] for r in history], [r["reward"] for r in history], color="#c8553d", marker="o", markersize=3)
left.set(xlabel="step", ylabel="mean reward", title="Reward (1.0 = exactly 5 words)", ylim=(0, 1.05))
right.plot([r["step"] for r in history], [r["words"] for r in history], color="#2a78d6", marker="o", markersize=3)
right.axhline(5, color="gray", linestyle="--", linewidth=1)
right.set(xlabel="step", ylabel="words per answer", title="Answer length (target 5)")
for axis in (left, right):
    axis.grid(alpha=0.3)
figure.tight_layout()
plot_path = outputs_dir() / f"toy_grpo{suffix}.png"
figure.savefig(plot_path, dpi=120)
print(f"saved {csv_path.name} and {plot_path.name} in the work folder ")

print("\nSelf-checks:")
check("reward rose: last-5 mean is at least 0.15 above the first-5 mean", last - first >= 0.15)
check("answers got shorter, towards 5 words", history[-1]["words"] < history[0]["words"])
if not NAIVE:
    check("the final answers are English (the Chinese loophole is closed)", all(ord(c) < 128 for c in example))
    check("the final answers are not the refusal (loophole 2 is closed)", "sorry" not in example.lower())
check("KL stayed finite (the policy moved but didn't explode)", all(r["kl"] == r["kl"] and r["kl"] < 10 for r in history))
check("most groups had some reward spread (frac_reward_zero_std < 0.5 on average)", mean(r["frac_reward_zero_std"] for r in history) < 0.5)
