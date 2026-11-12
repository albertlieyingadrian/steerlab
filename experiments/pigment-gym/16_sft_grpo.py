"""
Exploratory arm `sft_grpo`: the fine-tuned model (modal_next.py --stage sft), then the published laptop GRPO recipe.

Run from the repo root on an Apple-silicon laptop (renders on the laptop: Modal's software WebGL times out):
    .venv/bin/python experiments/pigment-gym/16_sft_grpo.py --steps 2    # smoke test
    .venv/bin/python experiments/pigment-gym/16_sft_grpo.py

Same as 7_train_laptop.py (sunset over green hills, 40 steps x 8 sketches, learning rate 2e-4, KL weight 0.04,
partial credit, the copy check), except that it starts from the sft LoRA and samples up to 1,024 tokens (the
fine-tuned model writes ~900-token sketches). The question: does a model that already paints still learn the paste
hack under the same reward? Output: the work folder next/sft_grpo/ (curve, rollouts, LoRA); then sample the arm with
    .venv/bin/modal run experiments/pigment-gym/modal_next.py --stage sample --arm sft_grpo
"""

import argparse
import shutil

import torch
from common import RESULTS, outputs_dir
from next_common import MAX_NEW_TOKENS, MODEL
from train_loop import train

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=40)
args = parser.parse_args()

work = outputs_dir() / "next"
out = work / ("sft_grpo" if args.steps == 40 else "sft_grpo_smoke")
summary = train(
    out,
    steps=args.steps,
    model_name=MODEL,
    learning_rate=2e-4,
    warmup=1,
    max_new_tokens=MAX_NEW_TOKENS,
    beta=0.04,
    pool_size=8,
    budget_s=4 * 3600,
    reserve_s=600,
    dtype=torch.float32,
    resume=False,
    init_adapter=work / "sft_lora",
)
if args.steps == 40:
    for name in ("curve.csv", "rollouts.jsonl", "summary.json"):
        shutil.copy(out / name, RESULTS / "next" / f"sft_grpo_{name}")
print(
    f"held-out sunset: pass {summary['before']['gate_pass']:.0%} -> {summary['after']['gate_pass']:.0%}, "
    f"reward {summary['before']['reward']:.3f} -> {summary['after']['reward']:.3f}"
)
