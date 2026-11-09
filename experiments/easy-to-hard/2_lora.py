"""
Step 2: easy-to-hard with LoRA fine-tuning of Qwen2.5-7B (the same recipe as scalable-oversight-101's w2s_lora.py:
rank 16, lr 1e-4, batch 8, 2 epochs, seed 0). Four runs of 299 student_train items each:

    easy-true    easy items, true labels          hard-true   hard items, true labels
    mixed-true   a random draw, true labels       mixed-weak  a random draw, the 0.5B teacher's probabilities

Each run writes results/lora/<run>.json with per-item scores (logit Yes - logit No) on dev and test.
The earlier full-size runs (1,300 items) from scalable-oversight-101 are re-scored on the same easy/hard split by
3_summarise.py, so nothing about them is rerun.

    .venv/bin/python experiments/easy-to-hard/2_lora.py            # every run not done yet
    .venv/bin/python experiments/easy-to-hard/2_lora.py --run easy-true
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from e2h import RESULTS, SEED, bucket, items, steps_by_id  # first: puts scalable-oversight-101 on the path

# isort: split
from common import device, judge_scores, load, verify_prompt, yes_no_ids  # scalable-oversight-101's helpers
from w2s_lora import DROPOUT, EPOCHS, LORA_ALPHA, LR, RANK, TARGETS, WARMUP, weak_labels

OUT = RESULTS / "lora"
RUNS = ("easy-true", "hard-true", "mixed-true", "mixed-weak")
N, BATCH = 299, 8


def run_data(run):
    rows = items("student_train")
    steps = steps_by_id()
    b = np.array([bucket(steps[r["id"]]) for r in rows])
    pool, lab = run.split("-")
    cand = np.arange(len(rows)) if pool == "mixed" else np.where(b == pool)[0]
    idx = np.random.default_rng(SEED).choice(cand, size=N, replace=False)
    y = np.array([r["label"] for r in rows], dtype=np.float32) if lab == "true" else weak_labels().astype(np.float32)
    return [rows[i] for i in idx], y[idx]


def train(run):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    rows, targets = run_data(run)
    model, tok = load("7b")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=RANK, lora_alpha=LORA_ALPHA, lora_dropout=DROPOUT, target_modules=TARGETS)
    )
    model.train()
    yes, no = yes_no_ids(tok)
    prompts = [verify_prompt(tok, r) for r in rows]
    steps = EPOCHS * math.ceil(len(prompts) / BATCH)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    warm = max(1, int(WARMUP * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm))
    )
    rng, started = np.random.default_rng(0), time.time()
    for _ in range(EPOCHS):
        order = rng.permutation(len(prompts))
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(device())
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, yes] - logits[:, no], torch.tensor(targets[idx], device=logits.device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    out = {"run": run, "train_items": len(prompts), "steps": steps, "seconds": time.time() - started}
    for split in ("dev", "test"):
        sr = items(split)
        out[split] = {
            "ids": [r["id"] for r in sr],
            "scores": judge_scores(model, tok, [verify_prompt(tok, r) for r in sr], batch=8),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps(out))
    print(f"{run}: {out['seconds'] / 60:.0f} min", flush=True)
    del model
    if device() == "mps":
        torch.mps.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=RUNS)
    a = p.parse_args()
    for run in [a.run] if a.run else [r for r in RUNS if not (OUT / f"{r}.json").exists()]:
        train(run)


if __name__ == "__main__":
    main()
