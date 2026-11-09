"""
Weak-to-strong with LoRA students: the same teacher and labels as w2s.py, but the student is fine-tuned, not probed.

    .venv/bin/python experiments/scalable-oversight-101/w2s_lora.py               # every run not done yet, then summary
    .venv/bin/python experiments/scalable-oversight-101/w2s_lora.py --student 7b --method naive    # one run
    .venv/bin/python experiments/scalable-oversight-101/w2s_lora.py --summarise   # only rebuild results/w2s_lora.json
    .venv/bin/python experiments/scalable-oversight-101/w2s_lora.py --shuffled    # random-label control runs

- Weak teacher: the 0.5B probe from w2s.py, unchanged, so its labels on student_train are the same 67.5%-right labels.
- Student: Qwen2.5-1.5B or Qwen2.5-7B Instruct with LoRA adapters, reading the same yes/no prompt the judges read.
  Its score is logit(Yes) - logit(No) at the answer token, trained with binary cross-entropy against the targets:
    ceiling      student_train's true labels
    naive        the teacher's probabilities as soft targets
    confident    the half of student_train the teacher is surest about, with its hard labels
    aux          Burns et al.'s confidence loss: (1 - a) CE(weak) + a CE(own prediction hardened at the batch median),
                 a ramped to 0.75 over the first fifth of training
  Unlike a probe, LoRA starts from the model's own zero-shot answer (the prompt is the zero-shot prompt), so the
  question is also whether weak labels pull it below where it started.

All settings are fixed before any run and nothing is tuned on dev or test: rank 16, alpha 32, dropout 0.05 on the
attention and MLP projections, AdamW lr 1e-4 with 10% warmup then linear decay, batch 8, 2 epochs, seed 0.
Each run writes results/lora/<student>_<method>.json (per-item test and dev scores); the summary writes
results/w2s_lora.json with PGR and 95% bootstrap intervals against the LoRA ceiling.

--shuffled is the random-label control (see random_labels.py): the teacher's probabilities are shuffled across
student_train items (seed 0) before training, so the labels keep their mix but say nothing about their items.
Those runs write results/lora/<student>_<method>_shuffled.json and appear in the summary as "<method> shuffled".
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from common import RESULTS, device, items, judge_scores, load, verify_prompt, yes_no_ids
from w2s import AUX_ALPHA, bootstrap_pgr, features, fit

OUT = RESULTS / "lora"
STUDENTS = ("1.5b", "7b")
METHODS = ("ceiling", "naive", "confident", "aux")
RANK, LORA_ALPHA, DROPOUT = 16, 32, 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR, WARMUP, BATCH, EPOCHS, SEED = 1e-4, 0.1, 8, 2, 0


def weak_labels():
    """The teacher's P(correct) on student_train, in items("student_train") order, exactly as in w2s.py."""
    W = features("0.5b")
    return fit(W["weak_train"][0], W["weak_train"][1])(W["student_train"][0])


def targets_for(method, rows, p_weak):
    """(rows, targets) the student trains on."""
    if method == "ceiling":
        return rows, np.array([r["label"] for r in rows], dtype=np.float32)
    if method == "confident":
        keep = np.abs(p_weak - 0.5) >= np.median(np.abs(p_weak - 0.5))
        return [r for r, k in zip(rows, keep, strict=True) if k], (p_weak[keep] > 0.5).astype(np.float32)
    return rows, p_weak.astype(np.float32)  # naive and aux


def train(student, method, shuffled=False):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    p_weak = weak_labels()
    if shuffled:
        p_weak = np.random.default_rng(SEED).permutation(p_weak)
    rows, targets = targets_for(method, items("student_train"), p_weak)
    model, tokenizer = load(student)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    config = LoraConfig(r=RANK, lora_alpha=LORA_ALPHA, lora_dropout=DROPOUT, target_modules=TARGETS)
    model = get_peft_model(model, config)
    model.train()
    yes, no = yes_no_ids(tokenizer)
    prompts = [verify_prompt(tokenizer, r) for r in rows]
    steps = EPOCHS * math.ceil(len(prompts) / BATCH)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm))
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rng = np.random.default_rng(SEED)
    started, step = time.time(), 0
    for _ in range(EPOCHS):
        order = rng.permutation(len(prompts))
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            enc = tokenizer([prompts[j] for j in idx], return_tensors="pt", padding=True).to(device())
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            score = logits[:, yes] - logits[:, no]
            y = torch.tensor(targets[idx], device=score.device)
            loss = bce(score, y)
            if method == "aux":
                alpha = AUX_ALPHA * min(1.0, step / (0.2 * steps))
                hard = (score > score.median()).float().detach()
                loss = (1 - alpha) * loss + alpha * bce(score, hard)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0 or step == steps:
                print(f"{student} {method} step {step}/{steps} loss {loss.item():.3f} {time.time() - started:.0f}s",
                      flush=True)  # fmt: skip
    seconds = time.time() - started
    model.eval()
    out = {"student": student, "method": method, "train_items": len(prompts), "steps": steps, "seconds": seconds}
    for split in ("dev", "test"):
        split_rows = items(split)
        out[split] = {
            "ids": [r["id"] for r in split_rows],
            "scores": judge_scores(model, tokenizer, [verify_prompt(tokenizer, r) for r in split_rows], batch=8),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    out["shuffled"] = shuffled
    (OUT / f"{student}_{run_name(method, shuffled)}.json").write_text(json.dumps(out))
    acc = np.mean([(s > 0) == r["label"] for s, r in zip(out["test"]["scores"], items("test"), strict=True)])
    print(f"{student} {method}: test acc {acc:.3f} ({seconds / 60:.0f} min)", flush=True)
    del model
    if device() == "mps":
        torch.mps.empty_cache()


def run_name(method, shuffled):
    return f"{method}_shuffled" if shuffled else method


def summarise():
    rng = np.random.default_rng(0)
    W = features("0.5b")
    teacher = fit(W["weak_train"][0], W["weak_train"][1])
    y_test = W["test"][1]
    weak_correct = ((teacher(W["test"][0]) > 0.5) == y_test).astype(float)
    probe = json.loads((RESULTS / "w2s.json").read_text())
    out = {"teacher_test_acc": weak_correct.mean(), "students": {}}
    for student in STUDENTS:
        paths = {m: OUT / f"{student}_{m}.json" for m in METHODS}
        paths |= {f"{m} shuffled": OUT / f"{student}_{m}_shuffled.json" for m in METHODS if m != "ceiling"}
        runs = {m: json.loads(p.read_text()) for m, p in paths.items() if p.exists()}
        if "ceiling" not in runs:
            continue
        correct = {m: ((np.array(r["test"]["scores"]) > 0) == y_test).astype(float) for m, r in runs.items()}
        zero_shot = features(student)["test"][2]
        correct["zero-shot"] = ((zero_shot > 0) == y_test).astype(float)
        result = {"ceiling_test_acc": correct["ceiling"].mean(), "methods": {}}
        print(f"student {student} (LoRA): ceiling {correct['ceiling'].mean():.3f}")
        for m, c in correct.items():
            if m == "ceiling":
                continue
            pgr, lo, hi = bootstrap_pgr(weak_correct, correct["ceiling"], c, rng)
            minutes = runs[m]["seconds"] / 60 if m in runs else 0.0
            result["methods"][m] = {"test_acc": c.mean(), "pgr": pgr, "pgr_ci": [lo, hi], "train_minutes": minutes}
            probe_acc = probe["students"][student]["methods"].get(m, {}).get("test_acc", float("nan"))
            print(f"  {m:18} acc {c.mean():.3f} (probe {probe_acc:.3f})   PGR {pgr:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        out["students"][student] = result
    (RESULTS / "w2s_lora.json").write_text(json.dumps(out, indent=1, default=float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", choices=STUDENTS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--shuffled", action="store_true", help="train on shuffled teacher labels (control)")
    args = parser.parse_args()
    if args.student and args.method:
        train(args.student, args.method, args.shuffled)
    elif not args.summarise:
        methods = [m for m in METHODS if not (args.shuffled and m == "ceiling")]
        for student in STUDENTS:
            for method in methods:
                if not (OUT / f"{student}_{run_name(method, args.shuffled)}.json").exists():
                    train(student, method, args.shuffled)
    summarise()


if __name__ == "__main__":
    main()
