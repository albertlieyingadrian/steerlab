"""Lesson 04: fine-tune a small Qwen into a one-number "how much AI" scorer, with the card's recipe.

    python 04_train.py --base Qwen/Qwen3-1.7B-Base --out runs/pilot-1.7b
    python 04_train.py --base Qwen/Qwen3-0.6B-Base --max-rows 2000 --epochs 1 --out runs/smoke   # a quick check

The recipe, from the voi-slop-detect-4B card:
- a sequence-classification head with ONE output on top of the base model; sigmoid(logit) is the score;
- LoRA r 16, alpha 32 on the attention and MLP projections; the head is trained in full;
- 3 epochs, learning rate 2e-4, batch 64, the first 512 tokens, fenced code emptied;
- loss: binary cross-entropy against the soft target (0 human .. 1 AI), each row weighted by its band weight.

Why BCE on a soft target: the target for a rewrite is "43% of the words changed", not a yes/no. BCE with a target
of 0.43 is smallest when sigmoid(logit) = 0.43, so the model learns to output the fraction itself.

Writes <out>/adapter (LoRA + head), <out>/log.jsonl (loss per step, val metrics per epoch) and <out>/val_scores.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from common import DATA, HERE, band, empty_fences, read_jsonl, three_way

ON_CHECKPOINT = None  # set by the Modal wrapper to commit the volume after each checkpoint


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(base: str, dev: str, lora_r: int = 16, num_labels: int = 1):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    if hasattr(model.config, "text_config"):  # Qwen3.5 reads it from the nested language-model config
        model.config.text_config.pad_token_id = tok.pad_token_id
    # attention and MLP projections of the language model. Qwen3 has q/k/v/o; Qwen3.5 adds linear-attention layers
    # (in_proj_qkv/z/a/b, out_proj) and an unused vision tower, which the regex leaves out.
    targets = (
        r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|"
        r"gate_proj|up_proj|down_proj)$"
    )
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.0,
        task_type="SEQ_CLS",
        target_modules=targets,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, cfg)
    return tok, model.to(dev)


def batches(rows: list[dict], max_tokens: int, shuffle: bool, seed: int = 0, max_rows: int = 512) -> list[list[int]]:
    """Group rows of similar length so little compute is spent on padding: at most max_tokens padded tokens and
    max_rows rows per micro-batch."""
    lens = [len(r["ids"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    out, cur = [], []
    for i in order:
        if cur and (
            lens[i] * (len(cur) + 1) > max_tokens or len(cur) >= max_rows
        ):  # sorted, so lens[i] is the longest so far
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def collate(rows: list[dict], idx: list[int], tok, dev: str):
    # lengths rounded up to a multiple of 64: fewer distinct shapes, so compiled GPU kernels are reused
    enc = tok.pad({"input_ids": [rows[i]["ids"] for i in idx]}, return_tensors="pt", pad_to_multiple_of=64)
    return {k: v.to(dev) for k, v in enc.items()}


def predict(model, rows: list[dict], tok, dev: str, max_tokens: int = 16384) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros(len(rows))
    with torch.no_grad():
        for idx in batches(rows, max_tokens, shuffle=False):
            logits = model(**collate(rows, idx, tok, dev)).logits.float()
            if logits.shape[-1] == 1:  # one number: sigmoid
                out[idx] = torch.sigmoid(logits.reshape(-1)).cpu().numpy()
            else:  # buckets: the expected bucket, scaled to 0..1
                probs = torch.softmax(logits, -1)
                k = torch.arange(logits.shape[-1], device=probs.device, dtype=probs.dtype)
                out[idx] = ((probs * k).sum(-1) / (logits.shape[-1] - 1)).cpu().numpy()
    model.train()
    return out


def encode(rows: list[dict], tok) -> list[dict]:
    texts = [empty_fences(r["text"]) for r in rows]
    ids = tok(texts, truncation=True, max_length=512)["input_ids"]
    return [{**r, "ids": i} for r, i in zip(rows, ids, strict=True)]


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=64, help="rows per optimiser step")
    ap.add_argument("--micro-tokens", type=int, default=8192, help="padded tokens per forward pass")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-minutes", type=float, default=0, help="save adapter + optimiser this often (0: per epoch)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly (with a checkpoint) after this long")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt/state.pt if it exists")
    ap.add_argument(
        "--head",
        choices=["score", "buckets"],
        default="score",
        help="score: one number, BCE on 'target'; buckets: 4 classes, cross-entropy on 'bucket'",
    )
    ap.add_argument(
        "--grad-ckpt", action="store_true", help="recompute activations in backward: less memory, ~30% slower"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    out = Path(args.out) if Path(args.out).is_absolute() else HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(args.base, dev, num_labels=4 if args.head == "buckets" else 1)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    train = read_jsonl(Path(args.data) / "train.jsonl")
    val = read_jsonl(Path(args.data) / "val.jsonl")
    if not train or not val:
        raise SystemExit(f"no rows in {args.data} (train {len(train)}, val {len(val)})")
    random.Random(args.seed).shuffle(train)
    if args.max_rows:
        train, val = train[: args.max_rows], val[: max(200, args.max_rows // 8)]
    train, val = encode(train, tok), encode(val, tok)
    mean_len = np.mean([len(r["ids"]) for r in train])
    print(f"{dev}: {len(train)} train rows, {len(val)} val rows, mean {mean_len:.0f} tokens", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def groups_for(epoch: int) -> list[list[list[int]]]:
        """One epoch: length-grouped micro-batches, regrouped into optimiser steps of at least args.batch rows."""
        groups, cur = [], []
        for m in batches(train, args.micro_tokens, shuffle=True, seed=args.seed + epoch, max_rows=args.batch):
            cur.append(m)
            if sum(len(x) for x in cur) >= args.batch:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return groups

    # steps come from the real grouping: a step holds whole micro-batches, so it can hold more than args.batch rows
    # (runs before 2026-11-12 counted ceil(rows / batch) steps per epoch and so trained ~1.3x the nominal epochs)
    total = int(len(groups_for(0)) * args.epochs)
    warm = max(1, int(0.03 * total))

    def lr_at(s: int) -> float:  # linear warm-up, then cosine down to zero
        return min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    state = out / "ckpt" / "state.pt"
    step, seen, epoch, skip, resumed = 0, 0, 0, 0, False
    if args.resume and state.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(out / "adapter" / "adapter_model.safetensors")))
        st = torch.load(state, map_location=dev, weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, seen, epoch, skip, resumed = st["step"], st["seen"], st["epoch"], st["gi"], True
        print(f"resumed at step {step} of {total} (epoch {epoch}, group {skip})", flush=True)
    log = (out / "log.jsonl").open("a" if resumed else "w")
    t0 = last_ck = time.time()
    seen_here, stopped, scores = 0, False, None

    def checkpoint(gi: int) -> None:
        """Adapter + optimiser + position, so a crash or a credit stop loses at most --ckpt-minutes of work."""
        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        state.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "seen": seen,
                "epoch": epoch,
                "gi": gi,
            },
            state,
        )
        log.write(
            json.dumps({"checkpoint": step, "epoch": epoch, "group": gi, "elapsed": round(time.time() - t0)}) + "\n"
        )
        log.flush()
        if ON_CHECKPOINT:
            ON_CHECKPOINT()

    while step < total and not stopped:
        for gi, g in enumerate(groups_for(epoch)):
            if gi < skip:
                continue
            if step >= total:
                break
            n = sum(len(x) for x in g)
            loss_sum = 0.0
            for idx in g:
                logits = model(**collate(train, idx, tok, dev)).logits.float()
                w = torch.tensor([train[i]["weight"] for i in idx], device=dev, dtype=torch.float32)
                if args.head == "buckets":
                    yb = torch.tensor([train[i]["bucket"] for i in idx], device=dev, dtype=torch.long)
                    per = torch.nn.functional.cross_entropy(logits, yb, reduction="none")
                else:
                    y = torch.tensor([train[i]["target"] for i in idx], device=dev, dtype=torch.float32)
                    per = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), y, reduction="none")
                loss = (per * w).sum() / n
                loss.backward()
                loss_sum += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if dev == "cuda" and step % 50 == 0:  # hand cached blocks back now and then (it costs a GPU sync)
                torch.cuda.empty_cache()
            step += 1
            seen += n
            seen_here += n
            if step % 10 == 0 or step == 1:
                rec = {
                    "step": step,
                    "of": total,
                    "loss": round(loss_sum, 4),
                    "lr": sched.get_last_lr()[0],
                    "rows_per_s": round(seen_here / (time.time() - t0), 1),
                    "elapsed": round(time.time() - t0),
                    "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else None,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
                checkpoint(gi + 1)
                log.write(json.dumps({"stopped": "time cap", "step": step, "of": total}) + "\n")
                print(f"stopped at the {args.max_minutes}-minute cap, step {step} of {total}", flush=True)
                stopped = True
                break
            if args.ckpt_minutes and time.time() - last_ck > args.ckpt_minutes * 60:
                checkpoint(gi + 1)
                last_ck = time.time()
        skip = 0
        if stopped:
            break
        epoch += 1
        scores = predict(model, val, tok, dev)
        m = {k: round(v, 4) for k, v in three_way(np.array([r["band"] for r in val]), band(scores)).items()}
        m["mae"] = round(float(np.mean(np.abs(scores - np.array([r["target"] for r in val])))), 4)
        log.write(json.dumps({"epoch": epoch, "val": m}) + "\n")
        log.flush()
        print(f"epoch {epoch} val {m}", flush=True)
        checkpoint(0)
    if scores is None or stopped:
        scores = predict(model, val, tok, dev)
        checkpoint(0)
    (out / "val_scores.json").write_text(json.dumps([round(float(s), 5) for s in scores]) + "\n")
    meta = {
        **vars(args),
        "device": dev,
        "seconds": round(time.time() - t0),
        "train_rows": len(train),
        "steps": step,
        "total_steps": total,
        "rows_seen": seen,
        "stopped_early": stopped,
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
