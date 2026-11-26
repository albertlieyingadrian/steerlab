"""Modal app for the RSI pilot (FIXED): one recipe evaluation = an update fine-tune of the round-0 detector, then scoring.

    MODAL_PROFILE=modal4 MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy experiments/slop-rsi/modal_rsi.py

Pilot 2 (plan amendment 3): app `slop-rsi-p2`; updates, adapters and the eval-p2 file live under rsi/p2/ on the volume;
`score_runs` rescores saved adapters (pilot 1's R0 seeds) on any text files without training.

The volume is the slop-detector note's (`slop-detector` on modal4). Round 0 is its `runs/b1b-1.7bB` adapter (Qwen3-1.7B
Base, LoRA r 16, 4-bucket head), the detector the hard-negative note started from. An update starts from that adapter,
trains on the recipe's rows (at most 15,000) with the slop-detector trainer's own batching, loss and schedule, then
scores the hard-negative note's frozen evaluation file, data v2b's validation split and 2,000 unlabeled pages.
The recipe never runs here: it only chose the rows and two numbers (learning rate, epochs) before this was called.
"""

from __future__ import annotations

import time

import modal

app = modal.App("slop-rsi-p2")
volume = modal.Volume.from_name("slop-detector")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.22.2",
        "accelerate>=1.10",
        "safetensors",
        "huggingface_hub",
        "peft",
        "scikit-learn",
        "numpy",
        "pyarrow",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_DISABLE_XET": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
train_image = (
    image.pip_install("flash-linear-attention==0.5.2")
    .env({"TORCHDYNAMO_DISABLE": "1"})
    .add_local_file("../slop-detector/common.py", "/root/sd/common.py")
    .add_local_file("../slop-detector/04_train.py", "/root/sd/train.py")
)

VOL = "/vol"
ROUND0 = "runs/b1b-1.7bB"
BASE = "Qwen/Qwen3-1.7B-Base"
VOLP = "rsi/p2"
EVALS = f"{VOLP}/evals.jsonl"


def _reload() -> None:
    try:
        volume.reload()
    except RuntimeError as e:
        print(f"volume reload skipped: {e}", flush=True)


@app.function(image=train_image, gpu="H100", volumes={VOL: volume}, timeout=25 * 60, max_containers=6)
def train_update(name: str, rows_path: str, lr: float, epochs: float, seed: int = 0, smoke: bool = False) -> dict:  # noqa: PLR0915
    """Update round 0 on the rows in `rows_path` (volume JSONL: text, bucket, weight); return every score."""
    import json
    import math
    import random
    import sys

    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sys.path.insert(0, "/root/sd")
    import train as tr

    t0 = time.time()
    _reload()
    torch.manual_seed(seed)
    tok, model = tr.load(BASE, "cuda", num_labels=4)
    set_peft_model_state_dict(model, load_file(f"{VOL}/{ROUND0}/adapter/adapter_model.safetensors"))
    rows = [json.loads(x) for x in open(f"{VOL}/{rows_path}") if x.strip()]
    if smoke:
        rows = rows[:300]
    random.Random(seed).shuffle(rows)
    rows = tr.encode(rows, tok)
    t_load = time.time() - t0

    # the slop-detector trainer's step: length-grouped micro-batches (6,144 padded tokens, as the round-0 run), at least
    # 64 rows per optimiser step, weighted cross-entropy on the bucket, AdamW, 3% warm-up then cosine, clip 1.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def groups_for(epoch: int) -> list:
        groups, cur = [], []
        for m in tr.batches(rows, 6144, shuffle=True, seed=seed + epoch, max_rows=64):
            cur.append(m)
            if sum(len(x) for x in cur) >= 64:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return groups

    total = max(1, int(len(groups_for(0)) * epochs))
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    model.train()
    step, epoch, losses, nonfinite = 0, 0, [], 0
    while step < total:
        for g in groups_for(epoch):
            if step >= total:
                break
            n = sum(len(x) for x in g)
            loss_sum = 0.0
            for idx in g:
                logits = model(**tr.collate(rows, idx, tok, "cuda")).logits.float()
                w = torch.tensor([rows[i]["weight"] for i in idx], device="cuda", dtype=torch.float32)
                yb = torch.tensor([rows[i]["bucket"] for i in idx], device="cuda", dtype=torch.long)
                loss = (torch.nn.functional.cross_entropy(logits, yb, reduction="none") * w).sum() / n
                loss.backward()
                loss_sum += float(loss)
            if not math.isfinite(loss_sum):  # skip a non-finite update (lesson C1)
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(round(loss_sum, 4))
        epoch += 1
    t_train = time.time() - t0 - t_load

    def score(texts: list[str]) -> list[float]:
        enc = tr.encode([{"text": t} for t in texts], tok)
        return [round(float(x), 6) for x in tr.predict(model, enc, tok, "cuda", max_tokens=16384)]

    ev = [json.loads(x) for x in open(f"{VOL}/{EVALS}")]
    if smoke:
        ev = ev[:500]
    s = score([r["text"] for r in ev])
    by: dict = {}
    for r, x in zip(ev, s):
        by.setdefault(r["set"], []).append(x)
    val = [json.loads(x)["text"] for x in open(f"{VOL}/data/b1b/val.jsonl")]
    cal = [json.loads(x)["text"] for x in open(f"{VOL}/rsi/cal_unlabeled.jsonl")]
    if smoke:
        val, cal = val[:300], cal[:300]
    out = {
        "name": name,
        "eval": by,
        "v2b_val": score(val),
        "unlabeled": score(cal),
        "rows": len(rows),
        "steps": step,
        "nonfinite_steps": nonfinite,
        "loss_first": float(np.mean(losses[:10])) if losses else None,
        "loss_last": float(np.mean(losses[-10:])) if losses else None,
        "load_seconds": round(t_load),
        "train_seconds": round(t_train),
        "score_seconds": round(time.time() - t0 - t_load - t_train),
        "seconds": round(time.time() - t0),
    }
    if not smoke:
        model.save_pretrained(f"{VOL}/{VOLP}/runs/{name}/adapter")
        volume.commit()
    return out


@app.function(image=train_image, gpu="H100", volumes={VOL: volume}, timeout=40 * 60, max_containers=1)
def score_runs(adapters: list[str], files: list[str]) -> dict:
    """Score volume JSONL files (field "text") with each saved adapter: {adapter: {file: [scores]}}. No training."""
    import json
    import sys

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sys.path.insert(0, "/root/sd")
    import train as tr

    t0 = time.time()
    _reload()
    tok, model = tr.load(BASE, "cuda", num_labels=4)
    enc = {f: tr.encode([{"text": json.loads(x)["text"]} for x in open(f"{VOL}/{f}") if x.strip()], tok) for f in files}
    out: dict = {"seconds": {}}
    for a in adapters:
        set_peft_model_state_dict(model, load_file(f"{VOL}/{a}/adapter_model.safetensors"))
        model.eval()
        out[a] = {
            f: [round(float(x), 6) for x in tr.predict(model, e, tok, "cuda", max_tokens=16384)] for f, e in enc.items()
        }
        out["seconds"][a] = round(time.time() - t0)
        print(a, "scored", out["seconds"][a], flush=True)
    return out
