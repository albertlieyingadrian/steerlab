"""Modal GPU functions for the backbone bake-off, in their own app on the slop-detector note's volume.

    MODAL_PROFILE=albertlieyingadrian modal deploy modal_bb.py

train(...)  one arm on an H100 with train_arm.py (04_train.py's recipe); the run lands in /vol/bb/runs/<out>.
score(...)  one trained arm on every eval split stored in /vol/bb/evals/<split>.json; returns scores and seconds.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SD = HERE.parent / "slop-detector"

app = modal.App("slop-backbones")
volume = modal.Volume.from_name("slop-detector", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
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
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHDYNAMO_DISABLE": "1",
        }
    )
    .add_local_file(str(SD / "common.py"), "/root/sd/common.py")
    .add_local_file(str(SD / "04_train.py"), "/root/sd/04_train.py")
    .add_local_file(str(HERE / "train_arm.py"), "/root/sd/train_arm.py")
)


@app.function(
    image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=20 * 60 * 60, single_use_containers=True
)
def train(
    base: str,
    kind: str,
    data: str,
    out: str,
    lr: float,
    epochs: float,
    micro: int,
    grad_ckpt: bool,
    max_rows: int = 0,
    max_minutes: float = 1150,
) -> dict:
    return _train(base, kind, data, out, lr, epochs, micro, grad_ckpt, max_rows, max_minutes, "H100")


@app.function(
    image=image, gpu="H200", volumes={"/vol": volume}, secrets=[hf], timeout=20 * 60 * 60, single_use_containers=True
)
def train_h200(
    base: str,
    kind: str,
    data: str,
    out: str,
    lr: float,
    epochs: float,
    micro: int,
    grad_ckpt: bool,
    max_rows: int = 0,
    max_minutes: float = 1150,
) -> dict:
    """Round 2's large arms (Gemma-3-12B, Qwen3-14B, Nemotron 30B-A3B): 141 GB instead of 80."""
    return _train(base, kind, data, out, lr, epochs, micro, grad_ckpt, max_rows, max_minutes, "H200")


def _train(base, kind, data, out, lr, epochs, micro, grad_ckpt, max_rows, max_minutes, gpu) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/root/sd")
    import train_arm

    tr = __import__("04_train")
    tr.ON_CHECKPOINT = volume.commit
    run = Path("/vol/bb/runs") / out
    argv = [
        "--kind",
        kind,
        "--base",
        base,
        "--data",
        f"/vol/bb/data/{data}",
        "--out",
        str(run),
        "--head",
        "buckets",
        "--lr",
        str(lr),
        "--epochs",
        str(epochs),
        "--micro-tokens",
        str(micro),
        "--ckpt-minutes",
        "30",
        "--max-minutes",
        str(max_minutes),
        "--resume",
    ]
    if grad_ckpt:
        argv.append("--grad-ckpt")
    if max_rows:
        argv += ["--max-rows", str(max_rows)]
    t0 = time.time()
    train_arm.main(argv)
    volume.commit()
    return {
        "log": [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines() if x.strip()],
        "val_scores": json.loads((run / "val_scores.json").read_text()),
        "config": json.loads((run / "config.json").read_text()),
        "wall": round(time.time() - t0),
        "gpu": gpu,
    }


@app.function(
    image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=4 * 60 * 60, single_use_containers=True
)
def score(out: str, splits: list[str]) -> dict:
    return _score(out, splits, "H100")


@app.function(
    image=image, gpu="H200", volumes={"/vol": volume}, secrets=[hf], timeout=4 * 60 * 60, single_use_containers=True
)
def score_h200(out: str, splits: list[str]) -> dict:
    return _score(out, splits, "H200")


def _score(out: str, splits: list[str], gpu: str) -> dict:
    """Score a trained arm on stored eval splits, timing each split (model load timed separately)."""
    import json
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/sd")
    tr = __import__("04_train")
    run = Path("/vol/bb/runs") / out
    cfg = json.loads((run / "config.json").read_text())
    kind = (run / "bb_kind.txt").read_text().strip() if (run / "bb_kind.txt").exists() else "lora"
    t0 = time.time()
    if kind in ("lora_mm", "pool"):  # round 2: rebuild with the same loader, then load the adapter and head
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        import train_arm

        tok, model = train_arm.LOADERS[kind](cfg["base"], "cuda", num_labels=4)
        set_peft_model_state_dict(model, load_file(str(run / "adapter" / "adapter_model.safetensors")))
    elif kind == "full" or (run / "adapter" / "model.safetensors").exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(run / "adapter"))
        tok.padding_side = "right"
        model = AutoModelForSequenceClassification.from_pretrained(str(run / "adapter"), dtype=torch.bfloat16)
        model = model.to("cuda")
    else:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        tok, model = tr.load(cfg["base"], "cuda", num_labels=4)
        set_peft_model_state_dict(model, load_file(str(run / "adapter" / "adapter_model.safetensors")))
    model.eval()
    load_s = round(time.time() - t0)
    out_scores, secs, ntok = {}, {}, {}
    for s in splits:
        texts = json.loads((Path("/vol/bb/evals") / f"{s}.json").read_text())
        t1 = time.time()
        rows = tr.encode([{"text": t} for t in texts], tok)
        out_scores[s] = [round(float(x), 6) for x in tr.predict(model, rows, tok, "cuda", max_tokens=4096)]
        secs[s] = round(time.time() - t1, 1)
        ntok[s] = int(sum(len(r["ids"]) for r in rows))
        print(f"{out} {s}: {len(texts)} texts in {secs[s]} s", flush=True)
    return {"scores": out_scores, "seconds": secs, "tokens": ntok, "load_seconds": load_s, "gpu": gpu}
