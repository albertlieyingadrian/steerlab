"""Modal GPU functions for the slop-detector replication, in their own app.

    modal deploy experiments/slop-detector/modal_slop.py

score(texts, model, revision): a one-logit sequence-classification detector on a rented L4 (else A10G).
Used for voi-slop-detect-4B (the model we try to match) and, later, for our own checkpoints on the volume.
One text per forward pass, as deslop-voi does, so no padding meets Qwen3.5's linear-attention layers.
"""

import time

import modal

app = modal.App("slop-detector")
volume = modal.Volume.from_name("slop-detector", create_if_missing=True)
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
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
train_image = (
    image.pip_install("flash-linear-attention==0.5.2")
    .env({"TORCHDYNAMO_DISABLE": "1"})
    .add_local_file("common.py", "/root/sd/common.py")
    .add_local_file("04_train.py", "/root/sd/train.py")
)


@app.function(image=train_image, gpu="H100", volumes={"/vol": volume}, timeout=3 * 60 * 60)
def train(
    data: str, base: str, out: str, evals: dict[str, list[str]], micro_tokens: int = 16384, extra: tuple[str, ...] = ()
) -> dict:
    return _train(data, base, out, evals, micro_tokens, extra)


@app.function(image=train_image, gpu="A100-80GB", volumes={"/vol": volume}, timeout=12 * 60 * 60)
def train_a100(
    data: str, base: str, out: str, evals: dict[str, list[str]], micro_tokens: int = 16384, extra: tuple[str, ...] = ()
) -> dict:
    """Qwen3.5 on an A100: flash-linear-attention's backward pass refuses Hopper GPUs with the Triton that ships with
    torch 2.9 (a known wrong-result bug), and the A100 has no such problem."""
    return _train(data, base, out, evals, micro_tokens, extra)


def _train(
    data: str, base: str, out: str, evals: dict[str, list[str]], micro_tokens: int, extra: tuple[str, ...]
) -> dict:
    """Lesson 04's trainer, unchanged, on one H100: data from /vol/data/<data>, run saved to /vol/runs/<out>.
    Then scores every text in `evals` (benchmark name -> texts) with the final adapter, so the benchmarks are
    scored once, by the same code the Mac uses."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/sd")
    import train as tr

    run = Path("/vol/runs") / out
    sys.argv = [
        "train.py",
        "--base",
        base,
        "--data",
        f"/vol/data/{data}",
        "--out",
        str(run),
        "--micro-tokens",
        str(micro_tokens),
        *extra,
    ]
    tr.ON_CHECKPOINT = volume.commit  # each checkpoint reaches the volume at once
    tr.main()
    volume.commit()
    heads = 4 if json.loads((run / "config.json").read_text()).get("head") == "buckets" else 1
    tok, model = tr.load(base, "cuda", num_labels=heads)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(str(run / "adapter" / "adapter_model.safetensors")))
    model.eval()
    scores = {}
    for name, texts in evals.items():
        rows = tr.encode([{"text": t} for t in texts], tok)
        scores[name] = [round(float(x), 6) for x in tr.predict(model, rows, tok, "cuda", max_tokens=16384)]
        print(f"scored {name}: {len(texts)}")
    return {
        "log": [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()],
        "val_scores": json.loads((run / "val_scores.json").read_text()),
        "config": json.loads((run / "config.json").read_text()),
        "scores": scores,
    }


@app.function(image=image, gpu=["L4", "A10G"], volumes={"/vol": volume}, timeout=2 * 60 * 60, max_containers=4)
def score(texts: list[str], model: str, revision: str | None = None) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    net = AutoModelForSequenceClassification.from_pretrained(model, revision=revision, dtype=torch.bfloat16)
    net = net.to("cuda").eval()
    volume.commit()
    t0, scores, tokens = time.time(), [], []
    for i, text in enumerate(texts):
        enc = tok([text], truncation=True, max_length=512, return_tensors="pt").to("cuda")
        scores.append(float(torch.sigmoid(net(**enc).logits.float()).reshape(-1)[0]))
        tokens.append(int(enc["input_ids"].shape[1]))
        if i % 500 == 0:
            print(f"{i}/{len(texts)} ({time.time() - t0:.0f} s)")
    return {
        "scores": scores,
        "tokens": tokens,
        "seconds": round(time.time() - t0),
        "gpu": torch.cuda.get_device_name(0),
    }


@app.function(image=train_image, gpu=["H100", "A100-80GB", "L40S"], volumes={"/vol": volume}, timeout=60 * 60)
def score_run(run: str, evals: dict[str, list[str]]) -> dict:
    """Score texts with a run already on the volume (/vol/runs/<run>): its base model plus its saved adapter."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/sd")
    import train as tr
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path("/vol/runs") / run
    cfg = json.loads((path / "config.json").read_text())
    base = cfg["base"]
    tok, model = tr.load(base, "cuda", num_labels=4 if cfg.get("head") == "buckets" else 1)
    set_peft_model_state_dict(model, load_file(str(path / "adapter" / "adapter_model.safetensors")))
    model.eval()
    out = {}
    for name, texts in evals.items():
        rows = tr.encode([{"text": t} for t in texts], tok)
        out[name] = [round(float(x), 6) for x in tr.predict(model, rows, tok, "cuda", max_tokens=16384)]
    return out
