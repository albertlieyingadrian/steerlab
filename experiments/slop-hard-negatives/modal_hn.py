"""Modal functions for the hard-negative rounds, in their own app (one app per experiment), on the modal4 workspace.

    MODAL_PROFILE=modal4 MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy experiments/slop-hard-negatives/modal_hn.py

The volume is the slop-detector note's (`slop-detector` on modal4): round 0 is its `runs/b1b-1.7bB` adapter and the
training data is its `data/b1b` (data v2b). Everything new lives under `hn/` on that volume.

- build_pools (CPU): FineWeb shards from dumps crawled before November 2022 -> mining pool, held-out web, dev, plus
  modern-topic seeds from a 2025 dump; split by host so no site is in two pools.
- train_round (H100): the slop-detector trainer (04_train.py) unchanged, on data v2b + the pages added so far; then
  `evaluate` in the same container.
- evaluate (H100): scores the frozen evaluation file and extracts the head input for the interp sets.
- score_pool (H100): scores one shard of the mining pool with a run's adapter (the loop runs four in parallel).
"""

from __future__ import annotations

import time

import modal

app = modal.App("slop-hard-negatives")
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
            "HF_HUB_DISABLE_XET": "1",  # xet keeps log files open on the volume, which blocks reload (lesson A19)
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
HN = "/vol/hn"


# ------------------------------------------------------------------ pools
def _host(url: str) -> str:
    from urllib.parse import urlsplit

    h = (urlsplit(url).hostname or "").lower()
    return h.removeprefix("www.")


def _bucket(host: str) -> int:
    import hashlib

    return int(hashlib.md5(host.encode()).hexdigest(), 16) % 10


@app.function(image=image, volumes={VOL: volume}, cpu=8, memory=32768, timeout=3 * 60 * 60)
def build_pools(old_dumps: list[str], new_dump: str, quota: dict, cutoff: str = "2022-11-01") -> dict:
    """Sample pages from one shard per dump. quota: per old dump {"pool", "heldout", "dev"}; plus {"modern"} for the
    new dump. Pages >= 80 words, deduplicated by URL and by normalised text; text cut to 4,000 characters."""
    import hashlib
    import json
    import os
    import random
    import re

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(f"{HN}/pools", exist_ok=True)
    seen_url, seen_txt = set(), set()
    out = {"pool": [], "heldout": [], "dev": [], "modern": []}
    stats = {}
    ws = re.compile(r"\s+")

    def take(dump: str, want: dict, is_new: bool) -> dict:
        path = hf_hub_download(
            "HuggingFaceFW/fineweb", f"data/{dump}/000_00000.parquet", repo_type="dataset", cache_dir="/tmp/hf"
        )
        pf = pq.ParquetFile(path)
        groups = list(range(pf.num_row_groups))
        random.Random(dump).shuffle(groups)  # spread the sample over the shard
        got = {k: 0 for k in want}
        st = {"rows_read": 0, "short": 0, "dup": 0, "late": 0, "dates": [None, None]}
        for g in groups:
            if all(got[k] >= want[k] for k in want):
                break
            t = pf.read_row_group(g, columns=["text", "id", "dump", "url", "date"]).to_pylist()
            for r in t:
                st["rows_read"] += 1
                d = str(r["date"])
                if not is_new and d >= cutoff:
                    st["late"] += 1
                    continue
                words = len(r["text"].split())
                if words < 80:
                    st["short"] += 1
                    continue
                h = hashlib.md5(ws.sub(" ", r["text"][:2000].lower()).encode()).hexdigest()
                if r["url"] in seen_url or h in seen_txt:
                    st["dup"] += 1
                    continue
                host = _host(r["url"])
                b = _bucket(host)
                key = "modern" if is_new else ("heldout" if b == 0 else "dev" if b == 1 else "pool")
                if got[key] >= want[key]:
                    continue
                seen_url.add(r["url"])
                seen_txt.add(h)
                got[key] += 1
                lo, hi = st["dates"]
                st["dates"] = [min(lo or d, d), max(hi or d, d)]
                out[key].append(
                    {
                        "id": r["id"],
                        "text": r["text"][:4000],
                        "words": words,
                        "url": r["url"],
                        "host": host,
                        "bucket": b,
                        "date": d,
                        "dump": r["dump"],
                    }
                )
        os.remove(path)
        st["got"] = got
        print(dump, st, flush=True)
        return st

    for d in old_dumps:
        stats[d] = take(d, {k: quota[k] for k in ("pool", "heldout", "dev")}, False)
    stats[new_dump] = take(new_dump, {"modern": quota["modern"]}, True)
    for k, rows in out.items():
        random.Random(k).shuffle(rows)
        with open(f"{HN}/pools/{k}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    volume.commit()
    return {"stats": stats, "sizes": {k: len(v) for k, v in out.items()}}


def _reload() -> None:
    """See files other containers committed; a busy volume (open files) is not fatal, the files we need are ours."""
    try:
        volume.reload()
    except RuntimeError as e:
        print(f"volume reload skipped: {e}", flush=True)


# ------------------------------------------------------------------ detector helpers
def _load(run: str):
    """A run on the volume (its config.json names the base and head) with its adapter: (tok, model, cfg)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/sd")
    import train as tr
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(VOL) / run
    cfg = json.loads((path / "config.json").read_text())
    tok, model = tr.load(cfg["base"], "cuda", num_labels=4 if cfg.get("head") == "buckets" else 1)
    set_peft_model_state_dict(model, load_file(str(path / "adapter" / "adapter_model.safetensors")))
    model.eval()
    return tok, model, cfg


def _score(tok, model, texts: list[str]):
    import sys

    sys.path.insert(0, "/root/sd")
    import train as tr
    from common import empty_fences

    rows = tr.encode([{"text": empty_fences(t)} for t in texts], tok)
    return tr.predict(model, rows, tok, "cuda", max_tokens=16384)


def _features(tok, model, texts: list[str]):
    """The head's input at the token the head reads (the last non-pad token), the head's weights, and the model's own
    score, so the score can be recomputed from the state (the check that the readout is the right one)."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root/sd")
    import train as tr
    from common import empty_fences

    rows = tr.encode([{"text": empty_fences(t)} for t in texts], tok)
    head = model.base_model.model.score
    W = [p for n, p in head.named_parameters() if "modules_to_save" in n and n.endswith("weight")][0]
    grab = {}
    hook = head.register_forward_hook(lambda m, inp, out: grab.__setitem__("h", inp[0]))
    H = np.zeros((len(rows), W.shape[1]), dtype=np.float32)
    S = np.zeros(len(rows), dtype=np.float32)
    with torch.no_grad():
        for idx in tr.batches(rows, 16384, shuffle=False):
            enc = tr.collate(rows, idx, tok, "cuda")
            logits = model(**enc).logits.float()
            ids = enc["input_ids"]
            pos = torch.arange(ids.shape[1], device=ids.device)
            last = (pos * (ids != tok.pad_token_id)).argmax(-1)
            h = grab["h"][torch.arange(len(idx), device=ids.device), last].float()
            H[idx] = h.cpu().numpy()
            p = torch.softmax(logits, -1)
            S[idx] = ((p * torch.arange(4, device=p.device)).sum(-1) / 3).cpu().numpy()
    hook.remove()
    return H, S, W.detach().float().cpu().numpy()


@app.function(image=train_image, gpu="H100", volumes={VOL: volume}, timeout=60 * 60)
def evaluate(run: str, out: str, evals: str = "hn/evals.jsonl", interp: str = "hn/interp.jsonl") -> dict:
    """Score `evals` and extract features for `interp` with `run`; write under `out` (all volume paths)."""
    return _evaluate(run, out, evals, interp)


def _evaluate(run: str, out: str, evals: str, interp: str) -> dict:
    import json
    import os
    from collections import defaultdict

    import numpy as np

    t0 = time.time()
    _reload()
    tok, model, _ = _load(run)
    os.makedirs(f"{VOL}/{out}", exist_ok=True)
    res = {}
    if os.path.exists(f"{VOL}/{evals}"):
        rows = [json.loads(x) for x in open(f"{VOL}/{evals}")]
        s = _score(tok, model, [r["text"] for r in rows])
        by = defaultdict(list)
        for r, x in zip(rows, s):
            by[r["set"]].append(round(float(x), 6))
        with open(f"{VOL}/{out}/eval_scores.json", "w") as f:
            json.dump(by, f)
        res.update(eval_rows=len(rows), eval_seconds=round(time.time() - t0))
    if os.path.exists(f"{VOL}/{interp}"):
        irows = [json.loads(x) for x in open(f"{VOL}/{interp}")]
        H, S, W = _features(tok, model, [r["text"] for r in irows])
        np.savez_compressed(f"{VOL}/{out}/feats.npz", h=H.astype(np.float32), score=S, W=W)
        res["interp_rows"] = len(irows)
    volume.commit()
    res["seconds"] = round(time.time() - t0)
    return res


@app.function(image=train_image, gpu="H100", volumes={VOL: volume}, timeout=60 * 60, max_containers=8)
def score_pool(run: str, out: str, shard: int, shards: int, src: str = "hn/pools/pool.jsonl") -> dict:
    """Score rows shard::shards of a pool file with `run`; save float32 scores to <out>/pool_scores_<shard>.npy."""
    import json
    import os

    import numpy as np

    t0 = time.time()
    rows = [json.loads(x) for i, x in enumerate(open(f"{VOL}/{src}")) if i % shards == shard]
    tok, model, _ = _load(run)
    s = _score(tok, model, [r["text"] for r in rows]).astype(np.float32)
    os.makedirs(f"{VOL}/{out}", exist_ok=True)
    np.save(f"{VOL}/{out}/pool_scores_{shard}.npy", s)
    volume.commit()
    return {"n": len(rows), "seconds": round(time.time() - t0)}


@app.function(
    image=train_image, gpu="H100", volumes={VOL: volume}, timeout=150 * 60, retries=modal.Retries(max_retries=1)
)
def train_round(
    name: str,
    added: list[str],
    seed: int,
    extra: list[str],
    evals: str = "hn/evals.jsonl",
    interp: str = "hn/interp.jsonl",
) -> dict:
    """Train `name` from the base model with the b1b recipe on data v2b (volume data/b1b) + the rows of every file in
    `added` (volume paths, JSONL in the training format); then evaluate. Resumes from its checkpoint on a retry."""
    import json
    import os
    import shutil
    import sys

    sys.path.insert(0, "/root/sd")
    import train as tr

    t0 = time.time()
    _reload()
    data = f"{HN}/data/{name}"
    run = f"{HN}/runs/{name}"
    os.makedirs(data, exist_ok=True)
    n_added = 0
    with open(f"{data}/train.jsonl", "w") as f:
        with open(f"{VOL}/data/b1b/train.jsonl") as src:
            shutil.copyfileobj(src, f)
        for a in added:
            for line in open(f"{VOL}/{a}"):
                if line.strip():
                    f.write(line if line.endswith("\n") else line + "\n")
                    n_added += 1
    shutil.copy(f"{VOL}/data/b1b/val.jsonl", f"{data}/val.jsonl")
    sys.argv = [
        "train.py",
        "--base",
        "Qwen/Qwen3-1.7B-Base",
        "--data",
        data,
        "--out",
        run,
        "--micro-tokens",
        "6144",
        "--epochs",
        "3.79",
        "--head",
        "buckets",
        "--seed",
        str(seed),
        "--ckpt-minutes",
        "20",
        "--resume",
        *extra,
    ]
    tr.ON_CHECKPOINT = volume.commit
    tr.main()
    volume.commit()
    train_s = time.time() - t0
    ev = _evaluate(f"hn/runs/{name}", f"hn/runs/{name}", evals, interp)
    cfg = json.loads(open(f"{run}/config.json").read())
    return {
        "name": name,
        "added_rows": n_added,
        "train_seconds": round(train_s),
        "config": cfg,
        **ev,
        "total_seconds": round(time.time() - t0),
    }
