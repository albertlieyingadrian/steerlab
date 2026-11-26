"""Amendment 8: Claim 1 with the post's exact rule and benchmark suite, on Qwen3.6-35B-A3B and on OLMoE-1B-7B
(runs in the Modal container).

Conditions: base, top-K (their rule: per token and layer, the K routed neurons with the largest |router-weighted
activation|; K = 320 of 4,096 on the 35B, 640 of 8,192 on OLMoE, both 7.8%), random-K. Shared experts untouched.
Metrics: lm-evaluation-harness zero-shot accuracy on ARC-Challenge, ARC-Easy, WinoGrande, HellaSwag (first 500 items
each) and MMLU (20 per subject), with per-item correctness kept for paired intervals; loss and KL on our 300 eval docs.
"""

# ruff: noqa: E501, RUF005  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import json
import time
from pathlib import Path

import neuronkit as nk
import torch

OUT = Path("/vol")
MODELS = {
    "qwen36": ("Qwen/Qwen3.6-35B-A3B", 320, 8),
    "olmoe": ("allenai/OLMoE-1B-7B-0125", 640, 16),
}
LIMITS = {"qwen36": (250, 10), "olmoe": (500, 20)}  # items per task, MMLU items per subject (35B cut for cost)
HARNESS_CONDS = {"qwen36": ("base", "top320"), "olmoe": ("base", "top640", "random640")}
TASKS = ["arc_challenge", "arc_easy", "winogrande", "hellaswag"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(key):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hub = MODELS[key][0]
    tok = AutoTokenizer.from_pretrained(hub)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # for the 35B this loads the text-only Qwen3_5MoeForCausalLM (transformers maps the multimodal checkpoint to it)
    model = AutoModelForCausalLM.from_pretrained(hub, dtype=torch.bfloat16, device_map="cuda")
    return model.eval(), tok


def harness(model, tok, bs, limit_items, smoke, mmlu_limit=20):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, backend="causal")
    out = {}
    for tasks, limit in ((TASKS, limit_items), (["mmlu"], 1 if smoke else mmlu_limit)):
        r = lm_eval.simple_evaluate(
            model=lm, tasks=tasks, num_fewshot=0, limit=limit, log_samples=True, bootstrap_iters=0
        )
        for name, res in r["results"].items():
            out[name] = {
                k.split(",")[0]: v for k, v in res.items() if isinstance(v, int | float) and not k.startswith("alias")
            }
        for name, samples in r.get("samples", {}).items():
            out.setdefault(name, {})["items"] = [int(s.get("acc", 0) > 0) for s in samples]
    return out


def run(key: str, smoke: bool = False):
    out = OUT / ("smoke35_" + key if smoke else key)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(key)
    k, bs = MODELS[key][1], MODELS[key][2]
    sites = nk.sites(model)
    log(
        f"{key}: {len(sites)} MoE layers, {sites[0].n_experts} experts x {sites[0].n_units}, top_k {sites[0].top_k}; keep {k}"
    )
    lm_model = model
    conds = {
        "base": None,
        f"top{k}": lambda: nk.keep_topk(model, k=k),
        f"random{k}": lambda: nk.keep_topk(model, k=k, mode="random"),
    }
    limit, mmlu_limit = (4, 1) if smoke else LIMITS[key]
    res_path = out / "harness.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name, make in conds.items():
        if name in results or name not in HARNESS_CONDS[key]:
            continue
        t0 = time.time()
        if make is None:
            results[name] = harness(lm_model, tok, bs, limit, smoke, mmlu_limit)
        else:
            with make():
                results[name] = harness(lm_model, tok, bs, limit, smoke, mmlu_limit)
        res_path.write_text(json.dumps(results))
        log(
            f"{key} {name}: "
            + ", ".join(
                f"{t} {v.get('acc', float('nan')):.3f}" for t, v in results[name].items() if t in TASKS + ["mmlu"]
            )
            + f" ({time.time() - t0:.0f} s)"
        )
    # our loss and KL on the 300 eval docs
    sp = out / "sparsity35.json"
    if not sp.exists():
        docs = [d["text"] for d in json.loads((OUT / "docs.json").read_text())][2000 : 2016 if smoke else 2300]
        pts = nk.sparsity_curve(
            lm_model, nk.batches(tok, docs, 256, bs), {n: (lambda m, f=f: f()) for n, f in conds.items() if f}
        )
        sp.write_text(json.dumps({n: {"loss": p.loss, "kl": p.kl, "n_tokens": p.n_tokens} for n, p in pts.items()}))
        log(f"{key} loss/KL: " + ", ".join(f"{n} {p.mean_loss:.3f}/{p.mean_kl:.3f}" for n, p in pts.items()))
    log("done", key)
