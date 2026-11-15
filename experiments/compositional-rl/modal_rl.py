"""
GRPO on data-structure tasks, on Modal: vLLM samples the rollouts, a hand-written GRPO step updates a LoRA adapter.

COSTS MONEY (one H100 per run; about $4/hour). Run from the repo root, for example:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/compositional-rl/modal_rl.py::screen --model Qwen/Qwen3-1.7B
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal run experiments/compositional-rl/modal_rl.py::train_many --config configs/l2.json

One training step (the recipe of experiments/pigment-gym/grpo_tools.py, with vLLM doing the sampling):

    32 prompts ──► vLLM samples 8 replies each, with the current LoRA ──► reward 1 if the boxed answer is right, else 0
                                                                                   │
    LoRA update: advantage = reward - mean(reward of its group of 8) ◄─────────────┘
    loss = -(advantage x log-prob of each reply token), summed and divided by all reply tokens in the batch (no KL)

After the update the adapter is saved and vLLM loads it under a new id, so the next step samples from the new policy.
Training batches are RIGHT-padded and masked with torch.where (left padding gave NaN logits in rl-hack-monitor).
Arms differ only in their data: `compound_frac` is the chance each prompt is a compound task rather than an atomic
one (0 = RL-Atomic, 1 = RL-Compound, in between = the mixtures).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
app = modal.App("compositional-rl")
volume = modal.Volume.from_name("compositional-rl", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "peft==0.17.1", "hf_transfer==0.1.9", "numpy")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(HERE / "tasks.py", "/root/tasks.py")
)
GPU = "H100"


def chat(tokenizer, model_name, prompt):
    kwargs = {"enable_thinking": False} if "Qwen3" in model_name else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kwargs
    )


def eval_tasks(spec, base_seed=1_000_000):
    """Fixed held-out items. spec: list of [level, domain, kind, length, n]. Evaluation seeds start at 1,000,000;
    training seeds are drawn below 1,000,000, so no evaluation prompt is trained on."""
    sys.path.insert(0, "/root")
    from tasks import make_task

    return [
        make_task(domain, level, kind, length, base_seed + i).to_json()
        for level, domain, kind, length, n in spec
        for i in range(n)
    ]


def generate(llm, tokenizer, model_name, prompts, max_tokens, temperature, n=1, lora=None):
    from vllm import SamplingParams

    params = SamplingParams(
        n=n, temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=max_tokens
    )
    texts = [chat(tokenizer, model_name, p) for p in prompts]
    return llm.generate(texts, params, lora_request=lora, use_tqdm=False)


def score_outputs(tasks, outs):
    sys.path.insert(0, "/root")
    from tasks import reward

    rows = []
    for t, o in zip(tasks, outs):
        c = o.outputs[0]
        rows.append(
            {
                "level": t["level"],
                "domain": t["domain"],
                "kind": t["kind"],
                "length": t["length"],
                "seed": t["seed"],
                "meta": t["meta"],
                "correct": reward(t, c.text),
                "n_tokens": len(c.token_ids),
                "truncated": c.finish_reason == "length",
                "text": c.text,
            }
        )
    return rows


def make_llm(model_name, enable_lora, gpu_util, max_len):
    from vllm import LLM

    kw = {"enable_lora": True, "max_lora_rank": 64, "max_loras": 2} if enable_lora else {}
    return LLM(
        model=model_name,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        seed=0,
        **kw,
    )


def summarise(rows):
    from collections import defaultdict

    acc, trunc, toks = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        k = (r["level"], r["kind"], r["domain"], r["length"])
        acc[k].append(r["correct"])
        trunc[k].append(r["truncated"])
        toks[k].append(r["n_tokens"])
    for k in sorted(acc):
        a = acc[k]
        print(
            f"L{k[0]} {k[1]:8s} {k[2]:8s} len {k[3]:2d}: acc {sum(a) / len(a):.2f}  trunc {sum(trunc[k])}/{len(a)}"
            f"  tokens {sum(toks[k]) / len(a):.0f}"
        )


# ---------------------------------------------------------------- screen (untrained models, before the pre-registration)


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={VOL: volume})
def screen_remote(model_name: str, spec: list, max_tokens: int, tag: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    llm = make_llm(model_name, False, 0.85, 8192)
    tasks = eval_tasks(spec, base_seed=2_000_000)  # screen items are disjoint from the evaluation items too
    t0 = time.time()
    rows = score_outputs(tasks, generate(llm, tok, model_name, [t["prompt"] for t in tasks], max_tokens, 0.0))
    print(f"{len(rows)} generations in {time.time() - t0:.0f}s")
    return rows


@app.local_entrypoint()
def screen(model: str = "Qwen/Qwen3-1.7B", max_tokens: int = 4096, n: int = 25, tag: str = ""):
    spec = []
    for kind in ["atomic", "compound"]:
        for level, domains in [
            (1, ["array", "heap", "bst", "hashmap", "bloom"]),
            (2, ["array", "heap", "bst", "hashmap", "bloom"]),
            (3, ["segment"]),
        ]:
            for d in domains:
                for L in [2, 4, 6, 8, 12]:
                    spec.append([level, d, kind, L, n if level != 1 else max(10, n // 2)])
    tag = tag or model.split("/")[-1]
    rows = screen_remote.remote(model, spec, max_tokens, tag)
    out = HERE / "results" / "screen" / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    summarise(rows)


# ---------------------------------------------------------------- training


def sample_train_task(rng, cfg):
    sys.path.insert(0, "/root")
    from tasks import make_task

    kind = "compound" if rng.random() < cfg["compound_frac"] else "atomic"
    level = cfg["level"]
    domain = "segment" if level == 3 else rng.choice(cfg["domains"])
    lengths = cfg["train_lengths_compound"] if kind == "compound" else cfg["train_lengths_atomic"]
    return make_task(domain, level, kind, rng.choice(lengths), rng.randrange(1_000_000)).to_json()


def token_logprobs(model, ids, attn):
    import torch

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    picked = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()
    return picked - torch.logsumexp(logits.float(), dim=-1)


def run_one(cfg):
    """One GRPO run, then the held-out evaluation of its final adapter. Everything goes to /vol/runs/<name>/."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, "/root")
    from tasks import reward

    name, model_name = cfg["name"], cfg["model"]
    run_dir = Path(VOL) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    llm = make_llm(model_name, True, cfg.get("gpu_util", 0.4), cfg.get("max_model_len", 8192))
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=2 * cfg["lora_rank"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapter weights and Adam state in float32; the frozen base stays bfloat16
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)

    def save_adapter(tag):
        d = run_dir / f"adapter_{tag}"
        model.save_pretrained(str(d))
        return d

    lora = LoRARequest(f"{name}-0", 1, str(save_adapter("step0")))
    curve = []
    t_start = time.time()
    for step in range(1, cfg["steps"] + 1):
        t0 = time.time()
        batch = [sample_train_task(rng, cfg) for _ in range(cfg["prompts_per_step"])]
        outs = generate(
            llm, tok, model_name, [t["prompt"] for t in batch], cfg["max_tokens"], 1.0, n=cfg["group"], lora=lora
        )
        seqs, advs, rewards_all, kinds = [], [], [], []
        n_trunc, n_tok = 0, 0
        for t, o in zip(batch, outs):
            rs = [reward(t, c.text) for c in o.outputs]
            mean_r = sum(rs) / len(rs)
            rewards_all.extend(rs)
            kinds.extend([t["kind"]] * len(rs))
            prompt_ids = list(o.prompt_token_ids)
            for c, r in zip(o.outputs, rs):
                n_trunc += c.finish_reason == "length"
                n_tok += len(c.token_ids)
                if r == mean_r:
                    continue  # zero advantage: no gradient, so skip the compute
                seqs.append((prompt_ids, list(c.token_ids)))
                advs.append(r - mean_r)
        t_gen = time.time() - t0
        model.train()
        opt.zero_grad(set_to_none=True)
        # normaliser: all reply tokens in the batch, including zero-advantage replies (as if they were computed)
        loss_val, gnorm = 0.0, 0.0
        if seqs:
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
            mb = cfg["micro_batch"]
            for s in range(0, len(order), mb):
                idx = order[s : s + mb]
                maxlen = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
                ids = torch.full((len(idx), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(idx), maxlen), dtype=torch.long)
                lmask = torch.zeros((len(idx), maxlen), dtype=torch.bool)
                for row, i in enumerate(idx):
                    p, c = seqs[i]
                    ids[row, : len(p) + len(c)] = torch.tensor(p + c)
                    attn[row, : len(p) + len(c)] = 1
                    lmask[row, len(p) : len(p) + len(c)] = True  # right padding: real tokens first
                ids, attn, lmask = ids.cuda(), attn.cuda(), lmask.cuda()
                adv = torch.tensor([advs[i] for i in idx], device="cuda").unsqueeze(1)
                logp = token_logprobs(model, ids, attn)
                per_tok = torch.where(lmask[:, 1:], -adv * logp, torch.zeros_like(logp))
                loss = per_tok.sum() / max(n_tok, 1)
                loss.backward()
                loss_val += float(loss.detach())
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if gnorm != gnorm or gnorm == float("inf"):
                opt.zero_grad(set_to_none=True)
                print(f"step {step}: non-finite grad norm, update skipped")
            else:
                opt.step()
        keep = step % cfg.get("save_every", 20) == 0 or step == cfg["steps"]
        adapter = save_adapter(f"step{step}" if keep else "cur")
        lora = LoRARequest(f"{name}-{step}", step + 1, str(adapter))
        ra = [r for r, k in zip(rewards_all, kinds) if k == "atomic"]
        rc = [r for r, k in zip(rewards_all, kinds) if k == "compound"]
        rec = {
            "step": step,
            "reward": sum(rewards_all) / len(rewards_all),
            "reward_atomic": sum(ra) / len(ra) if ra else None,
            "reward_compound": sum(rc) / len(rc) if rc else None,
            "n_update_seqs": len(seqs),
            "loss": loss_val,
            "grad_norm": gnorm,
            "trunc_frac": n_trunc / len(rewards_all),
            "mean_tokens": n_tok / len(rewards_all),
            "t_gen": round(t_gen, 1),
            "t_step": round(time.time() - t0, 1),
        }
        curve.append(rec)
        print(name, json.dumps(rec), flush=True)
        (run_dir / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in curve))
        volume.commit()
    print(f"{name}: training done in {(time.time() - t_start) / 60:.1f} min")
    rows = []
    if cfg.get("eval_spec"):
        rows = evaluate_with(llm, tok, model_name, cfg["eval_spec"], lora, cfg.get("eval_max_tokens", 4096))
        (run_dir / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        volume.commit()
    print(f"{name}: all done in {(time.time() - t_start) / 60:.1f} min")
    return {"name": name, "curve": curve, "eval": rows}


def evaluate_with(llm, tok, model_name, spec, lora, max_tokens):
    tasks = eval_tasks(spec)
    return score_outputs(
        tasks, generate(llm, tok, model_name, [t["prompt"] for t in tasks], max_tokens, 0.0, lora=lora)
    )


@app.function(image=image, gpu=GPU, timeout=4 * 60 * 60, volumes={VOL: volume}, max_containers=4)
def train_remote(cfg: dict):
    return run_one(cfg)


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={VOL: volume})
def eval_base_remote(model_name: str, spec: list, max_tokens: int, tag: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    llm = make_llm(model_name, False, 0.85, 8192)
    rows = evaluate_with(llm, tok, model_name, spec, None, max_tokens)
    d = Path(VOL) / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    volume.commit()
    return {"name": tag, "eval": rows}


# ---------------------------------------------------------------- the step test and adapter swaps (mechanisms)


def step_tasks(spec, base_seed=1_000_000):
    """Split each compound evaluation item into its steps: step t is an atomic prompt on the TRUE state after t-1 ops.

    Greedy decoding is deterministic, so a scaffold that chains atomic calls on the model's own outputs gets exactly
    these prompts while it is still right; it succeeds if and only if every step here is right. The share of items
    with all steps right is therefore the scaffold's accuracy, with no extra calls."""
    sys.path.insert(0, "/root")
    from tasks import atomic_from_state

    out = []
    for item in eval_tasks(spec, base_seed):
        for t, op in enumerate(item["ops"]):
            out.append(
                {
                    "domain": item["domain"],
                    "level": item["level"],
                    "kind": "step",
                    "length": item["length"],
                    "seed": item["seed"],
                    "meta": {
                        "step": t,
                        "op": op[0],
                        "n_steps": len(item["ops"]),
                        "prev_op": item["ops"][t - 1][0] if t else None,
                    },
                    "prompt": atomic_from_state(item["domain"], item["states"][t], op),
                    "answer": item["states"][t + 1],
                }
            )
    return out


def mix_adapters(a_dir, b_dir, out_dir, layers_from_b):
    """A LoRA adapter whose layers in `layers_from_b` come from adapter B and all others from adapter A."""
    import re
    import shutil

    from safetensors.torch import load_file, save_file

    a = load_file(str(Path(a_dir) / "adapter_model.safetensors"))
    b = load_file(str(Path(b_dir) / "adapter_model.safetensors"))
    mixed = {}
    for k, v in a.items():
        m = re.search(r"layers\.(\d+)\.", k)
        mixed[k] = b[k] if (m and int(m.group(1)) in layers_from_b) else v
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_file(mixed, str(Path(out_dir) / "adapter_model.safetensors"))
    shutil.copy(Path(a_dir) / "adapter_config.json", Path(out_dir) / "adapter_config.json")


@app.function(image=image, gpu=GPU, timeout=2 * 60 * 60, volumes={VOL: volume})
def mechanism_remote(model_name: str, jobs: list, max_tokens: int):
    """jobs: list of {"tag", "adapter" (run name or null for the base model), "mix" (optional: {"a", "b", "layers"}),
    "compound_spec" (items to score in context), "step_spec" (items to split into steps)}."""
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(model_name)
    llm = make_llm(model_name, True, 0.85, 8192)
    results = []
    for j, job in enumerate(jobs):
        lora = None
        if job.get("mix"):
            mx = job["mix"]
            out = Path(VOL) / "mixed" / job["tag"]
            mix_adapters(
                Path(VOL) / "runs" / mx["a"] / mx["a_ckpt"],
                Path(VOL) / "runs" / mx["b"] / mx["b_ckpt"],
                out,
                set(mx["layers"]),
            )
            lora = LoRARequest(job["tag"], 1000 + j, str(out))
        elif job.get("adapter"):
            lora = LoRARequest(job["tag"], 1000 + j, str(Path(VOL) / "runs" / job["adapter"] / job["ckpt"]))
        rows = []
        if job.get("compound_spec"):
            rows += evaluate_with(llm, tok, model_name, job["compound_spec"], lora, max_tokens)
        if job.get("step_spec"):
            st = step_tasks(job["step_spec"])
            outs = generate(llm, tok, model_name, [t["prompt"] for t in st], max_tokens, 0.0, lora=lora)
            rows += score_outputs(st, outs)
        for r in rows:
            r["tag"] = job["tag"]
        d = Path(VOL) / "mechanism"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{job['tag']}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        volume.commit()
        results.append({"tag": job["tag"], "rows": rows})
        print(job["tag"], "done", len(rows), flush=True)
    return results


@app.local_entrypoint()
def mechanism(jobs: str, model: str = "Qwen/Qwen3-1.7B", max_tokens: int = 4096, split: int = 1):
    """jobs: a JSON file with a list of jobs (see mechanism_remote). split: how many GPUs to spread them over."""
    job_list = json.loads((HERE / jobs).read_text())
    chunks = [job_list[i::split] for i in range(split)]
    out_dir = HERE / "results" / "mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in mechanism_remote.starmap([(model, c, max_tokens) for c in chunks if c], return_exceptions=True):
        if isinstance(res, Exception):
            print("FAILED:", repr(res))
            continue
        for r in res:
            (out_dir / f"{r['tag']}.jsonl").write_text("\n".join(json.dumps(x) for x in r["rows"]))
            acc = {}
            for x in r["rows"]:
                acc.setdefault(x["kind"], []).append(x["correct"])
            print(r["tag"], {k: round(sum(v) / len(v), 3) for k, v in acc.items()})


def load_runs(config, only):
    spec = json.loads((HERE / config).read_text())
    runs = []
    for r in spec["runs"]:
        if only and r["name"] not in only.split(","):
            continue
        cfg = dict(spec["common"])
        cfg.update(r)
        runs.append(cfg)
    return spec, runs


@app.local_entrypoint()
def train_many(config: str, only: str = "", steps: int = 0, no_eval: bool = False):
    """config: JSON {"common": {...}, "runs": [{"name": ..., overrides}, ...]}. Runs in parallel, one H100 each;
    results land in results/runs/<name>/. --steps and --no-eval override for smoke tests."""
    _, runs = load_runs(config, only)
    for r in runs:
        if steps:
            r["steps"] = steps
        if no_eval:
            r["eval_spec"] = []
    print(f"{len(runs)} runs: {[r['name'] for r in runs]}")
    for res in train_remote.map(runs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("RUN FAILED:", repr(res))
            continue
        d = HERE / "results" / "runs" / res["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in res["curve"]))
        if res["eval"]:
            (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in res["eval"]))
        acc = {}
        for row in res["eval"]:
            acc.setdefault(row["kind"], []).append(row["correct"])
        print(res["name"], {k: round(sum(v) / len(v), 3) for k, v in acc.items()})


@app.local_entrypoint()
def eval_base(config: str, tag: str = "base"):
    spec, _ = load_runs(config, "")
    c = spec["common"]
    res = eval_base_remote.remote(c["model"], c["eval_spec"], c.get("eval_max_tokens", 4096), tag)
    d = HERE / "results" / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in res["eval"]))
    summarise(res["eval"])
