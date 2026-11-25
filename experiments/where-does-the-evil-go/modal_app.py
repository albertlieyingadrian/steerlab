"""GPU worker for the where-does-the-evil-go note: Qwen2.5-7B-Instruct on a Modal H100. COSTS MONEY; the drivers in
run.py check the budget before every launch.

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/where-does-the-evil-go/modal_app.py

Everything reads and writes the volume `where-does-the-evil-go` (mounted at /vol):

    sample(model, jobs)   replies for each job, with optional steering, written to /vol/replies/<model>/<job id>.json;
                          with job["acts"], also the response-average activation at every hidden-state index (fp16,
                          [n, 29, 3584]) to <job id>.npy
    finetune(cfg)         a LoRA fine-tune (the paper's recipe), optionally with preventative steering; /vol/adapters/<name>
    gain(model, rows, directions)   inject each direction at block 19 on fixed text; KL of the next-token distribution
                          and how far the push survives through later layers

Layer convention: the post's "layer 20" is hidden_states[20], the output of decoder block 19 (model.model.layers[19]).
Activations are stored by hidden-state index (0 = embeddings), so index 20 is the post's layer 20.

Steering at inference uses the pipeline's "response" mode: the push is added at the LAST position of every forward
pass, i.e. the last prompt token during prefill and each generated token after it. Sampling matches what the pipeline
ran: temperature 1, top_p 1, and the model's generation_config defaults that it left on (top_k 20,
repetition_penalty 1.05).
"""

import json
import math
import time
from pathlib import Path

import modal

BASE = "Qwen/Qwen2.5-7B-Instruct"
HOOK_BLOCK = 19  # output of block 19 = hidden_states[20] = the post's layer 20
app = modal.App("where-does-the-evil-go")
volume = modal.Volume.from_name("where-does-the-evil-go", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.20.2",
        "accelerate>=1.10",
        "peft==0.21.2",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
    )
    .env(
        {"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
)
GPU_KW = {"image": image, "gpu": "H100", "volumes": {"/vol": volume}, "secrets": [hf]}


def _load(model_name):
    """(model, tokenizer); model_name 'base' or a fine-tune name whose LoRA adapter is in /vol/adapters/<name>."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    if model_name != "base":
        from peft import PeftModel

        volume.reload()
        model = PeftModel.from_pretrained(model, f"/vol/adapters/{model_name}").merge_and_unload()
    model.eval()
    return model, tok


def _prompt(tok, system, question):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _block(model):
    return model.model.layers[HOOK_BLOCK]


class _LastPositionPush:
    """Add `push` to block 19's output at the last position of every forward pass (the pipeline's 'response' mode)."""

    def __init__(self, model, push):
        self.model, self.push, self.handle = model, push, None

    def hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] += self.push
        return (h, *output[1:]) if isinstance(output, tuple) else h

    def __enter__(self):
        if self.push is not None:
            self.handle = _block(self.model).register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


def _vector(model, values, coef):
    import torch

    if values is None or coef == 0:
        return None
    return coef * torch.tensor(values, dtype=torch.bfloat16, device=model.device)


def _generate(model, tok, prompts, push, *, max_new_tokens, seed, batch):
    import torch

    replies, n_tokens, hit_cap = [None] * len(prompts), [0] * len(prompts), [False] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
    eos_cfg = model.generation_config.eos_token_id
    eos = set(eos_cfg if isinstance(eos_cfg, list) else [eos_cfg])
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        enc = tok([prompts[i] for i in chunk], return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed + start)
        with torch.no_grad(), _LastPositionPush(model, push):
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=20, repetition_penalty=1.05,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)  # fmt: skip
        width = enc["input_ids"].shape[1]
        for row, i in enumerate(chunk):
            ids = out[row, width:].tolist()
            n = next((k for k, t in enumerate(ids) if t in eos), len(ids))
            n_tokens[i], hit_cap[i] = n, n >= max_new_tokens
            replies[i] = tok.decode(ids[:n], skip_special_tokens=True).strip()
    return replies, n_tokens, hit_cap


def _response_acts(model, tok, prompts, responses, batch=16):
    """Mean over the reply's tokens of every hidden state (index 0..28), fp16 [n, 29, hidden]. Prompt and reply are
    tokenized separately and joined, so the boundary is exact (as in the persona-vectors note)."""
    import numpy as np
    import torch

    n_hidden = model.config.num_hidden_layers + 1
    out = np.zeros((len(prompts), n_hidden, model.config.hidden_size), dtype=np.float16)
    pad = tok.pad_token_id
    for start in range(0, len(prompts), batch):
        idx = list(range(start, min(start + batch, len(prompts))))
        p_ids = [tok(prompts[i], add_special_tokens=False)["input_ids"] for i in idx]
        r_ids = [tok(responses[i] or " ", add_special_tokens=False)["input_ids"] for i in idx]
        seqs = [p + r for p, r in zip(p_ids, r_ids, strict=True)]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for row, s in enumerate(seqs):
            ids[row, width - len(s) :] = torch.tensor(s)
            mask[row, width - len(s) :] = 1
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad():
            hs = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), position_ids=pos.cuda(),
                             output_hidden_states=True).hidden_states  # fmt: skip
        assert len(hs) == n_hidden, len(hs)
        for row, i in enumerate(idx):
            n_r = len(r_ids[row])
            out[i] = torch.stack([h[row, width - n_r :].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16)
        del hs
    return out


@app.function(**GPU_KW, timeout=4 * 60 * 60)
def sample(model_name, jobs, batch=200):
    """jobs: [{"id", "rows": [{"system", "question", ...}], "vector": [3584] or None, "coef", "max_new_tokens", "seed",
    "acts": bool}]. Skips a job whose output already exists. Returns per-job seconds."""
    import numpy as np

    t0 = time.time()
    model, tok = _load(model_name)
    folder = Path(f"/vol/replies/{model_name}")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"loaded {model_name} in {time.time() - t0:.0f} s", flush=True)
    done = {}
    for job in jobs:
        path = folder / f"{job['id']}.json"
        if path.exists() and (not job.get("acts") or (folder / f"{job['id']}.npy").exists()):
            print(f"  {job['id']}: exists, skipped", flush=True)
            continue
        t1 = time.time()
        rows = job["rows"]
        prompts = [_prompt(tok, r.get("system"), r["question"]) for r in rows]
        push = _vector(model, job.get("vector"), job.get("coef", 0.0))
        replies, n_tok, cap = _generate(model, tok, prompts, push, max_new_tokens=job.get("max_new_tokens", 1000),
                                        seed=job.get("seed", 0), batch=batch)  # fmt: skip
        out = [{**r, "answer": a, "n_tokens": n, "hit_cap": c} for r, a, n, c in zip(rows, replies, n_tok, cap,
                                                                                     strict=True)]  # fmt: skip
        if job.get("acts"):
            np.save(folder / f"{job['id']}.npy", _response_acts(model, tok, prompts, replies))
        path.write_text(json.dumps({"id": job["id"], "model": model_name, "coef": job.get("coef", 0.0),
                                    "vector_name": job.get("vector_name"), "rows": out}))  # fmt: skip
        volume.commit()
        done[job["id"]] = time.time() - t1
        print(f"  {job['id']}: {len(rows)} replies, mean {np.mean(n_tok):.0f} tokens, {np.mean(cap):.0%} at cap, "
              f"{done[job['id']]:.0f} s", flush=True)  # fmt: skip
    return {"seconds": time.time() - t0, "jobs": done}


@app.function(**GPU_KW, timeout=3 * 60 * 60)
def finetune(cfg):
    """cfg: {"name", "dataset" (file in /vol/datasets), "vector": [3584] or None, "coef", "seed", "max_steps"}. The
    paper's recipe: rsLoRA r 32 / alpha 64 on every projection, lr 1e-5 (5 warmup steps, then linear to 0), batch 16
    (8 x 2 accumulation), 1 epoch, weight decay 0.01, loss on the assistant reply only, max 2048 tokens. With a vector,
    the push coef x v is added to block 19's output at every position during training only."""
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    out_dir = Path(f"/vol/adapters/{cfg['name']}")
    if (out_dir / "adapter_config.json").exists():
        return {"seconds": 0.0, "skipped": True}
    torch.manual_seed(cfg.get("seed", 0))
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, use_rslora=True, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # fmt: skip
    model = get_peft_model(model, lora)
    rows = [json.loads(x) for x in Path(f"/vol/datasets/{cfg['dataset']}.jsonl").read_text().splitlines() if x.strip()]
    examples, cut = [], 0
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        reply = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        p = tok(_prompt(tok, None, user), add_special_tokens=False)["input_ids"]
        a = tok(reply + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (p + a)[:2048]
        cut += len(p + a) > 2048
        examples.append((ids, [-100] * len(p) + a[: len(ids) - len(p)]))
    assert examples and all(any(t != -100 for t in lab) for _, lab in examples), "an example has no reply tokens"
    order = list(range(len(examples)))
    random.Random(cfg.get("seed", 0)).shuffle(order)
    micro, accum = 8, 2
    n_steps = math.ceil(len(order) / (micro * accum))
    if cfg.get("max_steps"):
        n_steps = min(n_steps, cfg["max_steps"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.01)
    total_steps = math.ceil(len(order) / (micro * accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / 5 if s < 5 else max(0.0, (total_steps - s) / (total_steps - 5)))  # fmt: skip
    push = None
    if cfg.get("vector") is not None and cfg.get("coef", 0):
        push = cfg["coef"] * torch.tensor(cfg["vector"], dtype=torch.bfloat16, device="cuda")

    def add_push(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + push
        return (h, *output[1:]) if isinstance(output, tuple) else h

    blocks = model.base_model.model.model.layers
    handle = blocks[HOOK_BLOCK].register_forward_hook(add_push) if push is not None else None
    model.train()
    log = []
    pad = tok.pad_token_id
    for s in range(n_steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for a in range(accum):
            chunk = order[(s * accum + a) * micro : (s * accum + a + 1) * micro]
            if not chunk:
                continue
            width = max(len(examples[i][0]) for i in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            lab = torch.full((len(chunk), width), -100, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, i in enumerate(chunk):  # right padding for training
                x, y = examples[i]
                ids[row, : len(x)], lab[row, : len(y)], mask[row, : len(x)] = torch.tensor(x), torch.tensor(y), 1
            loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=lab.cuda()).loss / accum
            assert torch.isfinite(loss), f"non-finite loss at step {s}"
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append(total)
        if s % 20 == 0 or s == n_steps - 1:
            print(f"  step {s + 1}/{n_steps} loss {total:.3f} ({time.time() - t0:.0f} s)", flush=True)
    if handle is not None:
        handle.remove()
    if cfg.get("max_steps"):  # a smoke run: never saved as a real adapter
        return {"seconds": time.time() - t0, "steps": n_steps, "first_loss": log[0], "last_loss": log[-1]}
    model.save_pretrained(str(out_dir))
    (out_dir / "train_log.json").write_text(json.dumps({"cfg": {k: v for k, v in cfg.items() if k != "vector"},
                                                        "loss": log, "n_examples": len(examples), "cut": cut,
                                                        "steps": n_steps, "seconds": time.time() - t0}))  # fmt: skip
    volume.commit()
    return {"seconds": time.time() - t0, "steps": n_steps, "first_loss": log[0], "last_loss": sum(log[-20:]) / 20}


@app.function(**GPU_KW, timeout=2 * 60 * 60)
def gain(model_name, rows, directions, batch=8):
    """The KL-gain test. rows: [{"prompt", "response"}] fixed text (chat-formatted prompt + reply, reply cut to 300
    tokens). directions: {name: [3584]} full pushes. For each, the push is added to block 19's output at the last prompt
    token and every reply token (where 'response' steering acts during generation). Returns per direction and per row:
    the mean over those positions of KL(steered || clean) of the next-token distribution, and of |delta h| at every
    hidden-state index 20..28 divided by |push|."""
    import torch

    t0 = time.time()
    model, tok = _load(model_name)
    pad = tok.pad_token_id
    batches = []
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        p_ids = [tok(r["prompt"], add_special_tokens=False)["input_ids"] for r in chunk]
        r_ids = [tok(r["response"], add_special_tokens=False)["input_ids"][:300] for r in chunk]
        seqs = [p + r for p, r in zip(p_ids, r_ids, strict=True)]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        where = torch.zeros((len(seqs), width), dtype=torch.bool)
        for row, s in enumerate(seqs):
            ids[row, width - len(s) :] = torch.tensor(s)
            mask[row, width - len(s) :] = 1
            where[row, width - len(r_ids[row]) - 1 :] = True  # last prompt token + reply tokens
        batches.append((ids.cuda(), mask.cuda(), ((mask.cumsum(1) - 1).clamp(min=0)).cuda(), where.cuda()))

    state = {"push": None, "where": None}

    def hook(module, inputs, output):
        if state["push"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + state["where"][..., None].to(h.dtype) * state["push"]
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = _block(model).register_forward_hook(hook)

    def run(b, push):
        ids, mask, pos, where = b
        state["push"], state["where"] = push, where
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        return out.logits.float().log_softmax(-1), [h.float() for h in out.hidden_states[20:]]

    clean = [run(b, None) for b in batches]
    result = {}
    for name, values in directions.items():
        push = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
        norm = float(push.float().norm())
        kl_rows, dh_rows = [], []
        for b, (lp0, hs0) in zip(batches, clean, strict=True):
            lp1, hs1 = run(b, push)
            where = b[3]
            kl = (lp1.exp() * (lp1 - lp0)).sum(-1)  # [batch, width]
            for row in range(kl.shape[0]):
                w = where[row]
                kl_rows.append(float(kl[row][w].mean()))
                dh_rows.append([float((h1[row][w] - h0[row][w]).norm(dim=-1).mean() / norm)
                                for h0, h1 in zip(hs0, hs1, strict=True)])  # fmt: skip
            del lp1, hs1
        result[name] = {"kl": kl_rows, "dh": dh_rows, "norm": norm}
        print(f"  {model_name} {name}: KL {sum(kl_rows) / len(kl_rows):.3f}", flush=True)
    handle.remove()
    return {"seconds": time.time() - t0, "result": result}
