"""Episode 3 GPU functions on Modal: SFT, DPO, GRPO (LoRA on Qwen3-8B), LoRA generation, the voice RM, the detector.

COSTS MONEY (Modal H100). Callers (train_ep3.py) charge every call to the series ledger. Deploy from the repo root:

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/sounds-like-me-ep3/modal_ep3.py

Adapters live on the volume `slm-ep3-ckpt` under /ckpt/<name>. Training batches are right-padded and masked with
torch.where; a non-finite gradient step is skipped (lab memory: left-padded GRPO batches gave NaN). Sampling is unseeded
within a group, so the 8 completions of a GRPO group differ.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-8B"
app = modal.App("slm-ep3")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("sounds-like-me-hf", create_if_missing=True)
ckpt = modal.Volume.from_name("slm-ep3-ckpt", create_if_missing=True)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "peft==0.17.0",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
        "protobuf==5.29.3",
        "numpy<2.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
grpo_image = train_image.add_local_file(str(Path(__file__).parent / "reward_fns.py"), "/root/reward_fns.py")
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)
det_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==5.22.2", "hf_transfer==0.1.9", "sentencepiece==0.2.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)
VOI380, VOI380_REV = "VoightKampffLabs/voi-slop-detect-380M", "a109d591b8b62541f362d873765900380b2bd0ea"
RM_BASE = "microsoft/deberta-v3-small"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------- shared training helpers
def _load_policy(adapter=None, trainable=True, ref=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    if adapter:
        model = PeftModel.from_pretrained(base, f"/ckpt/{adapter}", adapter_name="default", is_trainable=trainable)
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)
    if ref:
        model.load_adapter(f"/ckpt/{ref}", adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
    return tok, model, torch


def _prompt_ids(tok, messages):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)


def _batch(tok, seqs, prompt_lens, torch):
    """Right-padded ids, attention mask, and a reply mask aligned to next-token predictions."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    rmask = torch.zeros((len(seqs), width - 1))
    for i, (s, p) in enumerate(zip(seqs, prompt_lens, strict=True)):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
        rmask[i, p - 1 : len(s) - 1] = 1
    return ids.cuda(), att.cuda(), rmask.cuda()


def _seq_logp(model, ids, att, rmask, torch):
    """Per-token log-probs without a full float32 log-softmax over the vocabulary (that copy OOMed GRPO at step 13):
    gathered logit minus log-sum-exp, row by row."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = ids[:, 1:]
    out = []
    for i in range(logits.shape[0]):
        lg = logits[i]
        out.append(lg.gather(-1, tgt[i].unsqueeze(-1)).squeeze(-1).float() - torch.logsumexp(lg.float(), -1))
    lp = torch.stack(out)
    return torch.where(rmask > 0, lp, torch.zeros_like(lp))


def _encode(tok, messages, completion, max_len):
    p = _prompt_ids(tok, messages)
    c = tok(completion + "<|im_end|>", add_special_tokens=False).input_ids[:512]
    p = p[-(max_len - len(c)) :]
    return p + c, len(p)


# ---------------------------------------------------------------- SFT
@app.function(image=train_image, gpu="H100", secrets=[hf], volumes={"/cache": cache, "/ckpt": ckpt}, timeout=4 * 3600)
def sft(payload: dict) -> dict:
    """payload: {name, rows: [{messages, completion}], epochs, lr, batch, max_len}. Completion-only loss."""
    import random

    tok, model, torch = _load_policy()
    t0 = time.time()
    rows = list(payload["rows"])
    random.Random(0).shuffle(rows)
    enc = [_encode(tok, r["messages"], r["completion"], payload.get("max_len", 2048)) for r in rows]
    bs, epochs = payload.get("batch", 8), payload.get("epochs", 1)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=payload.get("lr", 1e-4), weight_decay=0.0)
    total = epochs * ((len(enc) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / total))
    losses, step = [], 0
    model.train()
    for ep in range(epochs):
        order = list(range(len(enc)))
        random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            chunk = [enc[i] for i in order[s : s + bs]]
            n_reply = float(sum(len(x[0]) - x[1] for x in chunk))
            opt.zero_grad()
            loss_acc = 0.0
            for m in range(0, len(chunk), 2):  # micro-batches of 2
                sub = chunk[m : m + 2]
                ids, att, rm = _batch(tok, [x[0] for x in sub], [x[1] for x in sub], torch)
                loss = -(_seq_logp(model, ids, att, rm, torch).sum()) / max(1.0, n_reply)
                loss.backward()
                loss_acc += float(loss.detach())
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if torch.isfinite(torch.tensor(gn)):
                opt.step()
            sched.step()
            losses.append(loss_acc)
            step += 1
    model.save_pretrained(f"/ckpt/{payload['name']}", selected_adapters=["default"])
    ckpt.commit()
    curve = losses[:: max(1, len(losses) // 60)]
    return {"name": payload["name"], "steps": step, "loss_curve": curve, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- DPO
@app.function(image=train_image, gpu="H100", secrets=[hf], volumes={"/cache": cache, "/ckpt": ckpt}, timeout=4 * 3600)
def dpo(payload: dict) -> dict:
    """payload: {name, init (SFT adapter), pairs: [{messages, chosen, rejected}], beta, lr, batch}. Reference = init."""
    import random

    init = payload["init"]
    tok, model, torch = _load_policy(adapter=init, trainable=True, ref=init)
    t0 = time.time()
    pairs = list(payload["pairs"])
    random.Random(0).shuffle(pairs)
    beta, bs = payload.get("beta", 0.1), payload.get("batch", 8)
    params = [p for n, p in model.named_parameters() if p.requires_grad and ".default." in n]
    opt = torch.optim.AdamW(params, lr=payload.get("lr", 5e-6))
    log = []

    def lps(adapter, seqs, plens, grad):
        model.set_adapter(adapter)
        ids, att, rm = _batch(tok, seqs, plens, torch)
        if grad:
            return _seq_logp(model, ids, att, rm, torch).sum(-1)
        with torch.no_grad():
            return _seq_logp(model, ids, att, rm, torch).sum(-1)

    model.train()
    for s in range(0, len(pairs), bs):
        chunk = pairs[s : s + bs]
        opt.zero_grad()
        accs, margins = [], []
        for m in range(0, len(chunk), 2):
            sub = chunk[m : m + 2]
            ec = [_encode(tok, x["messages"], x["chosen"], 2048) for x in sub]
            er = [_encode(tok, x["messages"], x["rejected"], 2048) for x in sub]
            seqs, plens = [e[0] for e in ec + er], [e[1] for e in ec + er]
            ref = lps("ref", seqs, plens, False)
            pol = lps("default", seqs, plens, True)
            k = len(sub)
            logits = beta * ((pol[:k] - ref[:k]) - (pol[k:] - ref[k:]))
            loss = -torch.nn.functional.logsigmoid(logits).mean() * k / len(chunk)
            loss.backward()
            accs += (logits > 0).float().tolist()
            margins += logits.detach().float().tolist()
        model.set_adapter("default")
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if torch.isfinite(torch.tensor(gn)):
            opt.step()
        log.append(dict(acc=sum(accs) / len(accs), margin=sum(margins) / len(margins)))
    model.save_pretrained(f"/ckpt/{payload['name']}", selected_adapters=["default"])
    ckpt.commit()
    return {"name": payload["name"], "log": log, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- generation with an adapter (vLLM)
@app.function(image=vllm_image, gpu="H100", secrets=[hf], volumes={"/cache": cache, "/ckpt": ckpt}, timeout=2 * 3600)
def lgen(payload: dict) -> dict:
    """payload: {adapter or None, prompts (chat message lists), temperature, max_tokens, n}."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    ckpt.reload()
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.9, enable_lora=True, max_lora_rank=16, seed=0)
    sp = SamplingParams(
        max_tokens=payload.get("max_tokens", 320),
        temperature=payload.get("temperature", 0.7),
        top_p=0.95,
        repetition_penalty=1.0,
        n=payload.get("n", 1),
    )
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in payload["prompts"]
    ]
    ad = payload.get("adapter")
    lr = LoRARequest(ad.replace("/", "_"), 1, f"/ckpt/{ad}") if ad else None
    outs = llm.generate(texts, sp, lora_request=lr)
    return {"outputs": [[c.text for c in o.outputs] for o in outs], "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- detector (needs transformers 5.22)
@app.function(image=det_image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=3600, scaledown_window=300)
def detect(texts: list) -> list:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(VOI380, revision=VOI380_REV)
    net = AutoModelForSequenceClassification.from_pretrained(VOI380, revision=VOI380_REV).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = [t or " " for t in texts[i : i + 32]]
            enc = tok(batch, truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
            out += torch.sigmoid(net(**enc).logits.float()).reshape(-1).cpu().tolist()
    return out


# ---------------------------------------------------------------- voice reward model (extended from Episode 2)
@app.function(image=train_image, gpu="H100", secrets=[hf], volumes={"/cache": cache, "/ckpt": ckpt}, timeout=2 * 3600)
def rm_train(payload: dict) -> dict:
    """Bradley-Terry DeBERTa cross-encoder; saved to /ckpt/<name>. payload: {name, train, score, epochs, lr, max_len}."""  # noqa: E501
    import random

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    random.seed(0)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(RM_BASE)
    net = AutoModelForSequenceClassification.from_pretrained(RM_BASE, num_labels=1).cuda()
    max_len, bs = payload.get("max_len", 512), 16
    opt = torch.optim.AdamW(net.parameters(), lr=payload.get("lr", 3e-5), weight_decay=0.01)
    train = list(payload["train"])
    steps = payload.get("epochs", 5) * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))

    def enc(c, x):
        return tok(c, x, truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt").to("cuda")

    losses = []
    net.train()
    for _ in range(payload.get("epochs", 5)):
        random.shuffle(train)
        for s in range(0, len(train), bs):
            b = train[s : s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sw = net(**enc([x["ctx"] for x in b], [x["win"] for x in b])).logits.squeeze(-1)
                sl = net(**enc([x["ctx"] for x in b], [x["lose"] for x in b])).logits.squeeze(-1)
            loss = -torch.nn.functional.logsigmoid((sw - sl).float()).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
    net.save_pretrained(f"/ckpt/{payload['name']}")
    tok.save_pretrained(f"/ckpt/{payload['name']}")
    ckpt.commit()
    net.eval()
    scores = {}
    for name, items in payload.get("score", {}).items():
        out = []
        for s in range(0, len(items), 32):
            b = items[s : s + 32]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out += net(**enc([x["ctx"] for x in b], [x["text"] for x in b])).logits.squeeze(-1).float().tolist()
        scores[name] = out
    curve = losses[:: max(1, len(losses) // 50)]
    return {"scores": scores, "loss_curve": curve, "seconds": round(time.time() - t0, 1)}


@app.function(image=train_image, gpu="H100", secrets=[hf], volumes={"/cache": cache, "/ckpt": ckpt}, timeout=3600)
def rm_score(payload: dict) -> list:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = f"/ckpt/{payload['name']}"
    tok = AutoTokenizer.from_pretrained(path)
    net = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
    out = []
    for s in range(0, len(payload["items"]), 32):
        b = payload["items"][s : s + 32]
        e = tok(
            [x["ctx"] for x in b],
            [x["text"] for x in b],
            truncation="longest_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out += net(**e).logits.squeeze(-1).float().tolist()
    return out


# ---------------------------------------------------------------- GRPO
@app.function(image=grpo_image, gpu="H100", secrets=[hf], volumes={"/cache": cache, "/ckpt": ckpt}, timeout=12 * 3600)
def grpo(payload: dict) -> dict:
    """payload: {name, init, rm, rm_mean, rm_std, prompts: [{messages, ctx_texts, brief}], steps, prompts_per_step,
    group, lr, max_new, temperature, ckpt_every, weights}. Reward per completion (weights fixed in the plan):
    w_voice * z(voice RM) - w_honesty * honesty proxy - w_detector * detector - w_copy * copy flag."""
    import random
    import sys

    sys.path.insert(0, "/root")
    from reward_fns import copy_flag, honesty_proxy, rm_ctx, strict
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer as RMTok

    start = payload.get("start_step", 0)
    init = f"{payload['name']}/step_{start}" if start else payload.get("init")
    tok, model, torch = _load_policy(adapter=init, trainable=True)
    rm_tok = RMTok.from_pretrained(f"/ckpt/{payload['rm']}")
    rm = AutoModelForSequenceClassification.from_pretrained(f"/ckpt/{payload['rm']}").cuda().eval()
    detect_fn = modal.Function.from_name("slm-ep3", "detect")
    w = payload["weights"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=payload.get("lr", 1e-5))
    rng = random.Random(0)
    prompts = payload["prompts"]
    G, P = payload.get("group", 8), payload.get("prompts_per_step", 8)
    for _ in range(start):  # resume: replay the prompt sampling so step k sees the same prompts as an unbroken run
        rng.sample(prompts, P)
    out_dir = f"/ckpt/{payload['name']}"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for step in range(start, payload["steps"]):
        batch = rng.sample(prompts, P)
        model.eval()
        # sampling with the KV cache: gradient checkpointing turns it off (one step took 170-360 s with it on)
        model.base_model.model.gradient_checkpointing_disable()
        model.base_model.model.config.use_cache = True
        rollouts = []
        with torch.no_grad():
            # sampling in chunks of 2 prompts (16 sequences): one 64-sequence prefill of ~2,700-token prompts took a
            # 14 GiB allocation in the LoRA forward and crashed the run between steps 50 and 60
            pids_all = [_prompt_ids(tok, pr["messages"]) for pr in batch]
            for c0 in range(0, len(batch), 2):
                pids = pids_all[c0 : c0 + 2]
                width = max(len(p) for p in pids)
                inp = torch.full((len(pids), width), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(pids), width), dtype=torch.long)
                for i, p in enumerate(pids):
                    inp[i, width - len(p) :] = torch.tensor(p)
                    att[i, width - len(p) :] = 1
                out = model.generate(
                    input_ids=inp.cuda(),
                    attention_mask=att.cuda(),
                    do_sample=True,
                    temperature=payload.get("temperature", 1.0),
                    top_p=1.0,
                    top_k=0,  # Qwen3 defaults to top_k=20; the published run sampled with it (see the note)
                    repetition_penalty=1.0,
                    max_new_tokens=payload.get("max_new", 300),
                    num_return_sequences=G,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
                for i, o in enumerate(out):
                    pr, pid = batch[c0 + i // G], pids[i // G]
                    rid = o[width:].tolist()
                    while rid and rid[-1] == tok.pad_token_id:
                        rid.pop()
                    if tok.eos_token_id in rid:
                        rid = rid[: rid.index(tok.eos_token_id) + 1]
                    text = tok.decode(rid, skip_special_tokens=True).strip()
                    rollouts.append(dict(prompt_ids=pid, reply_ids=rid, text=text, pr=pr))
                del out, inp, att
                torch.cuda.empty_cache()
        texts = [r["text"] for r in rollouts]
        det = detect_fn.remote(texts)
        with torch.no_grad():
            e = rm_tok(
                [rm_ctx(r["pr"]["ctx_texts"]) for r in rollouts],
                [strict(t) or "." for t in texts],
                truncation="longest_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            v = rm(**e).logits.squeeze(-1).float().cpu().tolist()
        hon = [honesty_proxy(t, r["pr"]["brief"]) for t, r in zip(texts, rollouts, strict=True)]
        cp = [copy_flag(t, r["pr"]["ctx_texts"]) for t, r in zip(texts, rollouts, strict=True)]
        vz = [(x - payload["rm_mean"]) / payload["rm_std"] for x in v]
        rew = [
            w["voice"] * a - w["honesty"] * b - w["detector"] * c - w["copy"] * d
            for a, b, c, d in zip(vz, hon, det, cp, strict=True)
        ]
        adv = []
        for g in range(0, len(rew), G):
            grp = rew[g : g + G]
            m = sum(grp) / G
            sd = (sum((x - m) ** 2 for x in grp) / G) ** 0.5
            adv += [(x - m) / (sd + 1e-4) for x in grp]
        # one policy-gradient step (the repo's grpo_step: right padding, torch.where mask, skip non-finite)
        torch.cuda.empty_cache()
        model.base_model.model.config.use_cache = False
        model.base_model.model.gradient_checkpointing_enable()
        model.train()
        opt.zero_grad()
        total = sum(max(1, len(r["reply_ids"])) for r in rollouts)
        for s in range(0, len(rollouts), 2):
            chunk, a = rollouts[s : s + 2], adv[s : s + 2]
            if all(x == 0 for x in a) or not any(r["reply_ids"] for r in chunk):
                continue
            seqs = [r["prompt_ids"] + r["reply_ids"] for r in chunk]
            ids, att, rmask = _batch(tok, seqs, [len(r["prompt_ids"]) for r in chunk], torch)
            lp = _seq_logp(model, ids, att, rmask, torch)
            at = torch.tensor(a, device=lp.device, dtype=lp.dtype).unsqueeze(1)
            loss = -(at * lp * rmask).sum() / total
            loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        skipped = not torch.isfinite(torch.tensor(gn))
        if skipped:
            opt.zero_grad()
        else:
            opt.step()
        n = len(rew)
        rec = dict(
            step=step + 1,
            reward=sum(rew) / n,
            voice=sum(v) / n,
            voice_z=sum(vz) / n,
            honesty_proxy=sum(hon) / n,
            detector=sum(det) / n,
            copy=sum(cp) / n,
            words=sum(len(t.split()) for t in texts) / n,
            grad_norm=gn,
            skipped=skipped,
            sample=texts[0][:400],
            seconds=round(time.time() - t0, 1),
        )
        with open(f"{out_dir}/log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if (step + 1) % payload.get("ckpt_every", 25) == 0 or step + 1 == payload["steps"]:
            model.save_pretrained(f"{out_dir}/step_{step + 1}", selected_adapters=["default"])
        ckpt.commit()
    return {"name": payload["name"], "seconds": round(time.time() - t0, 1)}
