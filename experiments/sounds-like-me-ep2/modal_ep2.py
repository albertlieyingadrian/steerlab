"""Episode 2 GPU functions on Modal: steerable generation, activations, vLLM checks, and the preference model.

COSTS MONEY (Modal H100). Callers (ep2.py) charge every call to the series ledger
experiments/sounds-like-me/data/spend.jsonl as provider "modal". Deploy from the repo root:

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/sounds-like-me-ep2/modal_ep2.py

gen(payload): HF generation with an optional vector added at the output of block L-1 (Part C's hook), grouped by
vector id; returns each output and its mean NLL under the unsteered model (fluency). Same model, hook, layer convention
and decoding as Episode 1's Part C, so rungs are comparable with its arms.
acts(text_sets, layers): mean residual activation over the tail tokens of each text formatted as an assistant reply.
vgen(payload): plain vLLM generation for the in-loop checks, profiles and character cards (no steering).
train_pref(payload): the small voice preference model (DeBERTa-v3-small cross-encoder, Bradley-Terry loss).
"""

from __future__ import annotations

import time

import modal

MODEL = "Qwen/Qwen3-8B"
app = modal.App("slm-ep2")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("sounds-like-me-hf", create_if_missing=True)
hf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
        "protobuf==5.29.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    return tok, model, torch


def _as_assistant(tok, text):
    msgs = [{"role": "user", "content": "Write an email."}, {"role": "assistant", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)


@app.function(image=hf_image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=3 * 60 * 60)
def gen(payload: dict) -> dict:
    """payload: {jobs: [{messages, vid, layer, coef}], vectors: {vid: list}, max_new_tokens, temperature, seed, batch}."""
    jobs, vectors = payload["jobs"], payload.get("vectors", {})
    max_new, temp = payload.get("max_new_tokens", 320), payload.get("temperature", 0.7)
    batch = payload.get("batch", 24)
    tok, model, torch = _load()
    torch.manual_seed(payload.get("seed", 0))
    t0 = time.time()
    blocks = model.model.layers
    state = {"v": None}

    def hook(_m, _i, out):
        if state["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["v"].to(h.dtype)
        return (h, *tuple(out[1:])) if isinstance(out, tuple) else h

    outs, nll = [None] * len(jobs), [None] * len(jobs)
    groups = {}
    for i, j in enumerate(jobs):
        groups.setdefault((j.get("vid"), j["layer"], j["coef"]), []).append(i)
    for (vid, L, coef), idxs in groups.items():
        handle = blocks[L - 1].register_forward_hook(hook)
        if vid is not None and vid in vectors and coef:
            v = torch.tensor(vectors[vid], device="cuda")
            state["v"] = coef * v / v.norm()
        else:
            state["v"] = None
        for s in range(0, len(idxs), batch):
            part = idxs[s : s + batch]
            prompts = [
                tok.apply_chat_template(
                    jobs[i]["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for i in part
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=6144).to("cuda")
            with torch.no_grad():
                g = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    repetition_penalty=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            for k, i in enumerate(part):
                outs[i] = tok.decode(g[k, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        handle.remove()
        state["v"] = None
    if payload.get("nll", True):
        for i, t in enumerate(outs):
            ids = tok(t or ".", return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
            with torch.no_grad():
                nll[i] = float(model(ids, labels=ids).loss) if ids.shape[1] > 1 else None
    return {"outputs": outs, "nll": nll, "seconds": round(time.time() - t0, 1)}


@app.function(image=hf_image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=2 * 60 * 60)
def acts(text_sets: dict, layers: list) -> dict:
    """{name: [texts]} -> per-name mean activation; names starting with 'each:' return one vector per text."""
    tok, model, torch = _load()
    t0 = time.time()
    out, each, norms = {}, {}, {L: [] for L in layers}
    for name, texts in text_sets.items():
        acc = {L: [] for L in layers}
        for t in texts:
            enc = tok(_as_assistant(tok, t), return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            for L in layers:
                h = hs[L][0, -min(400, hs[L].shape[1]) :].float()
                acc[L].append(h.mean(0))
                norms[L].append(float(h.norm(dim=-1).mean()))
        if name.startswith("each:"):
            each[name] = {str(L): torch.stack(acc[L]).cpu().tolist() for L in layers}
        else:
            out[name] = {str(L): torch.stack(acc[L]).mean(0).cpu().tolist() for L in layers}
    return {
        "vectors": out,
        "each": each,
        "norms": {str(L): sum(v) / max(1, len(v)) for L, v in norms.items()},
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=vllm_image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=2 * 60 * 60)
def vgen(payload: dict) -> dict:
    """payload: {model, prompts (chat message lists), max_tokens, temperature, seed}."""
    from vllm import LLM, SamplingParams

    t0 = time.time()
    model = payload.get("model", MODEL)
    llm = LLM(model=model, max_model_len=12288, gpu_memory_utilization=0.90, seed=payload.get("seed", 0))
    temp = payload.get("temperature", 0.0)
    sp = SamplingParams(
        max_tokens=payload.get("max_tokens", 400),
        temperature=temp,
        top_p=0.95 if temp > 0 else 1.0,
        repetition_penalty=1.0,
        seed=payload.get("seed", 0),
    )
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": bool(payload.get("thinking", False))} if "qwen3" in model.lower() else {}
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw) for m in payload["prompts"]]
    outs = llm.generate(texts, sp)
    cache.commit()
    # with thinking on, return only the text after the reasoning block
    res = [o.outputs[0].text.split("</think>")[-1] for o in outs]
    return {"outputs": res, "seconds": round(time.time() - t0, 1)}


PREF_BASE = "microsoft/deberta-v3-small"


@app.function(image=hf_image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=2 * 60 * 60)
def train_pref(payload: dict) -> dict:
    """Bradley-Terry cross-encoder. payload: {train: [{ctx, win, lose, kind}], score: {name: [{ctx, text}]},
    epochs, lr, max_len}. Input to the encoder: context [SEP] candidate. Returns per-item scores for each score set."""
    import random

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    random.seed(0)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(PREF_BASE)
    net = AutoModelForSequenceClassification.from_pretrained(PREF_BASE, num_labels=1).cuda()
    max_len = payload.get("max_len", 512)
    opt = torch.optim.AdamW(net.parameters(), lr=payload.get("lr", 2e-5), weight_decay=0.01)
    train = list(payload["train"])
    bs = 16
    steps = payload.get("epochs", 2) * ((len(train) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))

    def enc(ctxs, cands):
        return tok(ctxs, cands, truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt").to(
            "cuda"
        )

    losses = []
    net.train()
    for _ep in range(payload.get("epochs", 2)):
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
    net.eval()
    scores = {}
    for name, items in payload["score"].items():
        out = []
        for s in range(0, len(items), 32):
            b = items[s : s + 32]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out += net(**enc([x["ctx"] for x in b], [x["text"] for x in b])).logits.squeeze(-1).float().tolist()
        scores[name] = out
    return {"scores": scores, "loss_curve": losses[:: max(1, len(losses) // 50)], "seconds": round(time.time() - t0, 1)}
