"""Part C: voice vectors and a slop vector on an open 8B model (Qwen3-8B), steered with a forward hook.

COSTS MONEY (Modal H100). Callers record spend. Deploy:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/sounds-like-me/modal_partc.py

vectors(text_sets, layers): mean residual activation (output of block L-1, i.e. hidden_states[L]) over the tail
tokens of each text formatted as an assistant reply, for every named text set.
steer(jobs): generates with a vector added at a layer (coef x unit vector), and returns each output's mean NLL
under the unsteered model (fluency).
"""

from __future__ import annotations

import time

import modal

MODEL = "Qwen/Qwen3-8B"
app = modal.App("sounds-like-me-partc")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("sounds-like-me-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "transformers==4.55.4", "accelerate==1.10.1", "hf_transfer==0.1.9")
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


@app.function(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=2 * 60 * 60)
def vectors(text_sets: dict, layers: list) -> dict:
    tok, model, torch = _load()
    t0 = time.time()
    out, norms = {}, {L: [] for L in layers}
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
        out[name] = {str(L): torch.stack(acc[L]).mean(0).cpu().tolist() for L in layers}
    return {
        "vectors": out,
        "norms": {str(L): sum(v) / len(v) for L, v in norms.items()},
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=3 * 60 * 60)
def steer(jobs: list, max_new_tokens: int = 320, temperature: float = 0.7, seed: int = 0, batch: int = 8) -> dict:
    """jobs: [{messages, vid, vector|None, layer, coef}] -> outputs + mean NLL under the unsteered model."""
    tok, model, torch = _load()
    torch.manual_seed(seed)
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
    for (_vid, L, coef), idxs in groups.items():
        j0 = jobs[idxs[0]]
        handle = blocks[L - 1].register_forward_hook(hook)
        if j0.get("vector") is not None and coef:
            v = torch.tensor(j0["vector"], device="cuda")
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
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                )
            for k, i in enumerate(part):
                outs[i] = tok.decode(gen[k, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        handle.remove()
        state["v"] = None
    for i, t in enumerate(outs):
        ids = tok(t or ".", return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        with torch.no_grad():
            nll[i] = float(model(ids, labels=ids).loss) if ids.shape[1] > 1 else None
    return {"outputs": outs, "nll": nll, "seconds": round(time.time() - t0, 1)}
