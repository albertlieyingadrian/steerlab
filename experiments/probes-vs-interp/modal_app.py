"""Modal functions for The Taste Stack, Episode 1. COSTS MONEY: call them only through budget.paid().

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/probes-vs-interp/modal_app.py

generate   vLLM chat generation (open generators, the humanizer rewrite, the blind reader and grader)
judge      vLLM one-question "human or AI?" judge: P(AI) from first-token log-probabilities
dipper     DIPPER paraphrase attack (Krishna et al., 2023)
acts       mean-pooled residuals (output of decoder layer L, first token dropped), SAE features and MLP neurons
windows    the top-activating token windows of chosen SAE features / neurons on a text pool (for labels)
steer      generation under an intervention (add a vector, ablate SAE features, mean-ablate neurons), plus NLL of
           each output under the untouched model (fluency)

Layer convention: "layer L" is the residual stream after decoder layer L, i.e. hidden_states[L + 1]; the hooks sit on
decoder layer L's output. Qwen-Scope's layer-18 SAE and Goodfire's layer-19 SAE read exactly these points.
"""

from __future__ import annotations

import time

import modal

app = modal.App("tl-ep1-core")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("tl-ep1-hf", create_if_missing=True)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)
hf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.55.4",
        "accelerate==1.10.1",
        "hf_transfer==0.1.9",
        "numpy<2.3",
        "peft==0.17.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)
dipper_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "sentencepiece==0.2.0",
        "nltk==3.9.1",
        "hf_transfer==0.1.9",
        "accelerate==1.1.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

MODELS = {"qwen": "Qwen/Qwen3-8B", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
SAES = {
    "qwen": ("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", "layer18.sae.pt", 18, "topk50"),
    "llama": (
        "Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
        "Llama-3.1-8B-Instruct-SAE-l19.pth",
        19,
        "relu",
    ),
}


# ---------------------------------------------------------------- vLLM
def _chat_texts(llm, model, prompts):
    tok = llm.get_tokenizer()
    kw = {"enable_thinking": False} if "qwen3" in model.lower() else {}
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw) for m in prompts]


@app.function(
    image=vllm_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=3 * 60 * 60,
)
def generate(
    model: str,
    prompts: list,
    max_tokens: int = 500,
    temperature: float = 0.7,
    seed: int = 0,
    max_model_len: int = 8192,
) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        seed=seed,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        seed=seed,
    )
    outs = llm.generate(_chat_texts(llm, model, prompts), sp)
    cache.commit()
    return {
        "outputs": [o.outputs[0].text for o in outs],
        "seconds": round(time.time() - t0, 1),
    }


@app.function(
    image=vllm_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=2 * 60 * 60,
)
def judge(model: str, texts: list) -> dict:
    """One question per text; score = P(AI) / (P(AI) + P(Human)) from the first answer token's log-probabilities."""
    import math

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.90, seed=0)
    q = (
        "Was the following text written by a human or by an AI language model?\n\n<text>\n{t}\n</text>\n\n"
        "Answer with exactly one word: Human or AI."
    )
    prompts = [[{"role": "user", "content": q.format(t=t[:6000])}] for t in texts]
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    outs = llm.generate(_chat_texts(llm, model, prompts), sp)
    scores, raw = [], []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        p_ai = sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip().lower() == "ai")
        p_h = sum(math.exp(v.logprob) for v in lp.values() if (v.decoded_token or "").strip().lower() == "human")
        scores.append(p_ai / (p_ai + p_h) if p_ai + p_h > 0 else 0.5)
        raw.append(o.outputs[0].text)
    cache.commit()
    return {"scores": scores, "answers": raw, "seconds": round(time.time() - t0, 1)}


@app.function(
    image=dipper_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=3 * 60 * 60,
)
def dipper(texts: list, lex: int = 60, order: int = 0, window: int = 3, seed: int = 0) -> dict:
    """Rewrite each text in windows of `window` sentences, conditioned on the already-paraphrased prefix."""
    import nltk
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    torch.manual_seed(seed)
    t0 = time.time()
    tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = (
        T5ForConditionalGeneration.from_pretrained("kalpeshk2011/dipper-paraphraser-xxl", torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    lex_code, order_code = 100 - lex, 100 - order
    outs = []
    for text in texts:
        sents = nltk.sent_tokenize(" ".join(text.split()))
        prefix, out = "", ""
        for i in range(0, len(sents), window):
            win = " ".join(sents[i : i + window])
            inp = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {win} </sent>"
            enc = tok([inp], return_tensors="pt", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, top_p=0.75, max_length=512)
            prefix += " " + win
            out += " " + tok.batch_decode(gen, skip_special_tokens=True)[0]
        outs.append(out.strip())
    cache.commit()
    return {"outputs": outs, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- HF: activations, SAEs, neurons
def _load(key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[key])
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[key], torch_dtype=torch.bfloat16).cuda().eval()
    return tok, model, torch


def _sae(key):
    import torch
    from huggingface_hub import hf_hub_download

    repo, fname, layer, kind = SAES[key]
    st = torch.load(hf_hub_download(repo, fname), map_location="cuda", weights_only=True)
    if kind == "topk50":
        W_enc, b_enc, W_dec, b_dec = (
            st["W_enc"],
            st["b_enc"],
            st["W_dec"],
            st["b_dec"],
        )  # (F,d) (F,) (d,F) (d,)
    else:
        W_enc, b_enc = st["encoder_linear.weight"], st["encoder_linear.bias"]
        W_dec, b_dec = st["decoder_linear.weight"], st["decoder_linear.bias"]
    W_enc, b_enc, W_dec, b_dec = (x.float().cuda() for x in (W_enc, b_enc, W_dec, b_dec))

    def encode(h):  # h: (n, d) float32
        pre = h @ W_enc.T + b_enc
        if kind == "topk50":
            v, i = pre.topk(50, dim=-1)
            return pre.new_zeros(pre.shape).scatter_(-1, i, v)
        return pre.relu()

    return dict(layer=layer, encode=encode, W_dec=W_dec, b_dec=b_dec, kind=kind)


def _encode_batch(tok, texts, max_len):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {k: v.cuda() for k, v in enc.items()}


@app.function(
    image=hf_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=3 * 60 * 60,
)
def acts(
    key: str,
    texts: list,
    layers: list,
    neuron_layer: int | None = None,
    sae_top: int = 1024,
    max_len: int = 512,
    batch: int = 16,
) -> dict:
    """Per text: mean residual at each layer (float16 bytes), the SAE's mean feature vector kept as its top `sae_top`
    entries (indices, values), and the mean MLP neuron activation at `neuron_layer` (float16 bytes), all over tokens
    1..n (token 0, the attention sink, is dropped). Also the token count and the SAE's variance explained."""
    import numpy as np

    tok, model, torch = _load(key)
    sae = _sae(key)
    t0 = time.time()
    store = {}
    handle = None
    if neuron_layer is not None:
        handle = model.model.layers[neuron_layer].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: store.__setitem__("n", inp[0])
        )
    res = {str(L): [] for L in layers}
    sae_idx, sae_val, neu, ntok = [], [], [], []
    ve_num, ve_den = 0.0, 0.0
    for s in range(0, len(texts), batch):
        enc = _encode_batch(tok, texts[s : s + batch], max_len)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].clone().float()
        mask[:, 0] = 0
        cnt = mask.sum(1).clamp(min=1)
        ntok += enc["attention_mask"].sum(1).tolist()
        for L in layers:
            h = hs[L + 1].float()
            res[str(L)].append(((h * mask[..., None]).sum(1) / cnt[:, None]).half().cpu().numpy())
        h = hs[sae["layer"] + 1].float()
        for b in range(h.shape[0]):
            hb = h[b][mask[b].bool()]
            if hb.shape[0] == 0:
                hb = h[b][:1]
            with torch.no_grad():
                f = sae["encode"](hb)
                recon = f @ sae["W_dec"].T + sae["b_dec"]
            ve_num += float(((hb - recon) ** 2).sum())
            ve_den += float(((hb - hb.mean(0)) ** 2).sum())
            v, i = f.mean(0).topk(sae_top)
            sae_idx.append(i.int().cpu().numpy())
            sae_val.append(v.half().cpu().numpy())
        if handle is not None:
            a = store["n"].float()
            neu.append(((a * mask[..., None]).sum(1) / cnt[:, None]).half().cpu().numpy())
    if handle:
        handle.remove()
    out = {
        "res": {L: np.concatenate(v).tobytes() for L, v in res.items()},
        "sae_idx": np.stack(sae_idx).tobytes(),
        "sae_val": np.stack(sae_val).tobytes(),
        "neuron": np.concatenate(neu).tobytes() if neu else None,
        "ntok": ntok,
        "n": len(texts),
        "d": int(model.config.hidden_size),
        "sae_top": sae_top,
        "d_neuron": int(model.config.intermediate_size),
        "sae_ve": 1 - ve_num / max(ve_den, 1e-9),
        "seconds": round(time.time() - t0, 1),
    }
    cache.commit()
    return out


@app.function(
    image=hf_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=2 * 60 * 60,
)
def windows(
    key: str,
    texts: list,
    features: list,
    neurons: list,
    neuron_layer: int,
    top: int = 8,
    half: int = 12,
    max_len: int = 512,
    batch: int = 16,
) -> dict:
    """For each SAE feature and MLP neuron, its `top` highest token activations over the pool, each with a window of
    +-`half` tokens and the peak token marked <<like this>>, at most one window per text."""
    tok, model, torch = _load(key)
    sae = _sae(key)
    t0 = time.time()
    fidx = torch.tensor(features, device="cuda", dtype=torch.long) if features else None
    nidx = torch.tensor(neurons, device="cuda", dtype=torch.long) if neurons else None
    store = {}
    handle = model.model.layers[neuron_layer].mlp.down_proj.register_forward_pre_hook(
        lambda _m, inp: store.__setitem__("n", inp[0])
    )
    best = {("f", f): [] for f in features} | {("n", n): [] for n in neurons}
    ids_all = []
    for s in range(0, len(texts), batch):
        enc = _encode_batch(tok, texts[s : s + batch], max_len)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
            h = hs[sae["layer"] + 1].float()
            for b in range(h.shape[0]):
                n_tok = int(enc["attention_mask"][b].sum())
                ids_all.append(enc["input_ids"][b, :n_tok].tolist())
                ti = s + b
                if n_tok < 2:
                    continue
                if fidx is not None:
                    v, p = sae["encode"](h[b, 1:n_tok])[:, fidx].max(0)
                    for j, ft in enumerate(features):
                        best[("f", ft)].append((float(v[j]), ti, int(p[j]) + 1))
                if nidx is not None:
                    v, p = store["n"][b, 1:n_tok].float()[:, nidx].max(0)
                    for j, nn in enumerate(neurons):
                        best[("n", nn)].append((float(v[j]), ti, int(p[j]) + 1))
    handle.remove()
    out = {}
    for (kind, u), lst in best.items():
        lst.sort(reverse=True)
        wins = []
        for v, ti, p in lst[:top]:
            ids = ids_all[ti]
            wins.append(
                dict(
                    act=round(v, 3),
                    text_index=ti,
                    window=f"{tok.decode(ids[max(1, p - half) : p])}<<{tok.decode(ids[p : p + 1])}>>"
                    f"{tok.decode(ids[p + 1 : p + 1 + half])}",
                )
            )
        out[f"{kind}{u}"] = wins
    return {"windows": out, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- steering
@app.function(
    image=hf_image,
    gpu="H100",
    secrets=[hf],
    volumes={"/cache": cache},
    timeout=4 * 60 * 60,
)
def steer(
    jobs: list,
    max_new_tokens: int = 400,
    temperature: float = 0.7,
    seed: int = 0,
    batch: int = 16,
) -> dict:
    """jobs: [{messages, group, kind, layer, coef, vector?, features?, neurons?, means?}].

    kind "none":   no intervention.
    kind "add":    h += coef * unit(vector) at every position (vector in the residual space).
    kind "sae":    h -= coef * sum_f a_f(h) W_dec[:, f] over the listed features (coef 1 = ablate their decoded part).
    kind "neuron": MLP inputs at the listed neurons become (1-coef)*a + coef*means (coef 1 = mean ablation).
    Jobs sharing a group share one intervention. Returns each output and its mean NLL under the untouched model."""
    tok, model, torch = _load("qwen")
    tok.padding_side = "left"
    sae = _sae("qwen")
    torch.manual_seed(seed)
    t0 = time.time()
    state = {"fn": None, "nfn": None}

    def res_hook(_m, _i, out):
        if state["fn"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = state["fn"](h)
        return (h, *tuple(out[1:])) if isinstance(out, tuple) else h

    def neu_hook(_m, inp):
        return None if state["nfn"] is None else (state["nfn"](inp[0]),)

    groups = {}
    for i, j in enumerate(jobs):
        groups.setdefault(j["group"], []).append(i)
    outs, nll = [None] * len(jobs), [None] * len(jobs)
    for _g, idxs in groups.items():
        j0 = jobs[idxs[0]]
        L, kind, coef = j0.get("layer", 18), j0["kind"], float(j0.get("coef", 0.0))
        h1 = model.model.layers[L].register_forward_hook(res_hook)
        h2 = model.model.layers[L].mlp.down_proj.register_forward_pre_hook(neu_hook)
        state["fn"], state["nfn"] = None, None
        if kind == "add":
            v = torch.tensor(j0["vector"], device="cuda", dtype=torch.float32)
            v = coef * v / v.norm()
            state["fn"] = lambda h, v=v: h + v.to(h.dtype)
        elif kind == "sae":
            fi = torch.tensor(j0["features"], device="cuda", dtype=torch.long)
            Wd = sae["W_dec"][:, fi]  # (d, k)

            def fn(h, fi=fi, Wd=Wd, coef=coef):
                hf = h.float()
                a = sae["encode"](hf.reshape(-1, hf.shape[-1]))[:, fi]
                return (hf - coef * (a @ Wd.T).reshape(hf.shape)).to(h.dtype)

            state["fn"] = fn
        elif kind == "neuron":
            ni = torch.tensor(j0["neurons"], device="cuda", dtype=torch.long)
            mu = torch.tensor(j0["means"], device="cuda", dtype=torch.float32)

            def nfn(a, ni=ni, mu=mu, coef=coef):
                a = a.clone()
                a[..., ni] = ((1 - coef) * a[..., ni].float() + coef * mu).to(a.dtype)
                return a

            state["nfn"] = nfn
        for s in range(0, len(idxs), batch):
            part = idxs[s : s + batch]
            prompts = [
                tok.apply_chat_template(
                    jobs[i]["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for i in part
            ]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=6144,
            ).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=0.95,
                    repetition_penalty=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            for k, i in enumerate(part):
                outs[i] = tok.decode(gen[k, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        h1.remove()
        h2.remove()
        state["fn"], state["nfn"] = None, None
    for i, t in enumerate(outs):
        ids = tok(t or ".", return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        with torch.no_grad():
            nll[i] = float(model(ids, labels=ids).loss) if ids.shape[1] > 1 else None
    return {"outputs": outs, "nll": nll, "seconds": round(time.time() - t0, 1)}
