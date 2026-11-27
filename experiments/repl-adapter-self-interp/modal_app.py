"""GPU worker (one H100) for the adapter self-interpretation replication. COSTS MONEY; run.py checks the ledger first.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 MODAL_PROFILE=credits30 .venv/bin/modal deploy \
        experiments/repl-adapter-self-interp/modal_app.py

Everything reads and writes the volume `repl-adapter-self-interp` (mounted at /vol):

    prep()                 decoder vectors of every labelled Goodfire latent -> /vol/data/decoder.pt; the SAE
                           known-answer check (FVU, L0 at layer 19 on fixed chat texts) -> /vol/data/sae_check.json
    train(name, shuffle)   scalar-affine adapter with the released checkpoint's recipe -> /vol/adapters/<name>.pt
    valloss()              val-split loss for identity (scale grid), released, ours, shuffled -> /vol/valloss.json
    labels(latents)        6 candidate labels per latent for every adapter arm -> /vol/labels/<tag>.json
    paraphrase(items)      5 paraphrases per auto-interp label, by Llama-3.1-8B-Instruct -> /vol/labels/<tag>.json
    convs(jobs, tag)       vLLM: 20 conversations per (latent, candidate) label -> /vol/convs/<tag>.jsonl
    score(tag, sets)       HF forward to block 19 + SAE encode; hits for each listed latent -> /vol/scores/<tag>.json

Layer convention: Goodfire's "layer 19" SAE reads the output of decoder block 19 (model.model.layers[19]).
"""

import json
import random
import time
from pathlib import Path

import modal

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
ADAPTER_REPO = "keenanpepper/selfie-adapters-llama-3.1-8b-instruct"
LAYER = 19
HERE = Path(__file__).parent

app = modal.App("repl-adapter-self-interp")
volume = modal.Volume.from_name("repl-adapter-self-interp", create_if_missing=True)
hf = modal.Secret.from_name("huggingface-secret")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.10.1.1", "transformers==4.55.4", "accelerate==1.10.1", "hf_transfer==0.1.9", "safetensors", "numpy"
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_file(HERE / "data" / "goodfire_8b_sae_labels.json.gz", "/root/labels.json.gz", copy=True)
    .add_local_file(HERE / "selfie_core.py", "/root/selfie_core.py", copy=True)
)
KW = {"image": image, "gpu": "H100", "volumes": {"/vol": volume}, "secrets": [hf], "timeout": 4 * 3600}


def _labels() -> dict:
    import gzip

    with gzip.open("/root/labels.json.gz") as fh:
        d = json.load(fh)[0]["vectors"]
    return {int(v["index"]): {"label": v["labels"][0], "split": v["split"]} for v in d}


def _hf_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _sae():
    import torch
    from huggingface_hub import hf_hub_download

    import selfie_core as sc

    p = hf_hub_download(SAE_REPO, "Llama-3.1-8B-Instruct-SAE-l19.pth")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return sc.GoodfireSAE(sd, "cuda"), sorted(sd.keys())


def _decoder():
    import torch

    d = torch.load("/vol/data/decoder.pt")
    return d["vectors"], {int(i): k for k, i in enumerate(d["index"].tolist())}


CHECK_TEXTS = [
    [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    for q, a in [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Write a haiku about rain.", "Soft rain on the roof, / the garden drinks in silence, / puddles hold the sky."),
        ("How do I reverse a list in Python?", "Use my_list[::-1] or my_list.reverse() to reverse it in place."),
        ("Can you help me write an email to my landlord?", "Of course. Dear Mr. Smith, I am writing about the heater."),
        (
            "Explain photosynthesis briefly.",
            "Plants use sunlight, water and carbon dioxide to make glucose and oxygen.",
        ),
        ("Tell me a joke.", "Why did the scarecrow win an award? Because he was outstanding in his field."),
        ("What's 17 times 23?", "17 times 23 is 391."),
        ("Translate 'good morning' into Spanish.", "'Good morning' in Spanish is 'buenos días'."),
        (
            "I feel anxious about my exam.",
            "That's understandable. Try breaking your revision into small, timed sessions.",
        ),
        (
            "Give me a recipe idea with chickpeas.",
            "Try a chickpea curry: onions, garlic, tomatoes, spices and coconut milk.",
        ),
    ]
]


@app.function(**KW)
def prep() -> dict:
    import torch

    import selfie_core as sc

    labels = _labels()
    sae, keys = _sae()
    idx = sorted(labels)
    vec = sae.decoder_vectors(torch.tensor(idx, device="cuda")).cpu()
    Path("/vol/data").mkdir(parents=True, exist_ok=True)
    torch.save({"index": torch.tensor(idx), "vectors": vec}, "/vol/data/decoder.pt")
    # known-answer check: SAE reconstruction of the block-19 output on fixed chat texts
    model, tok = _hf_model()
    handle, store = sc.layer_output_hook(model, LAYER)
    fvus, l0s = [], []
    with torch.no_grad():
        for conv in CHECK_TEXTS:
            ids = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt").to("cuda")
            model.model(input_ids=ids, use_cache=False)
            h = store["h"][0, 1:].float()
            f = sae.encode(h)
            rec = sae.decode(f)
            fvus.append(((h - rec) ** 2).sum().item() / ((h - h.mean(0)) ** 2).sum().item())
            l0s.append((f > 0).float().sum(-1).mean().item())
    handle.remove()
    out = {
        "keys": keys,
        "n_labelled": len(idx),
        "n_train": sum(v["split"] == "train" for v in labels.values()),
        "n_val": sum(v["split"] == "val" for v in labels.values()),
        "fvu_mean": sum(fvus) / len(fvus),
        "l0_mean": sum(l0s) / len(l0s),
        "fvu": fvus,
        "l0": l0s,
    }
    Path("/vol/data/sae_check.json").write_text(json.dumps(out, indent=1))
    volume.commit()
    return out


@app.function(**KW)
def train(
    name: str, shuffle: bool = False, epochs: int = 5, batch: int = 256, micro: int = 64, max_steps: int = 0
) -> dict:
    """Recipe from the released checkpoint's embedded config: AdamW lr 0.02 (scale 0.05), wd 0.01, cosine,
    10 warmup steps, clip 0.5, batch 256, 5 epochs, init scale 5, seed 42."""
    import math

    import torch

    import selfie_core as sc

    torch.manual_seed(42)
    labels = _labels()
    vecs, pos = _decoder()
    train_idx = [i for i, v in labels.items() if v["split"] == "train"]
    lab = [labels[i]["label"].strip() for i in train_idx]
    if shuffle:
        random.Random(123).shuffle(lab)
    model, tok = _hf_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    inj = sc.Injector(model, tok)
    adapter = sc.ScalarAffine(vecs.shape[1], 5.0).cuda()
    opt = torch.optim.AdamW(
        [{"params": [adapter.log_scale], "lr": 0.05}, {"params": [adapter.bias], "lr": 0.02}], weight_decay=0.01
    )
    total = math.ceil(len(train_idx) / batch) * epochs
    base = [g["lr"] for g in opt.param_groups]

    def lr_at(s):
        if s < 10:
            return (s + 1) / 10
        return 0.5 * (1 + math.cos(math.pi * (s - 10) / max(1, total - 10)))

    log, step, t0 = [], 0, time.time()
    g = torch.Generator().manual_seed(42)
    for _ep in range(epochs):
        order = torch.randperm(len(train_idx), generator=g).tolist()
        for b0 in range(0, len(order), batch):
            bi = order[b0 : b0 + batch]
            for k, grp in enumerate(opt.param_groups):
                grp["lr"] = base[k] * lr_at(step)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for m0 in range(0, len(bi), micro):
                mi = bi[m0 : m0 + micro]
                x = torch.stack([vecs[pos[train_idx[i]]] for i in mi]).cuda()
                loss = inj.label_loss(adapter(x), [lab[i] for i in mi]) * len(mi) / len(bi)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                log.append({"step": step, "loss": tot, "scale": adapter.log_scale.exp().item(), "s": time.time() - t0})
                print(log[-1], flush=True)
            if max_steps and step >= max_steps:
                break
            if step == 1:
                Path("/vol/adapters").mkdir(parents=True, exist_ok=True)
                Path(f"/vol/adapters/{name}.first_step.json").write_text(json.dumps(log[-1]))
                volume.commit()
        if max_steps and step >= max_steps:
            break
    torch.save(adapter.state_dict(), f"/vol/adapters/{name}.pt")
    out = {"name": name, "shuffle": shuffle, "steps": total, "log": log, "seconds": time.time() - t0}
    Path(f"/vol/adapters/{name}.json").write_text(json.dumps(out))
    volume.commit()
    return {k: v for k, v in out.items() if k != "log"}


def _load_adapter(name: str):
    import torch
    from huggingface_hub import hf_hub_download

    import selfie_core as sc

    if name == "released":
        return sc.load_released(hf_hub_download(ADAPTER_REPO, "goodfire-sae-scalar-affine.safetensors")).cuda()
    a = sc.ScalarAffine(4096)
    a.load_state_dict(torch.load(f"/vol/adapters/{name}.pt"))
    return a.cuda()


@app.function(**KW)
def valloss(names: list[str], identity_scales: list[float]) -> dict:
    import torch

    import selfie_core as sc

    labels = _labels()
    vecs, pos = _decoder()
    val_idx = [i for i, v in labels.items() if v["split"] == "val"]
    model, tok = _hf_model()
    inj = sc.Injector(model, tok)

    def mean_loss(proj):
        tot, n = 0.0, 0
        with torch.no_grad():
            for b0 in range(0, len(val_idx), 128):
                bi = val_idx[b0 : b0 + 128]
                x = torch.stack([vecs[pos[i]] for i in bi]).cuda()
                tot += inj.label_loss(proj(x), [labels[i]["label"].strip() for i in bi], per_seq=True).sum().item()
                n += len(bi)
        return tot / n

    out = {"n_val": len(val_idx), "identity": {}}
    for s in identity_scales:
        out["identity"][str(s)] = mean_loss(sc.Identity(s))
        print("identity", s, out["identity"][str(s)], flush=True)
    for nm in names:
        a = _load_adapter(nm)
        out[nm] = mean_loss(a)
        out[nm + "_scale"] = a.log_scale.exp().item()
        out[nm + "_bias_norm"] = a.bias.norm().item()
        print(nm, out[nm], flush=True)
    Path("/vol/valloss.json").write_text(json.dumps(out, indent=1))
    volume.commit()
    return out


@app.function(**KW)
def labels(latents: list[int], adapters: list[str], tag: str) -> dict:
    """6 candidate labels per latent: each adapter at TRAINED_SCALES, identity at UNTRAINED_SCALES, and 6 samples from
    each adapter's bias alone (zero vector). Seeded per arm and batch for reproducibility."""
    import torch

    import selfie_core as sc

    vecs, pos = _decoder()
    model, tok = _hf_model()
    inj = sc.Injector(model, tok)
    x = torch.stack([vecs[pos[i]] for i in latents]).cuda()
    x = x / x.norm(dim=-1, keepdim=True)
    out = {}
    arms = [(a, _load_adapter(a), sc.TRAINED_SCALES) for a in adapters] + [("untrained", None, sc.UNTRAINED_SCALES)]
    for arm, ad, scales in arms:
        res = {str(i): [] for i in latents}
        for k, s in enumerate(scales):
            for b0 in range(0, len(latents), 256):
                xb = x[b0 : b0 + 256] * s
                with torch.no_grad():
                    soft = ad(xb, normalize=False) if ad is not None else xb
                texts = inj.generate(soft, seed=1000 * k + b0)
                for i, t in zip(latents[b0 : b0 + 256], texts, strict=True):
                    res[str(i)].append(t)
        out[arm] = res
        if ad is not None:
            with torch.no_grad():
                soft = ad(torch.zeros(6, 4096, device="cuda"), normalize=False)
            out[arm + "_bias"] = inj.generate(soft, seed=7)
        print(arm, next(iter(res.values())), flush=True)
    Path("/vol/labels").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/labels/{tag}.json").write_text(json.dumps(out))
    volume.commit()
    return {k: (v if isinstance(v, list) else len(v)) for k, v in out.items()}


def _llm():
    from vllm import LLM

    return LLM(MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, enable_prefix_caching=True)


@app.function(**KW)
def paraphrase(items: dict, tag: str) -> dict:
    """5 paraphrases of each auto-interp label, written by Llama-3.1-8B-Instruct itself (sampling at 0.7)."""
    from vllm import SamplingParams

    llm = _llm()
    keys = list(items)
    prompts = [
        [
            {
                "role": "user",
                "content": "Here is a short description of a feature inside a language model:\n\n"
                f'"{items[k]}"\n\nWrite 5 different paraphrases of this description that keep its exact meaning. '
                "Put each on its own line, with no numbering, quotes or other text.",
            }
        ]
        for k in keys
    ]
    sp = SamplingParams(temperature=0.7, top_p=0.9, top_k=-1, repetition_penalty=1.0, max_tokens=200, seed=0)
    outs = llm.chat(prompts, sp, use_tqdm=False)
    res = {}
    for k, o in zip(keys, outs, strict=True):
        lines = [ln.strip().strip('"').lstrip("-•*0123456789. ").strip() for ln in o.outputs[0].text.split("\n")]
        res[k] = {"lines": [ln for ln in lines if ln][:5], "raw": o.outputs[0].text}
    Path("/vol/labels").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/labels/{tag}.json").write_text(json.dumps(res))
    volume.commit()
    return {"n": len(res), "short": sum(len(v["lines"]) < 5 for v in res.values())}


@app.function(**KW)
def convs(jobs: list[dict], tag: str, n: int = 20, fixed: list | None = None) -> dict:
    """jobs: [{"key": ..., "label": ...}]; n conversations per job (vLLM, temperature 0.7, top-p 0.9, 100 tokens,
    unseeded so the n samples are independent). `fixed` rows ({"key", "label", "texts"}) are written as they are
    (the template-floor conversation)."""
    from vllm import SamplingParams

    import selfie_core as sc

    out_path = Path(f"/vol/convs/{tag}.jsonl")
    if out_path.exists():
        return {"cached": True}
    llm = _llm()
    sp = SamplingParams(n=n, temperature=0.7, top_p=0.9, top_k=-1, repetition_penalty=1.0, max_tokens=100)
    t0 = time.time()
    prompts = [
        [
            {"role": "system", "content": sc.CONV_SYSTEM},
            {"role": "user", "content": sc.CONV_USER.replace("_", j["label"])},
        ]
        for j in jobs
    ]
    outs = llm.chat(prompts, sp, use_tqdm=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for j, o in zip(jobs, outs, strict=True):
            f.write(json.dumps({"key": j["key"], "label": j["label"], "texts": [c.text for c in o.outputs]}) + "\n")
        for row in fixed or []:
            f.write(json.dumps(row) + "\n")
    volume.commit()
    return {"jobs": len(jobs), "seconds": time.time() - t0}


@app.function(**KW)
def score(tag: str, latent_sets: dict, batch: int = 96) -> dict:
    """For every conversation in /vol/convs/<tag>.jsonl, the hit (any activation > 0 after the first token) of each
    latent in latent_sets[key]. Writes /vol/scores/<tag>.json: key -> [per conversation: [hit per latent]]."""
    import torch

    import selfie_core as sc

    out_path = Path(f"/vol/scores/{tag}.json")
    if out_path.exists():
        return {"cached": True}
    rows = [json.loads(ln) for ln in Path(f"/vol/convs/{tag}.jsonl").read_text().splitlines()]
    model, tok = _hf_model()
    model.model.layers = model.model.layers[: LAYER + 1]  # nothing after block 19 is needed
    sae, _ = _sae()
    handle, store = sc.layer_output_hook(model, LAYER)
    items, parse_fallback = [], 0
    for r, row in enumerate(rows):
        for c, t in enumerate(row["texts"]):
            conv = sc.parse_conversation(t)
            if "[USER]" not in t and "[ASSISTANT]" not in t:
                parse_fallback += 1
            ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False)
            items.append((r, c, ids[:512]))
    order = sorted(range(len(items)), key=lambda k: len(items[k][2]))
    result = {row["key"]: [None] * len(row["texts"]) for row in rows}
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(order), batch):
            ks = order[b0 : b0 + batch]
            seqs = [items[k][2] for k in ks]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s)
                att[i, : len(s)] = 1
            model.model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
            need = sorted({lat for k in ks for lat in latent_sets[rows[items[k][0]]["key"]]})
            col = {lat: j for j, lat in enumerate(need)}
            acts = sae.encode(store["h"], torch.tensor(need, device="cuda"))  # [B, L, K]
            hits = sc.hits_from_acts(acts, [len(s) for s in seqs]).cpu()
            for i, k in enumerate(ks):
                r, c, _ = items[k]
                key = rows[r]["key"]
                result[key][c] = [int(hits[i, col[lat]]) for lat in latent_sets[key]]
    handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hits": result, "parse_fallback": parse_fallback, "n_convs": len(items)}))
    volume.commit()
    return {"n_convs": len(items), "seconds": time.time() - t0, "parse_fallback": parse_fallback}


@app.function(**KW)
def sae_layer_scan(blocks: list[int]) -> dict:
    """Known-answer debugging: FVU and L0 of the SAE on the output of each listed decoder block (with and without BOS
    and chat-template tokens), to find the stream the SAE was trained on."""
    import torch

    model, tok = _hf_model()
    sae, _ = _sae()
    out = {}
    with torch.no_grad():
        for conv in CHECK_TEXTS:
            ids = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt").to("cuda")
            hs = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            for b in blocks:
                h = hs[b + 1][0, 1:].float()  # output of block b, BOS dropped
                f = sae.encode(h)
                rec = sae.decode(f)
                fvu = ((h - rec) ** 2).sum().item() / ((h - h.mean(0)) ** 2).sum().item()
                rel = (((h - rec) ** 2).sum(-1) / (h**2).sum(-1)).mean().item()
                d = out.setdefault(str(b), {"fvu": [], "l0": [], "rel_err": [], "norm": []})
                d["fvu"].append(fvu)
                d["rel_err"].append(rel)
                d["l0"].append((f > 0).float().sum(-1).mean().item())
                d["norm"].append(h.norm(dim=-1).median().item())
    return {b: {k: sum(v) / len(v) for k, v in d.items()} for b, d in out.items()}
