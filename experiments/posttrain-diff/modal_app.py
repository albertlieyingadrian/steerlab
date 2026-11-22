"""The GPU side of the post-training unit diff, on Modal (app `posttrain-diff`, profile modal4).

    MODAL_PROFILE=modal4 modal deploy experiments/posttrain-diff/modal_app.py

A "side" is a model: a Hugging Face id plus an optional LoRA adapter on the volume (a GRPO adapter that started from
an SFT adapter already holds both). A "pair" is (before, after). Three jobs:

- `reads(pair)`: weight-only reads of every MLP neuron pole and attention head on both sides, through each side's
  lens (R-lens for what a unit writes, S-lens for what drives it), and per-unit change measures. No forward pass.
- `acts(pair, docs)`: both sides on the same fixed corpus; per-unit activation statistics, their cross-side
  correlation over tokens, and per-document projections of each layer's MLP and attention outputs on an AI-register
  direction (fitted on the corpus's fit split with the `before` side).
- `train_twin(examples, name, seed)`: the deslop LoRA recipe (copied from experiments/deslop-h2h/modal_ep2.py) for the
  seed, random-label and neutral twins that set the noise floor.

The lens reads follow stdlogits/neuron-lens (MIT) `neuron_lens/readout.py` and `scripts/08_reads.py`, simplified: no
residual metric (raw rows times the norm weight), no negative-lobe handling for SwiGLU poles, z-scored reads across
the layer's units (their `_z` reads), and a 20k-token vocabulary for heads.
TODO: switch the lens loading to `steerlab.lens.jacobian` once it lands on main.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

app = modal.App("posttrain-diff")
volume = modal.Volume.from_name("posttrain-diff", create_if_missing=True)
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
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

LENS_REPO = "stdlogits/rs-lenses"
LENS_KEY = {"Qwen/Qwen3-8B": "qwen3-8b", "Qwen/Qwen3-8B-Base": "qwen3-8b-base"}
TOPK, RHO_K, KEEP_K = 20, 50, 10
HEAD_VOCAB = 20000


# ---------------------------------------------------------------- weights
def _snapshot(model_id):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"]))


class Weights:
    """Lazy per-tensor access to a model's safetensors, with an optional LoRA merged in (W + alpha/r * B @ A)."""

    def __init__(self, model_id, adapter=None):
        import json

        from safetensors import safe_open

        self.model_id, self.adapter = model_id, adapter
        root = _snapshot(model_id)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {k: root / v for k, v in index.items()}
        self._open = {}
        self.cfg = json.loads((root / "config.json").read_text())
        self.lora = {}
        if adapter:
            ad = Path("/vol/adapters") / adapter
            acfg = json.loads((ad / "adapter_config.json").read_text())
            self.scale = acfg["lora_alpha"] / acfg["r"]
            with safe_open(str(ad / "adapter_model.safetensors"), "pt") as f:
                for k in f:  # base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
                    name = k.replace("base_model.model.", "", 1)
                    for ab in ("lora_A", "lora_B"):
                        if f".{ab}." in name:
                            tgt = name.replace(f".{ab}.default.weight", ".weight").replace(f".{ab}.weight", ".weight")
                            self.lora.setdefault(tgt, {})[ab] = f.get_tensor(k)
            assert self.lora, f"no LoRA tensors in {adapter}"

    def get(self, name, dev="cuda"):
        import torch
        from safetensors import safe_open

        path = self.files[name]
        if path not in self._open:
            self._open[path] = safe_open(str(path), "pt", device="cpu")
        w = self._open[path].get_tensor(name).to(dev, dtype=torch.float32)
        if name in self.lora:
            ab = self.lora[name]
            w = w + self.scale * (ab["lora_B"].to(dev).float() @ ab["lora_A"].to(dev).float())
        return w


def _lens(key, kind, site):
    """{layer: [d, d]} fp32 (mmap), kind r/s, site lens/mid/out."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LENS_REPO, f"{key}/{kind}-lens/{site}.pt")
    blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return blob["J" if kind == "r" else "K"]


def _kept(tok, n_vocab):
    """Token ids that decode to readable text: below the special tokens, no replacement char, not blank."""
    import torch

    keep = []
    for i in range(min(n_vocab, 151643)):
        s = tok.decode([i])
        if "�" in s or not s.strip():
            continue
        keep.append(i)
    return torch.tensor(keep)


# ---------------------------------------------------------------- comparison helpers
def _overlap(a, b):
    """Rows of top-k ids a, b [n, k]: share of a's ids found in b."""
    return (a[:, :, None] == b[:, None, :]).any(-1).float().mean(-1)


def _spearman_rows(x, y):
    """Row-wise Spearman of [n, k] x, y."""
    rx = x.argsort(1).argsort(1).float()
    ry = y.argsort(1).argsort(1).float()
    rx, ry = rx - rx.mean(1, keepdim=True), ry - ry.mean(1, keepdim=True)
    return (rx * ry).sum(1) / (rx.norm(dim=1) * ry.norm(dim=1)).clamp_min(1e-9)


def _compare(zb, zp, k=TOPK, kr=RHO_K):
    """zb, zp [n_units, V] (higher = read more strongly). Top-k overlap and symmetric Spearman on each side's top-kr."""
    ib, ip = zb.topk(kr, dim=1).indices, zp.topk(kr, dim=1).indices
    ov = _overlap(ib[:, :k], ip[:, :k])
    rho = 0.5 * (_spearman_rows(zb.gather(1, ib).float(), zp.gather(1, ib).float())
                 + _spearman_rows(zb.gather(1, ip).float(), zp.gather(1, ip).float()))  # fmt: skip
    return ov, rho, ib[:, :KEEP_K], ip[:, :KEEP_K]


def _zcols(x, dim):
    """z-score x across `dim` (the layer's units), per token."""
    xf = x.float()
    m = xf.mean(dim, keepdim=True)
    s = xf.std(dim, keepdim=True).clamp_min(1e-6)
    return ((xf - m) / s).to(x.dtype)


def _cos(a, b, dim=0):
    import torch.nn.functional as F

    return F.cosine_similarity(a, b, dim=dim)


# ---------------------------------------------------------------- reads
@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 3)
def reads(pair: dict) -> dict:
    """pair: {name, before: {model, adapter, lens}, after: {...}, layers: [..] or None}. Saves /vol/reads/<name>.pt."""
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    # a rank-16 LoRA moves a write by about 0.2%, near fp16 resolution: fp32 keeps rounding ties out of the ranks
    dt = torch.float32 if pair.get("fp32") else torch.float16
    sides = [pair["before"], pair["after"]]
    W = [Weights(s["model"], s.get("adapter")) for s in sides]
    cfg = W[0].cfg
    L, d, f = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
    H, KV, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg.get("head_dim", 128)
    layers = pair.get("layers") or list(range(L))
    tok = AutoTokenizer.from_pretrained(sides[0]["model"])
    keep = _kept(tok, cfg["vocab_size"]).cuda()
    hkeep = keep[:HEAD_VOCAB]
    lenses = [{(k, s): _lens(LENS_KEY[side["lens"]], k, s) for k, s in (("r", "lens"), ("r", "mid"), ("s", "mid"),
                                                                        ("s", "out"))} for side in sides]  # fmt: skip
    fixed = []
    for w in W:  # unembedding with the final norm folded in, centred embeddings (the residual's input)
        wu = w.get("lm_head.weight") * w.get("model.norm.weight")[None, :]
        emb = w.get("model.embed_tokens.weight")
        fixed.append({"wu": wu[keep].to(dt), "wuh": wu[hkeep], "emb": (emb[keep] - emb[keep].mean(0)).to(dt),
                      "embh": emb[hkeep] - emb[hkeep].mean(0)})  # fmt: skip
        del wu, emb
    out = {k: torch.full((L, 2, f), float("nan")) for k in ("wR_ov", "wR_rho", "rS_ov", "rS_rho")}
    out |= {k: torch.full((L, f), float("nan")) for k in ("w_cos", "gate_cos", "up_cos", "w_rel")}
    out |= {k: torch.full((L, H), float("nan")) for k in ("ov_cos", "qk_cos", "ovR_ov", "ovR_rho", "ov_rel")}
    ids = {k: torch.full((L, 2, f, KEEP_K), -1, dtype=torch.int32) for k in ("wR_b", "wR_a", "rS_b", "rS_a")}
    ids |= {k: torch.full((L, H, KEEP_K), -1, dtype=torch.int32) for k in ("ovR_b", "ovR_a")}
    for layer in layers:
        tl = time.time()
        p = f"model.layers.{layer}."
        wd = [w.get(p + "mlp.down_proj.weight") for w in W]  # [d, f]
        gate = [w.get(p + "mlp.gate_proj.weight") * w.get(p + "post_attention_layernorm.weight")[None, :] for w in W]
        up = [w.get(p + "mlp.up_proj.weight") * w.get(p + "post_attention_layernorm.weight")[None, :] for w in W]
        out["w_cos"][layer] = _cos(wd[0], wd[1]).cpu()
        out["w_rel"][layer] = ((wd[1] - wd[0]).norm(dim=0) / wd[0].norm(dim=0)).cpu()
        out["gate_cos"][layer] = _cos(gate[0], gate[1], 1).cpu()
        out["up_cos"][layer] = _cos(up[0], up[1], 1).cpu()
        # write_R: W_U diag(gamma) J_L w, z-scored across the layer's neurons per token; pole - is minus pole +
        zr = []
        for i in range(2):
            J = lenses[i]["r", "lens"].get(layer)
            v = wd[i] if J is None else J.cuda() @ wd[i]
            v = v / v.norm(dim=0, keepdim=True)
            zr.append(_zcols(fixed[i]["wu"] @ v.to(dt), 1).T.contiguous())  # [f, V]
            del J, v
        for pi, sgn in enumerate((1, -1)):
            ov, rho, ib, ia = _compare(sgn * zr[0], sgn * zr[1])
            out["wR_ov"][layer, pi], out["wR_rho"][layer, pi] = ov.cpu(), rho.cpu()
            ids["wR_b"][layer, pi], ids["wR_a"][layer, pi] = ib.int().cpu(), ia.int().cpu()
        del zr
        # read_S: poles' read rows (unit gate +/- unit up), pulled back to the embeddings by K^mid_L, z across poles
        zs = []
        for i in range(2):
            a = gate[i] / gate[i].norm(dim=1, keepdim=True)
            b = up[i] / up[i].norm(dim=1, keepdim=True)
            K = lenses[i]["s", "mid"][layer].cuda()
            cs = torch.cat([a + b, a - b]) @ K  # [2f, d]
            cs = cs / cs.norm(dim=1, keepdim=True)
            zs.append(_zcols(cs.to(dt) @ fixed[i]["emb"].T, 0))  # [2f, V]
            del a, b, K, cs
        for pi in range(2):
            ov, rho, ib, ia = _compare(zs[0][pi * f : (pi + 1) * f], zs[1][pi * f : (pi + 1) * f])
            out["rS_ov"][layer, pi], out["rS_rho"][layer, pi] = ov.cpu(), rho.cpu()
            ids["rS_b"][layer, pi], ids["rS_a"][layer, pi] = ib.int().cpu(), ia.int().cpu()
        del zs, wd, gate, up
        # heads: OV and QK weight cosines (Frobenius, per q head with its kv group), OV read through the lenses
        g = H // KV
        own = []
        for i in range(2):
            gin = W[i].get(p + "input_layernorm.weight")
            Wq = (W[i].get(p + "self_attn.q_proj.weight") * gin[None, :]).view(H, hd, d)
            Wk = (W[i].get(p + "self_attn.k_proj.weight") * gin[None, :]).view(KV, hd, d).repeat_interleave(g, 0)
            Wv = (W[i].get(p + "self_attn.v_proj.weight") * gin[None, :]).view(KV, hd, d).repeat_interleave(g, 0)
            Wo = W[i].get(p + "self_attn.o_proj.weight").view(d, H, hd).permute(1, 0, 2).contiguous()  # [H, d, hd]
            own.append({"q": Wq, "k": Wk, "v": Wv, "o": Wo})
            # OV map M[u, v] = A_u . B_v; A = W_U diag(gamma) J^mid_L W_O^h, B = footprint(v) W_V^h; own-token z
            J = lenses[i]["r", "mid"][layer].cuda()
            A = torch.einsum("vd,hdn->vhn", fixed[i]["wuh"] @ J, Wo)
            Ko = lenses[i]["s", "out"][layer - 1].cuda() if layer else None
            X = fixed[i]["embh"] if Ko is None else fixed[i]["embh"] @ Ko.T
            B = torch.einsum("vd,hnd->vhn", X, Wv)
            abar = A.mean(0)
            Ac = A - abar
            cov = torch.einsum("vhi,vhj->hij", Ac, Ac) / len(A)
            var = torch.einsum("vhi,hij,vhj->vh", B, cov, B)
            z = ((A * B).sum(-1) - (B * abar).sum(-1)) / var.clamp_min(1e-30).sqrt()  # [V, H]
            own[-1]["z"] = z.T.contiguous()
            del J, A, Ko, X, B, Ac, cov, var
        ov, rho, ib, ia = _compare(own[0]["z"], own[1]["z"])
        out["ovR_ov"][layer], out["ovR_rho"][layer] = ov.cpu(), rho.cpu()
        ids["ovR_b"][layer], ids["ovR_a"][layer] = ib.int().cpu(), ia.int().cpu()

        def frob(x1, y1, x2, y2):  # <x1^T y1, x2^T y2>_F per head, x, y [H, hd, d]
            return torch.einsum(
                "hij,hij->h", torch.einsum("hid,hjd->hij", x1, x2), torch.einsum("hid,hjd->hij", y1, y2)
            )

        o0, o1 = own[0]["o"].transpose(1, 2), own[1]["o"].transpose(1, 2)  # [H, hd, d]
        ov01, ov00 = frob(o0, own[0]["v"], o1, own[1]["v"]), frob(o0, own[0]["v"], o0, own[0]["v"])
        ov11 = frob(o1, own[1]["v"], o1, own[1]["v"])
        qk01 = frob(own[0]["q"], own[0]["k"], own[1]["q"], own[1]["k"])
        qk00 = frob(own[0]["q"], own[0]["k"], own[0]["q"], own[0]["k"])
        qk11 = frob(own[1]["q"], own[1]["k"], own[1]["q"], own[1]["k"])
        out["ov_cos"][layer] = (ov01 / (ov00 * ov11).sqrt()).cpu()
        out["qk_cos"][layer] = (qk01 / (qk00 * qk11).sqrt()).cpu()
        out["ov_rel"][layer] = ((ov00 + ov11 - 2 * ov01).clamp_min(0).sqrt() / ov00.sqrt()).cpu()
        del own, o0, o1
        torch.cuda.empty_cache()
        print(f"{pair['name']} L{layer} {time.time() - tl:.0f}s", flush=True)
    res = {"meta": {**pair, "keep": keep.cpu(), "hkeep": hkeep.cpu(), "seconds": time.time() - t0}, **out, **ids}
    out_dir = Path("/vol/reads32" if pair.get("fp32") else "/vol/reads")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res, out_dir / f"{pair['name']}.pt")
    volume.commit()
    med = {k: float(v[~v.isnan()].median()) for k, v in out.items() if (~v.isnan()).any()}
    return {"name": pair["name"], "seconds": round(time.time() - t0), "median": med}


# ---------------------------------------------------------------- activations
class Sides:
    """Both sides of a pair as callables on the same token batch. Sides that share a model id share one copy in GPU
    memory, with LoRA adapters kept unmerged (merging a small update into bf16 weights would round much of it away)."""

    def __init__(self, sides):
        import torch
        from transformers import AutoModelForCausalLM

        self.sides = sides
        ids = {s["model"] for s in sides}
        self.models = {}
        for mid in ids:
            m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
            ads = [s["adapter"] for s in sides if s["model"] == mid and s.get("adapter")]
            if ads:
                from peft import PeftModel

                m = PeftModel.from_pretrained(m, f"/vol/adapters/{ads[0]}", adapter_name=ads[0].replace("/", "_"))
                for ad in ads[1:]:
                    if ad != ads[0]:
                        m.load_adapter(f"/vol/adapters/{ad}", adapter_name=ad.replace("/", "_"))
                m.eval()
            self.models[mid] = m

    def core(self, i):
        """The decoder (`.layers`) of side i's model."""
        m = self.models[self.sides[i]["model"]]
        return m.base_model.model.model if hasattr(m, "peft_config") else m.model

    def run(self, i, **kw):
        from contextlib import nullcontext

        s = self.sides[i]
        m = self.models[s["model"]]
        if hasattr(m, "peft_config"):
            if s.get("adapter"):
                m.set_adapter(s["adapter"].replace("/", "_"))
                ctx = nullcontext()
            else:
                ctx = m.disable_adapter()
        else:
            ctx = nullcontext()
        with ctx:
            return m(**kw)

    def o_weight(self, i, layer):
        """Side i's effective W_O at a layer (base weight plus its LoRA update), fp32 [d, H*hd]."""
        mod = self.core(i).layers[layer].self_attn.o_proj
        s = self.sides[i]
        if not hasattr(mod, "base_layer"):
            return mod.weight.float()
        w = mod.base_layer.weight.float()
        ad = (s.get("adapter") or "").replace("/", "_")
        if ad:
            w = w + mod.scaling[ad] * (mod.lora_B[ad].weight.float() @ mod.lora_A[ad].weight.float())
        return w


def _encode(tok, docs, max_reply=256):
    """Each doc as the assistant reply to a fixed request, thinking off. Returns (ids, reply-token mask) per doc."""
    prompt = tok.apply_chat_template([{"role": "user", "content": "Write a short text."}], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)  # fmt: skip
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    rows = []
    for d in docs:
        r = tok(d["text"], add_special_tokens=False)["input_ids"][:max_reply]
        rows.append((p_ids + r, [0] * len(p_ids) + [1] * len(r)))
    return rows


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 3)
def acts(pair: dict, docs: list[dict], rulers: dict | None = None) -> dict:
    """docs: [{id, text, label (1 AI, 0 human), split (fit/eval)}]. Saves /vol/acts/<name>.pt.

    Pass 1 (fit split, `before` side): reply-token residual means by label per layer -> AI direction d_L (unit).
    Pass 2 (all docs, both sides on the same tokens): per neuron (the down_proj input) and per head output norm, sums
    by split x label of x and x^2 per side, and of x_before * x_after; per doc, the mean projection on d_L of each
    layer's MLP output, attention output and residual, per side (and of the residual on any extra `rulers`).
    """
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    sides = [pair["before"], pair["after"]]
    tok = AutoTokenizer.from_pretrained(sides[0]["model"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 151643
    rows = _encode(tok, docs)
    M = Sides(sides)
    cfg = M.core(0).config
    L, d, f, H, hd = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, 128
    split = torch.tensor([0 if x["split"] == "fit" else 1 for x in docs])
    label = torch.tensor([int(x["label"]) for x in docs])
    BS = 8

    def batches(idx):
        for s in range(0, len(idx), BS):
            chunk = idx[s : s + BS]
            width = max(len(rows[i][0]) for i in chunk)
            ids = torch.tensor([rows[i][0] + [pad] * (width - len(rows[i][0])) for i in chunk]).cuda()
            att = torch.tensor([[1] * len(rows[i][0]) + [0] * (width - len(rows[i][0])) for i in chunk]).cuda()
            rm = torch.tensor([rows[i][1] + [0] * (width - len(rows[i][1])) for i in chunk]).cuda().bool()
            yield chunk, ids, att, rm

    # pass 1: AI direction per layer from the before side's residual (output of decoder layer L) on the fit split
    fit = [i for i in range(len(docs)) if docs[i]["split"] == "fit"]
    rsum = torch.zeros(2, L, d, dtype=torch.float64, device="cuda")
    rcnt = torch.zeros(2, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        for chunk, ids, att, rm in batches(fit):
            hs = M.run(0, input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
            w = rm.double() / rm.double().sum(1, keepdim=True)
            lab = label[chunk].cuda()
            for layer in range(L):
                doc_mean = torch.einsum("bt,btd->bd", w, hs[layer + 1].double())
                for c in range(2):
                    rsum[c, layer] += doc_mean[lab == c].sum(0)
            for c in range(2):
                rcnt[c] += float((lab == c).sum())
    mu = rsum / rcnt[:, None, None]
    dirs = {"fit": (mu[1] - mu[0]).float()}
    for k, v in (rulers or {}).items():  # extra rulers {name: {layer: [d]}}
        dd = torch.zeros(L, d, device="cuda")
        for layer, vec in v.items():
            dd[int(layer)] = torch.tensor(vec, device="cuda")
        dirs[k] = dd
    dirs = {k: v / v.norm(dim=1, keepdim=True).clamp_min(1e-9) for k, v in dirs.items()}
    D = dirs["fit"]

    # pass 2: both sides on the same tokens
    groups = [(sp, lab) for sp in range(2) for lab in range(2)]
    nsum = {k: {g: torch.zeros(2, L, f, dtype=torch.float64, device="cuda") for g in groups} for k in ("x", "x2")}
    hsum = {k: {g: torch.zeros(2, L, H, dtype=torch.float64, device="cuda") for g in groups} for k in ("x", "x2")}
    ncross = torch.zeros(L, f, dtype=torch.float64, device="cuda")
    hcross = torch.zeros(L, H, dtype=torch.float64, device="cuda")
    ntok = {g: 0 for g in groups}
    proj = {k: torch.zeros(2, len(docs), L) for k in ("mlp", "attn", "resid")}
    proj_x = {k: torch.zeros(2, len(docs), L) for k in dirs if k != "fit"}
    cap = {}

    cur = {"side": 0}

    def hooks(core):
        hs = []
        for layer, blk in enumerate(core.layers):

            def pre_down(mod, args, layer=layer):
                cap[cur["side"], "a", layer] = args[0]

            def post_mlp(mod, args, out, layer=layer):
                cap[cur["side"], "mlp", layer] = out

            def pre_o(mod, args, layer=layer):
                cap[cur["side"], "z", layer] = args[0]

            def post_o(mod, args, out, layer=layer):
                cap[cur["side"], "attn", layer] = out

            hs += [blk.mlp.down_proj.register_forward_pre_hook(pre_down), blk.mlp.register_forward_hook(post_mlp),
                   blk.self_attn.o_proj.register_forward_pre_hook(pre_o),
                   blk.self_attn.o_proj.register_forward_hook(post_o)]  # fmt: skip
        return hs

    Wo = [[M.o_weight(s, layer).view(d, H, hd) for layer in range(L)] for s in range(2)]
    cores = {id(M.core(s)): M.core(s) for s in range(2)}
    handles = [h for c in cores.values() for h in hooks(c)]
    with torch.no_grad():
        for chunk, ids, att, rm in batches(list(range(len(docs)))):
            hs = []
            for s in range(2):
                cur["side"] = s
                hs.append(M.run(s, input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states)
            sp = split[chunk].cuda()
            lb = label[chunk].cuda()
            tok_split = sp[:, None].expand_as(rm)[rm]
            tok_label = lb[:, None].expand_as(rm)[rm]
            sels = {g: (tok_split == g[0]) & (tok_label == g[1]) for g in groups}
            for layer in range(L):
                a = [cap[s, "a", layer][rm].float() for s in range(2)]  # [N, f]
                hn = []
                for s in range(2):
                    z = cap[s, "z", layer][rm].float().view(-1, H, hd)
                    hn.append(torch.einsum("nhk,dhk->nhd", z, Wo[s][layer]).norm(dim=-1))  # [N, H]
                ncross[layer] += (a[0] * a[1]).sum(0).double()
                hcross[layer] += (hn[0] * hn[1]).sum(0).double()
                for g, sel in sels.items():
                    if not sel.any():
                        continue
                    for s in range(2):
                        nsum["x"][g][s, layer] += a[s][sel].sum(0).double()
                        nsum["x2"][g][s, layer] += a[s][sel].pow(2).sum(0).double()
                        hsum["x"][g][s, layer] += hn[s][sel].sum(0).double()
                        hsum["x2"][g][s, layer] += hn[s][sel].pow(2).sum(0).double()
                w = rm.float() / rm.float().sum(1, keepdim=True)  # [B, T] mean over each doc's reply tokens
                for s in range(2):
                    mlp = torch.einsum("bt,btd->bd", w, cap[s, "mlp", layer].float())
                    att_o = torch.einsum("bt,btd->bd", w, cap[s, "attn", layer].float())
                    res = torch.einsum("bt,btd->bd", w, hs[s][layer + 1].float())
                    proj["mlp"][s, chunk, layer] = (mlp @ D[layer]).cpu()
                    proj["attn"][s, chunk, layer] = (att_o @ D[layer]).cpu()
                    proj["resid"][s, chunk, layer] = (res @ D[layer]).cpu()
                    for k, px in proj_x.items():
                        px[s, chunk, layer] = (res @ dirs[k][layer]).cpu()
            for g, sel in sels.items():
                ntok[g] += int(sel.sum())
            cap.clear()
    for h in handles:
        h.remove()
    res = {"meta": {**pair, "n_docs": len(docs), "ids": [x["id"] for x in docs], "seconds": time.time() - t0,
                    "ntok": {f"{a}{b}": v for (a, b), v in ntok.items()}},
           "dirs": {k: v.cpu() for k, v in dirs.items()}, "ncross": ncross.cpu(), "hcross": hcross.cpu(),
           "nsum": {k: {f"{a}{b}": v.cpu() for (a, b), v in dd.items()} for k, dd in nsum.items()},
           "hsum": {k: {f"{a}{b}": v.cpu() for (a, b), v in dd.items()} for k, dd in hsum.items()},
           "proj": proj, "proj_x": proj_x}  # fmt: skip
    Path("/vol/acts").mkdir(parents=True, exist_ok=True)
    torch.save(res, f"/vol/acts/{pair['name']}.pt")
    volume.commit()
    return {"name": pair["name"], "seconds": round(time.time() - t0), "ntok": res["meta"]["ntok"]}


# ---------------------------------------------------------------- twins (deslop-h2h's LoRA recipe)
def _tokenize(tok, prompt: str, reply: str):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)  # fmt: skip
    p_ids = tok(text, add_special_tokens=False)["input_ids"]
    r_ids = tok(reply + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return p_ids + r_ids, [-100] * len(p_ids) + r_ids


@app.function(image=image, gpu="H100", volumes={"/vol": volume}, secrets=[hf], timeout=60 * 60 * 2)
def train_twin(examples: list[dict], name: str, seed: int = 0, epochs: int = 2, lr: float = 2e-4, rank: int = 16,
               alpha: int = 32, batch: int = 8) -> dict:  # fmt: skip
    """Copied from deslop-h2h `train_sft`: every projection, loss on the reply only, bare prompt, right padding."""
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    base = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    rows = [_tokenize(tok, ex["prompt"], ex["reply"]) for ex in examples]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * ((len(rows) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    rng = random.Random(seed)
    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            chunk = [rows[i] for i in order[b : b + batch]]
            width = max(len(x) for x, _ in chunk)
            ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x, _ in chunk], device="cuda")
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in chunk], device="cuda")
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in chunk], device="cuda")
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, "loss": float(loss)})
    model.save_pretrained(f"/vol/adapters/{name}")
    volume.commit()
    return {"name": name, "steps": steps, "examples": len(rows), "seconds": round(time.time() - t0), "log": log,
            "seed": seed}  # fmt: skip
