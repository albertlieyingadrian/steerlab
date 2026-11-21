"""GPU work for Voice Alchemy, all through TransformerLens 4.3's TransformerBridge on Gemma-2-2B-it (runs inside the
Modal container; see modal_app.py).

- `find(out)`: set F replies (neutral + one per trait), teacher-forced after the plain question; per-reply activation
  means (residual stream, MLP neurons, Gemma Scope SAE features); then every unit and its shape-matched random control
  for each candidate layer L* in {6, 9, 12}. Writes units.pt, units_meta.json, f_replies.jsonl.
- `generate(rows, units)`: greedy steered generation, a different push per batch row.

A unit is stored as {layer: vector}, scaled so that sum over layers of |v_L| / R_L = 1, where R_L is the mean residual
norm at layer L (BOS excluded). A push of strength s adds s * v_L to blocks.L.hook_resid_post at every position.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

import va

CAND_LAYERS = va.CFG["cand_layers"]
MLP_LAYERS = va.CFG["mlp_layers"]  # covers every community window L*-2..L*+2
N_CAND = 3
POOL_PER_TRAIT = 40
EDGE_R = 0.2
SAE_REPO = "google/gemma-scope-2b-pt-res"
QWEN_SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
QWEN_K = 50


def boot():
    from transformer_lens import TransformerBridge

    torch.set_grad_enabled(False)
    model = TransformerBridge.boot_transformers(va.MODEL, device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model


def chat(tok, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )  # Qwen3: no thinking block; Gemma ignores the flag


def user_text(q: str, prefix: str | None) -> str:
    return q if not prefix else f"{q}\n\n{prefix}"


# --- generation ---------------------------------------------------------------------------------------------------
def generate(model, rows: list[dict], units: dict, bs: int = 64, max_new: int = va.MAX_NEW) -> list[dict]:
    """rows: {rid, q, prefix (or None), push: [[unit_id, s], ...]}. Returns rows with text, n_tokens, hit_eos."""
    tok = model.tokenizer
    eos = {tok.eos_token_id} | {
        i
        for i in (tok.convert_tokens_to_ids(t) for t in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>"))
        if isinstance(i, int) and i != tok.unk_token_id
    }
    out = []
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        prompts = [chat(tok, user_text(r["q"], r.get("prefix"))) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to("cuda")
        per_layer: dict[int, torch.Tensor] = {}
        for b, r in enumerate(batch):
            for uid, s in r.get("push", []):
                for L, v in units[uid]["parts"].items():
                    if L not in per_layer:
                        per_layer[L] = torch.zeros(len(batch), model.cfg.d_model, device="cuda")
                    per_layer[L][b] += s * v.to("cuda")
        hooks = []
        for L, V in per_layer.items():
            Vb = V.to(torch.bfloat16)[:, None, :]

            def add(x, hook, Vb=Vb):
                return x + Vb[: x.shape[0]]

            hooks.append((f"blocks.{L}.hook_resid_post", add))
        with model.hooks(fwd_hooks=hooks):
            ids = model.hf_generate(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.0,
                return_type="tokens",
            )
        n = enc.input_ids.shape[1]
        for b, r in enumerate(batch):
            new = ids[b, n:].tolist()
            cut = next((k for k, t in enumerate(new) if t in eos), None)
            new = new[:cut] if cut is not None else new
            out.append(
                {
                    **r,
                    "text": tok.decode(new, skip_special_tokens=True).strip(),
                    "n_tokens": len(new),
                    "hit_eos": cut is not None,
                }
            )
    return out


# --- unit finding -------------------------------------------------------------------------------------------------
def load_sae(L: int):
    from huggingface_hub import HfApi, hf_hub_download

    if va.TAG == "qwen":  # Qwen-Scope: a TopK SAE (k = 50), one file per layer, W_dec stored (d_model, n_features)
        st = torch.load(hf_hub_download(QWEN_SAE_REPO, f"layer{L}.sae.pt"), map_location="cuda", weights_only=True)
        return {
            "kind": "topk",
            "W_enc": st["W_enc"].float().T.contiguous(),
            "b_enc": st["b_enc"].float(),
            "W_dec": st["W_dec"].float().T.contiguous(),
            "l0": QWEN_K,
        }

    files = [
        f
        for f in HfApi().list_repo_files(SAE_REPO)
        if f.startswith(f"layer_{L}/width_16k/") and f.endswith("params.npz")
    ]
    l0s = sorted(int(f.split("average_l0_")[1].split("/")[0]) for f in files)
    l0 = min(l0s, key=lambda x: abs(x - 100))  # Gemma Scope's canonical choice: the L0 closest to 100
    p = np.load(hf_hub_download(SAE_REPO, f"layer_{L}/width_16k/average_l0_{l0}/params.npz"))
    sae = {k: torch.tensor(p[k], device="cuda", dtype=torch.float32) for k in ("W_enc", "b_enc", "W_dec", "threshold")}
    sae["l0"] = l0
    return sae


def sae_encode(sae, x: torch.Tensor) -> torch.Tensor:
    pre = x @ sae["W_enc"] + sae["b_enc"]
    if sae.get("kind") == "topk":
        vals, idx = pre.topk(QWEN_K, dim=-1)
        return pre.new_zeros(pre.shape).scatter_(-1, idx, vals.clamp(min=0))
    return pre * (pre > sae["threshold"])


def cohen_d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a, b: (n, k). Returns (d, mean difference) per column."""
    diff = a.mean(0) - b.mean(0)
    sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-6
    return diff / sd, diff


def find(out: Path) -> None:
    import networkx as nx

    model = boot()
    tok = model.tokenizer
    out.mkdir(parents=True, exist_ok=True)
    styles = [None, *va.TRAITS]
    rows = [
        {
            "rid": f"F{pi}-{s or 'neutral'}",
            "q": q,
            "style": s or "neutral",
            "prefix": va.instruction(s) if s else None,
            "push": [],
        }
        for pi, q in enumerate(va.F_PROMPTS)
        for s in styles
    ]
    rows = generate(model, rows, {})
    with (out / "f_replies.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("rid", "q", "style", "text", "n_tokens")}) + "\n")

    saes = {L: load_sae(L) for L in CAND_LAYERS}
    n_layers = model.cfg.n_layers
    resid_mean = {L: [] for L in CAND_LAYERS}
    mlp_mean = {L: [] for L in MLP_LAYERS}
    sae_mean = {L: [] for L in CAND_LAYERS}
    norm_sum = np.zeros(n_layers)
    rms_sum = np.zeros(n_layers)
    n_tok = 0
    seqs = []
    for r in rows:
        p = tok(chat(tok, r["q"]), add_special_tokens=False).input_ids
        a = tok(r["text"], add_special_tokens=False).input_ids or tok(".", add_special_tokens=False).input_ids
        seqs.append((torch.tensor([p + a], device="cuda"), len(p), len(p) + len(a)))

    for ids, s0, s1 in seqs:
        cache = {}

        def grab(name, cache=cache):
            def f(x, hook):
                cache[name] = x[0].float()
                return x

            return f

        hooks = [(f"blocks.{L}.hook_resid_post", grab(("r", L))) for L in range(n_layers)]
        hooks += [(f"blocks.{L}.mlp.hook_post", grab(("m", L))) for L in MLP_LAYERS]
        if va.TAG == "gemma":  # only Gemma-2 renormalises the MLP output
            hooks += [(f"blocks.{L}.ln2_post.hook_in", grab(("pre", L))) for L in range(n_layers)]
        model.run_with_hooks(ids, fwd_hooks=hooks)
        for L in range(n_layers):
            norm_sum[L] += cache[("r", L)][1:].norm(dim=-1).sum().item()
            if ("pre", L) in cache:
                rms_sum[L] += cache[("pre", L)][1:].pow(2).mean(-1).sqrt().sum().item()
        n_tok += ids.shape[1] - 1
        for L in CAND_LAYERS:
            resid_mean[L].append(cache[("r", L)][s0:s1].mean(0).cpu().numpy())
            sae_mean[L].append(sae_encode(saes[L], cache[("r", L)][s0:s1]).mean(0).cpu().numpy())
        for L in MLP_LAYERS:
            mlp_mean[L].append(cache[("m", L)][s0:s1].mean(0).cpu().numpy())

    R = norm_sum / n_tok
    rms = rms_sum / n_tok
    style_of = np.array([r["style"] for r in rows])
    neutral = style_of == "neutral"
    lm = model.original_model.model.layers

    def write_vec(L: int, idx: int) -> torch.Tensor:
        """What raising neuron idx by 1 writes into the residual stream, through Gemma-2's post-MLP norm gain."""
        col = lm[L].mlp.down_proj.weight[:, idx].float()
        if not hasattr(lm[L], "post_feedforward_layernorm"):  # Qwen: the neuron's output column is its write
            return col.cpu()
        g = 1.0 + lm[L].post_feedforward_layernorm.weight.float()
        return (col * g / rms[L]).cpu()

    def scaled(parts: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        tot = sum(v.norm().item() / R[L] for L, v in parts.items())
        return {L: (v / tot).float() for L, v in parts.items()}

    WU = model.original_model.lm_head.weight.float()

    def top_tokens(parts) -> list[str]:
        v = sum(p.to("cuda") for p in parts.values())
        return [tok.decode([i]) for i in (WU @ v).topk(12).indices.tolist()]

    units: dict[str, dict] = {}
    meta: dict = {"R": R.tolist(), "rms": rms.tolist(), "sae_l0": {L: saes[L]["l0"] for L in CAND_LAYERS}, "units": {}}
    rng = random.Random(0)
    torch.manual_seed(0)
    mlp_arr = {L: np.stack(mlp_mean[L]) for L in MLP_LAYERS}
    sae_arr = {L: np.stack(sae_mean[L]) for L in CAND_LAYERS}
    resid_arr = {L: np.stack(resid_mean[L]) for L in CAND_LAYERS}
    alive = {L: np.where(sae_arr[L].max(0) > 0)[0] for L in CAND_LAYERS}

    def add(uid, typ, trait, Lstar, rank, parts, info):
        parts = scaled(parts)
        units[uid] = {"parts": parts}
        meta["units"][uid] = {
            "type": typ,
            "trait": trait,
            "L": Lstar,
            "rank": rank,
            "layers": sorted(parts),
            "top_tokens": top_tokens(parts),
            **info,
        }

    for Lstar in CAND_LAYERS:
        window = [L for L in MLP_LAYERS if abs(L - Lstar) <= 2]
        stats = {}
        for t in va.TRAITS:
            is_t = style_of == t
            stats[t] = {
                "mlp": {L: cohen_d(mlp_arr[L][is_t], mlp_arr[L][neutral]) for L in window},
                "sae": cohen_d(sae_arr[Lstar][is_t], sae_arr[Lstar][neutral]),
            }
            # direction (baseline) and its random control
            dvec = resid_arr[Lstar][is_t].mean(0) - resid_arr[Lstar][neutral].mean(0)
            uid = f"L{Lstar}-direction-{t}"
            add(uid, "direction", t, Lstar, 0, {Lstar: torch.tensor(dvec)}, {})
            add(
                f"L{Lstar}-rand_direction-{t}",
                "rand_direction",
                t,
                Lstar,
                0,
                {Lstar: torch.randn(model.cfg.d_model)},
                {"control_of": uid},
            )
            # single neurons
            d, diff = stats[t]["mlp"][Lstar]
            for rank, i in enumerate(int(x) for x in np.argsort(-np.abs(d))[:N_CAND]):
                uid = f"L{Lstar}-neuron-{t}-{rank}"
                add(
                    uid,
                    "neuron",
                    t,
                    Lstar,
                    rank,
                    {Lstar: write_vec(Lstar, i) * float(np.sign(diff[i]))},
                    {"neuron": [Lstar, i], "d": float(d[i]), "diff": float(diff[i])},
                )
                j = rng.randrange(model.cfg.d_mlp)
                add(
                    f"L{Lstar}-rand_neuron-{t}-{rank}",
                    "rand_neuron",
                    t,
                    Lstar,
                    rank,
                    {Lstar: write_vec(Lstar, j)},
                    {"neuron": [Lstar, j], "control_of": uid},
                )
            # SAE features
            d, diff = stats[t]["sae"]
            order = [int(i) for i in np.argsort(-d) if diff[i] > 0][:N_CAND]
            for rank, f_ in enumerate(order):
                uid = f"L{Lstar}-sae-{t}-{rank}"
                add(
                    uid,
                    "sae",
                    t,
                    Lstar,
                    rank,
                    {Lstar: saes[Lstar]["W_dec"][f_].cpu()},
                    {"feature": [Lstar, f_], "d": float(d[f_]), "diff": float(diff[f_]), "l0": saes[Lstar]["l0"]},
                )
                fr = int(rng.choice(list(alive[Lstar])))
                add(
                    f"L{Lstar}-rand_sae-{t}-{rank}",
                    "rand_sae",
                    t,
                    Lstar,
                    rank,
                    {Lstar: saes[Lstar]["W_dec"][fr].cpu()},
                    {"feature": [Lstar, fr], "control_of": uid},
                )

        # communities: pool of top neurons per trait in the window, co-activation graph, Louvain
        pool: set = set()
        for t in va.TRAITS:
            cand = []
            for L in window:
                dL = stats[t]["mlp"][L][0]
                cand += [(abs(dL[i]), L, int(i)) for i in np.argsort(-np.abs(dL))[:POOL_PER_TRAIT]]
            pool.update((L, i) for _, L, i in sorted(cand, reverse=True)[:POOL_PER_TRAIT])
        pool_l = sorted(pool)
        acts = []
        for ids, s0, s1 in seqs:
            got = {}

            def grab2(L, s0=s0, s1=s1, got=got):
                def f(act, hook):
                    got[L] = act[0, s0:s1].float()
                    return act

                return f

            model.run_with_hooks(ids, fwd_hooks=[(f"blocks.{L}.mlp.hook_post", grab2(L)) for L in window])
            acts.append(torch.stack([got[L][:, i] for L, i in pool_l], 1).cpu().numpy())
        A = np.concatenate(acts, 0)
        C = np.nan_to_num(np.corrcoef(A.T))
        G = nx.Graph()
        G.add_nodes_from(range(len(pool_l)))
        for a in range(len(pool_l)):
            for b in range(a + 1, len(pool_l)):
                if C[a, b] >= EDGE_R:
                    G.add_edge(a, b, weight=float(C[a, b]))
        comms = [sorted(c) for c in nx.community.louvain_communities(G, weight="weight", seed=0) if 3 <= len(c) <= 60]
        meta[f"L{Lstar}_communities"] = {
            "pool": len(pool_l),
            "edges": G.number_of_edges(),
            "tokens": int(A.shape[0]),
            "sizes": sorted((len(c) for c in comms), reverse=True),
        }
        for t in va.TRAITS:
            scored = []
            for c in comms:
                ds = [stats[t]["mlp"][pool_l[k][0]][0][pool_l[k][1]] for k in c]
                scored.append((float(np.mean(ds)), c))
            scored = [x for x in sorted(scored, key=lambda x: -x[0]) if x[0] > 0][:N_CAND]
            for rank, (score, c) in enumerate(scored):
                parts: dict = {}
                rparts: dict = {}
                members = []
                for k in c:
                    L, i = pool_l[k]
                    diff = float(stats[t]["mlp"][L][1][i])
                    parts[L] = parts.get(L, 0) + diff * write_vec(L, i)
                    rparts[L] = rparts.get(L, 0) + diff * write_vec(L, rng.randrange(model.cfg.d_mlp))
                    members.append([L, i, round(float(stats[t]["mlp"][L][0][i]), 3)])
                uid = f"L{Lstar}-community-{t}-{rank}"
                add(uid, "community", t, Lstar, rank, parts, {"members": members, "mean_d": score, "size": len(c)})
                add(
                    f"L{Lstar}-rand_community-{t}-{rank}",
                    "rand_community",
                    t,
                    Lstar,
                    rank,
                    rparts,
                    {"size": len(c), "control_of": uid},
                )

    torch.save(
        {u: {"parts": {int(L): v.half() for L, v in d["parts"].items()}} for u, d in units.items()}, out / "units.pt"
    )
    (out / "units_meta.json").write_text(json.dumps(meta, indent=1))


def load_units(path: Path) -> dict:
    raw = torch.load(path)
    return {u: {"parts": {int(L): v.float() for L, v in d["parts"].items()}} for u, d in raw.items()}
