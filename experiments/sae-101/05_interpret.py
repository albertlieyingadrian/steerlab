"""
Step 5: what do the latents mean? Four readings, for latents drawn at random (not picked for looking good):

    top snippets     the 20 places in 5M held-out tokens where each latent fires hardest, with the words around them
    logit lens       which next tokens each latent's decoder direction pushes up and down (through the final norm and
                     the unembedding)
    histogram        how strongly it fires when it fires
    use test         remove the latent from the residual stream where it fires hardest: does the model's next-token
                     prediction change more than when a random direction of the same size is removed?

Plus a picture of superposition in the real model: how close each decoder direction is to its nearest neighbour,
against as many random directions in the same 896 dimensions.

Writes results/<size>/interpret.json (the dashboard and the optional auto-labels read it).

    .venv/bin/python experiments/sae-101/05_interpret.py --size small
"""

import argparse

import numpy as np
import torch

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
ap.add_argument("--n-random", type=int, default=48, help="latents drawn at random for the dashboard")
ap.add_argument("--n-note", type=int, default=12, help="of those, how many the note shows and the use test covers")
args = ap.parse_args()
work = s.WORK / args.size
TOP, HELD, CTX_BEFORE, CTX_AFTER = 20, 10, 12, 4

model, tok, device = s.load_lm()
sae, saved = s.load_sae(work / "sae_handwritten.pt", device)
scale = saved["scale"]
interp = s.load_tokens(args.size, "interp")
freq = np.load(work / "freq_handwritten.npy")

# ---- the draw: live, not dense, seeded (so nobody chose the latents) ----
rng = np.random.default_rng(s.SEED)
eligible = np.flatnonzero((freq > 1e-5) & (freq < 0.1))
drawn = rng.choice(eligible, size=min(args.n_random, len(eligible)), replace=False)
print(
    f"1. Drew {len(drawn)} of {len(eligible):,} latents that fire on between 1 in 100,000 and 1 in 10 tokens "
    f"(seed {s.SEED}); the first {args.n_note} are the ones in the note."
)
# latents step 6 picked for a concept (code, digits) get snippets too, marked as chosen, not drawn
chosen = {}
compare = s.read_json(s.RESULTS / args.size / "compare.json") or {}
for concept, r in compare.items():
    for d in r["sae_top5"][:3]:
        if d["id"] not in drawn:
            chosen.setdefault(d["id"], f"chosen in step 6: {r['title']} (held-out AUROC {d['test_auroc']:.3f})")
if chosen:
    print(f"   plus {len(chosen)} latents step 6 chose for its concepts: {sorted(chosen)}")
shown = np.concatenate([drawn, np.array(sorted(chosen), dtype=drawn.dtype)])

# ---- one pass over the interp windows: the top activations of every latent ----
d_sae = sae.d_sae
top_val = torch.full((d_sae, TOP + HELD), -1.0, device=device)
top_pos = torch.zeros((d_sae, TOP + HELD), dtype=torch.long, device=device)
drawn_t = torch.as_tensor(shown, device=device)
hist_vals = {int(j): [] for j in shown}
n_tok = 0
with torch.no_grad():
    for w0 in range(0, len(interp), 16):
        toks = torch.as_tensor(np.asarray(interp[w0 : w0 + 16], dtype=np.int64), device=device)
        mask = s.text_mask(toks)
        z = sae.encode(s.residual(model, toks)[mask] * scale)  # [n, d_sae]
        flat_pos = (torch.arange(toks.numel(), device=device).view_as(toks) + w0 * s.CTX)[mask]
        vals, idx = z.topk(min(TOP + HELD, len(z)), dim=0)  # [TOP+HELD, d_sae]
        cat_v = torch.cat([top_val, vals.T], 1)
        cat_p = torch.cat([top_pos, flat_pos[idx].T], 1)
        best = cat_v.topk(TOP + HELD, dim=1)
        top_val, top_pos = best.values, cat_p.gather(1, best.indices)
        sub = z[:, drawn_t].cpu()
        for col, j in enumerate(shown):
            if len(hist_vals[int(j)]) < 20000:
                v = sub[:, col]
                hist_vals[int(j)] += v[v > 0].tolist()
        n_tok += len(z)
print(f"   scanned {n_tok:,} held-out tokens")

flat = np.asarray(interp).reshape(-1)


def snippet(pos, acts=None):
    """Tokens around a position, the firing token marked by "at", with this latent's activation on each token."""
    w, p = divmod(int(pos), s.CTX)
    lo, hi = max(s.SKIP, p - CTX_BEFORE), min(s.CTX, p + CTX_AFTER + 1)
    ids = flat[w * s.CTX + lo : w * s.CTX + hi]
    a = acts[lo:hi] if acts is not None else [0.0] * len(ids)
    return {"tokens": [tok.decode([int(t)]) for t in ids], "acts": [round(float(x), 3) for x in a], "at": p - lo}


@torch.no_grad()
def window_acts(window_index, latent):
    toks = torch.as_tensor(np.asarray(interp[window_index : window_index + 1], dtype=np.int64), device=device)
    z = sae.encode(s.residual(model, toks)[0] * scale)[:, latent]
    z[~s.text_mask(toks)[0]] = 0
    return z.cpu().numpy()


# logit lens: the decoder direction (raw units) through the final RMSNorm weight and the unembedding
W_U = model.get_output_embeddings().weight  # [vocab, d_model]
norm_w = model.model.norm.weight

latents = []
for j in shown:
    j = int(j)
    vals, poss = top_val[j].cpu().numpy(), top_pos[j].cpu().numpy()
    examples = []
    for rank in range(TOP + HELD):
        if vals[rank] <= 0:
            break
        sn = snippet(poss[rank], window_acts(int(poss[rank]) // s.CTX, j))
        sn["value"] = round(float(vals[rank]), 3)
        examples.append(sn)
    with torch.no_grad():
        logits = (sae.W_dec[j] / scale * norm_w) @ W_U.T
        up, down = logits.topk(10).indices.tolist(), (-logits).topk(10).indices.tolist()
    counts, edges = np.histogram(np.array(hist_vals[j] or [0.0]), bins=20)
    latents.append(
        {
            "id": j,
            "chosen": chosen.get(j),
            "freq": float(freq[j]),
            "max": float(vals[0]),
            "top": examples[:TOP],
            "held_out": examples[TOP:],
            "logit_up": [tok.decode([t]) for t in up],
            "logit_down": [tok.decode([t]) for t in down],
            "hist": {"counts": counts.tolist(), "edges": [round(float(e), 3) for e in edges]},
        }
    )
print("\n2. The first latents of the draw (the firing token in [brackets]):")
for lat in latents[: args.n_note]:
    ex = lat["top"][0]
    text = "".join(t if i != ex["at"] else f"[{t}]" for i, t in enumerate(ex["tokens"])).replace("\n", "⏎")
    print(f"   latent {lat['id']:>5}  fires on {lat['freq']:.3%} of tokens  top: {text!r}")
    print(f"                 pushes up: {lat['logit_up'][:6]}")

# ---- negatives for the auto-label check: random snippets from the first 400 windows ----
text_positions = np.flatnonzero(
    s.text_mask(torch.as_tensor(np.asarray(interp[:400], dtype=np.int64))).numpy().reshape(-1)
)
negatives = [snippet(p) for p in rng.choice(text_positions, size=400, replace=False)]

# ---- the use test: remove the latent at its top positions vs a same-norm random direction ----
gen = torch.Generator().manual_seed(s.SEED)


@torch.no_grad()
def effect(window_index, p, delta):
    """KL from the clean next-token distribution at position p when `delta` is subtracted from the residual there."""
    toks = torch.as_tensor(np.asarray(interp[window_index : window_index + 1], dtype=np.int64), device=device)
    clean = torch.log_softmax(model(toks).logits[0, p].float(), -1)

    def hook(module, inputs, output):
        h = (output[0] if isinstance(output, tuple) else output).clone()
        h[0, p] -= delta
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = model.model.layers[s.LAYER].register_forward_hook(hook)
    try:
        ablated = torch.log_softmax(model(toks).logits[0, p].float(), -1)
    finally:
        handle.remove()
    return float((clean.exp() * (clean - ablated)).sum())


use = []
for lat in latents[: args.n_note]:
    j = lat["id"]
    direction = (sae.W_dec[j] / scale).detach()
    kl_f, kl_r = [], []
    for rank in range(len(lat["top"])):
        w, p = divmod(int(top_pos[j, rank]), s.CTX)
        delta = float(top_val[j, rank]) * direction  # exactly this latent's share of the rebuild, in raw units
        u = torch.randn(delta.shape, generator=gen).to(device)
        kl_f.append(effect(w, p, delta))
        kl_r.append(effect(w, p, u / u.norm() * delta.norm()))
    use.append(
        {
            "id": j,
            "kl_feature": float(np.mean(kl_f)),
            "kl_random": float(np.mean(kl_r)),
            "n": len(kl_f),
            "all_feature": kl_f,
            "all_random": kl_r,
        }
    )
pooled_f = np.concatenate([u_["all_feature"] for u_ in use])
pooled_r = np.concatenate([u_["all_random"] for u_ in use])
ratio = float(pooled_f.mean() / pooled_r.mean())
boot = np.random.default_rng(s.SEED).integers(0, len(pooled_f), (1000, len(pooled_f)))
ratio_ci = [float(np.percentile(pooled_f[boot].mean(1) / pooled_r[boot].mean(1), q)) for q in (2.5, 97.5)]
wins = sum(u_["kl_feature"] > u_["kl_random"] for u_ in use)
print(
    f"\n3. Use test at each latent's top {TOP} positions: removing the latent moves the next-token distribution by "
    f"KL {pooled_f.mean():.4f} on average; a random direction of the same length {pooled_r.mean():.4f} "
    f"({ratio:.1f}x, 95% CI [{ratio_ci[0]:.1f}, {ratio_ci[1]:.1f}]). The latent beats its random twin for {wins} of "
    f"{len(use)} latents."
)

# ---- superposition in the real model: nearest-neighbour cosine of decoder directions ----
with torch.no_grad():
    D = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
    R = torch.randn(D.shape, generator=torch.Generator().manual_seed(1)).to(device)
    R = R / R.norm(dim=1, keepdim=True)

    def nearest(M):
        best = torch.empty(len(M), device=device)
        for i in range(0, len(M), 2048):
            c = (M[i : i + 2048] @ M.T).abs()
            c[torch.arange(len(c)), torch.arange(i, i + len(c))] = 0
            best[i : i + 2048] = c.max(1).values
        return best.cpu().numpy()

    nn_dec, nn_rand = nearest(D), nearest(R)
bins = np.linspace(0, 1, 51)
print(
    f"\n4. Superposition: {d_sae:,} decoder directions in {sae.d_in} dimensions. Median |cosine| to the nearest other "
    f"direction {np.median(nn_dec):.3f} (random directions: {np.median(nn_rand):.3f}); "
    f"{(nn_dec > 0.9).mean():.1%} have a near-twin above 0.9."
)

s.save_json(
    s.RESULTS / args.size / "interpret.json",
    {
        "drawn": [int(j) for j in drawn],
        "n_note": args.n_note,
        "tokens_scanned": n_tok,
        "latents": latents,
        "negatives": negatives,
        "use_test": {
            "per_latent": [{k: v for k, v in u_.items() if not k.startswith("all_")} for u_ in use],
            "kl_feature": float(pooled_f.mean()),
            "kl_random": float(pooled_r.mean()),
            "ratio": ratio,
            "ratio_ci": ratio_ci,
            "wins": wins,
            "n_latents": len(use),
            "positions": len(pooled_f),
        },
        "superposition": {
            "bins": bins.tolist(),
            "decoder": np.histogram(nn_dec, bins)[0].tolist(),
            "random": np.histogram(nn_rand, bins)[0].tolist(),
            "median_decoder": float(np.median(nn_dec)),
            "median_random": float(np.median(nn_rand)),
            "share_above_0.9": float((nn_dec > 0.9).mean()),
        },
    },
)

print("\nSelf-checks:")
s.check(
    f"every drawn latent has {TOP} top examples", all(len(lat["top"]) == TOP for lat in latents if not lat["chosen"])
)
s.check(
    f"removing a latent where it fires moves the prediction more than a random push of the same size "
    f"(KL ratio {ratio:.1f}x, CI low end {ratio_ci[0]:.1f} > 1)",
    ratio_ci[0] > 1,
)
s.check(
    f"decoder directions sit closer to each other than random ones (median {np.median(nn_dec):.3f} vs "
    f"{np.median(nn_rand):.3f})",
    np.median(nn_dec) > np.median(nn_rand),
)
