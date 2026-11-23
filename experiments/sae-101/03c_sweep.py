"""
Step 3c (exploratory): many small SAEs on a stored slice of activations, for two pictures.

    A. ReLU + L1 against TopK: the older recipe makes latents sparse with a penalty on their size (L1); TopK simply
       keeps the k biggest. For each, the trade-off between sparsity (L0) and rebuild error (FVU).
    B. Common mistakes, each one changed on its own against the same k = 32 TopK baseline: no AuxK, a learning rate
       10x too high or too low, no normalisation, the attention sinks left in, far fewer tokens; and two mistakes made
       when *using* the main SAE: reading the wrong layer, and decoding without the decoder bias.

Every SAE here has 4,096 latents and sees 2 passes over 3M stored tokens, so each takes seconds on a GPU. These numbers
are exploratory (nothing was fixed in advance beyond the list of runs) and labelled so in the note.

    (full size on Modal: modal_app.py spawn sweep, about 15 minutes)
"""

import argparse
import time

import numpy as np
import torch

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
args = ap.parse_args()
cfg = s.SIZES[args.size]
work = s.WORK / args.size
torch.manual_seed(s.SEED)

D_SAE, EPOCHS, BATCH = 4096, 2, 4096
N_TRAIN = 3_000_000 if args.size == "full" else 600_000
N_EVAL = 250_000 if args.size == "full" else 100_000

model, _, device = s.load_lm()
stats = torch.load(work / "stats.pt", weights_only=False)
scale = stats["scale"]
train_w = s.load_tokens(args.size, "train")
eval_w = s.load_tokens(args.size, "eval")


@torch.no_grad()
def stored(windows, n_tokens, mask_fn=s.text_mask, layer=s.LAYER):
    """Raw activations from the first windows until n_tokens are collected, kept in bfloat16 to fit on the GPU."""
    out, have = [], 0
    for i in range(0, len(windows), 32):
        toks = torch.as_tensor(np.asarray(windows[i : i + 32], dtype=np.int64), device=device)
        h = s.residual(model, toks, layer)[mask_fn(toks)]
        out.append(h.to(torch.bfloat16).cpu())
        have += len(h)
        if have >= n_tokens:
            break
    return torch.cat(out)[:n_tokens]


# windows from the end of the training split (the main run reads them too, in a shuffled order; the eval set is held out)
tail = train_w[len(train_w) - (N_TRAIN // 400 + 200) :]
X = stored(tail, N_TRAIN).to(device)  # 5.4 GB in bfloat16 on the GPU
E = stored(eval_w, N_EVAL).float().to(device)
X_sink = stored(tail, N_TRAIN, mask_fn=lambda t: t != s.EOT)  # the attention sinks left in; kept on the CPU


def max_norm(data):
    return max(float(data[i : i + 262144].float().norm(dim=-1).max()) for i in range(0, len(data), 262144))


def mean_sq(data):
    return sum(float((data[i : i + 262144].float() ** 2).sum()) for i in range(0, len(data), 262144)) / len(data)
print(f"1. Stored {len(X):,} training and {len(E):,} held-out activations from the residual stream after block {s.LAYER}")
print(f"   with the sinks left in, the largest norm in the training slice is {max_norm(X_sink):,.0f} "
      f"(without: {max_norm(X):,.0f})")


def batches(data, scale_, epochs=EPOCHS, seed=s.SEED):
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=g).to(data.device)
        for i in range(0, len(data) - BATCH + 1, BATCH):
            yield data[order[i : i + BATCH]].to(device).float() * scale_, 0


@torch.no_grad()
def score(encode_decode, scale_):
    """FVU, L0 and the share of latents that never fire, on the held-out stored activations (raw units)."""
    err = tot = l0 = 0.0
    fired = torch.zeros(D_SAE, dtype=torch.bool, device=device)
    mean = E.mean(0)
    for i in range(0, len(E), 8192):
        x = E[i : i + 8192]
        x_hat, z = encode_decode(x * scale_)
        err += float(((x_hat / scale_ - x) ** 2).sum())
        tot += float(((x - mean) ** 2).sum())
        l0 += float((z > 0).sum())
        fired |= (z > 0).any(0)
    return {"fvu": err / tot, "l0": l0 / len(E), "dead_share": float((~fired).float().mean())}


def steps_for(data, epochs=EPOCHS):
    return epochs * (len(data) // BATCH)


def report(r):
    print(f"   {r['name']:<36} L0 {r['l0']:>6.1f}  FVU {r['fvu']:.4f}  never fired {r['dead_share']:>6.1%}  "
          f"({r['seconds']}s)", flush=True)
    return r


def run_topk(name, k=32, *, lr=4e-4, aux=1 / 32, data=None, scale_=None, epochs=EPOCHS, batch_topk=False, group="topk"):
    data = X if data is None else data
    scale_ = scale if scale_ is None else scale_
    torch.manual_seed(s.SEED)
    sae = s.TopKSAE(s.D_MODEL, D_SAE, k).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(data[:200_000].float().mean(0).to(device) * scale_)
    t0 = time.monotonic()
    s.train_sae(sae, batches(data, scale_, epochs), total_steps=steps_for(data, epochs), lr=lr, aux_coef=aux,
                batch_topk=batch_topk)
    enc_dec = sae
    if batch_topk:  # at inference, BatchTopK becomes a threshold: the average smallest kept value per batch
        with torch.no_grad():
            mins = [float(torch.relu(sae.pre(x)).flatten().topk(k * len(x)).values.min())
                    for x, _ in batches(data[:400_000], scale_, 1)]
        theta = float(np.mean(mins))

        def enc_dec(x):
            pre = sae.pre(x)
            z = torch.where(pre > theta, pre, torch.zeros_like(pre))
            return sae.decode(z), z

    return report({"name": name, "group": group, "k": k, "lr": lr, "aux": aux,
                   "seconds": round(time.monotonic() - t0, 1), **score(enc_dec, scale_)})


def run_l1(name, coef, *, lr=4e-4):
    """The older recipe: z = ReLU(encoder), loss = squared error + coef x sum(z) (decoder rows kept at length 1)."""
    torch.manual_seed(s.SEED)
    sae = s.TopKSAE(s.D_MODEL, D_SAE, k=D_SAE).to(device)  # the same parameters; TopK is simply not used
    with torch.no_grad():
        sae.b_dec.copy_(X[:200_000].float().mean(0) * scale)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    total = steps_for(X)
    t0 = time.monotonic()
    for step, (x, _) in enumerate(batches(X, scale)):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (total - step) / (0.2 * total))
        z = torch.relu(sae.pre(x))
        loss = ((sae.decode(z) - x) ** 2).sum(-1).mean() + coef * z.sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalise_decoder()

    def enc_dec(x):
        z = torch.relu(sae.pre(x))
        return sae.decode(z), z

    return report({"name": name, "group": "relu_l1", "l1": coef, "lr": lr, "seconds": round(time.monotonic() - t0, 1),
                   **score(enc_dec, scale)})


runs = []
print("\n2. A. TopK at five values of k, BatchTopK at three, and ReLU + L1 at seven penalty strengths")
for k in (8, 16, 32, 64, 128):
    runs.append(run_topk(f"TopK k={k}", k))
for k in (16, 32, 64):
    runs.append(run_topk(f"BatchTopK k={k}", k, batch_topk=True, group="batchtopk"))
for coef in (0.5, 1, 2, 4, 8, 16, 32):
    runs.append(run_l1(f"ReLU + L1, coefficient {coef}", coef))

print("\n3. B. Mistakes, each against TopK k=32 (lr 4e-4, AuxK on, normalised, sinks dropped, 2 passes)")
runs.append(run_topk("no AuxK", aux=0.0, group="mistake"))
runs.append(run_topk("learning rate 10x too high (4e-3)", lr=4e-3, group="mistake"))
runs.append(run_topk("learning rate 10x too low (4e-5)", lr=4e-5, group="mistake"))
runs.append(run_topk("no normalisation (raw activations)", scale_=1.0, group="mistake"))
sink_scale = float((s.D_MODEL / mean_sq(X_sink)) ** 0.5)
runs.append(run_topk("attention sinks left in", data=X_sink, scale_=sink_scale, group="mistake"))
runs.append(run_topk("1/20 of the tokens", data=X[: len(X) // 10], epochs=1, group="mistake"))

# two mistakes made when using the main SAE (the full one from step 3a)
main_path = work / "sae_handwritten.pt"
usage = {}
if main_path.exists():
    main = s.raw_from_handwritten(main_path, device)
    usage["right layer (block 11 output)"] = s.fvu_of(main, E)[0]
    for wrong, label in ((s.LAYER - 1, "hidden_states[11]: block 10's output"),
                         (s.LAYER + 1, "hidden_states[13]: block 12's output")):
        usage[f"wrong layer, {label}"] = s.fvu_of(main, stored(eval_w, N_EVAL, layer=wrong))[0]
    sae, saved = s.load_sae(main_path, device)
    sc = saved["scale"]
    no_bias = s.RawSAE(lambda x: sae.encode(x * sc), lambda z: (z @ sae.W_dec) / sc, sae.d_sae, sae.k, sae.W_dec,
                       "decoder bias forgotten")
    usage["decoder bias forgotten"] = s.fvu_of(no_bias, E)[0]
    print("\n4. Using the main SAE wrongly (held-out FVU):")
    for name, v in usage.items():
        print(f"   {name:<44} FVU {v:.4f}")

s.save_json(s.RESULTS / args.size / "sweep.json", {"d_sae": D_SAE, "epochs": EPOCHS, "train_tokens": len(X),
                                                   "eval_tokens": len(E), "runs": runs, "usage": usage})
topk = [r for r in runs if r["group"] == "topk"]
l1 = [r for r in runs if r["group"] == "relu_l1"]


def fvu_at(curve, l0):  # a curve's FVU at a given L0, interpolated on a log L0 axis
    pts = sorted((r["l0"], r["fvu"]) for r in curve if r["l0"] > 0)
    return float(np.interp(np.log(l0), np.log([p[0] for p in pts]), [p[1] for p in pts]))


print("\nSelf-checks:")
s.check(f"TopK's FVU falls as k rises ({', '.join(format(r['fvu'], '.3f') for r in topk)})",
        all(a["fvu"] > b["fvu"] for a, b in zip(topk, topk[1:])))
s.check(f"the L1 sweep spans L0 {min(r['l0'] for r in l1):.0f} to {max(r['l0'] for r in l1):.0f} (covers 16-64)",
        min(r["l0"] for r in l1) < 16 and max(r["l0"] for r in l1) > 64)
s.check(f"at L0 = 32, TopK rebuilds better than ReLU + L1 (FVU {fvu_at(topk, 32):.3f} vs {fvu_at(l1, 32):.3f})",
        fvu_at(topk, 32) < fvu_at(l1, 32))
if usage:
    s.check("the wrong layers and the missing bias are all worse than the right use",
            all(v > usage["right layer (block 11 output)"] for name, v in usage.items() if not name.startswith("right")))
