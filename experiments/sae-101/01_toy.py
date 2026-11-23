"""
Step 1: what an SAE is, on numbers small enough to check by hand, then on a 2-D toy you can see.

Part A, by hand. A 3-number activation, an SAE with 4 latents and k = 2:
    pre = (x - b_dec) @ W_enc + b_enc,  keep the 2 biggest (ReLU), x̂ = z @ W_dec + b_dec.
The script prints every intermediate so you can redo it on paper.

Part B, the toy. Five "concepts" (arrows) squeezed into 2 dimensions: more concepts than dimensions is called
superposition. Each point is 0, 1 or 2 concepts added together, plus a little noise. A TopK SAE with 5 latents is
trained for k = 1, 2 and 3. With k = 1 its decoder rows land on the five arrows; with k = 2 it rebuilds every point
perfectly, but the arrows are no longer pinned down (two arrows can make any 2-D point), so they slide.
Then the four linear tools of step 6, fitted on the same points for the label "concept A is on":
PCA (directions of most variance), a logistic probe, the difference of class means, and the SAE latent for A.

Writes results/toy.json, which the note's interactive figure draws. Runs in a few seconds on a laptop.

    .venv/bin/python experiments/sae-101/01_toy.py
"""

import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import sae101 as s

torch.manual_seed(s.SEED)
rng = np.random.default_rng(s.SEED)

# ---------------------------------------------------------------- Part A: by hand
print("Part A. One activation through a tiny SAE, by hand")
x = torch.tensor([2.0, 0.0, 1.0])
W_dec = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # latent 0: "the first number"
        [0.0, 1.0, 0.0],  # latent 1: "the second number"
        [0.0, 0.0, 1.0],  # latent 2: "the third number"
        [0.6, 0.0, 0.8],
    ]
)  # latent 3: a mix of the first and third (length 1: 0.36 + 0.64 = 1)
W_enc = W_dec.T.clone()  # read each latent by its own direction
b_enc = torch.tensor([0.0, 0.0, 0.0, -0.5])  # latent 3 must clear a small bar before it switches on
b_dec = torch.zeros(3)
pre = (x - b_dec) @ W_enc + b_enc
vals, idx = pre.topk(2)
z = torch.zeros(4).scatter(0, idx, torch.relu(vals))
x_hat = z @ W_dec + b_dec
print(f"   x = {x.tolist()}")
print(f"   pre = x @ W_enc + b_enc = {[round(v, 2) for v in pre.tolist()]}   (latent 3: 2*0.6 + 1*0.8 - 0.5 = 1.5)")
print(f"   keep the 2 biggest: latents {sorted(idx.tolist())}, z = {[round(v, 2) for v in z.tolist()]}")
print(
    f"   x̂ = z @ W_dec = {[round(v, 2) for v in x_hat.tolist()]}   error |x - x̂|² = {float(((x - x_hat) ** 2).sum()):.2f}"
)
print("   Two parts explain x: 'the first number' (2.0) and the mixed latent (1.5 x [0.6, 0, 0.8]).")
print("   With k = 3, latent 2 would join and the error would change; that is the sparsity/accuracy trade-off.")
by_hand = {"x": x.tolist(), "pre": pre.tolist(), "z": z.tolist(), "x_hat": x_hat.tolist()}

# ---------------------------------------------------------------- Part B: the 2-D toy
print("\nPart B. Five concepts in two dimensions")
N_CONCEPTS, N = 5, 600
ANGLES = [10, 80, 150, 215, 290]
concepts = np.stack([np.cos(np.deg2rad(ANGLES)), np.sin(np.deg2rad(ANGLES))], 1)  # five unit arrows
counts = rng.choice([0, 1, 2], size=N, p=[0.1, 0.6, 0.3])
active = np.zeros((N, N_CONCEPTS), dtype=bool)
amounts = np.zeros((N, N_CONCEPTS))
for i, c in enumerate(counts):
    on = rng.choice(N_CONCEPTS, size=c, replace=False)
    active[i, on] = True
    amounts[i, on] = rng.uniform(0.5, 1.5, size=c)
points = amounts @ concepts + rng.normal(0, 0.04, size=(N, 2))
X = torch.tensor(points, dtype=torch.float32)
print(
    f"   {N} points; {int((counts == 0).sum())} with no concept, {int((counts == 1).sum())} with one, "
    f"{int((counts == 2).sum())} with two"
)


def train_toy(k, steps=4000, seeds=5):
    """A TopK SAE with 5 latents. The decoder bias stays at 0 so every arrow starts at the origin (here "no concept
    on" really is the point 0), and decoder rows start on random data points. Of 5 seeds, keep the lowest error."""
    best = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        sae = s.TopKSAE(2, N_CONCEPTS, k)
        with torch.no_grad():
            pick = X[torch.randperm(N)[:N_CONCEPTS]]
            sae.W_dec.copy_(pick / pick.norm(dim=1, keepdim=True))
            sae.W_enc.copy_(sae.W_dec.T)
        opt = torch.optim.Adam(sae.parameters(), lr=5e-3)
        for _ in range(steps):
            x_hat, _ = sae(X)
            loss = ((x_hat - X) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalise_decoder()
            with torch.no_grad():
                sae.b_dec.zero_()
        with torch.no_grad():
            x_hat, z = sae(X)
            fvu = float(((x_hat - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
        if best is None or fvu < best[1]:
            best = (sae, fvu, x_hat.numpy(), z.numpy(), seed)
    return best


saes = {}
for k in (1, 2, 3):
    sae, fvu, x_hat, z, seed = train_toy(k)
    dec = sae.W_dec.detach().numpy()
    cos = concepts @ dec.T  # each true concept against each decoder row
    saes[k] = {
        "fvu": fvu,
        "seed": seed,
        "decoder": dec.round(4).tolist(),
        "x_hat": x_hat.round(3).tolist(),
        "z": z.round(3).tolist(),
        "match": cos.argmax(1).tolist(),
        "match_cos": cos.max(1).round(3).tolist(),
    }
    print(
        f"   TopK SAE, 5 latents, k = {k}: FVU {fvu:.3f}; best cosine of each concept to a decoder row "
        f"{[round(c, 2) for c in cos.max(1).tolist()]}"
    )

# the four linear tools for the label "concept A (the first arrow) is on"
label = active[:, 0]
_, _, vt = np.linalg.svd(points - points.mean(0), full_matrices=False)
probe = LogisticRegression(C=10.0).fit(points, label)
probe_dir = probe.coef_[0] / np.linalg.norm(probe.coef_[0])
dmeans = points[label].mean(0) - points[~label].mean(0)
dmeans_dir = dmeans / np.linalg.norm(dmeans)
sae_dir = np.array(saes[1]["decoder"][saes[1]["match"][0]])


def angle_to_a(v):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(v, concepts[0]))))))


print("\n   Angle between each tool's direction and concept A's true arrow (0 = the same line):")
for name, v in (
    ("PCA, first component", vt[0]),
    ("logistic probe", probe_dir),
    ("difference of means", dmeans_dir),
    ("SAE latent for A (k = 1)", sae_dir),
):
    print(f"   {name:<28} {angle_to_a(v):5.1f} degrees")

s.save_json(
    s.RESULTS / "toy.json",
    {
        "by_hand": by_hand,
        "concepts": concepts.round(4).tolist(),
        "angles_deg": ANGLES,
        "points": points.round(3).tolist(),
        "active": active.astype(int).tolist(),
        "saes": saes,
        "tools": {
            "pca1": vt[0].tolist(),
            "pca2": vt[1].tolist(),
            "probe": probe_dir.tolist(),
            "probe_bias": float(probe.intercept_[0] / np.linalg.norm(probe.coef_[0])),
            "dmeans": dmeans_dir.tolist(),
            "sae_a": sae_dir.tolist(),
            "mean": points.mean(0).tolist(),
        },
        "angles_to_a": {
            "pca1": angle_to_a(vt[0]),
            "probe": angle_to_a(probe_dir),
            "dmeans": angle_to_a(dmeans_dir),
            "sae_a": angle_to_a(sae_dir),
        },
    },
)
print("\n   wrote results/toy.json")

print("\nSelf-checks:")
s.check(
    "by hand: the rebuild is 2.0 x latent 0 + 1.5 x latent 3 = [2.9, 0, 1.2]",
    torch.allclose(torch.tensor(by_hand["x_hat"]), torch.tensor([2.9, 0.0, 1.2])),
)
s.check(
    f"the toy's rebuild error falls with k ({', '.join(format(saes[k]['fvu'], '.3f') for k in (1, 2, 3))})",
    saes[1]["fvu"] > saes[2]["fvu"] >= saes[3]["fvu"] - 1e-3,
)
s.check(
    f"with k = 1 every concept has a decoder row within 10 degrees (cosines {saes[1]['match_cos']})",
    min(saes[1]["match_cos"]) > math.cos(math.radians(10)),
)
s.check(
    f"with k = 2 the rebuild is near perfect (FVU {saes[2]['fvu']:.4f}) but the arrows slide (lowest cosine "
    f"{min(saes[2]['match_cos']):.2f} < k = 1's {min(saes[1]['match_cos']):.2f})",
    saes[2]["fvu"] < 0.01 and min(saes[2]["match_cos"]) < min(saes[1]["match_cos"]),
)
s.check(
    f"PCA's first direction is not concept A's arrow ({angle_to_a(vt[0]):.0f} degrees off; bar > 20)",
    angle_to_a(vt[0]) > 20,
)
