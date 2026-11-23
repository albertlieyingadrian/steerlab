"""Shared analysis code: data from the CLIP cache, interventions on the block-23 CLS residual, retrieval skew, bootstraps,
LEACE, probes. Nothing here touches the GPU; `clip_head()` downloads the CLIP weights once, to read post_layernorm and
visual_projection.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "results" / "cache"
torch.set_num_threads(8)

# ------------------------------------------------------------------ labels and groups
AGE3 = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])  # 0-2, 3-9, 10-19 | 20-29, 30-39 | 40-49 ... 70+
AGE3_NAMES = ["0-19", "20-39", "40+"]


class Split:
    """One split of the cache: residuals, labels and derived groupings."""

    def __init__(self, tag, split):
        d = CACHE / tag
        self.r = np.load(d / f"{split}_cls23.npy")
        self.emb = np.load(d / f"{split}_emb.npy")
        self.index = np.load(d / f"{split}_index.npy")
        lab = json.loads((d / f"{split}_labels.json").read_text())
        self.n = len(self.r)
        if split == "imagenette":
            self.label = np.array(lab["label"])
            return
        self.gender = np.array(lab["gender"])
        self.age9 = np.array(lab["age"])
        self.race = np.array(lab["race"])
        self.age3 = AGE3[self.age9]
        self.age7 = np.where((self.age9 >= 1) & (self.age9 <= 7), self.age9 - 1, -1)  # paper: drop 0-2 and 70+
        self.cell = self.gender * 21 + self.age3 * 7 + self.race  # 42 joint cells

    def groups(self, attr):
        """(labels, n_groups); label -1 = excluded."""
        return {"gender": (self.gender, 2), "age7": (self.age7, 7), "age3": (self.age3, 3), "race": (self.race, 7),
                "joint": (self.cell, 42)}[attr]  # fmt: skip


def names(tag):
    return json.loads((CACHE / tag / "label_names.json").read_text())


def cell_name(c, nm):
    g, rest = divmod(int(c), 21)
    a, r = divmod(rest, 7)
    return f"{nm['gender'][g]}, {AGE3_NAMES[a]}, {nm['race'][r]}"


# ------------------------------------------------------------------ CLIP head (post_layernorm + projection)
@lru_cache(1)
def clip_head():
    from huggingface_hub import hf_hub_download

    keys = ("vision_model.post_layernorm.weight", "vision_model.post_layernorm.bias", "visual_projection.weight")
    small = CACHE / "clip_head.pt"
    if not small.exists():  # the repo ships only pytorch_model.bin; keep the three tensors we need
        sd = torch.load(hf_hub_download("openai/clip-vit-large-patch14-336", "pytorch_model.bin"), map_location="cpu",
                        weights_only=True, mmap=True)  # fmt: skip
        small.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: sd[k].float().clone() for k in keys}, small)
    w = torch.load(small, weights_only=True)
    return tuple(w[k] for k in keys)


def embed(r, normalise=True):
    """CLS residual [N, 1024] -> image embedding [N, 768] (float32 torch)."""
    g, b, P = clip_head()
    r = torch.as_tensor(r, dtype=torch.float32)
    e = torch.nn.functional.layer_norm(r, (1024,), g, b, eps=1e-5) @ P.T
    return torch.nn.functional.normalize(e, dim=-1) if normalise else e


# ------------------------------------------------------------------ SAE interventions
def sae_blend(sae, r, zero=(), alpha=0.6, codes=None):
    """DeBiasLens: h' = (1 - alpha) h + alpha decode(encode(h) with `zero` latents set to 0)."""
    r = torch.as_tensor(r, dtype=torch.float32)
    f = sae.encode(r) if codes is None else codes.clone()
    if len(zero):
        f[:, list(zero)] = 0
    return (1 - alpha) * r + alpha * sae.decode(f)


def codes_of(sae, r, chunk=8192):
    r = torch.as_tensor(r, dtype=torch.float32)
    with torch.no_grad():
        return torch.cat([sae.encode(r[i : i + chunk]) for i in range(0, len(r), chunk)])


# ------------------------------------------------------------------ the paper's selection rule
def select_social(codes, labels, n_groups, tau=0.9, taus=(0.9, 0.7, 0.5, 0.3, 0.1)):
    """Per group: latents firing on >= tau of the group's images and on < tau of every other group's; keep the one
    with the highest mean activation on the group. Steps tau down for a group with no candidate (recorded)."""
    labels = np.asarray(labels)
    keep = labels >= 0
    A = codes[torch.as_tensor(keep)]
    lab = labels[keep]
    fire = torch.stack([(A[torch.as_tensor(lab == g)] > 0).float().mean(0) for g in range(n_groups)])  # [G, L]
    mean = torch.stack([A[torch.as_tensor(lab == g)].mean(0) for g in range(n_groups)])
    out = []
    for g in range(n_groups):
        for t in [tau] + [x for x in taus if x < tau]:
            eff = fire >= t
            cand = eff[g] & (eff.sum(0) == 1)
            if cand.any():
                j = int(torch.where(cand, mean[g], torch.full_like(mean[g], -1)).argmax())
                others = torch.cat([fire[:g, j], fire[g + 1 :, j]])
                out.append({"group": g, "latent": j, "tau": t, "fire": float(fire[g, j]),
                            "fire_other_max": float(others.max()), "n_candidates": int(cand.sum())})  # fmt: skip
                break
        else:
            out.append({"group": g, "latent": None, "tau": None, "n_candidates": 0})
    return out


def fire_by_group(codes, labels, n_groups):
    labels = np.asarray(labels)
    return torch.stack([(codes[torch.as_tensor(labels == g)] > 0).float().mean(0) for g in range(n_groups)])


def freq_matched(freq, chosen, n_draws, rng, n_bins=20, exclude=()):
    """For each chosen latent, draw a latent from the same firing-frequency bin (quantiles over latents that fire)."""
    freq = np.asarray(freq)
    live = np.where(freq > 0)[0]
    edges = np.quantile(freq[live], np.linspace(0, 1, n_bins + 1))
    binof = np.clip(np.searchsorted(edges, freq, side="right") - 1, 0, n_bins - 1)
    banned = set(int(c) for c in chosen) | set(int(e) for e in exclude)
    draws = []
    for _ in range(n_draws):
        pick = []
        for j in chosen:
            pool = [int(x) for x in live[binof[live] == binof[j]] if int(x) not in banned and int(x) not in pick]
            pick.append(int(rng.choice(pool)))
        draws.append(pick)
    return draws


# ------------------------------------------------------------------ retrieval skew
def topk_counts(sims, labels, n_groups, K):
    """sims [P, N] (torch), labels [N] (-1 excluded) -> counts [P, G] among the top K of the included images."""
    labels = torch.as_tensor(np.asarray(labels))
    keep = labels >= 0
    s = sims[:, keep]
    lab = labels[keep]
    idx = torch.topk(s, K, dim=1).indices
    g = lab[idx]
    out = torch.zeros(sims.shape[0], n_groups)
    out.scatter_add_(1, g, torch.ones_like(g, dtype=torch.float32))
    return out


def maxskew_eo(counts, K):
    """The released metric: uniform reference, p = 1/K for an absent group, max over groups floored at 0."""
    G = counts.shape[1]
    p = torch.clamp(counts / K, min=1.0 / K)
    return torch.clamp((torch.log(p) - np.log(1.0 / G)).max(1).values, min=0)


def skew_dp(counts, K, share):
    """Smoothed demographic-parity skew per group: log((n + 0.5) / (K p + 0.5))."""
    share = torch.as_tensor(share, dtype=torch.float32)
    return torch.log((counts + 0.5) / (K * share + 0.5))


def group_share(labels, n_groups):
    labels = np.asarray(labels)
    lab = labels[labels >= 0]
    return np.bincount(lab, minlength=n_groups) / len(lab)


def random_counts(labels, n_groups, K, n=200, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    lab = labels[labels >= 0]
    counts = torch.zeros(n, n_groups)
    for i in range(n):
        pick = rng.choice(len(lab), K, replace=False)
        counts[i] = torch.as_tensor(np.bincount(lab[pick], minlength=n_groups), dtype=torch.float32)
    return counts


def additive_rss(skew, share):
    """Weighted least squares of 42 cell skews on gender + age3 + race main effects; returns RSS per row."""
    skew = torch.as_tensor(skew, dtype=torch.float64)
    cells = np.arange(42)
    g, rest = np.divmod(cells, 21)
    a, r = np.divmod(rest, 7)
    X = np.zeros((42, 10))
    X[:, 0] = 1
    X[g == 1, 1] = 1
    X[a == 1, 2] = 1
    X[a == 2, 3] = 1
    for k in range(1, 7):
        X[r == k, 3 + k] = 1
    w = torch.as_tensor(np.asarray(share), dtype=torch.float64)
    Xt = torch.as_tensor(X)
    beta = torch.linalg.solve(Xt.T @ (w[:, None] * Xt), Xt.T @ (w[:, None] * skew.T))  # [10, P]
    res = skew - (Xt @ beta).T
    return (res**2 * w).sum(1)


# ------------------------------------------------------------------ bootstraps
def cluster_boot(values, clusters, n=10000, seed=0):
    """Mean of `values` [P] (or [P, M] paired columns) under a bootstrap over clusters. Returns draws [n] or [n, M]."""
    values = np.asarray(values, dtype=np.float64)
    one = values.ndim == 1
    if one:
        values = values[:, None]
    cl = np.unique(np.asarray(clusters), return_inverse=True)[1]
    C = cl.max() + 1
    sums = np.zeros((C, values.shape[1]))
    np.add.at(sums, cl, values)
    cnt = np.bincount(cl, minlength=C).astype(np.float64)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(C, np.full(C, 1.0 / C), size=n).astype(np.float64)  # [n, C]
    draws = (w @ sums) / (w @ cnt)[:, None]
    return draws[:, 0] if one else draws


def ci(draws):
    lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5], axis=0)
    return [float(lo), float(hi)] if np.ndim(lo) == 0 else [lo.tolist(), hi.tolist()]


def image_boot_mean(x, n=10000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(0, n, 500):
        idx = rng.integers(0, len(x), size=(min(500, n - i), len(x)))
        out[i : i + len(idx)] = x[idx].mean(1)
    return out


# ------------------------------------------------------------------ LEACE (closed form; "LEACE: Perfect linear concept erasure in closed form", NeurIPS 2023)
class Leace:
    """x' = x - W+ P W (x - mu); W = Sigma^{-1/2}, P = projection onto the column span of W Sigma_xz."""

    def __init__(self, mu, sigma, sigma_xz, eps=1e-6):
        mu = torch.as_tensor(mu, dtype=torch.float64)
        S = torch.as_tensor(sigma, dtype=torch.float64)
        Sxz = torch.as_tensor(sigma_xz, dtype=torch.float64)
        L, V = torch.linalg.eigh(S)
        keep = L > eps * L.max()
        Ls = torch.where(keep, L, torch.ones_like(L))
        W = (V * torch.where(keep, Ls**-0.5, torch.zeros_like(L))) @ V.T
        Wp = (V * torch.where(keep, Ls**0.5, torch.zeros_like(L))) @ V.T
        U, s, _ = torch.linalg.svd(W @ Sxz, full_matrices=False)
        U = U[:, s > 1e-8 * s.max()]
        self.mu = mu.float()
        self.M = (Wp @ U @ U.T @ W).float()  # [D, D]
        self.rank = int(U.shape[1])

    @classmethod
    def fit(cls, x, z_onehot):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
        z = torch.as_tensor(np.asarray(z_onehot), dtype=torch.float64)
        mu = x.mean(0)
        xc = x - mu
        return cls(mu, xc.T @ xc / len(x), xc.T @ (z - z.mean(0)) / len(x))

    @classmethod
    def from_stats(cls, S1, S2, Sg, Ng):
        """From sums over tokens: S1 = sum x, S2 = sum x x^T, Sg[c] = sum of x over class c, Ng[c] = count."""
        S1, S2, Sg, Ng = (torch.as_tensor(np.asarray(a), dtype=torch.float64) for a in (S1, S2, Sg, Ng))
        N = Ng.sum()
        mu = S1 / N
        sigma = S2 / N - torch.outer(mu, mu)
        pz = Ng / N
        sigma_xz = (Sg / N - torch.outer(pz, mu)).T  # cov(x, z_c) = E[x 1{c}] - mu p_c
        return cls(mu, sigma, sigma_xz)

    def __call__(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x - (x - self.mu) @ self.M.T


def onehot(labels, n):
    labels = np.asarray(labels)
    out = np.zeros((len(labels), n))
    out[np.arange(len(labels)), labels] = 1
    return out


# ------------------------------------------------------------------ probes and ability
def probe(train_x, train_y, test_x, test_y):
    """Logistic regression (C=1, standardised). Returns accuracy, balanced accuracy and per-image correctness."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(np.asarray(train_x), np.asarray(train_y))
    pred = clf.predict(np.asarray(test_x))
    y = np.asarray(test_y)
    return {"acc": float((pred == y).mean()), "bal_acc": float(balanced_accuracy_score(y, pred)), "correct": pred == y}


def zero_shot(img_emb, class_text, labels):
    t = torch.nn.functional.normalize(torch.as_tensor(np.asarray(class_text), dtype=torch.float32), dim=-1)
    pred = (torch.as_tensor(img_emb) @ t.T).argmax(1).numpy()
    return pred == np.asarray(labels)
