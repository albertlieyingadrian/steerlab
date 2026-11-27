"""Self-modeling as an auxiliary task, the measures the paper used, and our controls. Pure PyTorch; runs on CPU or GPU.

Replicates "Unexpected Benefits of Self-Modeling in Neural Systems" (arXiv 2407.10188). The paper released no code,
so everything here is a minimal reimplementation of its Methods section:

    loss = CE(logits, y) + AW * mean((a_hat - a)^2)

where the output layer emits both the logits and a_hat, a prediction of the hidden activations a, and gradient flows
into both a_hat and a. Before measuring, the self-model rows of the output layer are pruned, so every arm is measured
on the same architecture.

Arms (`target`):
    "self"      the paper's self-model (gradient into the target too)
    "detached"  same, but the target a is detached: the net cannot make itself easier to predict
    "other"     predict the hidden activations of a frozen, randomly initialised net of the same shape
    "none"      no auxiliary task (AW = 0 is the paper's baseline); `l2` adds a plain weight penalty

Measures:
    width       SD of the classification weights of the final layer (bias excluded), as in the paper
    llc         local learning coefficient (RLCT) by SGLD, following Lau et al. (arXiv 2308.12108)
    plus test accuracy, train CE, hidden-weight SD, mean hidden activation, dead-unit share, width x activation
"""

from __future__ import annotations

import copy
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------------------------------------------------
# models


class MLP(nn.Module):
    """MNIST: input -> hidden (ReLU) -> [10 logits | n_aux self-model outputs]."""

    def __init__(self, d_in: int, hidden: int, n_classes: int, n_aux: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.out = nn.Linear(hidden, n_classes + n_aux)
        self.n_classes = n_classes

    def forward(self, x):
        a = F.relu(self.fc1(x))
        z = self.out(a)
        return z[:, : self.n_classes], z[:, self.n_classes :], a


class BagNet(nn.Module):
    """IMDB: mean word embedding -> linear hidden (no activation, the paper says "linear") -> output."""

    def __init__(self, vocab: int, emb: int, hidden: int, n_classes: int, n_aux: int):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, emb, mode="mean")
        self.fc1 = nn.Linear(emb, hidden)
        self.out = nn.Linear(hidden, n_classes + n_aux)
        self.n_classes = n_classes

    def forward(self, x):
        ids, offsets = x
        a = self.fc1(self.emb(ids, offsets))
        z = self.out(a)
        return z[:, : self.n_classes], z[:, self.n_classes :], a


class _BagOther(nn.Module):
    def __init__(self, emb, lin):
        super().__init__()
        self.emb, self.lin = emb, lin

    def forward(self, x):
        ids, offsets = x
        return self.lin(self.emb(ids, offsets))


def prune(model: nn.Module) -> nn.Module:
    """Copy of the model whose output layer keeps only the classification rows (what the paper measured)."""
    m = copy.deepcopy(model)
    k = m.n_classes
    new = nn.Linear(m.out.in_features, k).to(m.out.weight.device)
    with torch.no_grad():
        new.weight.copy_(m.out.weight[:k])
        new.bias.copy_(m.out.bias[:k])
    m.out = new
    return m


# ---------------------------------------------------------------------------------------------------------------------
# training


def aux_loss(target: str, a_hat, a, x, other: nn.Module | None):
    if target == "self":
        t = a
    elif target == "detached":
        t = a.detach()
    elif target == "other":
        with torch.no_grad():
            t = other(x)
    else:
        raise ValueError(target)
    return F.mse_loss(a_hat, t)


def l2_penalty(model: nn.Module) -> torch.Tensor:
    return sum((p**2).sum() for n, p in model.named_parameters() if n.endswith("weight"))


def width(model: nn.Module) -> float:
    return model.out.weight[: model.n_classes].detach().float().std().item()


def build(cfg: dict, data: dict, device: str):
    """The model for one arm, plus the frozen 'other' net for the other-network control."""
    torch.manual_seed(cfg["seed"])
    target = "none" if cfg["aw"] == 0 else cfg["target"]
    n_aux = cfg["hidden"] if target != "none" else 0
    other = None
    if cfg["task"] == "mnist":
        model = MLP(784, cfg["hidden"], 10, n_aux)
        if target == "other":
            torch.manual_seed(10_000 + cfg["seed"])
            other = nn.Sequential(nn.Linear(784, cfg["hidden"]), nn.ReLU())
    elif cfg["task"] == "imdb":
        emb = cfg.get("emb", 64)
        model = BagNet(data["vocab"], emb, cfg["hidden"], 2, n_aux)
        if target == "other":
            torch.manual_seed(10_000 + cfg["seed"])
            other = _BagOther(nn.EmbeddingBag(data["vocab"], emb, mode="mean"), nn.Linear(emb, cfg["hidden"]))
    else:
        raise ValueError(cfg["task"])
    if other is not None:
        other = other.to(device).eval()
        for p in other.parameters():
            p.requires_grad_(False)
    return model.to(device), other, target


def train_run(cfg: dict, data: dict, device: str = "cpu") -> dict:
    """Train one arm and return its measures. cfg keys: task, hidden, aw, target, l2, seed, epochs, batch, lr,
    llc (dict name -> SGLD settings, or None), checkpoints (epochs at which to measure; default [epochs])."""
    model, other, target = build(cfg, data, device)
    aw, l2 = cfg["aw"], cfg.get("l2", 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    checkpoints = sorted(cfg.get("checkpoints", [cfg["epochs"]]))
    out = {"cfg": cfg, "width_by_epoch": [], "aux_by_epoch": [], "checkpoints": {}}
    g = torch.Generator().manual_seed(cfg["seed"])
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        aux_sum, nb = 0.0, 0
        for xb, yb in batches(data, "train", cfg["batch"], device, g):
            logits, a_hat, a = model(xb)
            loss = F.cross_entropy(logits, yb)
            if target != "none":
                la = aux_loss(target, a_hat, a, xb, other)
                loss = loss + aw * la
                aux_sum += la.item()
                nb += 1
            if l2:
                loss = loss + l2 * l2_penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        out["width_by_epoch"].append(width(model))
        out["aux_by_epoch"].append(aux_sum / nb if nb else None)
        if epoch in checkpoints:
            out["checkpoints"][str(epoch)] = measure(model, data, device, cfg.get("llc"), cfg["seed"])
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------------------------------------------------
# data iteration (MNIST: dense tensors; IMDB: token-id lists packed for EmbeddingBag)


def batches(data: dict, split: str, bs: int, device: str, g: torch.Generator | None, shuffle: bool = True):
    y = data[f"y_{split}"]
    n = len(y)
    idx = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i : i + bs]
        yield take(data, split, j, device), y[j].to(device)


def take(data: dict, split: str, j: torch.Tensor, device: str):
    if f"x_{split}" in data:
        return data[f"x_{split}"][j].to(device)
    ids, offs, lens = data[f"ids_{split}"], data[f"off_{split}"], data[f"len_{split}"]
    parts = [ids[offs[k] : offs[k] + lens[k]] for k in j.tolist()]
    flat = torch.cat(parts)
    offsets = torch.tensor([0] + [len(p) for p in parts[:-1]]).cumsum(0)
    return flat.to(device), offsets.to(device)


# ---------------------------------------------------------------------------------------------------------------------
# measures


@torch.no_grad()
def full_loss_acc(model: nn.Module, data: dict, split: str, device: str) -> tuple[float, float]:
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for xb, yb in batches(data, split, 4096, device, None, shuffle=False):
        logits = model(xb)[0]
        tot += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
        n += len(yb)
    return tot / n, correct / n


@torch.no_grad()
def hidden_stats(model: nn.Module, data: dict, device: str) -> dict:
    model.eval()
    acts = [model(xb)[2].float().cpu() for xb, _ in batches(data, "test", 4096, device, None, shuffle=False)]
    a = torch.cat(acts)
    alive = (a.abs() > 1e-6).any(0)
    return {
        "mean_abs_act": a.abs().mean().item(),
        "dead_share": 1 - alive.float().mean().item(),
        "hidden_w_sd": model.fc1.weight.detach().float().std().item(),
    }


def measure(model: nn.Module, data: dict, device: str, llc_cfg: dict | None, seed: int) -> dict:
    pruned = prune(model)
    test_loss, test_acc = full_loss_acc(pruned, data, "test", device)
    train_loss, train_acc = full_loss_acc(pruned, data, "train", device)
    hs = hidden_stats(pruned, data, device)
    w = width(pruned)
    m = {
        "width": w,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_loss": train_loss,
        "train_acc": train_acc,
        **hs,
        "width_x_act": w * hs["mean_abs_act"],
    }
    for name, kw in (llc_cfg or {}).items():
        r = estimate_llc(pruned, data, device, seed=seed, **kw)
        m[f"llc_{name}"] = r["llc"]
        m[f"llc_{name}_chains"] = r["chains"]
        m[f"llc_{name}_trace"] = r["trace"]
    return m


# ---------------------------------------------------------------------------------------------------------------------
# local learning coefficient by SGLD


def estimate_llc(
    model: nn.Module,
    data: dict,
    device: str,
    lr: float,
    gamma: float = 100.0,
    batch: int = 512,
    chains: int = 4,
    burn: int = 100,
    draws: int = 300,
    seed: int = 0,
    loss_fn=None,
) -> dict:
    """llc = n*beta*(E_w[L_n(w)] - L_n(w*)), w sampled by localised SGLD around w*, beta = 1/log n.

    Update (Lau et al. 2023): w <- w - (lr/2) * (n*beta*grad L_batch(w) + gamma*(w - w*)) + N(0, lr).
    L_n(w*) is the exact training loss; E_w[L_n(w)] is the mean of minibatch losses over the draws (unbiased).
    `trace` is the chain-mean minibatch loss at every step, for checking that the chain is stationary.
    """
    loss_fn = loss_fn or (lambda m, xb, yb: F.cross_entropy(m(xb)[0], yb))
    n = len(data["y_train"])
    nbeta = n / math.log(n)
    w_star = [p.detach().clone() for p in model.parameters()]
    l_star = _exact_train_loss(model, data, device, loss_fn)
    g = torch.Generator().manual_seed(1_000 + seed)
    gn = torch.Generator(device=device).manual_seed(2_000 + seed)
    chain_vals, trace = [], [0.0] * (burn + draws)
    for _c in range(chains):
        m = copy.deepcopy(model).to(device).train()
        params = list(m.parameters())
        vals = []
        it = _forever(data, batch, device, g)
        for t in range(burn + draws):
            xb, yb = next(it)
            loss = loss_fn(m, xb, yb)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, gr, p0 in zip(params, grads, w_star, strict=True):
                    drift = nbeta * gr + gamma * (p - p0)
                    noise = torch.randn(p.shape, generator=gn, device=device) * math.sqrt(lr)
                    p.add_(-0.5 * lr * drift + noise)
            lv = loss.item()
            trace[t] += lv / chains
            if t >= burn:
                vals.append(lv)
        chain_vals.append(nbeta * (sum(vals) / len(vals) - l_star))
    return {"llc": sum(chain_vals) / len(chain_vals), "chains": chain_vals, "trace": trace, "l_star": l_star}


@torch.no_grad()
def _exact_train_loss(model, data, device, loss_fn) -> float:
    tot, n = 0.0, 0
    for xb, yb in batches(data, "train", 4096, device, None, shuffle=False):
        tot += loss_fn(model, xb, yb).item() * len(yb)
        n += len(yb)
    return tot / n


def _forever(data, bs, device, g):
    while True:
        yield from batches(data, "train", bs, device, g)


# ---------------------------------------------------------------------------------------------------------------------
# data loading (run once inside the Modal image build; cached as .pt)


def load_mnist(cache_dir: str) -> dict:
    """MNIST from the Hugging Face `ylecun/mnist` files, normalised by the train mean and SD."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("ylecun/mnist", cache_dir=cache_dir)
    out = {}
    for split in ("train", "test"):
        imgs = np.stack([np.array(im, dtype=np.uint8) for im in ds[split]["image"]])
        out[f"x_{split}"] = torch.tensor(imgs, dtype=torch.float32).reshape(len(imgs), -1) / 255.0
        out[f"y_{split}"] = torch.tensor(ds[split]["label"], dtype=torch.long)
    mu, sd = out["x_train"].mean(), out["x_train"].std()
    for split in ("train", "test"):
        out[f"x_{split}"] = (out[f"x_{split}"] - mu) / sd
    return out


def load_imdb(cache_dir: str, vocab_size: int = 10_000) -> dict:
    """IMDB from `stanfordnlp/imdb`: lowercased word tokens, the 10,000 most frequent train words, OOV dropped."""
    import re
    from collections import Counter

    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)
    tok = re.compile(r"[a-z0-9']+")

    def words(t):
        return tok.findall(t.lower().replace("<br />", " "))

    train_words = [words(t) for t in ds["train"]["text"]]
    counts = Counter(w for ws in train_words for w in ws)
    vocab = {w: i for i, (w, _) in enumerate(counts.most_common(vocab_size))}
    out = {"vocab": len(vocab)}
    for split, wl in (("train", train_words), ("test", [words(t) for t in ds["test"]["text"]])):
        seqs = [[vocab[w] for w in ws if w in vocab] or [0] for ws in wl]
        lens = torch.tensor([len(s) for s in seqs])
        out[f"ids_{split}"] = torch.tensor([i for s in seqs for i in s], dtype=torch.long)
        out[f"len_{split}"] = lens
        out[f"off_{split}"] = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]])
        out[f"y_{split}"] = torch.tensor(ds[split]["label"], dtype=torch.long)
    return out
