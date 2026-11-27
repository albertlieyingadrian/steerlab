"""Scaled-down CIFAR-10 arm of the self-modeling replication (paper Figure 3), on one GPU.

ResNet-18 (CIFAR stem: 3x3 conv, no max-pool) + one linear hidden layer of 256 units between the pooled features and
the output, as the paper describes. The self-model target is the hidden layer and its input (512 pooled features +
256 hidden = 768 outputs; the paper's input also carried a skip connection from block 3, which we do not add).
30 epochs and 3 seeds instead of 150-250 epochs and 10 runs. SGD momentum 0.9 (Nesterov), one-cycle lr peaking at
0.05, weight decay 5e-4, batch 256, random crop and flip, bf16 autocast. LLC: the MNIST estimator with BatchNorm
statistics frozen, 2 chains x (100 burn-in + 200 draws), at four steps from 1e-6 to 3e-8
(plan amendment 2: the MNIST step diverged here).
"""

from __future__ import annotations

import copy
import math
import time

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

import selfmodel


class Net(nn.Module):
    def __init__(self, n_aux: int):
        super().__init__()
        r = torchvision.models.resnet18(num_classes=10)
        r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        r.maxpool = nn.Identity()
        r.fc = nn.Identity()
        self.body = r
        self.fc1 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10 + n_aux)
        self.n_classes = 10

    def forward(self, x):
        z = self.body(x)
        a = self.fc1(z)
        o = self.out(a)
        return o[:, :10], o[:, 10:], torch.cat([z, a], 1)


def load(cache_dir: str, device: str) -> dict:
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    out = {}
    for split in ("train", "test"):
        x = np.stack([np.array(im, dtype=np.uint8) for im in ds[split]["img"]])
        x = torch.tensor(x).permute(0, 3, 1, 2).float() / 255.0
        out[f"x_{split}"] = ((x - mean) / std).to(device)
        out[f"y_{split}"] = torch.tensor(ds[split]["label"], dtype=torch.long).to(device)
    return out


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop of a 4-pixel reflect-padded image, then a random horizontal flip (on the GPU)."""
    n = x.shape[0]
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), generator=g).to(x.device)
    j = torch.randint(0, 9, (n,), generator=g).to(x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None, :])[:, None, :, None].expand(n, 3, 32, 40)
    cols = (j[:, None] + ar[None, :])[:, None, None, :].expand(n, 3, 32, 32)
    out = pad.gather(2, rows).gather(3, cols)
    flip = (torch.rand(n, generator=g) < 0.5).to(x.device)
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def train_run(cfg: dict, cache_dir: str, device: str = "cuda") -> dict:
    torch.manual_seed(cfg["seed"])
    data = load(cache_dir, device)
    target = "none" if cfg["aw"] == 0 else cfg["target"]
    model = Net(768 if target != "none" else 0).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = cfg["epochs"] * math.ceil(50_000 / 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=steps, pct_start=0.15)
    g = torch.Generator().manual_seed(cfg["seed"])
    out = {"cfg": cfg, "width_by_epoch": [], "checkpoints": {}}
    t0 = time.time()
    for _ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(50_000, generator=g).to(device)
        for b in range(0, 50_000, 256):
            idx = perm[b : b + 256]
            xb = augment(data["x_train"][idx], g).contiguous(memory_format=torch.channels_last)
            yb = data["y_train"][idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, a_hat, a = model(xb)
            loss = F.cross_entropy(logits.float(), yb)
            if target != "none":
                t = a if target == "self" else a.detach()
                loss = loss + cfg["aw"] * F.mse_loss(a_hat.float(), t.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        out["width_by_epoch"].append(selfmodel.width(model))
    out["checkpoints"][str(cfg["epochs"])] = measure(model, data, cfg)
    out["seconds"] = time.time() - t0
    out["state_dict"] = {k: v.detach().cpu() for k, v in selfmodel.prune(model).state_dict().items()}
    return out


def measure(model, data, cfg):
    pruned = selfmodel.prune(model).eval()
    m = {"width": selfmodel.width(pruned)}
    with torch.no_grad():
        for split in ("train", "test"):
            tot, cor = 0.0, 0
            n = len(data[f"y_{split}"])
            for b in range(0, n, 1000):
                xb = data[f"x_{split}"][b : b + 1000].contiguous(memory_format=torch.channels_last)
                yb = data[f"y_{split}"][b : b + 1000]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = pruned(xb)[0].float()
                tot += F.cross_entropy(lg, yb, reduction="sum").item()
                cor += (lg.argmax(-1) == yb).sum().item()
            m[f"{split}_loss"], m[f"{split}_acc"] = tot / n, cor / n
    for name, kw in (cfg.get("llc") or {}).items():
        r = llc_frozen_bn(pruned, data, seed=cfg["seed"], **kw)
        m[f"llc_{name}"], m[f"llc_{name}_chains"], m[f"llc_{name}_trace"] = r["llc"], r["chains"], r["trace"]
    return m


def llc_frozen_bn(model, data, lr, gamma=100.0, batch=512, chains=2, burn=100, draws=200, seed=0):
    """selfmodel.estimate_llc with BatchNorm in eval mode (fixed running statistics) during SGLD."""
    n = 50_000
    nbeta = n / math.log(n)
    w_star = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        tot = 0.0
        for b in range(0, n, 1000):
            xb = data["x_train"][b : b + 1000].contiguous(memory_format=torch.channels_last)
            tot += F.cross_entropy(model(xb)[0].float(), data["y_train"][b : b + 1000], reduction="sum").item()
    l_star = tot / n
    g = torch.Generator().manual_seed(1000 + seed)
    vals_all, trace = [], [0.0] * (burn + draws)
    for _c in range(chains):
        m = copy.deepcopy(model).eval()
        params = list(m.parameters())
        vals = []
        for t in range(burn + draws):
            idx = torch.randint(0, n, (batch,), generator=g).to(data["x_train"].device)
            xb = data["x_train"][idx].contiguous(memory_format=torch.channels_last)
            loss = F.cross_entropy(m(xb)[0].float(), data["y_train"][idx])
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, gr, p0 in zip(params, grads, w_star, strict=True):
                    p.add_(-0.5 * lr * (nbeta * gr + gamma * (p - p0)) + torch.randn_like(p) * math.sqrt(lr))
            trace[t] += loss.item() / chains
            if t >= burn:
                vals.append(loss.item())
        vals_all.append(nbeta * (sum(vals) / len(vals) - l_star))
    return {"llc": sum(vals_all) / len(vals_all), "chains": vals_all, "trace": trace}
