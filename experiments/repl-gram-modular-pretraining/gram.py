"""GRAM at toy scale: model, data, and every method of the paper's Simple Stories experiment.

Follows the released code (github.com/agencyenterprise/modular-pretraining): `src/model/base.py`, `src/model/moe.py`,
`src/run/train/{base,routed,maxent,finetune}.py`. Differences are listed in PREREGISTRATION.md.

Called by `modal_app.py` (one seed per GPU job). Nothing here touches the network.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

AUX_LABELS = [
    "a-deadline-or-time-limit",
    "alien-encounters",
    "bygone-eras",
    "cultural-traditions",
]
LABELS = ["core", *AUX_LABELS]
# a profile is the set of auxiliary topics served; index i -> AUX_LABELS[i]
PROFILES = [(), (0,), (1,), (2,), (3,)]


def profile_name(p: tuple[int, ...]) -> str:
    return "core" if not p else "core+" + "+".join(AUX_LABELS[i] for i in p)


@dataclass
class Cfg:
    vocab: int = 4096
    ctx: int = 256
    layers: int = 8
    width: int = 512
    heads: int = 8
    kv: int = 2
    mlp: int = 2048
    aux_dim: int = 192
    eos: int = 1
    lr: float = 5e-3
    betas: tuple[float, float] = (0.9, 0.95)
    wd: float = 0.01
    batch: int = 128
    warmup: float = 0.1
    decay: float = 0.1
    p_as: float = 0.3  # auxiliary spread: share of aux batches that also update the core
    p_cr: float = 0.5  # core robustness: share of core batches run with one random module on
    curve_evals: int = 100
    curve_batches: int = 16
    val_batches: int = 16
    test_batches: int = 40
    max_steps: int | None = None  # smoke tests only


# --------------------------------------------------------------------------------------------------------------- #
# Model (same blocks as the paper's code)                                                                          #
# --------------------------------------------------------------------------------------------------------------- #


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        freq = torch.cat([freq, freq.new_zeros(dim // 4)])
        theta = torch.einsum("i,j -> ij", torch.arange(max_seq_len, dtype=torch.float32), freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), 3).type_as(x)


class Attention(nn.Module):
    def __init__(self, c: Cfg) -> None:
        super().__init__()
        self.h, self.kv, self.hd = c.heads, c.kv, c.width // c.heads
        self.rotary = Rotary(self.hd)
        self.q = nn.Linear(c.width, c.width, bias=True)
        self.kvp = nn.Linear(c.width, 2 * c.kv * self.hd, bias=True)
        self.o = nn.Linear(c.width, c.width, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape
        q = self.q(x).view(B, T, self.h, self.hd)
        k, v = self.kvp(x).split(self.kv * self.hd, dim=2)
        k, v = k.view(B, T, self.kv, self.hd), v.view(B, T, self.kv, self.hd)
        q, k = self.rotary(q), self.rotary(k)
        rep = self.h // self.kv  # grouped-query attention: each KV head serves rep query heads
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, E))


class MLP(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(width, hidden, bias=True)
        self.proj = nn.Linear(hidden, width, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, c: Cfg, n_aux: int) -> None:
        super().__init__()
        self.attn = Attention(c)
        self.n1, self.n2 = nn.RMSNorm(c.width), nn.RMSNorm(c.width)
        self.mlp = MLP(c.width, c.mlp)  # the core MLP
        self.aux = nn.ModuleList([MLP(c.width, c.aux_dim) for _ in range(n_aux)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, aux_on: tuple[int, ...]) -> torch.Tensor:
        x = x + self.attn(self.n1(x), mask)
        h = self.n2(x)
        y = self.mlp(h)
        for i in aux_on:  # GRAM: core output plus each switched-on module's output
            y = y + self.aux[i](h)
        return x + y


def doc_mask(tok: torch.Tensor, eos: int) -> torch.Tensor:
    """Causal mask that also stops attention across story boundaries. (B, 1, T, T), True = allowed."""
    e = (tok == eos).int()
    seg = torch.cumsum(e, dim=1) - e
    same = seg.unsqueeze(2).eq(seg.unsqueeze(1))
    causal = torch.tril(torch.ones(tok.shape[1], tok.shape[1], dtype=torch.bool, device=tok.device))
    return (same & causal).unsqueeze(1)


class GPT(nn.Module):
    def __init__(self, c: Cfg, n_aux: int = 0) -> None:
        super().__init__()
        self.c, self.n_aux = c, n_aux
        self.embed = nn.Embedding(c.vocab, c.width)
        self.blocks = nn.ModuleList([Block(c, n_aux) for _ in range(c.layers)])
        self.norm = nn.RMSNorm(c.width)
        self.unembed = nn.Linear(c.width, c.vocab, bias=True)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                m.bias.data.zero_()

    def forward(self, tok: torch.Tensor, aux_on: tuple[int, ...] = ()) -> torch.Tensor:
        mask = doc_mask(tok, self.c.eos)
        h = self.embed(tok)
        for b in self.blocks:
            h = b(h, mask, aux_on)
        return self.unembed(self.norm(h))

    def core_params(self) -> list[nn.Parameter]:
        aux = {id(p) for i in range(self.n_aux) for p in self.aux_params(i)}
        return [p for p in self.parameters() if id(p) not in aux]

    def aux_params(self, i: int) -> list[nn.Parameter]:
        return [p for b in self.blocks for p in b.aux[i].parameters()]


def n_params(ps) -> int:
    return sum(p.numel() for p in ps)


def ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))


# --------------------------------------------------------------------------------------------------------------- #
# Data: one uint16 token stream per (label, split) on the GPU, cut into windows of ctx + 1 at stride ctx           #
# --------------------------------------------------------------------------------------------------------------- #


class Stream:
    def __init__(self, path: Path, ctx: int, device: str) -> None:
        arr = np.fromfile(path, dtype=np.uint16)
        self.tok = torch.from_numpy(arr.astype(np.int64)).to(device)
        self.ctx = ctx
        self.n = (len(arr) - 1) // ctx
        self.off = torch.arange(ctx + 1, device=device)

    def windows(self, idx: torch.Tensor) -> torch.Tensor:
        return self.tok[idx.unsqueeze(1) * self.ctx + self.off]


class Data:
    """train streams per label; test windows split once by a fixed permutation into val and test halves."""

    def __init__(self, root: Path, c: Cfg, device: str) -> None:
        self.c, self.device = c, device
        self.train = {lab: Stream(root / f"{lab}_train.bin", c.ctx, device) for lab in LABELS}
        self.heldout = {lab: Stream(root / f"{lab}_test.bin", c.ctx, device) for lab in LABELS}
        self.val_idx, self.test_idx = {}, {}
        for lab in LABELS:
            g = torch.Generator().manual_seed(1234)
            perm = torch.randperm(self.heldout[lab].n, generator=g).to(device)
            half = len(perm) // 2
            self.val_idx[lab], self.test_idx[lab] = perm[:half], perm[half:]

    def n_batches(self, lab: str) -> int:
        return self.train[lab].n // self.c.batch

    def eval_batches(self, lab: str, split: str, n: int):
        idx = (self.val_idx if split == "val" else self.test_idx)[lab]
        B = self.c.batch
        for j in range(min(n, len(idx) // B)):
            yield self.heldout[lab].windows(idx[j * B : (j + 1) * B])


class Sampler:
    """Per-label epoch-shuffled window order, like a DistributedSampler with shuffle=True."""

    def __init__(self, data: Data, seed: int) -> None:
        self.data, self.seed, self.pos, self.perm, self.epoch = data, seed, {}, {}, {}
        for lab in LABELS:
            self.epoch[lab] = 0
            self._reshuffle(lab)

    def _reshuffle(self, lab: str) -> None:
        g = torch.Generator().manual_seed(self.seed * 1000 + LABELS.index(lab) * 10 + self.epoch[lab])
        self.perm[lab] = torch.randperm(self.data.train[lab].n, generator=g).to(self.data.device)
        self.pos[lab] = 0

    def next(self, lab: str, B: int) -> torch.Tensor:
        if self.pos[lab] + B > len(self.perm[lab]):
            self.epoch[lab] += 1
            self._reshuffle(lab)
        idx = self.perm[lab][self.pos[lab] : self.pos[lab] + B]
        self.pos[lab] += B
        return self.data.train[lab].windows(idx)


@torch.no_grad()
def eval_loss(model: GPT, data: Data, lab: str, aux_on: tuple[int, ...], split: str, n: int) -> float:
    model.eval()
    tot, k = 0.0, 0
    for w in data.eval_batches(lab, split, n):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(w[:, :-1], aux_on)
        tot += ce(logits, w[:, 1:]).item()
        k += 1
    model.train()
    return tot / max(k, 1)


def eval_all(model: GPT, data: Data, aux_on: tuple[int, ...], split: str = "test") -> dict[str, float]:
    n = data.c.test_batches if split == "test" else data.c.val_batches
    return {lab: eval_loss(model, data, lab, aux_on, split, n) for lab in LABELS}


# --------------------------------------------------------------------------------------------------------------- #
# Pretraining loop shared by baseline, filtering and GRAM                                                          #
# --------------------------------------------------------------------------------------------------------------- #

# a batch spec: (data label, modules on in the forward pass, update the core?)
Spec = tuple[str, tuple[int, ...], bool]


def wsd(step: int, total: int, c: Cfg) -> float:
    w, d = max(1, round(c.warmup * total)), max(1, round(c.decay * total))
    end = 1e-8 / c.lr
    if step < w:
        return end + (1 - end) * step / w
    if step < total - d:
        return 1.0
    return 1 - (1 - end) * (step - (total - d)) / d


def pretrain(
    model: GPT,
    data: Data,
    specs: list[Spec],
    c: Cfg,
    seed: int,
    log,
    curve: bool = False,
) -> dict:
    """One optimiser per parameter partition (core, each module); a partition steps only when it got a gradient."""
    random.Random(seed).shuffle(specs)
    if c.max_steps:
        specs = specs[: c.max_steps]
    total = len(specs)
    core_ps = model.core_params()
    aux_ps = [model.aux_params(i) for i in range(model.n_aux)]
    opts = {"core": torch.optim.AdamW(core_ps, lr=c.lr, betas=c.betas, weight_decay=c.wd, fused=True)}
    for i in range(model.n_aux):
        opts[i] = torch.optim.AdamW(aux_ps[i], lr=c.lr, betas=c.betas, weight_decay=c.wd, fused=True)
    sampler = Sampler(data, seed)
    fwd = torch.compile(model)
    curve_pts = []
    every = max(1, total // c.curve_evals)
    t0 = time.time()
    model.train()
    for step, (lab, aux_on, upd_core) in enumerate(specs):
        lr = c.lr * wsd(step, total, c)
        for o in opts.values():
            o.param_groups[0]["lr"] = lr
        w = sampler.next(lab, c.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ce(fwd(w[:, :-1], aux_on), w[:, 1:])
        loss.backward()
        if not upd_core:  # gradient routing: this batch may only change its module
            for p in core_ps:
                p.grad = None
        stepped = (["core"] if upd_core else []) + list(aux_on)
        params = [p for k in stepped for p in (core_ps if k == "core" else aux_ps[k])]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for k in stepped:
            opts[k].step()
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        if curve and ((step + 1) % every == 0 or step + 1 == total):
            pt = {"step": step + 1}
            for lab2 in LABELS:
                pt[lab2] = eval_loss(model, data, lab2, (), "test", c.curve_batches)
            curve_pts.append(pt)
        if step % 1000 == 0 or step + 1 == total:
            log(f"step {step + 1}/{total} loss {loss.item():.3f} lr {lr:.2e} {time.time() - t0:.0f}s")
    return {"steps": total, "seconds": time.time() - t0, "curve": curve_pts}


def baseline_specs(data: Data) -> list[Spec]:
    return [(lab, (), True) for lab in LABELS for _ in range(data.n_batches(lab))]


def filtering_specs(data: Data, keep: tuple[int, ...]) -> list[Spec]:
    labs = ["core"] + [AUX_LABELS[i] for i in keep]
    return [(lab, (), True) for lab in labs for _ in range(data.n_batches(lab))]


def gram_specs(data: Data, c: Cfg, P_base: int, P_core: int, P_aux: float) -> tuple[list[Spec], dict]:
    """Batch list of `do_routed_unordered` with equal_compute=True, label_prc=1, aux_factor=1."""
    n = {lab: data.n_batches(lab) for lab in LABELS}
    len_all = sum(n.values())
    len_aux = sum(n[a] for a in AUX_LABELS)
    len_core = len_all - len_aux
    len_core_ft = round(len_core * c.p_cr)
    len_base = len_core - len_core_ft
    len_ft = len_aux + len_core_ft
    prc_core_ft = len_core_ft / len_ft
    base_fl, ft_fl = len_base * P_core, len_ft * (P_core + P_aux)
    diff = len_all * P_base - (base_fl + ft_fl)
    len_base += round(base_fl / (base_fl + ft_fl) * diff / P_core)
    len_ft += round(ft_fl / (base_fl + ft_fl) * diff / (P_core + P_aux))
    len_core_ft = round(len_ft * prc_core_ft)
    len_aux_ft = len_ft - len_core_ft
    specs: list[Spec] = [("core", (), True)] * len_base
    prop = {a: n[a] / len_aux for a in AUX_LABELS}
    for i, a in enumerate(AUX_LABELS):
        cur_aux, cur_core = round(len_aux_ft * prop[a]), round(len_core_ft * prop[a])
        routed = round(cur_aux * c.p_as)
        specs += [(a, (i,), True)] * routed  # aux batch, core also updated (p_as)
        specs += [(a, (i,), False)] * (cur_aux - routed)  # aux batch, only its module updated
        specs += [("core", (i,), True)] * cur_core  # robust core batch: module on, both updated
    info = {
        "len_all": len_all,
        "len_base": len_base,
        "len_ft": len_ft,
        "len_aux_ft": len_aux_ft,
        "len_core_ft": len_core_ft,
        "total": len(specs),
    }
    return specs, info


# --------------------------------------------------------------------------------------------------------------- #
# MaxEnt post-hoc unlearning (src/run/train/maxent.py)                                                             #
# --------------------------------------------------------------------------------------------------------------- #


def kl_to_uniform(logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits.float(), dim=-1)
    return (lp.exp() * lp).sum(-1) + math.log(logits.size(-1))  # KL(p || U) = log V - H(p)


def maxent(
    model: GPT,
    data: Data,
    profile: tuple[int, ...],
    lr: float,
    alpha: float,
    seed: int,
    steps: int = 2000,
    stop: float = 2.0,
) -> dict:
    forget = [AUX_LABELS[i] for i in range(len(AUX_LABELS)) if i not in profile]
    retain_aux = [AUX_LABELS[i] for i in profile]
    rng = random.Random(seed)
    sampler = Sampler(data, seed + 77)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    half = data.c.batch // 2
    model.train()
    done = 0
    for step in range(steps):
        fx = sampler.next(forget[step % len(forget)], half)
        rlab = "core" if (rng.random() < 0.75 or not retain_aux) else rng.choice(retain_aux)
        rx = sampler.next(rlab, half)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(torch.cat([fx[:, :-1], rx[:, :-1]]))
        unif = kl_to_uniform(logits[:half]).mean()
        loss = unif + alpha * ce(logits[half:], rx[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        if unif.item() <= stop:
            break
    return {"steps": done, "final_kl": unif.item()}


# --------------------------------------------------------------------------------------------------------------- #
# Elicitation: the relearning attack (src/run/train/finetune.py, do_finetune)                                       #
# --------------------------------------------------------------------------------------------------------------- #


def elicit(
    model: GPT,
    data: Data,
    lab: str,
    aux_on: tuple[int, ...],
    seed: int,
    num_seq: int = 512,
    rounds: int = 200,
    patience: int = 10,
    lr: float | None = None,
) -> dict:
    """Fine-tune `model` in place on num_seq sequences of `lab`, one full-batch step per round; keep the best round
    on val, report test. Pass a copy."""
    c = data.c
    lr = lr if lr is not None else c.lr / 4
    g = torch.Generator().manual_seed(seed * 7919 + LABELS.index(lab))
    idx = torch.randperm(data.train[lab].n, generator=g)[:num_seq].to(data.device)
    seqs = data.train[lab].windows(idx)
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr, fused=True)
    val0 = eval_loss(model, data, lab, aux_on, "val", c.val_batches)
    best_val, best_round, bad = val0, 0, 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    vals = [val0]
    nmb = (num_seq + c.batch - 1) // c.batch
    model.train()
    for r in range(1, rounds + 1):
        for j in range(nmb):
            w = seqs[j * c.batch : (j + 1) * c.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ce(model(w[:, :-1], aux_on), w[:, 1:]) / nmb
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        v = eval_loss(model, data, lab, aux_on, "val", c.val_batches)
        vals.append(v)
        if v <= best_val:
            best_val, best_round, bad = v, r, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    test = eval_loss(model, data, lab, aux_on, "test", c.test_batches)
    return {
        "label": lab,
        "num_seq": num_seq,
        "best_round": best_round,
        "val_curve": vals,
        "test_loss": test,
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
