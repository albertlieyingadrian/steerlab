"""
The Part 18 world model: a tiny GPT that sees ball positions and predicts the next frame (KTD2, KTD3).

    WorldModel(...)                 the transformer (one token per frame, 6 numbers in, 6 numbers out)
    train(model, objective, ...)    AdamW training with one of three objectives (KTD7):
                                      "next"         next frame only                        (Model A)
                                      "rollout8"     8 frames ahead, feeding its own guesses (Model B)
                                      "next+energy"  next frame plus a total-energy head    (Model C)
    predict(model, positions)       next-frame predictions for whole episodes, in batches
    rollout(model, positions, k)    give the model the first k true frames, then let it run on its own guesses
    copy_last / constant_velocity   the two baselines that need no learning (R5)
    frame_kinds(collides_next)      which frames are free flight, which have a collision next, which just bounced

Shapes: positions are [episodes, frames, 3 balls, 2] in the box's units, or flattened to [episodes, frames, 6].
The model's prediction at frame t is its guess for frame t + 1, made from frames 0..t only.

How the output works: the head predicts a velocity-sized number per coordinate, and the predicted next position is
    x̂_{t+1} = x_t + DT · head(h_t)
so "do nothing" (copy the last frame) is the starting point and the head only has to learn the step. The loss is
the MSE of that step in velocity units, (x̂_{t+1} - x_{t+1}) / DT, so a loss of 1e-4 means about 0.0005 box units.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DT = 0.05  # frame spacing, as in physics.py
N_COORDS = 6  # 3 balls x 2 coordinates
CONTEXT = 63  # 64 frames per episode: inputs are frames 0..62, targets 1..63
OBJECTIVES = ("next", "rollout8", "next+energy")
ROLLOUT_STEPS = 8
ENERGY_WEIGHT = 0.1  # KTD7: Model C's energy loss weight
ENERGY_MEAN, ENERGY_STD = 1.25, 0.433  # the training range 0.5..2.0 as a uniform: standardises the energy target


# --- 1. The model --------------------------------------------------------------------------------------------------


class Attention(nn.Module):
    """Causal multi-head self-attention: each frame may look at itself and earlier frames only."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (z.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for z in (q, k, v))
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(mixed.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    """Pre-norm transformer block: x + attention(norm(x)), then + MLP(norm(x)). Its output is the residual stream."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class WorldModel(nn.Module):
    """One token per frame. Blocks live in `self.layers`, which sandbagkit's capture_layers/register_hooks find."""

    def __init__(self, d_model=128, n_layers=4, n_heads=4, d_mlp=512, context=CONTEXT, energy_head=False):
        super().__init__()
        self.config = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_mlp=d_mlp, context=context,
                           energy_head=energy_head)
        self.embed = nn.Linear(N_COORDS, d_model)
        self.pos = nn.Parameter(torch.randn(context, d_model) * 0.02)
        self.layers = nn.ModuleList([Block(d_model, n_heads, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_COORDS)
        self.energy = nn.Linear(d_model, 1) if energy_head else None

    def forward(self, x):
        """x: [B, T, 6] raw positions. Returns (next positions [B, T, 6], standardised energy [B, T] or None)."""
        T = x.shape[1]
        h = self.embed((x - 0.5) / 0.3) + self.pos[:T]  # centre the box and scale it to about unit size
        for block in self.layers:
            h = block(h)
        h = self.ln_f(h)
        nxt = x + DT * self.head(h)
        energy = self.energy(h).squeeze(-1) if self.energy is not None else None
        return nxt, energy

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(config: dict, seed: int) -> WorldModel:
    torch.manual_seed(seed)
    return WorldModel(**config)


# --- 2. Training ---------------------------------------------------------------------------------------------------


def to_tensor(positions, device) -> torch.Tensor:
    """[E, T, 3, 2] float64 numpy -> [E, T, 6] float32 tensor on `device`."""
    p = np.asarray(positions, dtype=np.float32)
    return torch.from_numpy(p.reshape(*p.shape[:2], -1)).to(device)


def step_loss(pred, target):
    """MSE of the predicted step in velocity units, skipping frame 0 (one frame says nothing about velocity)."""
    return (((pred[:, 1:] - target[:, 1:]) / DT) ** 2).mean()


def rollout_loss(model, x, rng: np.random.Generator):
    """Model B's objective: from a random cut c, predict c+1..c+8 feeding back its own guesses; mean over the 8."""
    c = int(rng.integers(1, x.shape[1] - ROLLOUT_STEPS))  # x has 64 frames; the last guess is frame c + 8 <= 63
    seq = x[:, : c + 1]
    losses = []
    for k in range(ROLLOUT_STEPS):
        pred, _ = model(seq[:, -CONTEXT:])
        guess = pred[:, -1:]
        losses.append((((guess - x[:, c + k + 1 : c + k + 2]) / DT) ** 2).mean())
        seq = torch.cat([seq, guess], dim=1)
    return torch.stack(losses).mean()


def loss_for(model, objective, x, energy, rng):
    """x: [B, 64, 6] whole episodes. energy: [B] true total energy. Returns (loss, next-frame loss)."""
    pred, e_hat = model(x[:, :-1])
    nxt = step_loss(pred, x[:, 1:])
    if objective == "next":
        return nxt, nxt
    if objective == "next+energy":
        target = ((energy - ENERGY_MEAN) / ENERGY_STD)[:, None].expand_as(e_hat)
        return nxt + ENERGY_WEIGHT * F.mse_loss(e_hat[:, 1:], target[:, 1:]), nxt
    if objective == "rollout8":
        return 0.5 * (nxt + rollout_loss(model, x, rng)), nxt
    raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")


def train(model, objective, train_pos, train_energy, *, val_pos=None, epochs=20, batch=128, lr=1e-3,
          weight_decay=0.01, warmup=200, seed=0, device="cpu", log_every=1):
    """AdamW with linear warmup then cosine decay. Positions are numpy [E, 64, 3, 2]; energies numpy [E].

    Returns a dict: per-epoch train loss, val next-frame loss, wall time and device. The model is trained in place.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    X = to_tensor(train_pos, device)
    Ev = torch.as_tensor(train_energy, dtype=torch.float32, device=device)
    steps_per_epoch = math.ceil(len(X) / batch)
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    curve = {"epoch": [], "train_loss": [], "train_next_loss": [], "val_next_loss": [], "seconds": []}
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.from_numpy(rng.permutation(len(X))).to(device)
        running, running_next = 0.0, 0.0
        for s in range(steps_per_epoch):
            idx = order[s * batch : (s + 1) * batch]
            loss, nxt = loss_for(model, objective, X[idx], Ev[idx], rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            running_next += nxt.item()
        curve["epoch"].append(epoch + 1)
        curve["train_loss"].append(running / steps_per_epoch)
        curve["train_next_loss"].append(running_next / steps_per_epoch)
        val = next_frame_loss(model, val_pos, device) if val_pos is not None else None
        curve["val_next_loss"].append(val)
        curve["seconds"].append(round(time.monotonic() - started, 1))
        model.train()
        if log_every and (epoch + 1) % log_every == 0:
            v = f", val next-frame {val:.2e}" if val is not None else ""
            print(f"   epoch {epoch + 1:>2}/{epochs}: train {curve['train_loss'][-1]:.2e}{v} "
                  f"({curve['seconds'][-1]:.0f} s)", flush=True)
    model.eval()
    curve["device"] = str(device)
    curve["wall_seconds"] = round(time.monotonic() - started, 1)
    curve["steps"] = total
    return curve


# --- 3. Using a trained model --------------------------------------------------------------------------------------


@torch.no_grad()
def predict(model, positions, device, batch=512) -> np.ndarray:
    """Next-frame predictions for whole episodes: positions [E, 64, 3, 2] -> [E, 63, 3, 2] (guess for frames 1..63)."""
    model.eval()
    out = []
    for s in range(0, len(positions), batch):
        x = to_tensor(positions[s : s + batch, :-1], device)
        pred, _ = model(x)
        out.append(pred.float().cpu().numpy())
    p = np.concatenate(out).astype(np.float64)
    return p.reshape(p.shape[0], p.shape[1], -1, 2)


def next_frame_loss(model, positions, device) -> float:
    """The training loss (velocity units, frames 1..62 as inputs) on a whole split, without gradients."""
    pred = predict(model, positions, device)
    return float((((pred[:, 1:] - positions[:, 2:]) / DT) ** 2).mean())


@torch.no_grad()
def rollout(model, positions, given: int, device, batch=512) -> np.ndarray:
    """Give the model frames 0..given-1, then feed its own guesses back until frame 63. Returns [E, 64, 3, 2]
    (the first `given` frames are the true ones)."""
    model.eval()
    out = []
    for s in range(0, len(positions), batch):
        seq = to_tensor(positions[s : s + batch, :given], device)
        while seq.shape[1] < positions.shape[1]:
            pred, _ = model(seq[:, -CONTEXT:])
            seq = torch.cat([seq, pred[:, -1:]], dim=1)
        out.append(seq.float().cpu().numpy())
    r = np.concatenate(out).astype(np.float64)
    return r.reshape(r.shape[0], r.shape[1], -1, 2)


def copy_last(positions) -> np.ndarray:
    """Baseline 1: the next frame is the same as this one. [E, 64, 3, 2] -> [E, 63, 3, 2]."""
    return positions[:, :-1].copy()


def constant_velocity(positions) -> np.ndarray:
    """Baseline 2: keep the last step, x̂_{t+1} = 2·x_t − x_{t−1}. Frame 0 has no previous frame, so it copies.
    Exact in free flight, wrong across any bounce. [E, 64, 3, 2] -> [E, 63, 3, 2]."""
    p = positions[:, :-1].copy()
    p[:, 1:] = 2 * positions[:, 1:-1] - positions[:, :-2]
    return p


def frame_kinds(collides_next) -> dict:
    """Boolean masks over prediction frames t = 1..62 (inputs that have a previous frame), shape [E, 62]:
      collision   a collision happens after t and at or before t + 1 (the bounce the model must predict)
      just_hit    a collision happened after t − 1 and at or before t, and none comes next
      free        no collision from t − 1 to t + 1: constant velocity is exact here
    collides_next is [E, 63, K] from physics.make_split."""
    any_next = collides_next.any(axis=2)  # [E, 63]: collision in (t, t+1]
    nxt = any_next[:, 1:]  # t = 1..62
    prev = any_next[:, :-1]  # collision in (t-1, t]
    return {"collision": nxt, "just_hit": prev & ~nxt, "free": ~prev & ~nxt}


def step_mse(pred, positions) -> np.ndarray:
    """Per-(episode, frame) squared error of the predicted step in velocity units, for t = 1..62. [E, 62]"""
    err = (pred[:, 1:] - positions[:, 2:]) / DT
    return (err**2).mean(axis=(2, 3))


def save_checkpoint(model, path, extra: dict | None = None) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path, device="cpu") -> WorldModel:
    """Load a WorldModel saved by save_checkpoint, or exit naming the script that makes it."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} is missing. Train it first with experiments/world-model-physics/train.py "
                         "(pre-trained models) or experiments/world-model-physics/exp_c.py train (post-trained arms).")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = WorldModel(**saved["config"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()
