"""Train the toy world model with teacher forcing on fresh random-walk episodes (a new random map every episode).

    python train.py --seed 1 --steps 20000 --out results/ckpt_s1.pt

Loss: cross-entropy over the 8 colours of each of the 4 tiles of each of the 16 patches of frames 1..T-1. Tiles never
seen before cannot be predicted (chance is 1/8), so the loss has a floor; what the model can learn is to redraw what
it has seen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from wm_core import Cfg, K, WorldModel, gaps, make_episodes, seq_mask


def batch(n, T, rng, dev):
    tiles, pose, act, loc = make_episodes(n, T, rng)
    g = gaps(loc)
    return [torch.from_numpy(x).to(dev) for x in (tiles, pose, act, loc, g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--out", default="results/ckpt.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    dev = a.device
    cfg = Cfg(T=a.T)
    model = WorldModel(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    # --steps 0 saves the untrained model (the floor for dream-mode self-consistency)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, a.steps), pct_start=0.05)
    mask = seq_mask(a.T, dev)
    log = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        tiles, pose, act, _loc, g = batch(a.bs, a.T, rng, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            lg = model.forward_seq(tiles, pose, act, mask)
        loss = F.cross_entropy(lg.float().reshape(-1, K), tiles[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == 1:
            with torch.no_grad():
                hit = (lg.argmax(-1) == tiles[:, 1:]).float().mean(-1)  # [B,T-1,16]
                gg = g[:, 1:]
                row = {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "sec": round(time.time() - t0, 1),
                    "acc_shift": round(hit[gg == 1].mean().item(), 4),
                    "acc_revisit": round(hit[gg >= 2].mean().item(), 4),
                    "acc_new": round(hit[gg == 0].mean().item(), 4),
                }
            log.append(row)
            print(json.dumps(row), flush=True)
    if not log:
        log.append({"step": 0})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state": model.state_dict(),
            "log": log,
            "params": nparam,
            "seed": a.seed,
            "steps": a.steps,
            "bs": a.bs,
            "lr": a.lr,
        },
        a.out,
    )
    print("saved", a.out, "params", nparam)


if __name__ == "__main__":
    main()
