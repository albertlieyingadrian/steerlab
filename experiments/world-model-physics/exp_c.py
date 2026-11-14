"""
Experiment C: post-train the pre-trained base-world models for physical consistency, with RL or with more data,
and check from inside what each one taught.

Arms, per pre-trained seed (2610, 2611, 2612):
    pre   the pre-trained model (experiment A's)
    rl    GRPO-style policy gradient on the model's own sampled rollouts, reward = energy conservation + wall legality
    sft   supervised next-frame training on 20,000 fresh simulator episodes, with the same compute as rl

The policy: the world model's prediction is the mean of a Gaussian with fixed sigma (SIGMA box units per coordinate).
From an 8-frame true prefix it samples G = 8 rollouts of N = 24 frames each; the reward scores each rollout; the
advantage is the reward standardised within its group of 8 (GRPO); the loss is -advantage x log-probability of the
sampled frames. Why a policy gradient and not backpropagating the reward: the wall-legality term is a hinge that is
flat (zero gradient) for every legal frame, and the point of the study is to see what a method that only sees a score
learns, as RL on a real world model would.

The reward (the "physics check" RL optimises), for a rollout x_0..x_31 whose first 8 frames are the true prefix:
    KE_t = 1/2 sum |(x_{t+1} - x_t) / DT|^2 over balls            finite-difference kinetic energy
    E_ref = median of KE_t over the prefix's 7 steps
    energy term = median over the 24 generated steps of |KE_t - E_ref| / E_ref
    wall term = mean over generated frames and coordinates of max(0, r - x) + max(0, x - (1 - r)), divided by r
    reward = -(energy term + wall term)
The median ignores the few frames where a bounce falls between frames (finite differences dip there even for the true
trajectory). `reward-check` scores the true trajectories and degenerate ones before any training.

Held-out checks (never rewarded), on deterministic rollouts of 56 frames from 8 true frames:
    position error vs the simulator at horizons 8, 24, 56; valid horizon (frames until any ball is 0.05 off)
    energy drift: mean |KE_t - E| / E over generated steps (E = the episode's true energy)
    ball overlap: share of frames where two balls are closer than 2r - 0.01 (they should have bounced)
    momentum: |change in total momentum| per step on steps where every ball is clear of the walls (should be ~0)
    speed ratio: mean speed in the rollout / in the truth (1 = no damping)

    .venv/bin/python experiments/world-model-physics/exp_c.py reward-check
    .venv/bin/python experiments/world-model-physics/exp_c.py pilot --seed 2610
    .venv/bin/python experiments/world-model-physics/exp_c.py train --arm rl --seed 2610
    .venv/bin/python experiments/world-model-physics/exp_c.py train --arm sft --seed 2610
    .venv/bin/python experiments/world-model-physics/exp_c.py evaluate
    .venv/bin/python experiments/world-model-physics/exp_c.py inside
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

import common as C
import worldmodel as wm

DT, RADIUS = wm.DT, 0.05
K, N, G = 8, 24, 8  # prefix frames, generated frames, rollouts per prefix
B = 64  # prefixes per step
SIGMA = 0.001  # policy noise, box units per coordinate (truth + this noise scores -0.031)
RL_STEPS = 300
RL_LR = 3e-5  # used only if the pilot has not run; the pilot's choice is in results/exp_c_pilot.json
PILOT_LRS = (1e-5, 3e-5, 1e-4)
PILOT_STEPS = 80
SFT_LR = 1e-4
SFT_BATCH = 128
EVAL_GIVEN, TOL = 8, 0.05


# --- reward ---------------------------------------------------------------------------------------------------------


def reward_parts(seq, k=K):
    """seq [M, T, 6] (torch or numpy). Returns (energy term, wall term), each [M]."""
    v = (seq[:, 1:] - seq[:, :-1]) / DT
    ke = 0.5 * (v**2).sum(-1)  # [M, T-1]
    g = seq[:, k:]
    if isinstance(seq, torch.Tensor):
        eref = ke[:, : k - 1].median(1).values
        et = ((ke[:, k - 1 :] - eref[:, None]).abs() / eref[:, None]).median(1).values
        wall = ((RADIUS - g).clamp(min=0) + (g - (1 - RADIUS)).clamp(min=0)).mean((1, 2)) / RADIUS
    else:
        eref = np.median(ke[:, : k - 1], 1)
        et = np.median(np.abs(ke[:, k - 1 :] - eref[:, None]) / eref[:, None], 1)
        wall = (np.maximum(0, RADIUS - g) + np.maximum(0, g - (1 - RADIUS))).mean((1, 2)) / RADIUS
    return et, wall


def reward(seq, k=K):
    et, wall = reward_parts(seq, k)
    return -(et + wall)


# --- rollouts --------------------------------------------------------------------------------------------------------


def prefixes(data, n, rng, k=K, horizon=N):
    """n random windows of k true frames (start c in [0, 64 - k - horizon]) -> (prefix [n, k, 6], window
    [n, k + horizon, 6], episode, start)."""
    e = rng.integers(0, len(data["positions"]), n)
    c = rng.integers(0, 64 - k - horizon + 1, n)
    P = data["positions"].reshape(len(data["positions"]), 64, 6)
    win = np.stack([P[ei, ci : ci + k + horizon] for ei, ci in zip(e, c)])
    return win[:, :k], win, e, c


@torch.no_grad()
def sample(model, prefix, n_gen=N, sigma=SIGMA):
    seq = prefix
    for _ in range(n_gen):
        mu, _ = model(seq[:, -wm.CONTEXT :])
        nxt = mu[:, -1:]
        if sigma > 0:
            nxt = nxt + sigma * torch.randn_like(nxt)
        seq = torch.cat([seq, nxt], 1)
    return seq


def logprob(model, seq, k=K, sigma=SIGMA):
    mu, _ = model(seq[:, :-1])
    gen_mu, gen_x = mu[:, k - 1 :], seq[:, k:]
    return -((gen_x - gen_mu) ** 2).sum((1, 2)) / (2 * sigma**2)


# --- compute accounting (token passes: forward = 1, backward = 2) ---------------------------------------------------

RL_STEP_COST = B * G * (sum(K + n for n in range(N)) + 3 * (K + N - 1))
SFT_STEP_COST = SFT_BATCH * wm.CONTEXT * 3
SFT_STEPS = round(RL_STEPS * RL_STEP_COST / SFT_STEP_COST)


# --- training --------------------------------------------------------------------------------------------------------


@torch.no_grad()
def val_reward(model, val, device):
    """Reward of deterministic (mean) rollouts from fixed validation prefixes."""
    model.eval()
    seq = sample(model, torch.as_tensor(val[0], dtype=torch.float32, device=device), sigma=0.0)
    return float(reward(seq).mean())


def val_set(n=512):
    return prefixes(C.base_split("val"), n, np.random.default_rng(C.SEEDS["rl"] + 99))


def rl_train(model, device, steps, lr, seed, log_every=20, val=None):
    data = C.base_split("rlprefix")
    rng = np.random.default_rng([C.SEEDS["rl"], seed])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    curve = {"step": [], "reward": [], "energy": [], "wall": [], "val_reward": []}
    t0 = time.monotonic()
    for step in range(steps):
        model.train()
        pre = prefixes(data, B, rng)[0]
        pre_t = torch.as_tensor(pre, dtype=torch.float32, device=device).repeat_interleave(G, 0)
        seq = sample(model, pre_t)
        et, wall = reward_parts(seq)
        R = -(et + wall)
        Rg = R.view(B, G)
        adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-6)).view(-1)
        loss = -(adv.detach() * logprob(model, seq)).mean() / (N * 6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            v = val_reward(model, val, device) if val is not None else None
            for key, x in (
                ("step", step),
                ("reward", R.mean().item()),
                ("energy", et.mean().item()),
                ("wall", wall.mean().item()),
                ("val_reward", v),
            ):
                curve[key].append(x)
            print(
                f"   rl step {step:>4}: reward {R.mean():+.4f} (energy {et.mean():.4f}, wall {wall.mean():.4f})"
                + (f"  val {v:+.4f}" if v is not None else "")
                + f"  {time.monotonic() - t0:.0f} s",
                flush=True,
            )
    model.eval()
    curve["seconds"] = round(time.monotonic() - t0, 1)
    curve["token_passes"] = steps * RL_STEP_COST
    return curve


PATH_STEP_COST = B * G * 3 * sum(K + n for n in range(N))  # 24 sequential forward passes with gradient, + backward
PATH_STEPS = round(RL_STEPS * RL_STEP_COST / PATH_STEP_COST)


def pathwise_train(model, device, steps, lr, seed, log_every=20, val=None):
    """Amendment 1 arm: the same reward, the same sampled rollouts (x = mu + sigma * eps), but the gradient flows through
    the samples into the model (reparameterisation), so each step follows d reward / d parameters directly."""
    data = C.base_split("rlprefix")
    rng = np.random.default_rng([C.SEEDS["rl"], seed])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    curve = {"step": [], "reward": [], "energy": [], "wall": [], "val_reward": []}
    t0 = time.monotonic()
    for step in range(steps):
        model.train()
        seq = torch.as_tensor(prefixes(data, B, rng)[0], dtype=torch.float32, device=device).repeat_interleave(G, 0)
        for _ in range(N):
            mu, _ = model(seq)
            seq = torch.cat([seq, mu[:, -1:] + SIGMA * torch.randn_like(mu[:, -1:])], 1)
        et, wall = reward_parts(seq)
        R = -(et + wall)
        loss = -R.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            v = val_reward(model, val, device) if val is not None else None
            for key, x in (
                ("step", step),
                ("reward", R.mean().item()),
                ("energy", et.mean().item()),
                ("wall", wall.mean().item()),
                ("val_reward", v),
            ):
                curve[key].append(x)
            print(
                f"   pathwise step {step:>4}: reward {R.mean():+.4f} (energy {et.mean():.4f}, wall "
                f"{wall.mean():.4f})"
                + (f"  val {v:+.4f}" if v is not None else "")
                + f"  {time.monotonic() - t0:.0f} s",
                flush=True,
            )
    model.eval()
    curve["seconds"] = round(time.monotonic() - t0, 1)
    curve["token_passes"] = steps * PATH_STEP_COST
    return curve


def sft_train(model, device, steps, lr, seed, log_every=200):
    data = C.base_split("sft")
    X = wm.to_tensor(data["positions"], device)
    rng = np.random.default_rng([C.SEEDS["sft"], seed])
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    curve = {"step": [], "loss": []}
    t0 = time.monotonic()
    order, pos = rng.permutation(len(X)), 0
    for step in range(steps):
        model.train()
        if pos + SFT_BATCH > len(order):
            order, pos = rng.permutation(len(X)), 0
        x = X[torch.as_tensor(order[pos : pos + SFT_BATCH], device=device)]
        pos += SFT_BATCH
        pred, _ = model(x[:, :-1])
        loss = wm.step_loss(pred, x[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            curve["step"].append(step)
            curve["loss"].append(loss.item())
            print(f"   sft step {step:>5}: loss {loss.item():.2e}  {time.monotonic() - t0:.0f} s", flush=True)
    model.eval()
    curve["seconds"] = round(time.monotonic() - t0, 1)
    curve["token_passes"] = steps * SFT_STEP_COST
    return curve


# --- evaluation ------------------------------------------------------------------------------------------------------


def metrics(roll, P, E, given=EVAL_GIVEN):
    """Rollout metrics against the simulator. roll, P: [E, 64, 3, 2]; E: true energies [E]."""
    dist = np.linalg.norm(roll - P, axis=-1)  # [E, 64, 3]
    mean_err = dist.mean(-1)
    off = (dist.max(-1) > TOL)[:, given:]
    valid = np.where(off.any(1), off.argmax(1), off.shape[1])
    v = (roll[:, 1:] - roll[:, :-1]) / DT  # [E, 63, 3, 2]
    vt = (P[:, 1:] - P[:, :-1]) / DT
    ke = 0.5 * (v**2).sum((-1, -2))[:, given - 1 :]
    ke_t = 0.5 * (vt**2).sum((-1, -2))[:, given - 1 :]
    drift = (np.abs(ke - E[:, None]) / E[:, None]).mean(1)
    drift_truth = (np.abs(ke_t - E[:, None]) / E[:, None]).mean(1)
    g = roll[:, given:]
    I, J = np.array([(0, 1), (0, 2), (1, 2)]).T
    sep = np.linalg.norm(g[:, :, J] - g[:, :, I], axis=-1)  # [E, T, 3]
    overlap = (sep < 2 * RADIUS - 0.01).any(-1)
    clear = ((g > RADIUS + 0.15) & (g < 1 - RADIUS - 0.15)).all((-1, -2))  # [E, T]: no wall within one step
    mom = v[:, given:].sum(2)  # total momentum per step (mass 1) [E, T-1, 2]
    dmom = np.linalg.norm(mom[:, 1:] - mom[:, :-1], axis=-1)  # [E, T-2]
    both_clear = clear[:, :-2] & clear[:, 1:-1] & clear[:, 2:]
    speed = np.linalg.norm(v[:, given:], axis=-1).mean((1, 2))
    speed_t = np.linalg.norm(vt[:, given:], axis=-1).mean((1, 2))
    wall = (np.maximum(0, RADIUS - g) + np.maximum(0, g - (1 - RADIUS))).mean((1, 2, 3)) / RADIUS
    out = {
        "episodes": len(P),
        "pos_err_h8": float(mean_err[:, given + 7].mean()),
        "pos_err_h24": float(mean_err[:, given + 23].mean()),
        "pos_err_h56": float(mean_err[:, 63].mean()),
        "pos_err_mean": float(mean_err[:, given:].mean()),
        "valid_horizon_median": float(np.median(valid)),
        "valid_horizon_mean": float(valid.mean()),
        "energy_drift": float(drift.mean()),
        "energy_drift_truth_floor": float(drift_truth.mean()),
        "ball_overlap_rate": float(overlap.mean()),
        "momentum_change_clear": float(dmom[both_clear].mean()) if both_clear.any() else None,
        "speed_ratio": float((speed / speed_t).mean()),
        "wall_term": float(wall.mean()),
        "reward_24": float(np.mean(reward(roll[:, given - K : given + N].reshape(len(P), K + N, 6)))),
    }
    per = {
        "pos_err_mean": mean_err[:, given:].mean(1),
        "energy_drift": drift,
        "valid": valid.astype(float),
        "overlap": overlap.mean(1),
        "speed_ratio": speed / speed_t,
    }
    return out, per


def evaluate(model, data, device, given=EVAL_GIVEN):
    roll = wm.rollout(model, data["positions"], given, device)  # [E, 64, 3, 2]
    return metrics(roll, data["positions"], data["energy"], given)


# --- commands --------------------------------------------------------------------------------------------------------


def cmd_reward_check():
    """Score true and degenerate trajectories with the reward before any training."""
    import physics as ph

    d = C.base_split("val")
    n = 500
    rng = np.random.default_rng(C.SEEDS["rl"] + 7)
    _, truth, e, c = prefixes(d, n, rng)
    ghost = truth.copy()
    for i in range(n):  # each ball simulated alone from the prefix's last frame: balls pass through each other
        for b in range(3):
            p0 = d["positions"][e[i], c[i] + K - 1, b].reshape(1, 2)
            v0 = d["velocities"][e[i], c[i] + K - 1, b].reshape(1, 2)
            ep = ph.simulate(p0, v0, n_frames=N + 1)
            ghost[i, K - 1 :, 2 * b : 2 * b + 2] = ep["positions"][:, 0]
    frozen = truth.copy()
    frozen[:, K:] = truth[:, K - 1 : K]
    damped = truth.copy()  # every generated step shrunk to 80%: speeds fall to 0.8x
    damped[:, K:] = truth[:, K - 1 : K] + np.cumsum(np.diff(truth[:, K - 1 :], axis=1) * 0.8, 1)
    noisy = truth.copy()
    noisy[:, K:] += SIGMA * rng.standard_normal(noisy[:, K:].shape)
    out = {}
    for name, s in (
        ("truth", truth),
        ("truth + policy noise", noisy),
        ("ghost: no ball-ball collisions", ghost),
        ("damped to 0.8x speed", damped),
        ("frozen", frozen),
    ):
        et, wall = reward_parts(s)
        out[name] = {
            "reward": float(-(et + wall).mean()),
            "energy_term": float(et.mean()),
            "wall_term": float(wall.mean()),
        }
        print(f"   {name:<32} reward {out[name]['reward']:+.4f}  energy {et.mean():.4f}  wall {wall.mean():.4f}")
    C.save_json("exp_c_reward_check.json", {"rollouts": n, "sigma": SIGMA, "scores": out})


def cmd_pilot(seed):
    device = C.pick_device()
    val = val_set()
    rows = []
    for lr in PILOT_LRS:
        model = C.load_model("base", seed, "pre", device)
        v0 = val_reward(model, val, device)
        print(f" lr {lr:g}: val reward before {v0:+.4f}", flush=True)
        curve = rl_train(model, device, PILOT_STEPS, lr, seed, log_every=20, val=val)
        rows.append({"lr": lr, "val_reward_before": v0, "val_reward_after": curve["val_reward"][-1], "curve": curve})
    best = max(rows, key=lambda r: r["val_reward_after"])
    C.save_json(
        "exp_c_pilot.json",
        {
            "seed": seed,
            "steps": PILOT_STEPS,
            "rows": rows,
            "chosen_lr": best["lr"],
            "rule": "highest validation reward (deterministic rollouts) after the pilot",
        },
    )
    print(f" chosen lr {best['lr']:g}")


AMEND_LRS = (1e-7, 3e-7, 1e-6, 3e-6)  # amendment 1: the registered grid was above the region where steps help


def cmd_pilot_amend(seed, method):
    """Amendment 1: the registered pilot rule (best validation reward, validation windows only) on a lower lr grid,
    for GRPO (80 steps) and for the pathwise arm (32 steps, about the same compute)."""
    device = C.pick_device()
    val = val_set()
    rows = []
    train_fn, steps, every = (rl_train, PILOT_STEPS, 20) if method == "grpo" else (pathwise_train, 32, 8)
    for lr in AMEND_LRS:
        model = C.load_model("base", seed, "pre", device)
        v0 = val_reward(model, val, device)
        print(f" {method} lr {lr:g}: val reward before {v0:+.4f}", flush=True)
        curve = train_fn(model, device, steps, lr, seed, log_every=every, val=val)
        rows.append({"lr": lr, "val_reward_before": v0, "val_reward_after": curve["val_reward"][-1], "curve": curve})
    best = max(rows, key=lambda r: r["val_reward_after"])
    C.save_json(
        f"exp_c_pilot_amend_{method}.json",
        {
            "seed": seed,
            "steps": steps,
            "rows": rows,
            "chosen_lr": best["lr"],
            "rule": "highest validation reward (deterministic rollouts) after the pilot; amendment 1 grid",
        },
    )
    print(f" {method}: chosen lr {best['lr']:g}")


def cmd_train(arm, seed):
    device = C.pick_device()
    model = C.load_model("base", seed, "pre", device)
    if arm == "rl_registered":  # the pre-registered arm, lr from the registered pilot
        lr = C.read_json("exp_c_pilot.json")["chosen_lr"]
        curve = rl_train(model, device, RL_STEPS, lr, seed, val=val_set())
    elif arm == "rl":  # amendment 1: the same GRPO, lr from the lower grid
        lr = C.read_json("exp_c_pilot_amend_grpo.json")["chosen_lr"]
        curve = rl_train(model, device, RL_STEPS, lr, seed, val=val_set())
    elif arm == "pathwise":
        lr = C.read_json("exp_c_pilot_amend_pathwise.json")["chosen_lr"]
        curve = pathwise_train(model, device, PATH_STEPS, lr, seed, val=val_set())
    elif arm == "sft":
        lr = SFT_LR
        curve = sft_train(model, device, SFT_STEPS, lr, seed)
    else:
        raise SystemExit(f"unknown arm {arm}")
    curve["lr"] = lr
    wm.save_checkpoint(model, C.model_path("base", seed, arm), {"arm": arm, "seed": seed})
    C.save_json(f"exp_c_train_{arm}_s{seed}.json", {"arm": arm, "seed": seed, **curve})


def cmd_evaluate(seeds, arms):
    device = C.pick_device()
    splits = {"test": C.base_split("test"), "ood": C.base_split("ood")}
    out = {
        "truth": {s: metrics(d["positions"].copy(), d["positions"], d["energy"])[0] for s, d in splits.items()},
        "models": {},
        "costs": {
            "rl_step_token_passes": RL_STEP_COST,
            "sft_step_token_passes": SFT_STEP_COST,
            "rl_steps": RL_STEPS,
            "sft_steps": SFT_STEPS,
        },
    }
    per = {}
    for seed in seeds:
        for arm in arms:
            if not C.model_path("base", seed, arm).exists():
                continue
            model = C.load_model("base", seed, arm, device)
            for s, d in splits.items():
                m, p = evaluate(model, d, device)
                out["models"].setdefault(arm, {}).setdefault(s, {})[str(seed)] = m
                per[(arm, s, seed)] = p
                print(
                    f" {arm:>4} s{seed} {s:>4}: err h8 {m['pos_err_h8']:.4f} h24 {m['pos_err_h24']:.4f} h56 "
                    f"{m['pos_err_h56']:.4f} valid {m['valid_horizon_median']:.0f}  drift {m['energy_drift']:.3f} "
                    f"overlap {m['ball_overlap_rate']:.3f} dmom {m['momentum_change_clear']:.3f} speed "
                    f"{m['speed_ratio']:.3f} wall {m['wall_term']:.4f} reward {m['reward_24']:+.4f}",
                    flush=True,
                )
    rng = np.random.default_rng(C.SEEDS["rl"] + 5)
    diffs = {}
    for arm in arms:
        if arm == "pre":
            continue
        for s in splits:
            ok = [sd for sd in seeds if (arm, s, sd) in per and ("pre", s, sd) in per]
            for key in (ok and ("pos_err_mean", "energy_drift", "valid", "overlap", "speed_ratio")) or ():
                d = np.stack([per[(arm, s, sd)][key] - per[("pre", s, sd)][key] for sd in ok])  # [seeds, E]
                boots = [d[:, rng.integers(0, d.shape[1], d.shape[1])].mean() for _ in range(2000)]
                diffs.setdefault(arm, {}).setdefault(s, {})[key] = {
                    "mean": float(d.mean()),
                    "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "per_seed": [float(x) for x in d.mean(1)],
                }
    out["paired_vs_pre"] = diffs
    C.save_json("exp_c_eval.json", out)


def cmd_inside(seeds, arms, n_random):
    import exp_a

    device = C.pick_device()
    train, test = C.base_split("train"), C.base_split("test")
    out = C.read_json("exp_c_inside.json") or {}
    for seed in seeds:
        for arm in arms:
            if not C.model_path("base", seed, arm).exists():
                continue
            print(f" inside: seed {seed} arm {arm}", flush=True)
            model = C.load_model("base", seed, arm, device)
            out.setdefault(arm, {})[str(seed)] = exp_a.analyse(
                model, seed, device, train, test, n_random=n_random, n_frames=500, verbose=True
            )
            C.save_json("exp_c_inside.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("reward-check", "pilot", "pilot-amend", "train", "evaluate", "inside"))
    ap.add_argument("--method", choices=("grpo", "pathwise"), default="grpo")
    ap.add_argument("--seed", type=int, default=C.SEEDS["model_base"][0])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS["model_base"]))
    ap.add_argument("--arm", default="rl")
    ap.add_argument("--arms", nargs="*", default=["pre", "sft", "rl", "rl_registered", "pathwise"])
    ap.add_argument("--n-random", type=int, default=50)
    a = ap.parse_args()
    if a.cmd == "reward-check":
        cmd_reward_check()
    elif a.cmd == "pilot":
        cmd_pilot(a.seed)
    elif a.cmd == "pilot-amend":
        cmd_pilot_amend(a.seed, a.method)
    elif a.cmd == "train":
        cmd_train(a.arm, a.seed)
    elif a.cmd == "evaluate":
        cmd_evaluate(a.seeds, a.arms)
    else:
        cmd_inside(a.seeds, a.arms, a.n_random)
