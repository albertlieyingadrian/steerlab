"""
Experiment B: is a physical constant (gravity g, drag gamma) linearly readable in context, steerable, and does it
extrapolate beyond the training range?

Three constants-world models (gravity in [0, 2], drag in [0, 0.5] during training). For each:
  1. Prediction error, one step and rollout, on test (in range), ood_g (g in [2.5, 3.5]) and ood_drag (gamma in
     [0.7, 1.0]), against constant velocity and constant acceleration (3 x_t - 3 x_{t-1} + x_{t-2}).
  2. The model's BEHAVIOURAL constants: least-squares g and gamma that explain its own predicted accelerations on
     free-flight frames (sim_const.fit_g_drag). Does its behaviour track the true constant, in and out of range?
  3. Probes: ridge from the residual stream (frames 16..62) to g and gamma at every site; input baselines = ridge on the
     raw last three frames (which contain the second difference) and on the last eight. OOD: does the reading
     extrapolate?
  4. Steering: (a) donor interchange of the g (or gamma) direction at every frame -> tau of the change in the
     behavioural constant against the true donor-minus-target change, vs 100 random directions; (b) additive dose:
     shift the probe's reading at every frame and measure the change in the behavioural constant.

    .venv/bin/python experiments/world-model-physics/exp_b.py
Writes results/exp_b_s<seed>.json and results/exp_b_rollout_demo.npz (figure data).
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch

import common as C
import probing as pr
import sim_const as sc
import worldmodel as wm

DT = sc.DT
T0 = 16  # first frame used for constants (the model needs some context to infer them)
N_RANDOM = 100
SPLITS = ("test", "ood_g", "ood_drag")
TRAIN_MAX = {"g": sc.G_TRAIN[1], "drag": sc.DRAG_TRAIN[1]}


def free_mask(data):
    """[E, 62, 3] for t = 1..62: ball b had no contact in (t-1, t+1]."""
    h = data["ball_hit_next"]  # [E, 63, 3], contact in (t, t+1]
    return ~h[:, :-1] & ~h[:, 1:]


@torch.no_grad()
def full_pred(model, pos, device, hook=None, layer=None, batch=500):
    """Next-frame predictions for frames 0..62 -> [E, 63, 3, 2], optionally with a forward hook at block `layer`.
    `hook` is a factory: hook(start, stop) returns the hook for episodes start..stop of this batch."""
    out = []
    for s in range(0, len(pos), batch):
        h = None
        if hook is not None:
            h = model.layers[layer].register_forward_hook(hook(s, min(s + batch, len(pos))))
        try:
            p, _ = model(wm.to_tensor(pos[s : s + batch, :-1], device))
        finally:
            if h is not None:
                h.remove()
        out.append(p.float().cpu().numpy())
    p = np.concatenate(out).astype(np.float64)
    return p.reshape(len(pos), 63, 3, 2)


def behavioural(pred, data, t0=T0):
    """Episode-level (g, gamma) implied by predictions pred [E, 63, 3, 2] (pred[:, t] = guess for frame t+1)."""
    P, V = data["positions"], data["velocities"]
    acc = sc.implied_accel(P[:, 0:62], P[:, 1:63], pred[:, 1:63])  # t = 1..62
    mask = free_mask(data)
    mask[:, : t0 - 1] = False
    E = len(P)
    return sc.fit_g_drag(acc.reshape(E, -1, 2), V[:, 1:63].reshape(E, -1, 2), mask.reshape(E, -1))


def errors(model, data, device):
    P = data["positions"]
    pred = full_pred(model, P, device)
    true = P[:, 1:]
    cv = np.concatenate([P[:, :1], 2 * P[:, 1:-1] - P[:, :-2]], 1)
    ca = np.concatenate([P[:, :2], 3 * P[:, 2:-1] - 3 * P[:, 1:-2] + P[:, :-3]], 1)
    coll = data["collides_next"][:, :, 0]  # [E, 63]
    out = {}
    for name, p in (("model", pred), ("constant_velocity", cv), ("constant_acceleration", ca)):
        e = (((p - true) / DT) ** 2).mean(axis=(2, 3))[:, 2:]  # t = 2..62
        c = coll[:, 2:]
        out[name] = {"all": float(e.mean()), "free": float(e[~c].mean()), "collision": float(e[c].mean())}
    roll = wm.rollout(model, P, T0, device)
    err = np.linalg.norm(roll - P, axis=-1).mean(-1)  # [E, 64] mean ball distance
    out["rollout_mean_error"] = {f"h{h}": float(err[:, T0 - 1 + h].mean()) for h in (8, 24, 47)}
    return out, pred


def acts_rows(model, data, eps, device, t0=T0):
    a = pr.capture(model, data["positions"][eps], device)[:, :, t0:63]  # [5, E, 63 - t0, d]
    return a.reshape(a.shape[0], -1, a.shape[-1])


def raw_rows(data, eps, t0=T0, k=3):
    """The raw positions of frames t-k+1..t for every row (6 k numbers)."""
    P = data["positions"][eps].reshape(len(eps), 64, -1)
    return np.concatenate([P[:, t0 - i : 63 - i] for i in range(k)], -1).reshape(-1, 6 * k)


def direction(probe):
    """A scalar ridge probe's weight vector in raw activation units (reading = w . h + b)."""
    coef = probe.ridge.coef_.reshape(-1, len(probe.x_mean))[0]
    return coef / probe.x_std * probe.y_std[0]


def vel_matrix(probe):
    return (probe.ridge.coef_.reshape(6, -1) / probe.x_std[None]) * probe.y_std[:, None]


@torch.no_grad()
def full_resid(model, layer, pos, device, batch=500):
    out = []
    for s in range(0, len(pos), batch):
        store = {}
        h = model.layers[layer].register_forward_hook(lambda m, i, o, st=store: st.__setitem__("h", o.detach()))
        try:
            model(wm.to_tensor(pos[s : s + batch, :-1], device))
        finally:
            h.remove()
        out.append(store["h"])
    return torch.cat(out)


def swap(u, donor_h, device):
    """Hook factory: replace the component along u with the donor's, at every frame (Part 18 KTD6)."""
    u_t = torch.as_tensor(u, dtype=torch.float32, device=device)

    def make(s, e):
        dproj = donor_h[s:e] @ u_t

        def hook(m, i, o):
            return o + (dproj - o @ u_t)[..., None] * u_t

        return hook

    return make


def add(delta, device):
    """Hook factory: add delta at every frame."""
    d_t = torch.as_tensor(delta, dtype=torch.float32, device=device)

    def make(s, e):
        def hook(m, i, o):
            return o + d_t

        return hook

    return make


def _band(x):
    x = np.asarray(x)[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _bins(gt, gb, dt_, db):
    out = {"g": [], "drag": []}
    for name, t, b, edges in (("g", gt, gb, np.linspace(0, 3.5, 15)), ("drag", dt_, db, np.linspace(0, 1.0, 11))):
        for lo, hi in itertools.pairwise(edges):
            m = (t >= lo) & (t < hi)
            if m.sum() >= 10:
                out[name].append(
                    {
                        "lo": round(float(lo), 3),
                        "hi": round(float(hi), 3),
                        "n": int(m.sum()),
                        "true": float(t[m].mean()),
                        "model": float(np.median(b[m])),
                        "q25": float(np.percentile(b[m], 25)),
                        "q75": float(np.percentile(b[m], 75)),
                    }
                )
    return out


def run(seed, device, n_random=N_RANDOM, demo=False):
    t_start = time.monotonic()
    model = C.load_model("const", seed, "pre", device)
    rng = np.random.default_rng([C.SEEDS["steer"], seed])
    data = {s: C.const_split(s) for s in ("train", *SPLITS)}
    res = {"seed": seed, "splits": {}}

    # 1-2. prediction error and behavioural constants
    for s in SPLITS:
        err, pred = errors(model, data[s], device)
        gb, db = behavioural(pred, data[s])
        gt, dt_ = data[s]["g"], data[s]["drag"]
        ok = np.isfinite(gb) & np.isfinite(db)
        res["splits"][s] = {
            "errors": err,
            "behavioural": {
                "episodes": int(ok.sum()),
                "g_corr": _corr(gb, gt),
                "drag_corr": _corr(db, dt_),
                "g_mean_true": float(gt[ok].mean()),
                "g_mean_model": float(gb[ok].mean()),
                "drag_mean_true": float(dt_[ok].mean()),
                "drag_mean_model": float(db[ok].mean()),
                "g_slope": float(np.polyfit(gt[ok], gb[ok], 1)[0]),
                "drag_slope": float(np.polyfit(dt_[ok], db[ok], 1)[0]),
                "g_mae": float(np.abs(gb[ok] - gt[ok]).mean()),
                "drag_mae": float(np.abs(db[ok] - dt_[ok]).mean()),
            },
            "behavioural_bins": _bins(gt[ok], gb[ok], dt_[ok], db[ok]),
        }
        b = res["splits"][s]["behavioural"]
        print(
            f"  {s}: one-step error model {err['model']['all']:.2e} (const. velocity "
            f"{err['constant_velocity']['all']:.2e}, const. accel. {err['constant_acceleration']['all']:.2e}); "
            f"behavioural g corr {b['g_corr']:.3f}, mean {b['g_mean_model']:.2f} vs true {b['g_mean_true']:.2f};"
            f" drag corr {b['drag_corr']:.3f}, mean {b['drag_mean_model']:.3f} vs {b['drag_mean_true']:.3f}",
            flush=True,
        )

    # 3. probes, fitted on 1,000 training episodes
    tr = np.sort(rng.choice(len(data["train"]["positions"]), 1000, replace=False))
    A_tr = acts_rows(model, data["train"], tr, device)
    nfr = 63 - T0
    grp = np.repeat(tr, nfr)
    y_tr = {k: np.repeat(data["train"][k][tr], nfr) for k in ("g", "drag")}
    probes = {k: [pr.fit_ridge(A_tr[s], y_tr[k], grp) for s in range(A_tr.shape[0])] for k in ("g", "drag")}
    vel_tr = data["train"]["velocities"][tr][:, T0:63].reshape(-1, 6)
    vprobes = [pr.fit_ridge(A_tr[s], vel_tr, grp) for s in range(A_tr.shape[0])]
    base3 = {k: pr.fit_ridge(raw_rows(data["train"], tr), y_tr[k], grp) for k in ("g", "drag")}
    base8 = {k: pr.fit_ridge(raw_rows(data["train"], tr, k=8), y_tr[k], grp) for k in ("g", "drag")}
    res["probe"] = {
        "cv_r2": {k: [round(p.cv_r2, 4) for p in probes[k]] for k in probes},
        "best_site": {k: int(np.argmax([p.cv_r2 for p in probes[k]][1:])) + 1 for k in probes},
    }
    for s in SPLITS:
        d = data[s]
        te = np.sort(rng.choice(len(d["positions"]), min(500, len(d["positions"])), replace=False))
        A = acts_rows(model, d, te, device)
        R3, R8 = raw_rows(d, te), raw_rows(d, te, k=8)
        block = {}
        for k in ("g", "drag"):
            y = np.repeat(d[k][te], nfr)
            preds_site = [probes[k][si].predict(A[si])[:, 0] for si in range(A.shape[0])]
            best = res["probe"]["best_site"][k]
            ep_mean = preds_site[best].reshape(len(te), nfr).mean(1)
            yy, pp = y.reshape(len(te), nfr), preds_site[best].reshape(len(te), nfr)
            block[k] = {
                "r2_by_site": [round(pr.r2(y, p), 4) for p in preds_site],
                "baseline_r2_last3": round(pr.r2(y, base3[k].predict(R3)[:, 0]), 4),
                "baseline_r2_last8": round(pr.r2(y, base8[k].predict(R8)[:, 0]), 4),
                "mean_true": float(d[k][te].mean()),
                "mean_probe": float(ep_mean.mean()),
                "slope_probe_vs_true": float(np.polyfit(d[k][te], ep_mean, 1)[0]),
                "corr_probe_vs_true": float(np.corrcoef(d[k][te], ep_mean)[0, 1]),
                "frac_above_train_max": float((ep_mean > TRAIN_MAX[k]).mean()),
                "frames": list(range(T0, 63))[::4],
                "r2_by_frame_best_site": [round(pr.r2(yy[:, j], pp[:, j]), 4) for j in range(0, nfr, 4)],
            }
        res["splits"][s]["probe"] = block
        print(
            f"  {s} probe: g R2 by site {block['g']['r2_by_site']} (raw last-3 {block['g']['baseline_r2_last3']},"
            f" last-8 {block['g']['baseline_r2_last8']}); probe mean {block['g']['mean_probe']:.2f} vs true "
            f"{block['g']['mean_true']:.2f}; drag R2 {block['drag']['r2_by_site']} mean "
            f"{block['drag']['mean_probe']:.3f} vs {block['drag']['mean_true']:.3f}",
            flush=True,
        )

    # 4. steering on the in-range test split: 1,000 target-donor pairs
    d = data["test"]
    n_pairs = 1000
    tgt = rng.integers(0, len(d["positions"]), n_pairs)
    don = (tgt + rng.integers(1, len(d["positions"]), n_pairs)) % len(d["positions"])
    Pt, Pd = d["positions"][tgt], d["positions"][don]
    tsub = {kk: d[kk][tgt] for kk in ("positions", "velocities", "ball_hit_next")}
    true_change = {k: d[k][don] - d[k][tgt] for k in ("g", "drag")}
    g0, dr0 = behavioural(full_pred(model, Pt, device), tsub)
    res["steer"] = {"pairs": n_pairs, "random_directions": n_random}

    def change(hook, layer, k):
        gp, dp = behavioural(full_pred(model, Pt, device, hook, layer), tsub)
        return (gp - g0, dp - dr0) if k == "g" else (dp - dr0, gp - g0)

    for layer in range(len(model.layers)):
        Wv = vel_matrix(vprobes[layer + 1])
        donor_h = full_resid(model, layer, Pd, device)
        row = {"block": layer}
        for k in ("g", "drag"):
            w = direction(probes[k][layer + 1])
            u = pr.orthogonalise(w / np.linalg.norm(w), Wv)  # velocity directions removed, as for the nulls
            own, cross = change(swap(u, donor_h, device), layer, k)
            nulls = [
                _corr(change(swap(pr.orthogonalise(r, Wv), donor_h, device), layer, k)[0], true_change[k])
                for r in pr.random_directions(n_random, len(u), rng)
            ]
            step = float(np.dot(w, u))  # probe reading change per unit step along u
            doses = [-1.0, -0.5, 0.5, 1.0] if k == "g" else [-0.25, -0.1, 0.1, 0.25]
            dose_out = [
                {"probe_shift": a, "behaviour_shift": float(np.nanmean(change(add(a / step * u, device), layer, k)[0]))}
                for a in doses
            ]
            top = doses[-1]
            null_dose = [
                float(np.nanmean(change(add(abs(top / step) * pr.orthogonalise(r, Wv), device), layer, k)[0]))
                for r in pr.random_directions(min(n_random, 50), len(u), rng)
            ]
            slope = float(np.polyfit(doses, [x["behaviour_shift"] for x in dose_out], 1)[0])
            row[k] = {
                "tau": round(_corr(own, true_change[k]), 4),
                "tau_cross": round(_corr(cross, true_change[k]), 4),
                "null_band": _band(nulls),
                "dose": dose_out,
                "dose_slope": round(slope, 4),
                "null_dose_band_at_max": _band(null_dose),
                "edit_norm_at_max": round(abs(top / step), 4),
            }
        res["steer"][f"block{layer}"] = row
        print(
            f"  steer block {layer}: g tau {row['g']['tau']:+.3f} null {row['g']['null_band']} dose slope "
            f"{row['g']['dose_slope']:+.3f} (random at +1: {row['g']['null_dose_band_at_max']}); drag tau "
            f"{row['drag']['tau']:+.3f} null {row['drag']['null_band']} slope {row['drag']['dose_slope']:+.3f}",
            flush=True,
        )
    res["seconds"] = round(time.monotonic() - t_start, 1)

    if demo:  # rollouts for the figure: steer g by -1 / +1 (probe units) during a rollout
        layer = res["probe"]["best_site"]["g"] - 1
        Wv = vel_matrix(vprobes[layer + 1])
        w = direction(probes["g"][layer + 1])
        u = pr.orthogonalise(w / np.linalg.norm(w), Wv)
        step = float(np.dot(w, u))
        idx = np.argsort(np.abs(d["g"] - 1.0) + 10 * (d["drag"] > 0.1))[:8]
        out = {
            "truth": d["positions"][idx],
            "g": d["g"][idx],
            "drag": d["drag"][idx],
            "layer": layer,
            "clean": wm.rollout(model, d["positions"][idx], T0, device),
        }
        for amt in (-1.0, 1.0):
            h = model.layers[layer].register_forward_hook(add(amt / step * u, device)(0, len(idx)))
            try:
                out[f"steer_{amt:+.0f}"] = wm.rollout(model, d["positions"][idx], T0, device)
            finally:
                h.remove()
        C.save_npz(C.RESULTS / "exp_b_rollout_demo.npz", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS["model_const"]))
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = C.pick_device()
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        r = run(seed, device, args.n_random, demo=(seed == args.seeds[0] and not args.tag))
        C.save_json(f"exp_b_s{seed}{args.tag}.json", r)
