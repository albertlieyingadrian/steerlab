"""
Experiment A, exploratory follow-ups (not pre-registered; added after reading the registered results).

The registered edits change one frame (t) only. Energy is constant through an episode, and the model can attend to
earlier frames, so a frame-t edit may be outvoted by the unedited earlier frames; the donor interchange (which patches
every frame) found energy used where the frame-t edit did not. Two follow-ups:

  1. all-frame edits: add the same velocity / energy edit at every frame 0..62, read the prediction at frame t.
  2. a dose curve for the collision edit: 1x, 4x and 16x the registered size, each against random directions of the
     same norm, with the edit's norm relative to the residual stream's typical norm at that block.

    .venv/bin/python experiments/world-model-physics/exp_a_extra.py
Writes results/exp_a_extra.json.
"""

from __future__ import annotations

import numpy as np
import torch

import common as C
import inside as I
import probing as pr
import worldmodel as wm

DT = wm.DT
N_RANDOM = 50


@torch.no_grad()
def pred_all(model, layer, pos, frames, delta, device, batch=500):
    """Prediction at frame t with `delta` (one vector) added at block `layer` at EVERY frame. [B, 6]"""
    d_t = None if delta is None else torch.as_tensor(delta, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(pos), batch):
        h = None
        if d_t is not None:
            h = model.layers[layer].register_forward_hook(lambda m, i, o: o + d_t)
        try:
            p, _ = model(wm.to_tensor(pos[s : s + batch, :-1], device))
        finally:
            if h is not None:
                h.remove()
        idx = torch.as_tensor(frames[s : s + batch], device=device)
        out.append(p[torch.arange(len(idx), device=device), idx].float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def all_frame_edits(model, probes, data, eps, frames, device, rng):
    pos = data["positions"][eps]
    n = len(eps)
    cur = pos[np.arange(n), frames].reshape(n, -1)
    clean = pred_all(model, 0, pos, frames, None, device)
    ke0 = I._implied_ke(clean, cur)
    out = []
    for layer in range(len(model.layers)):
        W = I.velocity_matrix(probes["velocity"][layer + 1])
        gains = []
        for c in range(6):
            for sign in (1, -1):
                delta = sign * 0.25 * W[c] / np.dot(W[c], W[c])
                E = DT * (W @ delta)
                O = pred_all(model, layer, pos, frames, delta, device) - clean
                gains.append(I._gain(O, np.tile(E, (n, 1))))
        norm = float(np.mean([0.25 / np.linalg.norm(W[c]) for c in range(6)]))
        vnull = []
        for u in pr.random_directions(N_RANDOM, W.shape[1], rng):
            c, sign = int(rng.integers(6)), (1 if rng.random() < 0.5 else -1)
            E = DT * (W @ (sign * 0.25 * W[c] / np.dot(W[c], W[c])))
            O = pred_all(model, layer, pos, frames, u * norm, device) - clean
            vnull.append(I._gain(O, np.tile(E, (n, 1))))
        we = I.energy_vector(probes["energy"][layer + 1])
        u = pr.orthogonalise(we / np.linalg.norm(we), W)
        step = float(np.dot(we, u))
        egains = [
            float(
                np.mean(I._implied_ke(pred_all(model, layer, pos, frames, s * 0.5 / step * u, device), cur) - ke0)
                / (s * 0.5)
            )
            for s in (1, -1)
        ]
        enull = []
        for r in pr.random_directions(N_RANDOM, len(u), rng):
            s = 1 if rng.random() < 0.5 else -1
            p = pred_all(model, layer, pos, frames, s * 0.5 / abs(step) * pr.orthogonalise(r, W), device)
            enull.append(float(np.mean(I._implied_ke(p, cur) - ke0) / (s * 0.5)))
        out.append(
            {
                "block": layer,
                "velocity_gain": round(float(np.mean(gains)), 4),
                "velocity_null_band": I._band(vnull),
                "energy_gain": round(float(np.mean(egains)), 4),
                "energy_null_band": I._band(enull),
            }
        )
    return out


def collision_dose(model, probes, data, eps, frames, device, rng):
    pos = data["positions"][eps]
    n = len(eps)
    clean = I._pred_at(model, 0, pos, frames, None, device)
    b0 = I._bounce_fraction(clean, data, eps, frames)
    out = []
    for layer in range(len(model.layers)):
        probe = probes["collision"][layer + 1]
        u, wn = probe.raw_direction()
        acts = pr.residual_rows(model, layer, pos, device)
        rows = acts[np.arange(n) * 62 + (frames - 1)]
        z = float(np.median(probe.logit(rows)))
        resid_norm = float(np.linalg.norm(rows, axis=1).mean())
        base = 2 * abs(z) / wn
        doses = []
        for mult in (1, 4, 16):
            norm = base * mult
            drop = b0 - I._bounce_fraction(
                I._pred_at(model, layer, pos, frames, np.tile(-norm * u, (n, 1)), device), data, eps, frames
            )
            null = [
                b0
                - I._bounce_fraction(
                    I._pred_at(model, layer, pos, frames, np.tile(-norm * r, (n, 1)), device), data, eps, frames
                )
                for r in pr.random_directions(30, len(u), rng)
            ]
            doses.append(
                {
                    "multiple": mult,
                    "edit_norm": round(norm, 4),
                    "edit_over_residual": round(norm / resid_norm, 4),
                    "drop": round(drop, 4),
                    "null_band": I._band(null),
                }
            )
        out.append({"block": layer, "bounce_clean": round(b0, 4), "doses": doses})
    return out


if __name__ == "__main__":
    device = C.pick_device()
    train, test = C.base_split("train"), C.base_split("test")
    res = {"note": "exploratory, not pre-registered", "random_directions": N_RANDOM, "seeds": {}}
    for seed in C.SEEDS["model_base"]:
        print(f"seed {seed}", flush=True)
        model = C.load_model("base", seed, "pre", device)
        rng = np.random.default_rng([C.SEEDS["steer"] + 100, seed])
        probes, _ = I.probes_for(model, train, test, device, seed=C.SEEDS["probe"])
        ef, ff = I.sample_frames(test, 500, rng, "free")
        ec, fc = I.sample_frames(test, 500, rng, "collision")
        r = {
            "all_frame_edits": all_frame_edits(model, probes, test, ef, ff, device, rng),
            "collision_dose": collision_dose(model, probes, test, ec, fc, device, rng),
        }
        for b in r["all_frame_edits"]:
            print(
                f"  all frames block {b['block']}: velocity gain {b['velocity_gain']:+.3f} {b['velocity_null_band']}"
                f"  energy gain {b['energy_gain']:+.3f} {b['energy_null_band']}",
                flush=True,
            )
        for b in r["collision_dose"]:
            print(
                f"  collision block {b['block']}: "
                + "; ".join(
                    f"x{d['multiple']} (|edit|/|h| {d['edit_over_residual']:.2f}) drop {d['drop']:+.3f} {d['null_band']}"
                    for d in b["doses"]
                ),
                flush=True,
            )
        res["seeds"][str(seed)] = r
        C.save_json("exp_a_extra.json", res)
