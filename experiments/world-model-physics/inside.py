"""
Looking inside one base-world model: presence (probes) and use (interventions). Experiments A and C both call this.

    probes_for(model, train, test, device)     velocity, energy and collision-imminent probes at every site,
                                               with test R^2 / AUROC and input baselines            (presence)
    velocity_edit(...)                         add a velocity probe direction at frame t: does the predicted next
                                               position move by Delta v * DT, in the probe's direction?   (use)
    energy_edit(...)                           add the energy direction (velocity directions removed): does the
                                               predicted step's kinetic energy move by Delta E?           (use)
    collision_edit(...)                        subtract the collision-imminent direction on frames with a bounce
                                               next: does the predicted bounce disappear?                 (use)
    interchange_taus(...)                      the base project's donor swap (tau) for velocity and energy

Every use measure has a random-direction null of the same norm (and the same constraint: random directions for the
energy edit have the velocity directions removed too). Site s = 0 is the embedding, s = 1..4 the outputs of blocks
0..3 (the base project's convention); edits happen at block outputs, layer = s - 1.

Gain: the edit's expected effect on the predicted next frame, E (6 numbers), is what the probe says should happen
(velocity: DT times the change in every velocity probe's reading). The observed effect O is the change in the
model's prediction. gain = (O . E) / (E . E): 1 = the prediction moved exactly as physics says, 0 = no movement in
that direction. The null applies the same formula to random edits of the same norm.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import probing as pr
import worldmodel as wm

DT = wm.DT
N_RANDOM = 100


# --- presence ----------------------------------------------------------------------------------------------------------


def _rows_and_labels(model, data, episodes, device):
    acts = pr.rows(pr.capture(model, data["positions"][episodes], device))  # [5, n, d]
    return acts, pr.frame_labels(data, episodes)


class Logit:
    """Logistic probe on standardised X; raw_direction() gives (unit vector, logit change per unit step)."""

    def __init__(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit((X - self.mu) / self.sd, y)

    def logit(self, X):
        return self.clf.decision_function((X - self.mu) / self.sd)

    def raw_direction(self):
        w = self.clf.coef_[0] / self.sd
        n = float(np.linalg.norm(w))
        return (w / n).astype(np.float32), n


def probes_for(model, train, test, device, n_train=1000, n_test=500, seed=0):
    """Fit probes on n_train training episodes, score on n_test test episodes. Returns (probes, presence)."""
    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(len(train["positions"]), n_train, replace=False))
    te = np.sort(rng.choice(len(test["positions"]), n_test, replace=False))
    A_tr, L_tr = _rows_and_labels(model, train, tr, device)
    A_te, L_te = _rows_and_labels(model, test, te, device)
    coll_tr, coll_te = L_tr["collides_next"].any(1), L_te["collides_next"].any(1)
    probes = {"velocity": [], "energy": [], "collision": []}
    presence = {"sites": list(pr.SITES), "velocity_r2": [], "energy_r2": [], "collision_auroc": []}
    for s in range(A_tr.shape[0]):
        pv = pr.fit_ridge(A_tr[s], L_tr["velocity"], L_tr["episode"])
        pe = pr.fit_ridge(A_tr[s], L_tr["energy"], L_tr["episode"])
        pc = Logit(A_tr[s], coll_tr)
        probes["velocity"].append(pv)
        probes["energy"].append(pe)
        probes["collision"].append(pc)
        presence["velocity_r2"].append(pr.r2(L_te["velocity"], pv.predict(A_te[s])))
        presence["energy_r2"].append(pr.r2(L_te["energy"], pe.predict(A_te[s])))
        presence["collision_auroc"].append(float(roc_auc_score(coll_te, pc.logit(A_te[s]))))
    # input baselines: the same probe family on the raw frames t-1 and t (12 numbers)
    bv = pr.fit_ridge(L_tr["last_two"], L_tr["velocity"], L_tr["episode"])
    be = pr.fit_ridge(L_tr["last_two"], L_tr["energy"], L_tr["episode"])
    bc = Logit(L_tr["last_two"], coll_tr)
    presence["baseline"] = {
        "velocity_r2": pr.r2(L_te["velocity"], bv.predict(L_te["last_two"])),
        "energy_r2": pr.r2(L_te["energy"], be.predict(L_te["last_two"])),
        "collision_auroc": float(roc_auc_score(coll_te, bc.logit(L_te["last_two"]))),
    }
    presence["collision_base_rate_test"] = float(coll_te.mean())
    presence["rows"] = {"train": int(A_tr.shape[1]), "test": int(A_te.shape[1])}
    for k in ("velocity_r2", "energy_r2", "collision_auroc"):
        presence[k] = [round(float(v), 4) for v in presence[k]]
    return probes, presence


def velocity_matrix(probe):
    """Raw velocity read-out matrix W [6, d]: reading = W h + b (target units)."""
    coef = probe.ridge.coef_.reshape(-1, len(probe.x_mean))
    return (coef / probe.x_std[None, :]) * probe.y_std[:, None]


def energy_vector(probe):
    coef = probe.ridge.coef_.reshape(-1, len(probe.x_mean))[0]
    return coef / probe.x_std * probe.y_std[0]


# --- frame samples --------------------------------------------------------------------------------------------------------


def sample_frames(data, n, rng, kind, t_range=(8, 62)):
    """n (episode, frame) pairs of one kind: 'free' (no collision in (t-1, t+1]) or 'collision' (one in (t, t+1])."""
    kinds = wm.frame_kinds(data["collides_next"])  # [E, 62] over t = 1..62
    mask = kinds[kind].copy()
    mask[:, : t_range[0] - 1] = False
    e, f = np.nonzero(mask)
    pick = rng.choice(len(e), size=min(n, len(e)), replace=False)
    return e[pick], f[pick] + 1


def _pred_at(model, layer, pos, frames, delta, device, batch=500):
    """The next-frame prediction at frame t of each episode, with delta[i] added at block `layer`, frame t only
    (delta None = no edit). Returns [B, 6]."""
    out = []
    for s in range(0, len(pos), batch):
        p, f = pos[s : s + batch], frames[s : s + batch]
        if delta is None:
            with torch.no_grad():
                pred, _ = model(wm.to_tensor(p[:, :-1], device))
            idx = torch.as_tensor(f, device=device)
            out.append(pred[torch.arange(len(idx), device=device), idx].float().cpu().numpy())
        else:
            out.append(pr.add_at_frame(model, layer, p, f, delta[s : s + batch], device))
    return np.concatenate(out).astype(np.float64)


def _gain(O, E):
    return float(np.sum(O * E) / np.sum(E * E))


def _band(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# --- use: velocity -----------------------------------------------------------------------------------------------------


def velocity_edit(model, probes, data, eps, frames, device, rng, dv=0.25, n_random=N_RANDOM):
    """At each block: for every coordinate c and sign, add dv / |w_c|^2 * w_c (changes probe c by exactly dv).
    Expected effect on the prediction: DT * (W delta) (all six probes' changes). Returns gains and nulls per block."""
    pos = data["positions"][eps]
    clean = _pred_at(model, 0, pos, frames, None, device)
    n = len(eps)
    out = []
    for layer in range(len(model.layers)):
        W = velocity_matrix(probes["velocity"][layer + 1])  # [6, d]
        gains, cosines, nulls, leak = [], [], [], []
        for c in range(6):
            w = W[c]
            for sign in (1, -1):
                delta = sign * dv * w / np.dot(w, w)
                E = DT * (W @ delta)  # expected change of each coordinate (target c gets DT * dv)
                O = _pred_at(model, layer, pos, frames, np.tile(delta, (n, 1)), device) - clean
                gains.append(_gain(O, np.tile(E, (n, 1))))
                om = O.mean(0)
                cosines.append(float(om @ E / (np.linalg.norm(om) * np.linalg.norm(E) + 1e-12)))
                others = [k for k in range(6) if k != c]
                leak.append(float(np.abs(O[:, others]).mean() / (np.abs(O[:, c]).mean() + 1e-12)))
        norm = float(np.mean([dv / np.linalg.norm(W[c]) for c in range(6)]))
        for u in pr.random_directions(n_random, W.shape[1], rng):
            O = _pred_at(model, layer, pos, frames, np.tile(u * norm, (n, 1)), device) - clean
            c = int(rng.integers(6))  # score each random edit against one coordinate's intended effect
            sign = 1 if rng.random() < 0.5 else -1
            E = DT * (W @ (sign * dv * W[c] / np.dot(W[c], W[c])))
            nulls.append(_gain(O, np.tile(E, (n, 1))))
        out.append(
            {
                "block": layer,
                "gain": round(float(np.mean(gains)), 4),
                "gain_min": round(float(np.min(gains)), 4),
                "gain_max": round(float(np.max(gains)), 4),
                "cosine": round(float(np.mean(cosines)), 4),
                "leak_ratio": round(float(np.mean(leak)), 4),
                "null_band": _band(nulls),
                "null_abs_max": round(float(np.max(np.abs(nulls))), 4),
                "edit_norm": round(norm, 4),
            }
        )
    return {"dv": dv, "frames": n, "random_directions": n_random, "blocks": out}


# --- use: energy -------------------------------------------------------------------------------------------------------


def _implied_ke(pred, cur):
    v = (pred - cur) / DT
    return 0.5 * (v**2).sum(1)


def energy_edit(model, probes, data, eps, frames, device, rng, de=0.5, n_random=N_RANDOM):
    """At each block: add the energy probe direction with the six velocity probe directions removed, sized so the
    energy probe's reading changes by +-de. Expected: the kinetic energy implied by the predicted step changes by de,
    with the step growing along the current velocity. Null: random directions, velocity directions removed, same norm."""
    pos = data["positions"][eps]
    n = len(eps)
    cur = pos[np.arange(n), frames].reshape(n, -1)
    vtrue = data["velocities"][eps, frames].reshape(n, -1)
    clean = _pred_at(model, 0, pos, frames, None, device)
    ke0 = _implied_ke(clean, cur)
    out = []
    for layer in range(len(model.layers)):
        W = velocity_matrix(probes["velocity"][layer + 1])
        we = energy_vector(probes["energy"][layer + 1])
        u = pr.orthogonalise(we / np.linalg.norm(we), W)
        step = float(np.dot(we, u))  # energy-probe change per unit step along u
        gains, cos = [], []
        for sign in (1, -1):
            pred = _pred_at(model, layer, pos, frames, np.tile(sign * de / step * u, (n, 1)), device)
            gains.append(float(np.mean(_implied_ke(pred, cur) - ke0) / (sign * de)))
            dvel = (pred - clean) / DT
            cos.append(
                float(
                    np.mean(
                        np.sum(dvel * vtrue, 1)
                        * sign
                        / (np.linalg.norm(dvel, axis=1) * np.linalg.norm(vtrue, axis=1) + 1e-12)
                    )
                )
            )
        norm = de / abs(step)
        nulls = []
        for r in pr.random_directions(n_random, len(u), rng):
            r = pr.orthogonalise(r, W)
            sign = 1 if rng.random() < 0.5 else -1
            pred = _pred_at(model, layer, pos, frames, np.tile(sign * norm * r, (n, 1)), device)
            nulls.append(float(np.mean(_implied_ke(pred, cur) - ke0) / (sign * de)))
        out.append(
            {
                "block": layer,
                "gain": round(float(np.mean(gains)), 4),
                "gain_up": round(gains[0], 4),
                "gain_down": round(gains[1], 4),
                "cos_with_velocity": round(float(np.mean(cos)), 4),
                "null_band": _band(nulls),
                "null_abs_max": round(float(np.max(np.abs(nulls))), 4),
                "edit_norm": round(float(norm), 4),
            }
        )
    return {"de": de, "frames": n, "random_directions": n_random, "blocks": out}


# --- use: collision-imminent ------------------------------------------------------------------------------------------


def _bounce_fraction(pred, data, eps, frames):
    """How much of the true bounce the prediction contains: projection of (pred - constant velocity) onto
    (truth - constant velocity), over |truth - constant velocity|^2, pooled. 1 = the full bounce, 0 = a straight line."""
    P = data["positions"][eps]
    i = np.arange(len(eps))
    cur = P[i, frames].reshape(len(eps), -1)
    prev = P[i, frames - 1].reshape(len(eps), -1)
    nxt = P[i, frames + 1].reshape(len(eps), -1)
    cv = 2 * cur - prev
    a, b = pred - cv, nxt - cv
    return float(np.sum(a * b) / np.sum(b * b))


def collision_edit(model, probes, data, eps_c, frames_c, eps_f, frames_f, device, rng, n_random=N_RANDOM):
    """At each block: subtract the collision-imminent direction on frames with a bounce next, sized to move the probe's
    logit from its median value on those frames to minus that value (z -> -z). Use = drop in bounce fraction.
    And add it on free frames by the same amount: how far (velocity units) does the prediction leave straight flight?"""
    pos_c, pos_f = data["positions"][eps_c], data["positions"][eps_f]
    clean_c = _pred_at(model, 0, pos_c, frames_c, None, device)
    clean_f = _pred_at(model, 0, pos_f, frames_f, None, device)
    b_clean = _bounce_fraction(clean_c, data, eps_c, frames_c)
    nc, nf = len(eps_c), len(eps_f)
    out = []
    for layer in range(len(model.layers)):
        probe = probes["collision"][layer + 1]
        u, wn = probe.raw_direction()
        acts = pr.residual_rows(model, layer, pos_c, device)  # rows for frames 1..62, episode-major
        z = float(np.median(probe.logit(acts[np.arange(nc) * 62 + (frames_c - 1)])))
        norm = 2 * abs(z) / wn
        b_edit = _bounce_fraction(
            _pred_at(model, layer, pos_c, frames_c, np.tile(-norm * u, (nc, 1)), device), data, eps_c, frames_c
        )
        pred_f = _pred_at(model, layer, pos_f, frames_f, np.tile(norm * u, (nf, 1)), device)
        dev_f = float(np.linalg.norm(pred_f - clean_f, axis=1).mean() / DT)
        null_drop, null_dev = [], []
        for r in pr.random_directions(n_random, len(u), rng):
            p = _pred_at(model, layer, pos_c, frames_c, np.tile(-norm * r, (nc, 1)), device)
            null_drop.append(b_clean - _bounce_fraction(p, data, eps_c, frames_c))
            p = _pred_at(model, layer, pos_f, frames_f, np.tile(norm * r, (nf, 1)), device)
            null_dev.append(float(np.linalg.norm(p - clean_f, axis=1).mean() / DT))
        out.append(
            {
                "block": layer,
                "bounce_clean": round(b_clean, 4),
                "bounce_edited": round(b_edit, 4),
                "drop": round(b_clean - b_edit, 4),
                "null_drop_band": _band(null_drop),
                "null_drop_max": round(float(np.max(null_drop)), 4),
                "free_deviation": round(dev_f, 4),
                "null_deviation_band": _band(null_dev),
                "median_logit": round(z, 3),
                "edit_norm": round(float(norm), 4),
            }
        )
    return {"collision_frames": nc, "free_frames": nf, "random_directions": n_random, "blocks": out}


# --- use: interchange, as in the base project ------------------------------------------------------------------------


def interchange_taus(model, probes, data, device, rng, n_pairs=2000, n_random=N_RANDOM):
    """tau for velocity (ball 0, x) and energy (velocity directions removed) at every block, with random nulls."""
    pairs = pr.make_pairs(len(data["positions"]), n_pairs, rng)
    runner = pr.PairRunner(model, data, pairs, device)
    out = []
    for layer in range(len(model.layers)):
        W = velocity_matrix(probes["velocity"][layer + 1])
        uv = (W[0] / np.linalg.norm(W[0])).astype(np.float32)
        we = energy_vector(probes["energy"][layer + 1])
        ue = pr.orthogonalise(we / np.linalg.norm(we), W)
        tv = runner.taus(layer, uv)["velocity"]
        te = runner.taus(layer, ue)["energy"]
        nv, ne = [], []
        for r in pr.random_directions(n_random, len(uv), rng):
            nv.append(runner.taus(layer, r)["velocity"])
            ne.append(runner.taus(layer, pr.orthogonalise(r, W))["energy"])
        out.append(
            {
                "block": layer,
                "tau_velocity": round(tv, 4),
                "velocity_band": _band(nv),
                "tau_energy_perp_velocity": round(te, 4),
                "energy_band": _band(ne),
            }
        )
    return {"pairs": n_pairs, "random_directions": n_random, "blocks": out}


def verdict(value, band):
    return "used" if value > band[1] else ("reversed" if value < band[0] else "inside the null")
