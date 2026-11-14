"""
Looking inside the Part 18 world model: capture activations, fit linear probes, and run interchange patching.

    capture(model, positions)          residual stream at every site, every frame: [5 sites, E, 63, d_model]
    frame_labels(data, episodes)       the answer key for each (episode, frame) row: position, velocity, energy,
                                       which balls just bounced, which collisions come next
    fit_ridge(X, Y, groups)            RidgeCV on standardised inputs and targets, alpha chosen by GroupKFold
    r2(y, y_hat)                       R² averaged over output columns
    raw_direction(probe, k)            output k's probe as a unit vector in raw activation space
    interchange(model, layer, u, ...)  overwrite the component along u with a donor's (KTD6) and finish the pass
    tau(dg, dphi)                      transfer correlation, the KTD6 metric

Sites: 0 is the input to block 0 (the frame's embedding plus the positional embedding); 1 to 4 are the outputs of
blocks 0 to 3. The blocks are pre-norm, so each block's output is the residual stream at that depth.

Rows: a "row" is one (episode, frame t) pair with t = 1..62. Frame 0 is left out everywhere, because one frame
says nothing about velocity, so no probe or readout could be fair there.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import worldmodel as wm

SITES = ("embed", "block 0", "block 1", "block 2", "block 3")
FRAMES = slice(1, 63)  # rows use frames t = 1..62
ALPHAS = np.logspace(-3, 3, 7)


def _register_hooks():
    from sandbagkit.probing.hooks import capture_layers, register_hooks

    return capture_layers, register_hooks


# --- 1. Capture ------------------------------------------------------------------------------------------------------


@torch.no_grad()
def capture(model, positions, device, batch=256) -> np.ndarray:
    """The residual stream at every site for whole episodes. positions [E, 64, 3, 2] -> float32 [5, E, 63, d]."""
    capture_layers, _ = _register_hooks()
    n_layers = len(model.layers)
    out = np.empty((n_layers + 1, len(positions), wm.CONTEXT, model.config["d_model"]), dtype=np.float32)
    embedded = {}
    handle = model.layers[0].register_forward_pre_hook(lambda m, args: embedded.__setitem__("h", args[0]))
    try:
        with capture_layers(model, list(range(n_layers))) as captured:
            for s in range(0, len(positions), batch):
                model(wm.to_tensor(positions[s : s + batch, :-1], device))
                out[0, s : s + batch] = embedded["h"].float().cpu().numpy()
                for i in range(n_layers):
                    out[i + 1, s : s + batch] = captured[i].float().cpu().numpy()
    finally:
        handle.remove()
    return out


def rows(acts) -> np.ndarray:
    """[sites, E, 63, d] -> [sites, E * 62, d], keeping frames 1..62 (episode-major: row = e * 62 + (t - 1))."""
    a = acts[:, :, FRAMES]
    return a.reshape(a.shape[0], -1, a.shape[-1])


def ball_just_hit(data) -> np.ndarray:
    """[E, 64, 3] bool: ball b's velocity at frame t differs from the step it took from t − 1 to t, because it hit
    something between t − 1 and t."""
    E, T = data["positions"].shape[:2]
    hit = np.zeros((E, T, data["positions"].shape[2]), dtype=bool)
    e, f = data["event_episode"], data["event_frame"] + 1
    hit[e, f, data["event_i"]] = True
    bb = data["event_j"] >= 0
    hit[e[bb], f[bb], data["event_j"][bb]] = True
    return hit


def frame_labels(data, episodes) -> dict:
    """The answer key for each row of `rows(capture(...))` on data[...][episodes]. Every array has E * 62 rows."""
    ep = np.asarray(episodes)
    pos = data["positions"][ep][:, FRAMES]  # [E, 62, 3, 2]
    vel = data["velocities"][ep][:, FRAMES]
    n = pos.shape[0] * pos.shape[1]
    hit = ball_just_hit(data)[ep][:, FRAMES]
    return {
        "position": pos.reshape(n, -1),
        "velocity": vel.reshape(n, -1),
        "energy": np.repeat(data["energy"][ep], pos.shape[1]),
        "just_hit_ball": hit.reshape(n, -1),
        "collides_next": data["collides_next"][ep][:, FRAMES].reshape(n, -1),
        "episode": np.repeat(ep, pos.shape[1]),
        "frame": np.tile(np.arange(1, 63), len(ep)),
        # raw inputs the baselines may use: frames t - 1 and t
        "last_two": np.concatenate([data["positions"][ep][:, 0:62].reshape(n, -1), pos.reshape(n, -1)], axis=1),
    }


# --- 2. Probes ---------------------------------------------------------------------------------------------------------


class Probe:
    """A ridge probe fitted on standardised X and Y. predict() works in the original units."""

    def __init__(self, ridge, x_mean, x_std, y_mean, y_std, cv_r2):
        self.ridge, self.x_mean, self.x_std, self.y_mean, self.y_std, self.cv_r2 = (
            ridge, x_mean, x_std, y_mean, y_std, cv_r2)

    @property
    def alpha(self):
        return float(self.ridge.alpha_)

    def predict(self, X):
        z = self.ridge.predict((X - self.x_mean) / self.x_std)
        return z.reshape(len(X), -1) * self.y_std + self.y_mean


def fit_ridge(X, Y, groups, *, folds=5, alphas=ALPHAS) -> Probe:
    """RidgeCV with alpha picked by GroupKFold over `groups` (episodes), so no episode is in two folds."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(len(X), -1)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = Y.mean(0), Y.std(0) + 1e-12
    Xs, Ys = (X - x_mean) / x_std, (Y - y_mean) / y_std
    splits = list(GroupKFold(n_splits=folds).split(Xs, Ys, groups))
    ridge = RidgeCV(alphas=alphas, cv=splits, scoring="r2").fit(Xs, Ys)
    return Probe(ridge, x_mean, x_std, y_mean, y_std, float(ridge.best_score_))


def r2(y, y_hat) -> float:
    """R² per output column, averaged (1 = perfect, 0 = no better than predicting the mean)."""
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    y_hat = np.asarray(y_hat, dtype=np.float64).reshape(len(y), -1)
    ss_res = ((y - y_hat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / ss_tot))


def raw_direction(probe: Probe, k: int = 0) -> tuple[np.ndarray, float]:
    """Output k's probe as (unit vector in raw activation space, the probe's change in target units per unit step
    along it). A probe reads y = w · x + b in raw units with w = coef / x_std · y_std."""
    coef = probe.ridge.coef_.reshape(-1, len(probe.x_mean))[k]
    w = coef / probe.x_std * probe.y_std[k]
    norm = float(np.linalg.norm(w))
    return (w / norm).astype(np.float32), norm


def orthogonalise(u, others) -> np.ndarray:
    """u with its component in the span of the rows of `others` removed, renormalised to unit length."""
    q, _ = np.linalg.qr(np.asarray(others, dtype=np.float64).T)
    v = u - q @ (q.T @ u)
    return (v / np.linalg.norm(v)).astype(np.float32)


# --- 3. Interchange (KTD6) ---------------------------------------------------------------------------------------------


@torch.no_grad()
def residual_at(model, layer, positions, device) -> torch.Tensor:
    """Block `layer`'s output for whole episodes, [E, 63, d], kept on `device`."""
    capture_layers, _ = _register_hooks()
    with capture_layers(model, [layer]) as captured:
        model(wm.to_tensor(positions[:, :-1], device))
        return captured[layer].clone()


def swap_hook(u_t, donor_proj, mask=None):
    """The KTD6 forward hook: replace the output's component along unit vector u_t with donor_proj [B, T] (only where
    mask [B, T] is 1, when given)."""

    def hook(module, inputs, output):
        shift = donor_proj - output @ u_t
        if mask is not None:
            shift = shift * mask
        return output + shift[..., None] * u_t

    return hook


@torch.no_grad()
def interchange(model, layer, u, target_pos, donor_h, frames, device, *, only_frame_t=False):
    """KTD6: at block `layer`'s output, replace the component along unit vector u with the donor's,
        h  ->  h + u (<u, h_donor> − <u, h>)
    then finish the forward pass. By default every frame 0..62 is patched (so later layers cannot read the old
    value back from an earlier frame); only_frame_t patches just the frame the readout is taken at.

    target_pos: [B, 64, 3, 2]; donor_h: [B, 63, d] (from residual_at on the donors); frames: [B] frame index t.
    Returns (next-frame prediction at t [B, 6], energy-head output at t [B] or None, as numpy)."""
    _, register_hooks = _register_hooks()
    u_t = torch.as_tensor(u, dtype=torch.float32, device=device)
    donor_proj = donor_h @ u_t  # [B, 63]
    idx = torch.as_tensor(frames, device=device)
    mask = None
    if only_frame_t:
        mask = torch.zeros(donor_proj.shape, device=device)
        mask[torch.arange(len(idx), device=device), idx] = 1.0

    with register_hooks(model, {layer: swap_hook(u_t, donor_proj, mask)}):
        pred, energy = model(wm.to_tensor(target_pos[:, :-1], device))
    rows_ = torch.arange(len(idx), device=device)
    e = energy[rows_, idx].float().cpu().numpy() if energy is not None else None
    return pred[rows_, idx].float().cpu().numpy(), e


@torch.no_grad()
def clean_at(model, target_pos, frames, device):
    """The unpatched (next-frame prediction [B, 6], energy head [B] or None) at frame t of each target."""
    pred, energy = model(wm.to_tensor(target_pos[:, :-1], device))
    idx = torch.as_tensor(frames, device=device)
    rows_ = torch.arange(len(idx), device=device)
    e = energy[rows_, idx].float().cpu().numpy() if energy is not None else None
    return pred[rows_, idx].float().cpu().numpy(), e


def implied_velocity(pred_next, current) -> np.ndarray:
    """The velocity the prediction implies, (x̂_{t+1} − x_t) / DT, per coordinate. [B, 6]"""
    return (pred_next - current) / wm.DT


def implied_energy(pred_next, current) -> np.ndarray:
    """Total kinetic energy implied by the predicted step, ½ Σ |(x̂_{t+1} − x_t) / DT|² (mass 1). [B]"""
    return 0.5 * (implied_velocity(pred_next, current) ** 2).sum(axis=1)


def tau(dg, dphi) -> float:
    """Transfer correlation (KTD6): Pearson correlation of the readout's change with the true change."""
    dg, dphi = np.asarray(dg, dtype=np.float64), np.asarray(dphi, dtype=np.float64)
    if dg.std() < 1e-12 or dphi.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(dg, dphi)[0, 1])


# --- 4. Target-donor pairs and readouts (lesson 04) -----------------------------------------------------------------------


def make_pairs(n_episodes, n_pairs, rng, frames=(8, 62)) -> dict:
    """KTD6 pairs: a target episode, a different donor episode, and one frame index t shared by both."""
    target = rng.integers(0, n_episodes, n_pairs)
    donor = (target + rng.integers(1, n_episodes, n_pairs)) % n_episodes  # never the target itself
    t = rng.integers(frames[0], frames[1] + 1, n_pairs)
    return {"target": target, "donor": donor, "t": t}


def true_changes(data, pairs) -> dict:
    """Δφ for each pair: the true donor-minus-target change in each quantity at frame t."""
    v = data["velocities"]
    return {
        "velocity": v[pairs["donor"], pairs["t"], 0, 0] - v[pairs["target"], pairs["t"], 0, 0],  # ball 0, x
        "energy": data["energy"][pairs["donor"]] - data["energy"][pairs["target"]],
    }


def readouts(pred, energy_head, current) -> dict:
    """The scalars g read from a prediction (KTD6): ball 0's implied x-velocity, the implied total energy, and the
    energy head's estimate in energy units (Model C only)."""
    out = {"velocity": implied_velocity(pred, current)[:, 0], "energy": implied_energy(pred, current)}
    if energy_head is not None:
        out["energy head"] = energy_head * wm.ENERGY_STD + wm.ENERGY_MEAN
    return out


class PairRunner:
    """Holds one model's pairs, clean readouts and donor activations, so many directions can be tested cheaply."""

    def __init__(self, model, data, pairs, device):
        self.model, self.device, self.pairs = model, device, pairs
        self.target_pos = data["positions"][pairs["target"]]
        self.donor_pos = data["positions"][pairs["donor"]]
        self.current = self.target_pos[np.arange(len(pairs["t"])), pairs["t"]].reshape(len(pairs["t"]), -1)
        self.dphi = true_changes(data, pairs)
        pred, e = clean_at(model, self.target_pos, pairs["t"], device)
        self.clean = readouts(pred, e, self.current)
        self.clean_pred = pred
        self._donor_h = {}

    def donor_h(self, layer):
        if layer not in self._donor_h:
            self._donor_h[layer] = residual_at(self.model, layer, self.donor_pos, self.device)
        return self._donor_h[layer]

    def patched(self, layer, u, *, only_frame_t=False, donor_h=None):
        h = self.donor_h(layer) if donor_h is None else donor_h
        pred, e = interchange(self.model, layer, u, self.target_pos, h, self.pairs["t"], self.device,
                              only_frame_t=only_frame_t)
        return pred, readouts(pred, e, self.current)

    def taus(self, layer, u, *, only_frame_t=False) -> dict:
        """τ for every readout, each against its own quantity's true change (energy head against energy)."""
        _, g = self.patched(layer, u, only_frame_t=only_frame_t)
        out = {}
        for name, value in g.items():
            quantity = "velocity" if name == "velocity" else "energy"
            out[name] = tau(value - self.clean[name], self.dphi[quantity])
        return out


def random_directions(n, d, rng) -> np.ndarray:
    """n uniformly random unit vectors in d dimensions (the KTD6 null; the paper also rescales to |w|, which a
    unit-vector swap does not need)."""
    r = rng.standard_normal((n, d))
    return (r / np.linalg.norm(r, axis=1, keepdims=True)).astype(np.float32)


def head_rows(model) -> np.ndarray:
    """The directions through which the last block's output reaches the prediction linearly: the six rows of W_head
    scaled by the final LayerNorm's gain, plus the all-ones direction (LayerNorm subtracts the mean, so a shift along
    the ones vector changes every coordinate's centred value). Removing these from a direction removes its linear
    routes to the prediction (KTD6). LayerNorm's rescaling by the standard deviation is not linear and cannot be
    projected out; a direction that changes |h| still reaches the output through it. [7, d]"""
    w = model.head.weight.detach().float().cpu().numpy()
    g = model.ln_f.weight.detach().float().cpu().numpy()
    ones = np.ones((1, w.shape[1]), dtype=w.dtype) / np.sqrt(w.shape[1])
    return np.vstack([w * g, ones])


def residual_rows(model, layer, positions, device, batch=512) -> np.ndarray:
    """Block `layer`'s output as rows (frames 1..62, episode-major), for many episodes in batches. [E * 62, d]"""
    out = [residual_at(model, layer, positions[s : s + batch], device)[:, FRAMES].float().cpu().numpy()
           for s in range(0, len(positions), batch)]
    a = np.concatenate(out)
    return a.reshape(-1, a.shape[-1])


@torch.no_grad()
def add_at_frame(model, layer, positions, frames, delta, device):
    """Add delta[i] to block `layer`'s output at frame frames[i] of episode i only, finish the pass, and return the
    next-frame prediction at that frame. positions [B, 64, 3, 2]; delta [B, d]. Returns numpy [B, 6]."""
    _, register_hooks = _register_hooks()
    idx = torch.as_tensor(frames, device=device)
    rows_ = torch.arange(len(idx), device=device)
    d_t = torch.as_tensor(delta, dtype=torch.float32, device=device)

    def hook(module, inputs, output):
        output = output.clone()
        output[rows_, idx] += d_t
        return output

    with register_hooks(model, {layer: hook}):
        pred, _ = model(wm.to_tensor(positions[:, :-1], device))
    return pred[rows_, idx].float().cpu().numpy()


def null_band(values) -> list[float]:
    """The 2.5th and 97.5th percentiles of a null distribution of τ."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def verdict(tau_value, band, *, present=True) -> str:
    """KTD6's verdict, two-sided: above the band is "used"; below it is "reversed" (the prediction moves AGAINST the
    donor more than any random direction makes it, which is not "unused"); inside it is "present but unused" when the
    probe beat its input baseline, else "inside the null"."""
    lo, hi = band
    if tau_value > hi:
        return "used"
    if tau_value < lo:
        return "reversed (below the null)"
    return "present but unused" if present else "inside the null"
