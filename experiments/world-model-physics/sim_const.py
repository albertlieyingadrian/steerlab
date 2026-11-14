"""
The constants world: Part 18's box of three balls, plus gravity g (pulling towards -y) and linear drag gamma.

    dv/dt = -g * y_hat - gamma * v        between collisions (exact solution per substep)
    walls and ball-ball contacts          elastic (restitution 1), resolved at substep ends

Unlike physics.py this is a time-stepped simulator (20 substeps per frame, vectorised over episodes): with gravity and
drag the flight paths are no longer straight lines, so the event-driven trick does not apply. Each substep moves every
ball along its exact gravity-plus-drag path, then reflects balls that crossed a wall (position mirrored, velocity
flipped) and swaps the line-of-centres velocity components of touching, approaching pairs. A substep moves a ball at
most ~0.006 box units against a radius of 0.05, so no contact is skipped.

Each episode draws its own g and gamma (uniform in the split's range), non-overlapping starting positions and random
velocities scaled to a kinetic energy in [0.5, 2.0], as in Part 18. The model sees positions only, so it has to infer
g and gamma from how the paths bend and slow within the episode.

    PLAN[split] = (episodes, g range, gamma range)
    make_split(n, seed, g_range, drag_range) -> arrays (positions, velocities, g, drag, collides_next, ...)
    fit_g_drag(acc, vel, mask)  least-squares g and gamma from accelerations: how we read the MODEL's constants
"""

from __future__ import annotations

import numpy as np

BOX, RADIUS, DT, N_FRAMES, N_BALLS, DIM = 1.0, 0.05, 0.05, 64, 3, 2
SUBSTEPS = 20
KE_RANGE = (0.5, 2.0)

G_TRAIN, DRAG_TRAIN = (0.0, 2.0), (0.0, 0.5)
PLAN = {
    "train": (20_000, G_TRAIN, DRAG_TRAIN),
    "val": (2_000, G_TRAIN, DRAG_TRAIN),
    "test": (2_000, G_TRAIN, DRAG_TRAIN),
    "ood_g": (1_000, (2.5, 3.5), DRAG_TRAIN),  # stronger gravity than training ever showed
    "ood_drag": (1_000, G_TRAIN, (0.7, 1.0)),  # stronger drag than training ever showed
}
PAIRS = [(0, 1), (0, 2), (1, 2)]


def _flight(pos, vel, g, drag, h):
    """Exact motion for time h under gravity g ([E]) and drag gamma ([E]). pos, vel: [E, N, 2].

    With a = -g y_hat and s = (1 - e^{-gamma h}) / gamma:  v = v0 e^{-gamma h} + a s,  x = x0 + v0 s + a (h - s) / gamma
    (both stay finite as gamma -> 0: s -> h and (h - s) / gamma -> h^2 / 2)."""
    gm = drag[:, None, None]
    small = gm < 1e-8
    safe = np.where(small, 1.0, gm)
    decay = np.exp(-gm * h)
    s = np.where(small, h, (1 - decay) / safe)
    w = np.where(small, h * h / 2, (h - s) / safe)
    acc = np.zeros_like(vel)
    acc[..., 1] = -g[:, None]
    return pos + vel * s + acc * w, vel * decay + acc * s


def _walls(pos, vel):
    lo, hi = RADIUS, BOX - RADIUS
    under, over = pos < lo, pos > hi
    pos = np.where(under, 2 * lo - pos, np.where(over, 2 * hi - pos, pos))
    hit = under | over
    return pos, np.where(hit, -vel, vel), hit


def _pairs(pos, vel):
    hit = np.zeros(pos.shape[:2], dtype=bool)
    for i, j in PAIRS:
        dp = pos[:, j] - pos[:, i]
        dist = np.linalg.norm(dp, axis=-1)
        n = dp / np.maximum(dist, 1e-12)[:, None]
        u = np.sum((vel[:, i] - vel[:, j]) * n, axis=-1)  # closing speed along the line of centres
        touch = (dist < 2 * RADIUS) & (u > 0)
        dv = np.where(touch[:, None], u[:, None] * n, 0.0)
        vel[:, i] -= dv
        vel[:, j] += dv
        hit[:, i] |= touch
        hit[:, j] |= touch
    return vel, hit


def random_starts(rng, n):
    pos = np.empty((n, N_BALLS, DIM))
    for e in range(n):
        for b in range(N_BALLS):
            while True:
                c = rng.uniform(RADIUS, BOX - RADIUS, size=DIM)
                if b == 0 or np.min(np.linalg.norm(pos[e, :b] - c, axis=1)) > 2 * RADIUS:
                    pos[e, b] = c
                    break
    vel = rng.normal(size=(n, N_BALLS, DIM))
    ke = rng.uniform(*KE_RANGE, size=n)
    vel *= np.sqrt(ke / (0.5 * (vel**2).sum(axis=(1, 2))))[:, None, None]
    return pos, vel, ke


def simulate(pos, vel, g, drag, n_frames=N_FRAMES):
    """Vectorised over episodes. Returns positions, velocities [E, T, N, 2] and hits [E, T-1, N] (ball b touched a
    wall or a ball after frame t and at or before frame t+1)."""
    pos, vel = pos.copy(), vel.copy()
    E = len(pos)
    P = np.empty((E, n_frames, N_BALLS, DIM))
    V = np.empty_like(P)
    hits = np.zeros((E, n_frames - 1, N_BALLS), dtype=bool)
    P[:, 0], V[:, 0] = pos, vel
    h = DT / SUBSTEPS
    for k in range(n_frames - 1):
        for _ in range(SUBSTEPS):
            pos, vel = _flight(pos, vel, g, drag, h)
            pos, vel, wall_hit = _walls(pos, vel)
            vel, pair_hit = _pairs(pos, vel)
            hits[:, k] |= wall_hit.any(axis=-1) | pair_hit
        P[:, k + 1], V[:, k + 1] = pos, vel
    return P, V, hits


def make_split(n, seed, g_range, drag_range):
    rng = np.random.default_rng(seed)
    pos, vel, ke = random_starts(rng, n)
    g = rng.uniform(*g_range, size=n)
    drag = rng.uniform(*drag_range, size=n)
    P, V, hits = simulate(pos, vel, g, drag)
    return {
        "positions": P,
        "velocities": V,
        "g": g,
        "drag": drag,
        "ke0": ke,
        "ball_hit_next": hits,
        "collides_next": hits.any(axis=-1, keepdims=True),
    }


def implied_accel(prev, cur, nxt):
    """Second difference (x_{t+1} - 2 x_t + x_{t-1}) / DT^2, per coordinate."""
    return (nxt - 2 * cur + prev) / DT**2


def fit_g_drag(acc, vel, mask):
    """Least-squares (g, gamma) from accelerations a = -g y_hat - gamma v over the masked balls.

    acc, vel: [..., N, 2]; mask [..., N] (True = use this ball). Equations per ball: a_x = -gamma v_x and
    a_y = -g - gamma v_y. Returns g, gamma of shape [...] (nan where no ball is usable)."""
    m = mask.astype(float)
    vx, vy, ax, ay = vel[..., 0], vel[..., 1], acc[..., 0], acc[..., 1]
    s11 = m.sum(-1)
    s12 = (m * vy).sum(-1)
    s22 = (m * (vx**2 + vy**2)).sum(-1)
    b1 = -(m * ay).sum(-1)
    b2 = -(m * (vx * ax + vy * ay)).sum(-1)
    det = s11 * s22 - s12**2
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (s22 * b1 - s12 * b2) / det
        gamma = (s11 * b2 - s12 * b1) / det
    bad = (s11 < 1) | (np.abs(det) < 1e-9)
    return np.where(bad, np.nan, g), np.where(bad, np.nan, gamma)
