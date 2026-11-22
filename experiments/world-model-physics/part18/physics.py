"""
The base project's world: balls bouncing elastically in a box, simulated exactly.

An event-driven simulator. Between collisions every ball moves in a straight line, so instead of stepping time in
small slices it asks "when is the next contact?", jumps straight there, applies the bounce, and repeats until the
next frame time. Nothing is ever approximated, so energy is conserved to floating-point rounding and every collision
gets an exact time.

    simulate(pos0, vel0, ...)         one episode from given starting positions and velocities
    random_start(rng, energy, ...)    non-overlapping random positions, velocities scaled to a total energy
    make_split(n, seed, energy_range) a whole split: arrays for the model (positions) and for us (everything else)
    matched_negatives(arrays, rng)    the "same distance, moving the other way" partner for every collision

Conventions: a unit box [0, 1]^dim, radius 0.05, mass 1, no gravity or friction, frames 0.05 time units
apart, 64 frames per episode. The collision code is written per axis, so `dim=3` works too (offered as is,
not a promise).

Collision types for dim 2: 0 = wall-x, 1 = wall-y, 2 = ball-ball (in general: axes 0..dim-1, then ball-ball = dim).
A label at frame t means "this collision happens after frame t and at or before frame t+1".
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

BOX = 1.0
RADIUS = 0.05
MASS = 1.0
DT = 0.05
N_FRAMES = 64
N_BALLS = 3
DIM = 2
AXIS_NAMES = "xyz"
MAX_EVENTS_PER_FRAME = 10_000  # a guard: a correct simulator never gets near it


def type_names(dim: int = DIM) -> list[str]:
    return [f"wall-{AXIS_NAMES[a]}" for a in range(dim)] + ["ball-ball"]


def pair_list(n_balls: int = N_BALLS) -> list[tuple[int, int]]:
    """Ball pairs in a fixed order: (0, 1), (0, 2), (1, 2) for three balls."""
    return list(combinations(range(n_balls), 2))


def kinetic_energy(vel, masses):
    """½ m |v|² per ball. vel [..., n_balls, dim], masses [n_balls] -> [..., n_balls]."""
    return 0.5 * masses * np.sum(vel * vel, axis=-1)


# --- 1. The event-driven step ---------------------------------------------------------------------------------------


def _wall_times(pos, vel, radius, box):
    """Time until each ball touches each wall it is moving towards; inf if it is moving away. [n_balls, dim]"""
    times = np.full(pos.shape, np.inf)
    right, left = vel > 0, vel < 0
    times[right] = (box - radius - pos[right]) / vel[right]
    times[left] = (radius - pos[left]) / vel[left]
    return np.maximum(times, 0.0)  # a ball already touching (rounding) collides now, never in the past


def _pair_times(pos, vel, radius, I, J):
    """Time until each pair's centres are 2r apart, for approaching pairs; inf otherwise. [n_pairs]

    With dp = p_j - p_i and dv = v_j - v_i, the distance at time t is |dp + dv t|. Setting it to 2r gives
    a t² + 2 b t + c = 0 with a = dv·dv, b = dp·dv, c = dp·dp - (2r)². The pair approaches when b < 0, and
    the first root is t = c / (-b + sqrt(b² - a c)) (the same root as (-b - sqrt(...)) / a, without cancellation).
    """
    dp, dv = pos[J] - pos[I], vel[J] - vel[I]
    a = np.sum(dv * dv, axis=1)
    b = np.sum(dp * dv, axis=1)
    c = np.sum(dp * dp, axis=1) - (2 * radius) ** 2
    disc = b * b - a * c
    times = np.full(len(I), np.inf)
    hit = (b < 0) & (disc >= 0)
    times[hit] = np.maximum(c[hit] / (-b[hit] + np.sqrt(disc[hit])), 0.0)
    return times


def _bounce_pair(pos, vel, masses, i, j):
    """Elastic collision: exchange momentum along the line of centres, leave the sideways part alone.

    n = unit vector from i to j. The closing speed along n is u = (v_i - v_j)·n. Ball i loses 2 m_j/(m_i+m_j) u
    along n and ball j gains 2 m_i/(m_i+m_j) u. For equal masses that is a swap of the two balls' n-components.
    Returns the change in total momentum (zero up to rounding) so callers can audit it.
    """
    n = pos[j] - pos[i]
    n = n / np.linalg.norm(n)
    u = np.dot(vel[i] - vel[j], n)
    before = masses[i] * vel[i] + masses[j] * vel[j]
    total = masses[i] + masses[j]
    vel[i] = vel[i] - (2 * masses[j] / total) * u * n
    vel[j] = vel[j] + (2 * masses[i] / total) * u * n
    return masses[i] * vel[i] + masses[j] * vel[j] - before


# --- 2. One episode -------------------------------------------------------------------------------------------------


def simulate(pos0, vel0, *, n_frames=N_FRAMES, dt=DT, radius=RADIUS, box=BOX, masses=None):
    """Run one episode exactly. pos0, vel0: [n_balls, dim].

    Returns a dict:
      positions, velocities  [n_frames, n_balls, dim]   the state at each frame time t = k·dt
      events                 list of (k, fraction, type, i, j): a collision after frame k, at time (k + fraction)·dt;
                             j = -1 for walls
      momentum_change        one vector per ball-ball collision: total momentum after minus before (should be ~0)
      contact_error          one number per ball-ball collision: |distance between centres - 2r| at the contact
    """
    pos = np.array(pos0, dtype=np.float64)
    vel = np.array(vel0, dtype=np.float64)
    n_balls, dim = pos.shape
    masses = np.full(n_balls, MASS) if masses is None else np.asarray(masses, dtype=np.float64)
    pairs = pair_list(n_balls)
    I = np.array([p[0] for p in pairs], dtype=int)
    J = np.array([p[1] for p in pairs], dtype=int)
    positions = np.empty((n_frames, n_balls, dim))
    velocities = np.empty((n_frames, n_balls, dim))
    events, momentum_change, contact_error = [], [], []
    positions[0], velocities[0] = pos, vel

    for k in range(n_frames - 1):
        elapsed = 0.0
        for _ in range(MAX_EVENTS_PER_FRAME):
            wall = _wall_times(pos, vel, radius, box)
            pair = _pair_times(pos, vel, radius, I, J) if len(pairs) else np.empty(0)
            t_wall = wall.min()
            t_pair = pair.min() if len(pair) else np.inf
            t_next = min(t_wall, t_pair)
            if elapsed + t_next > dt:  # nothing else happens before the next frame
                pos += vel * (dt - elapsed)
                break
            pos += vel * t_next
            elapsed += t_next
            if t_wall <= t_pair:
                ball, axis = np.unravel_index(np.argmin(wall), wall.shape)
                pos[ball, axis] = radius if vel[ball, axis] < 0 else box - radius  # exactly on the wall
                vel[ball, axis] = -vel[ball, axis]
                events.append((k, elapsed / dt, int(axis), int(ball), -1))
            else:
                p = int(np.argmin(pair))
                contact_error.append(abs(np.linalg.norm(pos[J[p]] - pos[I[p]]) - 2 * radius))
                momentum_change.append(_bounce_pair(pos, vel, masses, I[p], J[p]))
                events.append((k, elapsed / dt, dim, int(I[p]), int(J[p])))
        else:
            raise RuntimeError(f"more than {MAX_EVENTS_PER_FRAME} collisions in frame {k}: the simulator is stuck")
        positions[k + 1], velocities[k + 1] = pos, vel

    return {
        "positions": positions,
        "velocities": velocities,
        "events": events,
        "momentum_change": np.array(momentum_change).reshape(-1, dim),
        "contact_error": np.array(contact_error),
    }


def random_start(rng, energy, *, n_balls=N_BALLS, dim=DIM, radius=RADIUS, box=BOX, masses=None):
    """Uniform non-overlapping positions, and random velocities rescaled so the total kinetic energy is `energy`."""
    masses = np.full(n_balls, MASS) if masses is None else np.asarray(masses, dtype=np.float64)
    pos = np.empty((n_balls, dim))
    for b in range(n_balls):
        while True:  # rejection sampling: redraw until this ball overlaps none of the earlier ones
            candidate = rng.uniform(radius, box - radius, size=dim)
            if b == 0 or np.min(np.linalg.norm(pos[:b] - candidate, axis=1)) > 2 * radius:
                pos[b] = candidate
                break
    vel = rng.normal(size=(n_balls, dim))
    vel *= np.sqrt(energy / kinetic_energy(vel, masses).sum())
    return pos, vel


# --- 3. Checks run on every episode -----------------------------------------------------------------


def audit(episode, *, radius=RADIUS, box=BOX, masses=None) -> dict:
    """The four audit numbers for one simulated episode (worst case over its frames and collisions).

    `overlap` is measured at every frame AND at every contact (where two balls are closest), so it is not
    trivially zero just because frames happen to fall between collisions."""
    pos, vel = episode["positions"], episode["velocities"]
    n_balls = pos.shape[1]
    masses = np.full(n_balls, MASS) if masses is None else np.asarray(masses, dtype=np.float64)
    energy = kinetic_energy(vel, masses).sum(axis=-1)
    pairs = pair_list(n_balls)
    if pairs:
        I, J = np.array(pairs).T
        separation = np.linalg.norm(pos[:, J] - pos[:, I], axis=-1)
        overlap = float(max(0.0, (2 * radius - separation).max()))
        if len(episode["contact_error"]):
            overlap = max(overlap, float(episode["contact_error"].max()))
    else:
        overlap = 0.0
    outside = float(max(0.0, (radius - pos).max(), (pos - (box - radius)).max()))
    dmom = episode["momentum_change"]
    return {
        "energy_drift": float(np.abs(energy - energy[0]).max() / energy[0]),
        "momentum_error": float(np.abs(dmom).max()) if len(dmom) else 0.0,
        "outside": outside,
        "overlap": overlap,
        "n_ball_ball": len(dmom),
    }


# --- 4. A whole split ------------------------------------------------------------------------------------------------


def make_split(n_episodes, seed, energy_range, *, n_frames=N_FRAMES, n_balls=N_BALLS, dim=DIM):
    """Simulate `n_episodes` episodes. Episode e is drawn from np.random.default_rng([seed, e]), so any one
    episode can be regenerated on its own and the split does not depend on the order episodes are run in.

    Returns (arrays, worst) where `arrays` is ready for save_npz and `worst` holds the worst audit numbers.
    Shapes (E episodes, T frames, N balls, D dims, P pairs, K collision types = D + 1):
      positions, velocities [E, T, N, D] float64; energy [E]; ke_ball [E, T, N]
      wall_next [E, T-1, N, D], pair_next [E, T-1, P], collides_next [E, T-1, K]   bool
      event_episode, event_frame, event_type, event_i, event_j  int32 [n_events]; event_fraction float64
    """
    pairs = pair_list(n_balls)
    positions = np.empty((n_episodes, n_frames, n_balls, dim))
    velocities = np.empty_like(positions)
    energies = np.empty(n_episodes)
    wall_next = np.zeros((n_episodes, n_frames - 1, n_balls, dim), dtype=bool)
    pair_next = np.zeros((n_episodes, n_frames - 1, len(pairs)), dtype=bool)
    ev = []
    worst = {"energy_drift": 0.0, "momentum_error": 0.0, "outside": 0.0, "overlap": 0.0, "n_ball_ball": 0}
    lo, hi = energy_range
    for e in range(n_episodes):
        rng = np.random.default_rng([seed, e])
        energies[e] = rng.uniform(lo, hi)
        pos0, vel0 = random_start(rng, energies[e], n_balls=n_balls, dim=dim)
        episode = simulate(pos0, vel0, n_frames=n_frames)
        positions[e], velocities[e] = episode["positions"], episode["velocities"]
        for k, fraction, kind, i, j in episode["events"]:
            ev.append((e, k, fraction, kind, i, j))
            if kind < dim:
                wall_next[e, k, i, kind] = True
            else:
                pair_next[e, k, pairs.index((i, j))] = True
        report = audit(episode)
        for key in ("energy_drift", "momentum_error", "outside", "overlap"):
            worst[key] = max(worst[key], report[key])
        worst["n_ball_ball"] += report["n_ball_ball"]
    collides_next = np.concatenate([wall_next.any(axis=2), pair_next.any(axis=2, keepdims=True)], axis=2)
    events = np.array(ev, dtype=np.float64).reshape(-1, 6)
    arrays = {
        "positions": positions,
        "velocities": velocities,
        "energy": energies,
        "ke_ball": kinetic_energy(velocities, np.full(n_balls, MASS)),
        "wall_next": wall_next,
        "pair_next": pair_next,
        "collides_next": collides_next,
        "event_episode": events[:, 0].astype(np.int32),
        "event_frame": events[:, 1].astype(np.int32),
        "event_fraction": events[:, 2],
        "event_type": events[:, 3].astype(np.int32),
        "event_i": events[:, 4].astype(np.int32),
        "event_j": events[:, 5].astype(np.int32),
    }
    return arrays, worst


# --- 5. Matched negatives -------------------------------------------------------------------------------------


def wall_gap_and_approach(positions, velocities, *, radius=RADIUS, box=BOX):
    """For every ball and axis: the gap between the ball's surface and the NEAREST wall on that axis, and +1 if
    the ball moves towards that wall, -1 if away (0 if still). Shapes match positions."""
    low_side = positions < box / 2
    gap = np.where(low_side, positions - radius, box - radius - positions)
    toward = np.where(low_side, -velocities, velocities)
    return gap, np.sign(toward)


def pair_gap_and_approach(positions, velocities, *, radius=RADIUS):
    """For every pair: the surface-to-surface gap, and +1 if the pair is closing, -1 if separating. [..., P]"""
    I, J = np.array(pair_list(positions.shape[-2])).T
    dp = positions[..., J, :] - positions[..., I, :]
    dv = velocities[..., J, :] - velocities[..., I, :]
    gap = np.linalg.norm(dp, axis=-1) - 2 * radius
    return gap, np.sign(-np.sum(dp * dv, axis=-1))


def _match(pos_index, pos_gap, neg_index, neg_gap, rng, tol):
    """Pair each positive with a different negative whose gap is in the same `tol`-wide bin (so |Δgap| < tol),
    without replacement. Positives in a bin with too few negatives stay unmatched and are counted."""
    pos_bin = np.floor(pos_gap / tol).astype(np.int64)
    neg_bin = np.floor(neg_gap / tol).astype(np.int64)
    order = np.argsort(neg_bin, kind="stable")
    neg_sorted = neg_bin[order]
    chosen_pos, chosen_neg = [], []
    for b in np.unique(pos_bin):
        p = np.flatnonzero(pos_bin == b)
        start, stop = np.searchsorted(neg_sorted, [b, b + 1])
        pool = order[start:stop]
        take = min(len(p), len(pool))
        if take:
            chosen_pos.append(rng.choice(p, size=take, replace=False))
            chosen_neg.append(rng.choice(pool, size=take, replace=False))
    if not chosen_pos:
        return np.empty((0, 2, pos_index.shape[1]), dtype=np.int32), len(pos_index)
    cp, cn = np.concatenate(chosen_pos), np.concatenate(chosen_neg)
    keep = np.argsort(cp, kind="stable")
    matched = np.stack([pos_index[cp[keep]], neg_index[cn[keep]]], axis=1).astype(np.int32)
    return matched, len(pos_index) - len(cp)


def matched_negatives(arrays, rng, *, tol=0.01):
    """The matched sets. A wall collision correlates with being near the wall, so each wall positive
    (a ball that hits a wall on that axis before the next frame, approaching it at frame t) gets a negative at the
    same gap to the nearest wall on that axis but moving away. Each ball-ball positive gets a pair at the same
    separation but moving apart.

    Returns {name: int32 array}:
      match_wall [M, 2, 4]: (episode, frame, ball, axis) of the positive, then of its negative
      match_pair [M, 2, 3]: (episode, frame, pair) of the positive, then of its negative
    and a dict of counts (positives, matched, unmatched, positives excluded because they were not approaching at
    frame t, which happens when another collision earlier in the same frame turns the ball around).
    """
    pos, vel = arrays["positions"][:, :-1], arrays["velocities"][:, :-1]  # frames that have a "next"
    out, counts = {}, {}

    gap, approach = wall_gap_and_approach(pos, vel)
    label = arrays["wall_next"]
    positive = label & (approach > 0)
    negative = ~label & (approach < 0)
    pos_index, neg_index = np.argwhere(positive), np.argwhere(negative)
    matched, unmatched = _match(pos_index, gap[positive], neg_index, gap[negative], rng, tol)
    out["match_wall"] = matched
    counts["wall"] = {"positives": int(label.sum()), "excluded_not_approaching": int((label & ~positive).sum()),
                      "matched": len(matched), "unmatched": int(unmatched)}

    gap, approach = pair_gap_and_approach(pos, vel)
    label = arrays["pair_next"]
    positive = label & (approach > 0)
    negative = ~label & (approach < 0)
    pos_index, neg_index = np.argwhere(positive), np.argwhere(negative)
    matched, unmatched = _match(pos_index, gap[positive], neg_index, gap[negative], rng, tol)
    out["match_pair"] = matched
    counts["ball-ball"] = {"positives": int(label.sum()), "excluded_not_approaching": int((label & ~positive).sum()),
                           "matched": len(matched), "unmatched": int(unmatched)}
    return out, counts
