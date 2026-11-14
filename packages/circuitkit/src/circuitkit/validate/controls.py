"""Control feature sets: what the same intervention does to features that are *not* the claimed circuit.

Two kinds, both the same size as the tested group and drawn from features active on the same prompt:

- **random**: for each tested feature, a random active feature at the same layer and position; failing that, the same
  layer at any position; failing that, any active feature.
- **matched-norm**: for each tested feature, one of the `k` active features (same position if possible) whose
  contribution norm, |activation| x decoder norm, is closest to it. Ablating such a feature removes a write of the
  same size from the residual stream, so a group that only "matters" because it is big will not beat this control.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from circuitkit.validate.backend import ActiveFeature
from circuitkit.validate.types import FeatureNode

Key = tuple[int, int, int]


def _node(a: ActiveFeature) -> FeatureNode:
    return FeatureNode(layer=a[0], pos=a[1], feature=a[2], activation=a[3])


def candidate_pool(
    active: Sequence[ActiveFeature], exclude: Sequence[FeatureNode], skip_pos0: bool = True
) -> list[ActiveFeature]:
    """Active features that may serve as controls: not in the tested group, and (by default) not at position 0,
    where circuit-tracer zeroes every feature."""
    taken = {n.key for n in exclude}
    return [a for a in active if (a[0], a[1], a[2]) not in taken and not (skip_pos0 and a[1] == 0)]


def sample_random(group: Sequence[FeatureNode], pool: Sequence[ActiveFeature], rng: random.Random) -> list[FeatureNode]:
    chosen: set[Key] = set()
    out: list[FeatureNode] = []
    for n in group:
        tiers = (
            [a for a in pool if a[0] == n.layer and a[1] == n.pos],
            [a for a in pool if a[0] == n.layer],
            list(pool),
        )
        for tier in tiers:
            free = [a for a in tier if (a[0], a[1], a[2]) not in chosen]
            if free:
                pick = rng.choice(free)
                chosen.add((pick[0], pick[1], pick[2]))
                out.append(_node(pick))
                break
    return out


def sample_matched_norm(
    group: Sequence[FeatureNode],
    pool: Sequence[ActiveFeature],
    rng: random.Random,
    decoder_norm: Callable[[int, int], float],
    k: int = 5,
) -> list[FeatureNode]:
    def size(layer: int, feature: int, act: float) -> float:
        return max(abs(act) * decoder_norm(layer, feature), 1e-12)

    chosen: set[Key] = set()
    out: list[FeatureNode] = []
    for n in group:
        target = math.log(size(n.layer, n.feature, n.activation))
        same_pos = [a for a in pool if a[1] == n.pos and (a[0], a[1], a[2]) not in chosen]
        cands = same_pos if len(same_pos) >= k else [a for a in pool if (a[0], a[1], a[2]) not in chosen]
        if not cands:
            continue
        cands.sort(key=lambda a: abs(math.log(size(a[0], a[2], a[3])) - target))
        pick = rng.choice(cands[:k])
        chosen.add((pick[0], pick[1], pick[2]))
        out.append(_node(pick))
    return out


SAMPLERS = ("random", "matched_norm")
