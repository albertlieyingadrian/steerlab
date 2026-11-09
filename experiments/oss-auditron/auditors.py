"""Auditors without a model: the yardsticks any trained auditor has to beat.

Each auditor is a function: observation -> (pick, flags, reason).

    random   : picks at random, flags each supplier with probability 1/4      (the floor)
    cheapest : picks the lowest bid, never flags                               (what a naive buyer does)
    history  : flags anyone with a failure so far, picks the cheapest other    (the rule Auditron's prompt hints at)
    price    : flags any bid more than 8% under the median, picks the cheapest other  (reads only today's bids)
    oracle   : reads the secret strengths through the game object              (the ceiling; checks the scoring)
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable

from real_auction_game import SUPPLIERS, Game

Decision = tuple[str, list[str], str]
Auditor = Callable[[dict], Decision]


def cheapest_of(bids: dict[str, float], allowed: list[str]) -> str:
    pool = allowed or list(bids)
    return min(pool, key=lambda s: bids[s])


def random_auditor(seed: int) -> Auditor:
    rng = random.Random(seed)

    def act(obs: dict) -> Decision:
        return rng.choice(SUPPLIERS), [s for s in SUPPLIERS if rng.random() < 0.25], ""

    return act


def cheapest_auditor(obs: dict) -> Decision:
    return cheapest_of(obs["bids"], []), [], ""


def history_auditor(obs: dict) -> Decision:
    history = obs.get("history", {})
    flags = [s for s in SUPPLIERS if history.get(s, {}).get("failures", 0) > 0]
    return cheapest_of(obs["bids"], [s for s in SUPPLIERS if s not in flags]), flags, ""


def price_auditor(obs: dict) -> Decision:
    bids = obs["bids"]
    median = statistics.median(bids.values())
    flags = [s for s in SUPPLIERS if bids[s] < 0.92 * median]
    return cheapest_of(bids, [s for s in SUPPLIERS if s not in flags]), flags, ""


def oracle_auditor(game: Game) -> Auditor:
    def act(obs: dict) -> Decision:
        cheaters = [s for s in SUPPLIERS if game.bids[s].cheats(game.required)]
        return cheapest_of(obs["bids"], [s for s in SUPPLIERS if s not in cheaters]), cheaters, ""

    return act
