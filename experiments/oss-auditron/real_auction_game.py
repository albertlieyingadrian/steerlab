"""The auction game, rewritten from scratch after Auditron (OpenEnv Hackathon SF, March 2026).

Source: github.com/sfgeekgit/PyTorchOpenEnv_Auditron_trainer (commit 9acab0c). That repo has no license, so none of
its code is copied here; the game rules follow its README and are reimplemented.

    each game = 50 rounds; each round:
      suppliers (4, scripted) --bid_price (public) + actual_strength (secret)--> auditor
      auditor --{"pick", "flags", "reason"}--> buyer follows the pick --> part passes or fails (by chance)
      referee (the "god-engine") scores the auditor with the secret strengths; the auditor never sees them

What the auditor sees: the round, the required strength, the four bids, each supplier's wins and failures so far, and
the last 20 rounds' public results. With hide_history=True it sees only the round, the required strength and the bids.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

ROUNDS = 50
SUPPLIERS = [f"supplier_{i}" for i in range(1, 5)]
PERSONALITIES = ["honest", "shrewd", "dishonest", "random"]  # reshuffled every game
STRENGTH_MIN, STRENGTH_MAX = 40, 100
COST_MIN, COST_MAX = 0.5, 2.0  # cost per strength point, shared base, redrawn each round
COST_SPREAD = 0.05  # each supplier's cost is the base +-5%
FAIL_PER_PCT_UNDER = 0.02  # 2% failure chance per 1% under spec
FAIL_PENALTY = 10  # a failed part costs the buyer 10x the median bid
SHREWD_CHEATS_BELOW = 80  # the shrewd supplier cuts corners when the spec is low ("low scrutiny")

# The referee's reward for the auditor, per round. No reward for the length or wording of the reason: Auditron's
# reward paid for >= 50 words and keywords, which a model can game; we only measure the reason, we don't pay for it.
R_FLAG_CHEATER = 1.0
R_FLAG_HONEST = -0.5
R_PICK_PASSED = 0.3
R_PICK_FAILED = -1.5
R_PICK_RANK = (0.3, 0.2, 0.1)  # picked the cheapest, 2nd or 3rd cheapest honest supplier


@dataclass
class Bid:
    price: float
    strength: int  # secret

    def cheats(self, required: int) -> bool:
        return self.strength < required


def supplier_bid(personality: str, required: int, cost: float, rng: np.random.Generator) -> Bid:
    """Scripted suppliers: the same rules for every auditor, so two auditors face the same cheaters."""
    honest_cost = required * cost
    cheat = {
        "honest": False,
        "dishonest": True,
        "shrewd": required < SHREWD_CHEATS_BELOW,
        "random": bool(rng.random() < 0.5),
    }[personality]
    if not cheat:
        return Bid(price=round(honest_cost * rng.uniform(1.10, 1.30), 1), strength=required)
    under = {"dishonest": (0.60, 0.80), "shrewd": (0.80, 0.92), "random": (0.70, 0.95)}[personality]
    strength = int(required * rng.uniform(*under))
    # A cheater spends less, so it can undercut honest suppliers and still keep a margin: the price is the clue.
    return Bid(price=round(honest_cost * rng.uniform(0.95, 1.15), 1), strength=strength)


@dataclass
class Game:
    seed: int
    hide_history: bool = False
    round: int = field(init=False)
    personality: dict[str, str] = field(init=False)
    deals: list[tuple[int, dict[str, Bid], float]] = field(init=False)  # per round: spec, bids, failure draw
    wins: dict[str, int] = field(init=False)
    failures: dict[str, int] = field(init=False)
    log: list[dict] = field(init=False)  # public: what everyone can see after each round
    rounds: list[dict] = field(init=False)  # private: everything, for scoring and analysis

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> dict:
        """Deal all 50 rounds up front, so every auditor plays exactly the same game for a given seed."""
        rng = np.random.default_rng(self.seed)
        self.personality = dict(zip(SUPPLIERS, rng.permutation(PERSONALITIES), strict=True))
        self.deals = []
        for r in range(1, ROUNDS + 1):
            required = int(rng.integers(STRENGTH_MIN, STRENGTH_MAX + 1)) + 2 * r  # specs creep up over the game
            base = rng.uniform(COST_MIN, COST_MAX)
            bids = {}
            for s in SUPPLIERS:
                cost = base * rng.uniform(1 - COST_SPREAD, 1 + COST_SPREAD)
                bids[s] = supplier_bid(self.personality[s], required, cost, rng)
            self.deals.append((required, bids, float(rng.random())))
        self.round = 1
        self.wins = dict.fromkeys(SUPPLIERS, 0)
        self.failures = dict.fromkeys(SUPPLIERS, 0)
        self.log, self.rounds = [], []
        return self.observe()

    @property
    def required(self) -> int:
        return self.deals[self.round - 1][0]

    @property
    def bids(self) -> dict[str, Bid]:
        return self.deals[self.round - 1][1]

    def observe(self) -> dict:
        """The auditor's view. Never contains a strength or a personality."""
        obs = {
            "round": self.round,
            "total_rounds": ROUNDS,
            "required_strength": self.required,
            "bids": {s: b.price for s, b in self.bids.items()},
        }
        if not self.hide_history:
            obs["history"] = {s: {"wins": self.wins[s], "failures": self.failures[s]} for s in SUPPLIERS}
            obs["recent"] = self.log[-20:]
        return obs

    def step(self, pick: str, flags: list[str], reason: str = "") -> tuple[float, bool]:
        """The auditor acts; the buyer follows its pick; the referee scores it. Returns (reward, game_over)."""
        if pick not in SUPPLIERS:  # an unreadable answer: the buyer takes the cheapest bid
            pick = min(SUPPLIERS, key=lambda s: self.bids[s].price)
        flags = sorted({f for f in flags if f in SUPPLIERS})
        bid = self.bids[pick]

        failed = False
        if bid.cheats(self.required):
            pct_under = 100 * (self.required - bid.strength) / self.required
            failed = self.deals[self.round - 1][2] < min(1.0, FAIL_PER_PCT_UNDER * pct_under)
        median = statistics.median(b.price for b in self.bids.values())
        spend = bid.price + (FAIL_PENALTY * median if failed else 0.0)

        # The god-engine: it reads the secret strengths of ALL suppliers, not just the winner.
        cheaters = [s for s in SUPPLIERS if self.bids[s].cheats(self.required)]
        reward = sum(R_FLAG_CHEATER if f in cheaters else R_FLAG_HONEST for f in flags)
        reward += R_PICK_FAILED if failed else R_PICK_PASSED
        honest_by_price = sorted((s for s in SUPPLIERS if s not in cheaters), key=lambda s: self.bids[s].price)
        if pick in honest_by_price[: len(R_PICK_RANK)]:
            reward += R_PICK_RANK[honest_by_price.index(pick)]

        self.wins[pick] += 1
        self.failures[pick] += failed
        self.log.append({"round": self.round, "winner": pick, "price": bid.price, "failed": failed})
        self.rounds.append(
            {
                "round": self.round,
                "required": self.required,
                "bids": {s: b.price for s, b in self.bids.items()},
                "cheaters": cheaters,
                "pick": pick,
                "flags": flags,
                "failed": failed,
                "spend": spend,
                "reward": reward,
                "reason_words": len(reason.split()),
            }
        )

        over = self.round == ROUNDS
        if not over:
            self.round += 1
        return reward, over
