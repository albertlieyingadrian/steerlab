"""Play held-out games with an auditor and report what the README claim needed: counts, intervals, both halves.

    python evaluate.py --auditor history                 # 20 held-out games (seeds 100000-100019)
    python evaluate.py --auditor history --hide-history  # ablation: bids only
    python evaluate.py --all                             # every model-free auditor, with and without history

Metrics (each a ratio of counts summed over games; 95% intervals by resampling whole games, since rounds inside one
game share a supplier line-up and are not independent):
    flag TPR   : of all cheating bids (every supplier, every round), the share flagged      <- the main measure
    flag FPR   : of all honest bids, the share flagged
    winner TPR : of rounds a cheater won, the share where the auditor had flagged the winner (Auditron's TPR)
    winner FPR : of rounds an honest supplier won, the share where the auditor flagged the winner (Auditron's FPR)
The winner metrics are kept only to compare with Auditron's README. The buyer follows the auditor's pick, so a cheater
wins a flagged round only when the auditor recommends a supplier it has itself flagged: a coherent auditor scores 0.
    failures   : parts that failed per game; spend: buyer's total spend per game, penalties included
Each is also split into rounds 1-25 and 26-50, the comparison Auditron's README reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import auditors
from real_auction_game import ROUNDS, SUPPLIERS, Game

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EVAL_SEED0 = 100_000  # training (if any) uses seeds below this, so evaluation games are never trained on
N_GAMES = 20
N_BOOT = 2000
MODEL_FREE = ["random", "cheapest", "history", "price", "oracle"]


def play(seed: int, make_auditor, hide_history: bool = False) -> list[dict]:
    game = Game(seed=seed, hide_history=hide_history)
    act = make_auditor(game)
    obs, over = game.observe(), False
    while not over:
        pick, flags, reason = act(obs)
        _, over = game.step(pick, flags, reason)
        obs = game.observe()
    return game.rounds


def counts(rounds: list[dict]) -> dict[str, float]:
    """Numerators and denominators for one game (or one half of it)."""
    c = dict.fromkeys(["cheat_wins", "cheat_wins_flagged", "honest_wins", "honest_wins_flagged",
                       "cheat_bids", "cheat_bids_flagged", "honest_bids", "honest_bids_flagged",
                       "failures", "spend", "reward", "reason_words"], 0.0)  # fmt: skip
    for r in rounds:
        won_by_cheater = r["pick"] in r["cheaters"]
        c["cheat_wins" if won_by_cheater else "honest_wins"] += 1
        if r["pick"] in r["flags"]:
            c["cheat_wins_flagged" if won_by_cheater else "honest_wins_flagged"] += 1
        for s in SUPPLIERS:
            kind = "cheat" if s in r["cheaters"] else "honest"
            c[f"{kind}_bids"] += 1
            c[f"{kind}_bids_flagged"] += s in r["flags"]
        c["failures"] += r["failed"]
        c["spend"] += r["spend"]
        c["reward"] += r["reward"]
        c["reason_words"] += r["reason_words"]
    return c


RATIOS = {
    "winner_tpr": ("cheat_wins_flagged", "cheat_wins"),
    "winner_fpr": ("honest_wins_flagged", "honest_wins"),
    "flag_tpr": ("cheat_bids_flagged", "cheat_bids"),
    "flag_fpr": ("honest_bids_flagged", "honest_bids"),
}
PER_GAME = ["failures", "spend", "reward"]


def summarise(per_game: list[dict[str, float]], seed: int = 0) -> dict:
    """Point estimates and game-level bootstrap intervals."""
    keys = list(per_game[0])
    table = np.array([[g[k] for k in keys] for g in per_game])
    col = {k: i for i, k in enumerate(keys)}

    def stats(t: np.ndarray) -> dict[str, float]:
        out = {}
        for name, (num, den) in RATIOS.items():
            d = t[:, col[den]].sum()
            out[name] = float(t[:, col[num]].sum() / d) if d else float("nan")
        for name in PER_GAME:
            out[name] = float(t[:, col[name]].mean())
        return out

    point = stats(table)
    rng = np.random.default_rng(seed)
    boots = [stats(table[rng.integers(0, len(table), len(table))]) for _ in range(N_BOOT)]
    out = {}
    for name, value in point.items():
        b = np.array([x[name] for x in boots])
        b = b[np.isfinite(b)]
        ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]
        out[name] = {"value": value if np.isfinite(value) else None, "ci95": ci}
    for name, (num, den) in RATIOS.items():
        out[name]["count"] = f"{int(table[:, col[num]].sum())}/{int(table[:, col[den]].sum())}"
    return out


def evaluate(name: str, make_auditor, n_games: int = N_GAMES, hide_history: bool = False) -> dict:
    games = [play(EVAL_SEED0 + i, make_auditor, hide_history) for i in range(n_games)]
    half = ROUNDS // 2
    result = {
        "auditor": name,
        "hide_history": hide_history,
        "games": n_games,
        "seeds": [EVAL_SEED0, EVAL_SEED0 + n_games - 1],
        "all": summarise([counts(g) for g in games]),
        "rounds_1_25": summarise([counts(g[:half]) for g in games]),
        "rounds_26_50": summarise([counts(g[half:]) for g in games]),
        "mean_reason_words": float(np.mean([counts(g)["reason_words"] / ROUNDS for g in games])),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}{'_nohistory' if hide_history else ''}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def model_free(name: str):
    """A factory: game -> auditor, so the random auditor gets a seed and the oracle gets the game."""
    return {
        "random": lambda game: auditors.random_auditor(game.seed),
        "cheapest": lambda game: auditors.cheapest_auditor,
        "history": lambda game: auditors.history_auditor,
        "price": lambda game: auditors.price_auditor,
        "oracle": auditors.oracle_auditor,
    }[name]


def line(r: dict) -> str:
    def f(block: dict, key: str) -> str:
        m = block[key]
        if m["value"] is None:
            return "  n/a"
        return f"{m['value']:.2f} [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]"

    a, h1, h2 = r["all"], r["rounds_1_25"], r["rounds_26_50"]
    tag = r["auditor"] + (" (bids only)" if r["hide_history"] else "")
    return (f"{tag:21s} flag TPR {f(a, 'flag_tpr')} FPR {f(a, 'flag_fpr')} | halves TPR {f(h1, 'flag_tpr')} -> "
            f"{f(h2, 'flag_tpr')} | winner TPR {f(a, 'winner_tpr')} ({a['winner_tpr']['count']}) "
            f"| failures/game {a['failures']['value']:.1f}  spend/game {a['spend']['value']:.0f}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auditor", choices=MODEL_FREE)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hide-history", action="store_true")
    ap.add_argument("--games", type=int, default=N_GAMES)
    args = ap.parse_args()
    runs = [(n, h) for n in MODEL_FREE for h in (False, True)] if args.all else [(args.auditor, args.hide_history)]
    for name, hide in runs:
        print(line(evaluate(name, model_free(name), args.games, hide)))


if __name__ == "__main__":
    main()
