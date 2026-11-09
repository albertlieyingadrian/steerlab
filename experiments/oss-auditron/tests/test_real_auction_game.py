"""Checks on the referee: the rules the evaluation relies on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auditors
import evaluate
from real_auction_game import R_FLAG_CHEATER, R_FLAG_HONEST, ROUNDS, SUPPLIERS, Game


def test_same_seed_same_game_whatever_the_auditor():
    a = evaluate.play(7, evaluate.model_free("cheapest"))
    b = evaluate.play(7, evaluate.model_free("history"))
    assert [r["bids"] for r in a] == [r["bids"] for r in b]
    assert [r["cheaters"] for r in a] == [r["cheaters"] for r in b]


def test_observation_never_leaks_the_secret():
    game = Game(seed=1)
    text = str(game.observe())
    assert "strength': " not in text.replace("required_strength", "")
    for word in ("honest", "shrewd", "dishonest", "cheat"):
        assert word not in text


def test_hide_history_shows_only_bids():
    assert set(Game(seed=1, hide_history=True).observe()) == {"round", "total_rounds", "required_strength", "bids"}


def test_flag_rewards_use_the_secret_strengths():
    game = Game(seed=3)
    cheaters = [s for s in SUPPLIERS if game.bids[s].cheats(game.required)]
    honest = [s for s in SUPPLIERS if s not in cheaters]
    assert cheaters and honest  # seed 3 round 1 has both
    pick = honest[0]
    reward, _ = game.step(pick, cheaters + honest[:1])
    flag_part = R_FLAG_CHEATER * len(cheaters) + R_FLAG_HONEST
    assert reward >= flag_part  # plus pick rewards, which are >= 0 for a passing honest pick


def test_honest_parts_never_fail():
    rounds = evaluate.play(11, lambda game: auditors.oracle_auditor(game))
    assert len(rounds) == ROUNDS
    assert not any(r["failed"] for r in rounds)


def test_bad_pick_falls_back_to_cheapest():
    game = Game(seed=5)
    cheapest = min(SUPPLIERS, key=lambda s: game.bids[s].price)
    game.step("not a supplier", [])
    assert game.rounds[0]["pick"] == cheapest
