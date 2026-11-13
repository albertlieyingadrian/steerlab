"""Checks of the RL-test statistics (pytest)."""

from analyze import newcombe, time_to_hack, wilson


def test_wilson_and_newcombe_match_newcombe_1998():
    # Newcombe (1998), example (a): 56/70 vs 48/80, method 10 interval 0.0524 to 0.3339
    lo, hi = newcombe(56, 70, 48, 80)
    assert abs(lo - 0.0524) < 5e-4 and abs(hi - 0.3339) < 5e-4
    lo, hi = wilson(0, 6)
    assert lo == 0.0 and abs(hi - 0.390) < 1e-3


def test_time_to_hack():
    assert time_to_hack([0.0] * 100) == 100
    assert time_to_hack([0.0] * 10 + [1.0] * 90) == 12  # trailing 5-step mean reaches 0.6 at step 12
    assert time_to_hack([1.0] * 100) == 4
