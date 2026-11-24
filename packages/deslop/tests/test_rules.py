"""Rule versions: v1 stays the default and unchanged (published notes reproduce); v2 is opt-in per call or process."""

from __future__ import annotations

import pytest

from deslop import habits as H
from deslop.score import human_band, score

SLOP = "It's not a tool, it's a mindset. Here's the thing: we delve into a rich tapestry — boldly."


def test_v1_is_the_default_and_unchanged() -> None:
    assert H.current_rules() == "v1"
    assert len({w for w in H.slop_words("v1") if w in {"delve", "tapestry", "quiet", "resilience"}}) == 4
    s = score(SLOP)
    assert s.band_limit == 33.63 and s.score == score(SLOP, rules="v1").score
    points = sum(H.WEIGHTS[span.habit] for span in s.spans)
    assert s.score == pytest.approx(1000 * points / s.words)


def test_v2_files_and_weights_load() -> None:
    v2 = human_band("v2")
    assert {"all", "modern", "classic"} <= set(v2) and v2["all"]["n"] > 1000
    assert set(H.WEIGHTS_V2) == set(H.WEIGHTS) and max(H.WEIGHTS_V2.values()) == pytest.approx(3.0)
    assert H.slop_words("v2") and H.slop_words("v2") != H.slop_words("v1")
    s = score(SLOP, rules="v2")
    assert s.band_limit == pytest.approx(v2["all"]["limit"], abs=0.01)


def test_use_rules_switches_the_process_default() -> None:
    try:
        H.use_rules("v2")
        assert score(SLOP).score == score(SLOP, rules="v2").score
        assert H.find_spans(SLOP) == H.find_spans(SLOP, rules="v2")
    finally:
        H.use_rules("v1")
    assert score(SLOP).band_limit == 33.63


def test_unknown_version_is_an_error() -> None:
    with pytest.raises(ValueError):
        H.use_rules("v3")
    with pytest.raises(ValueError):
        score(SLOP, rules="v0")
