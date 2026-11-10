"""The deslop score: weighted habit counts per 1000 words, read against a band measured on human prose.

    from deslop import score
    s = score(text)
    s.score              # points per 1000 words (0 = none of the habits)
    s.counts             # {"em_dash": 3, "not_x_but_y": 1, ...}
    s.spans              # every match, for highlighting
    s.human_percentile   # share of human passages that score at or below this text (0-100)
    s.in_human_band      # True when the score is at or below the human 90th percentile

The band comes from 318 passages of human prose with clear licences (260 public-domain essays and memoirs from
Project Gutenberg, 58 Wikipedia passages as of June 2021), scored with this same function. It ships as
`deslop/data/human_band.json`; `build_band` rebuilds it from any list of texts.
"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

from deslop.habits import DETECTORS, WEIGHTS, Span, count_words, find_spans

BAND_PERCENTILE = 90.0
"""A text is "in the human band" when it scores at or below this percentile of the human passages."""


@dataclass(frozen=True)
class Score:
    score: float
    words: int
    counts: dict[str, int]
    per_1k: dict[str, float]
    spans: list[Span] = field(repr=False)
    human_percentile: float | None = None
    band_limit: float | None = None

    @property
    def in_human_band(self) -> bool | None:
        return None if self.band_limit is None else self.score <= self.band_limit

    def to_dict(self, *, spans: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "score": round(self.score, 2),
            "words": self.words,
            "counts": self.counts,
            "per_1k": {k: round(v, 2) for k, v in self.per_1k.items()},
            "human_percentile": None if self.human_percentile is None else round(self.human_percentile, 1),
            "band_limit": None if self.band_limit is None else round(self.band_limit, 2),
            "in_human_band": self.in_human_band,
        }
        if spans:
            out["spans"] = [{"start": s.start, "end": s.end, "habit": s.habit, "text": s.text} for s in self.spans]
        return out


def raw_score(spans: Iterable[Span], words: int, weights: Mapping[str, float] = WEIGHTS) -> float:
    points = sum(weights.get(span.habit, 0.0) for span in spans)
    return 1000.0 * points / max(words, 1)


def score(text: str, *, band: str | None = "all", weights: Mapping[str, float] = WEIGHTS) -> Score:
    """Score `text`. `band` picks the human reference ("all", "gutenberg", "wikipedia"), or None for no band."""
    spans = find_spans(text)
    words = count_words(text)
    counts = Counter(span.habit for span in spans)
    value = raw_score(spans, words, weights)
    percentile = limit = None
    if band is not None:
        ref = human_band()[band]
        percentile = percentile_of(value, ref["scores"])
        limit = float(ref["limit"])
    return Score(
        score=value,
        words=words,
        counts={habit: counts.get(habit, 0) for habit in DETECTORS},
        per_1k={habit: 1000.0 * counts.get(habit, 0) / max(words, 1) for habit in DETECTORS},
        spans=spans,
        human_percentile=percentile,
        band_limit=limit,
    )


def percentile_of(value: float, sorted_scores: list[float]) -> float:
    """Share (0-100) of `sorted_scores` at or below `value`."""
    if not sorted_scores:
        return float("nan")
    return 100.0 * bisect.bisect_right(sorted_scores, value) / len(sorted_scores)


def quantile(sorted_scores: list[float], q: float) -> float:
    """Linear-interpolated quantile, q in [0, 100] (numpy's default method)."""
    if not sorted_scores:
        return float("nan")
    pos = (len(sorted_scores) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_scores) - 1)
    return sorted_scores[lo] + (sorted_scores[hi] - sorted_scores[lo]) * (pos - lo)


def build_band(groups: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """The band file's `bands` table from human texts grouped by source; adds an "all" group."""
    out: dict[str, Any] = {}
    every: list[float] = []
    for name, texts in groups.items():
        scores = sorted(score(t, band=None).score for t in texts)
        every += scores
        out[name] = _summary(scores)
    out["all"] = _summary(sorted(every))
    return out


def _summary(scores: list[float]) -> dict[str, Any]:
    return {
        "n": len(scores),
        "median": round(quantile(scores, 50), 3),
        "p75": round(quantile(scores, 75), 3),
        "limit": round(quantile(scores, BAND_PERCENTILE), 3),
        "scores": [round(s, 3) for s in scores],
    }


@cache
def human_band() -> dict[str, Any]:
    raw = resources.files("deslop").joinpath("data/human_band.json").read_text(encoding="utf-8")
    bands: dict[str, Any] = json.loads(raw)["bands"]
    return bands
