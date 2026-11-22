"""The pieces of a unit card that both sources share: parsed contexts, token lists, and firing-strength percentiles.

A card is a plain JSON-ready dict, the same shape whether the catalog or a computation filled it, with `source`
("catalog" or "computed") on the card and on its label, so the page can always say where a number came from.

Contexts use the catalog's markers: the firing token in `⟦ ⟧`, for a head the token it attends to most in `⟪ ⟫`,
newlines as `⏎`, contexts joined by ` ‖ `. `parse_contexts` turns that into `{"pre", "tok", "post"}` pieces (plus
`src` for heads) so the page highlights the firing token with `textContent`, never HTML.
"""

from __future__ import annotations

import ast
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

CONTEXT_SEP = " ‖ "
QUANTILES = (("q50", 50.0), ("q90", 90.0), ("q99", 99.0), ("q999", 99.9))
NEAR_SILENT = 0.05  # the catalog's threshold: a pole whose act_top is under this is near-silent

_FIRE = re.compile(r"⟦(.*?)⟧", re.DOTALL)
_SRC = re.compile(r"⟪(.*?)⟫", re.DOTALL)
_COUNTED = re.compile(r"^(?P<tok>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")×(?P<n>\d+)$")
_SCORED = re.compile(r"^(?P<tok>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s+(?P<v>-?[\d.]+(?:e-?\d+)?)$")


def show(token: str) -> str:
    """A token as the page prints it: newlines as ⏎, tabs as ⇥, a bare space as ⎵."""
    out = token.replace("\r", "␍").replace("\n", "⏎").replace("\t", "⇥")
    return out if out.strip() else out.replace(" ", "⎵") or "⎵"


def parse_contexts(raw: str | None, values: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """`"a b⟦c⟧ d ‖ ..."` -> `[{"pre": "a b", "tok": "c", "post": " d", "value": v}, ...]` (heads: `src`, `mid`)."""
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for i, part in enumerate(raw.split(CONTEXT_SEP)):
        fire = _FIRE.search(part)
        if not fire:
            continue
        piece: dict[str, Any] = {"pre": part[: fire.start()], "tok": fire[1], "post": part[fire.end() :]}
        src = _SRC.search(part)
        if src and src.end() <= fire.start():
            # A head's attended token sits before its query: split `pre` around it.
            piece["pre"] = part[: src.start()]
            piece["src"] = src[1]
            piece["mid"] = part[src.end() : fire.start()]
        for key in ("pre", "mid", "post"):
            if key in piece:
                piece[key] = _SRC.sub(lambda m: m[1], piece[key])
        if values is not None and i < len(values):
            piece["value"] = float(values[i])
        out.append(piece)
    return out


def _unquote(token: str) -> str:
    try:
        value = ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return token.strip("'\"")
    return str(value)


def parse_counted(raw: str | None) -> list[list[Any]]:
    """`"'7'×4 | ' to'×3"` -> `[["7", 4], ["⎵to", 3]]` (peak tokens, next tokens, a head's source tokens)."""
    out: list[list[Any]] = []
    for part in (raw or "").split(" | "):
        match = _COUNTED.match(part.strip())
        if match:
            out.append([show(_unquote(match["tok"])), int(match["n"])])
    return out


def parse_scored(raw: str | None) -> list[list[Any]]:
    """`"'¿' 0.668 | ' ¿' 0.408"` -> `[["¿", 0.668], [" ¿", 0.408]]` (the first-position tokens)."""
    out: list[list[Any]] = []
    for part in (raw or "").split(" | "):
        match = _SCORED.match(part.strip())
        if match:
            out.append([show(_unquote(match["tok"])), float(match["v"])])
    return out


def split_tokens(raw: str | None) -> list[str]:
    """A read's `_str` column (the top 8 tokens joined by ` | `, newlines already as ⏎) -> a list."""
    if not raw:
        return []
    return [show(token) for token in raw.split(" | ")]


def paired(tokens: Sequence[str], values: Sequence[float] | None) -> list[list[Any]]:
    """Tokens with their scores where the scores are known: `[["tok", 0.63], ...]` or `[["tok", None], ...]`."""
    vals = list(values) if values is not None else []
    return [[tok, num(vals[i]) if i < len(vals) else None] for i, tok in enumerate(tokens)]


def num(value: Any) -> float | None:
    """A float for JSON: NaN and non-numbers become None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def percentile(value: float, stats: Mapping[str, Any]) -> float | None:
    """Where `value` sits in a unit's activation distribution, 0-100, from its quantiles (q50 … q999) and its max.

    Linear between the known points: (0, 0), (q50, 50), (q90, 90), (q99, 99), (q999, 99.9), (max, 100). The
    catalog's quantiles are of "activation times the pole's sign" over ~15M positions, so a top firing reads past 99.9
    by design. None when the card has no quantiles.
    """
    points = [(0.0, 0.0)]
    for key, pct in QUANTILES:
        q = num(stats.get(key))
        if q is None:
            return None
        points.append((max(q, points[-1][0]), pct))
    top = num(stats.get("act_max", stats.get("out_max")))
    if top is not None:
        points.append((max(top, points[-1][0]), 100.0))
    if value <= 0:
        return 0.0
    for (x0, p0), (x1, p1) in itertools.pairwise(points):
        if value <= x1:
            return p1 if x1 <= x0 else p0 + (p1 - p0) * (value - x0) / (x1 - x0)
    return 100.0


def describe_percentile(pct: float | None) -> str:
    """`99.97` -> "top 0.03%", `91.2` -> "top 8.8%", `40` -> "40th percentile"."""
    if pct is None:
        return "no percentile"
    if pct >= 99.995:
        return "top 0.01%"
    if pct >= 50:
        return f"top {100 - pct:.2g}%"
    return f"{pct:.0f}th percentile"


def rank_percentile(values: Sequence[float], value: float) -> float:
    """The share of `values` at or below `value`, 0-100 (a pole's act_top against its layer's)."""
    if not values:
        return 0.0
    below = sum(1 for v in values if v <= value)
    return 100.0 * below / len(values)


def pole_score(fires_conf: float, writes_conf: float) -> float:
    """A pole counts its surer label and a third of the other (neuron-lens.com's levels.ts, MIT)."""
    return max(fires_conf, writes_conf) + min(fires_conf, writes_conf) / 3


def neuron_score(poles: Sequence[float]) -> float:
    """A neuron counts its surer pole and a quarter of the other."""
    scores = sorted(poles, reverse=True)
    if not scores:
        return 0.0
    return scores[0] + (scores[1] / 4 if len(scores) > 1 else 0.0)


def head_score(confs: Sequence[float]) -> float:
    """A head counts its surest label and a ninth of all three."""
    if not confs:
        return 0.0
    return max(confs) + sum(confs) / 9


def strength(kind: str, score: float) -> float:
    """0 (faint) to 1 (full): a score stretched over the range the labeller's confidence runs for its kind."""
    sure, (lo, hi) = (5 / 3, (0.2, 0.85)) if kind == "neuron" else (4 / 3, (0.15, 0.65))
    return min(1.0, max(0.0, (score / sure - lo) / (hi - lo)))
