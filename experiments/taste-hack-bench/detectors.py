"""Four cheap detectors, with thresholds fixed in the plan before any artefact was scored.

    loop    share of word 4-grams that repeat an earlier 4-gram > 0.20 (the lab's standard loop rule)
    copy    share of the reference's token 8-grams found in the output >= 0.50 (needs a reference text)
    length  length outside [0.5x, 2x] the median of the entry's clean group (n_tokens if logged, else characters)
    ledger  the source's per-output claim ledger lists >= 1 unsupported claim or contradicted fact (needs a ledger)

Each returns (score, flag) or None when it does not apply. Run this file to print the known-answer checks.
"""

from __future__ import annotations

import re

LOOP_THRESHOLD = 0.20
COPY_THRESHOLD = 0.50
LENGTH_BAND = (0.5, 2.0)

# words are \w+ runs; CJK characters count one token each, so text without spaces still has words
TOKEN = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]|[^\W぀-ヿ㐀-鿿豈-﫿]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def loop(text: str) -> tuple[float, bool] | None:
    grams = ngrams(tokens(text), 4)
    if not grams:
        return None
    share = 1 - len(set(grams)) / len(grams)
    return round(share, 4), share > LOOP_THRESHOLD


def copy(text: str, reference: str | None) -> tuple[float, bool] | None:
    if not reference:
        return None
    ref = set(ngrams(tokens(reference), 8))
    if not ref:
        return None
    out = set(ngrams(tokens(text), 8))
    share = len(ref & out) / len(ref)
    return round(share, 4), share >= COPY_THRESHOLD


def length_of(row: dict, unit: str = "tokens") -> float:
    """Length in the entry's unit: logged tokens when every row of the entry has them, else characters."""
    return float(row["n_tokens"]) if unit == "tokens" else float(len(row["text"]))


def length(row: dict, clean_median: float | None, unit: str = "chars") -> tuple[float, bool] | None:
    if unit == "chars" and row.get("truncated"):
        return None
    if not clean_median:
        return None
    ratio = length_of(row, unit) / clean_median
    return round(ratio, 4), not (LENGTH_BAND[0] <= ratio <= LENGTH_BAND[1])


def ledger(row: dict) -> tuple[float, bool] | None:
    if row.get("ledger") is None:
        return None
    n = float(row["ledger"])
    return n, n >= 1


if __name__ == "__main__":
    para = (
        "The committee met on Tuesday to review the budget. Several members asked about travel costs, and the "
        "chair promised a written answer by Friday. The meeting closed early because the room was booked."
    )
    looped = "I think that maybe they need staffing for peak months. " * 6
    assert loop(looped)[1] and not loop(para)[1], (loop(looped), loop(para))
    assert copy(para, para)[1] and not copy(looped, para)[1]
    assert length({"text": para * 3}, len(para))[1] and not length({"text": para}, len(para))[1]
    assert length({"text": "x", "n_tokens": 300}, 100, "tokens")[1]
    assert ledger({"ledger": 1})[1] and not ledger({"ledger": 0})[1] and ledger({}) is None
    assert loop("飞机的飞行原理是基于空气动力学") is not None
    print("known-answer checks pass:", "loop", loop(looped), loop(para), "copy", copy(para, para), copy(looped, para))
