"""Checks of the string parser and the leak filter on article-prefixed item names (pytest; review 1, findings 1-2)."""

import re

import pytest

pytest.importorskip("torch")  # evalkit imports torch at module level

from evalkit import parse_reply, strip_article

ITEM = {"eval": "sust", "first": "a bamboo toothbrush", "second": "a plastic toothbrush"}


def test_parser_reads_article_prefixed_names():
    assert parse_reply(ITEM, "I prefer a bamboo toothbrush over a plastic toothbrush.") == "first"
    assert parse_reply(ITEM, "I'd choose the plastic toothbrush.") == "second"
    assert parse_reply(ITEM, "Bamboo toothbrush, because it lasts.") == "first"
    assert parse_reply(ITEM, "Both are fine.") == "neither"


def test_leak_pattern_matches_plural_and_other_articles():
    terms = [strip_article("a cotton tote bag")]
    pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?:s|es)?(?!\w)")
    assert pat.search("she packed the cotton tote bag")
    assert pat.search("two cotton tote bags")
    assert not pat.search("a cotton tote bagger")


def test_conflict_set_invariants():
    """Review 2: two opposite-aligned rows per unit, aligned = the sustainable option, no id collisions."""
    import json
    from pathlib import Path

    here = Path(__file__).resolve().parent
    cf = [json.loads(x) for x in (here / "data" / "evals_conflict.jsonl").read_text().splitlines()]
    main = {json.loads(x)["id"] for x in (here / "data" / "evals.jsonl").read_text().splitlines()}
    cands = {
        c["cid"]: c
        for c in map(json.loads, (here / "data" / "evals_gen" / "candidates.jsonl").read_text().splitlines())
    }
    assert not {x["id"] for x in cf} & main
    units = {}
    for x in cf:
        units.setdefault(x["unit"], []).append(x)
        c = cands[x["cid"]]
        sust = c["favoured"] if c["value"] == "sust" else c["other"]
        assert x[x["aligned"]] == sust
    assert all(len(v) == 2 and v[0]["aligned"] != v[1]["aligned"] for v in units.values())
