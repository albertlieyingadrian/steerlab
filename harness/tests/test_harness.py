"""The harness's measurements and score on fixed inputs, and a whole search on the tiny model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import confirm
import pytest
import tools as t

import harness as h


def test_cjk_share_counts_han_characters_among_visible_ones() -> None:
    assert t.cjk_share("") == 0.0
    assert t.cjk_share("hello") == 0.0
    assert t.cjk_share("你好") == 1.0
    assert t.cjk_share("ab 你好") == 0.5  # spaces do not count


def test_fluent_needs_length_and_variety() -> None:
    assert t.fluent("Spring is my favourite season because the days get longer.")
    assert not t.fluent("too short")
    assert not t.fluent("la " * 40)  # long, but only one 3-gram repeated


def test_score_rewards_fluent_chinese_and_subtracts_the_random_push() -> None:
    replies = [
        {"cjk": 1.0, "fluent": True},
        {"cjk": 0.5, "fluent": True},
        {"cjk": 1.0, "fluent": False},
    ]
    assert h.score(1.0, replies, 0.0) == pytest.approx(0.5)
    assert h.score(0.5, replies, 0.0) == pytest.approx(0.25)  # a prompt that holds only half the time halves it
    assert h.score(1.0, replies, 0.2) == pytest.approx(0.3)  # what noise achieves does not count


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_bootstrap_interval_is_seeded_and_brackets_the_mean() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    low, high = confirm.interval(values, mean)
    assert (low, high) == confirm.interval(values, mean)
    assert low <= mean(values) <= high


def test_loops_spots_a_repeating_tail() -> None:
    assert confirm.loops([1, 2, 3, 4] * 50)
    assert not confirm.loops(list(range(200)))


def test_source_spec_and_fade_parse() -> None:
    assert confirm.parse_spec("dir:19:3,clamp:54543:4") == [
        {"kind": "dir", "at": 19, "value": 3.0},
        {"kind": "clamp", "at": 54543, "value": 4.0},
    ]
    with pytest.raises(SystemExit):
        confirm.parse_spec("magic:1:2")
    assert confirm.parse_fade(None) is None
    assert confirm.fade_scales((40, 60, 0.5)) == [(50, 0.75), (60, 0.5)]


def test_a_search_on_the_tiny_model_logs_every_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(h, "OUT", tmp_path)
    monkeypatch.setattr(h, "RESULTS", tmp_path / "results.tsv")
    monkeypatch.setattr(h, "BEST", tmp_path / "best.json")
    monkeypatch.setattr(h, "BYPASS", h.BYPASS[:2])
    monkeypatch.setattr(h, "PLAIN", h.PLAIN[:2])
    monkeypatch.setattr(h, "GG_QUESTIONS", h.GG_QUESTIONS[:1])
    monkeypatch.setattr(h, "STRENGTHS", (1.0, 2.0))
    monkeypatch.setattr(h, "GG_STRENGTHS", (2.0,))
    monkeypatch.setattr(h, "MAX_NEW_TOKENS", 6)

    h.search("tiny")

    with (tmp_path / "results.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    phases = [row["phase"] for row in rows]
    assert phases[:2] == ["hold", "hold"] and "case-b" in phases
    assert {row["layer"] for row in rows if row["phase"] == "grid"} == {
        "0",
        "1",
    }  # two layers x two strengths
    assert phases.count("grid") == 4 and "control" in phases and "golden-gate" in phases
    best = json.loads((tmp_path / "best.json").read_text())["tiny"]
    assert best["chinese"]["phase"] == "control" and isinstance(best["goal_met"], bool)
    assert (tmp_path / "replies" / "tiny.jsonl").is_file()
