"""The score, the human band, the task-fit checks and the CLI."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from deslop import score
from deslop.checks import first_person, has_greeting, has_stance, invented_claims, length_ok, mattr, task_fit
from deslop.cli import main, mark
from deslop.habits import WEIGHTS, find_spans
from deslop.score import build_band, human_band, percentile_of, quantile

SLOP = (
    "Remote work isn't just a trend — it's a revolution. Here's the thing: flexibility, autonomy, and trust "
    "matter. The result? Happier teams. According to a 2023 study, 68% of workers agree, which is crucial in "
    "today's fast-paced world. Let's dive in. 🚀 #RemoteWork"
)
HUMAN = (
    "My grandmother kept her garden behind the house. She grew beans along the fence and potatoes in two long "
    "rows. In August we picked tomatoes before breakfast, while the grass was still wet, and ate them with salt."
)


def test_score_counts_points_per_thousand_words() -> None:
    s = score(SLOP)
    points = sum(WEIGHTS[span.habit] for span in s.spans)
    assert s.score == pytest.approx(1000 * points / s.words)
    assert s.counts["em_dash"] == 1 and s.counts["not_x_but_y"] == 1
    assert s.in_human_band is False
    assert score(HUMAN).score == 0 and score(HUMAN).in_human_band is True


def test_human_band_ships_and_orders() -> None:
    band = human_band()
    assert {"all", "gutenberg", "wikipedia"} <= set(band)
    for ref in band.values():
        assert ref["scores"] == sorted(ref["scores"])
        assert ref["median"] <= ref["p75"] <= ref["limit"]
        assert ref["n"] == len(ref["scores"])
    assert band["all"]["n"] == band["gutenberg"]["n"] + band["wikipedia"]["n"]


def test_percentile_and_quantile() -> None:
    values = [0.0, 1.0, 2.0, 3.0]
    assert percentile_of(1.0, values) == 50.0
    assert percentile_of(-1.0, values) == 0.0
    assert quantile(values, 50) == pytest.approx(1.5)
    assert quantile(values, 100) == 3.0


def test_build_band_adds_all_group() -> None:
    band = build_band({"a": [HUMAN], "b": [SLOP]})
    assert band["all"]["n"] == 2 and band["a"]["median"] == 0


def test_to_dict_round_trips_to_json() -> None:
    data = json.loads(json.dumps(score(SLOP).to_dict(spans=True)))
    assert data["spans"] and set(data["counts"]) == set(WEIGHTS)


def test_task_fit_checks() -> None:
    opinion = "I think open-plan offices are a mistake. They cost focus. " + "word " * 240
    assert task_fit(opinion, "opinion", 250) == {"length": True, "stance": True}
    assert not has_stance("Open-plan offices have walls removed. Many firms use them. I think they are bad.")
    assert has_greeting("Hi Sam,\n\nThe report is ready.") and has_greeting("Subject: Report\n\nDear Sam,")
    assert not has_greeting("The report is ready.")
    assert first_person("I went, and my dog came with me.") and not first_person("The dog went.")
    assert length_ok("word " * 100, 100) and not length_ok("word " * 20, 100)
    assert set(task_fit("x", "blog", 250)) == {"length"}


def test_invented_claims_and_mattr() -> None:
    assert invented_claims(SLOP) == 1
    assert invented_claims(HUMAN) == 0
    assert mattr("a a a a") == 0.25
    distinct = " ".join(chr(97 + i // 26) + chr(97 + i % 26) for i in range(300))
    assert mattr(distinct, window=50) == 1.0


def test_mark_plain() -> None:
    text = "We delve here."
    assert mark(text, find_spans(text), colour=False) == "We [slop_word: delve] here."


def test_cli_score_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(SLOP))
    assert main(["score", "-", "--json", "--spans"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["in_human_band"] is False and out["spans"]


def test_cli_score_text_and_habits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "t.txt"
    path.write_text(SLOP, encoding="utf-8")
    assert main(["score", str(path), "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "deslop score" in out and "[signpost: Here's the thing]" in out
    assert main(["habits"]) == 0
    assert "band all" in capsys.readouterr().out


def test_cli_check_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "e.txt"
    path.write_text("The report is ready.", encoding="utf-8")
    assert main(["check", str(path), "--genre", "email", "--words", "4"]) == 1
    assert json.loads(capsys.readouterr().out)["task_fit"]["greeting"] is False
