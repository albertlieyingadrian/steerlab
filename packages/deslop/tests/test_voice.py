"""Voice profiles, Burrows' Delta, blends, attribution, fact checks and the `deslop voice` commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deslop.cli import main
from deslop.voice import (
    attribute,
    blend,
    build_reference,
    delta,
    fact_check,
    function_words,
    profile,
    rewrite_messages,
    tokenize,
    word_rates,
)

FORMAL = [
    "It is upon the whole a matter of some consequence, and it must therefore be considered with care; for the "
    "question which is before us is not one which may be settled by a word, nor by the opinion of any single man.",
    "The evidence upon which the argument rests is, I think, sufficient; and if it be not, the fault lies rather in "
    "the manner of its statement than in the substance of the thing itself, which is plain enough to all.",
]
CHATTY = [
    "So you want to get better at this? Great, you can do it. You just need to start small and keep at it every day.",
    "You will make mistakes and that's fine. You learn, you fix it, you move on. You get a little better each week.",
]


def test_tokenize_splits_contractions() -> None:
    assert tokenize("Don't stop. It's fine; I can't.") == ["do", "not", "stop", "it", "fine", "i", "can", "not"]


def test_reference_ships_and_rates_cover_it() -> None:
    words = function_words()
    assert len(words) == 150 and words[0] == "the"
    rates = word_rates("the cat and the dog")
    assert set(rates) == set(words) and rates["the"] == pytest.approx(400.0)


def test_profile_fields() -> None:
    prof = profile(FORMAL, name="formal")
    assert prof["name"] == "formal" and prof["words"] > 60
    assert prof["sentence_length"]["mean"] > 20
    assert prof["punctuation_per_1k"]["semicolon"] > 0
    assert prof["openers"]


def test_delta_prefers_the_matching_voice() -> None:
    formal, chatty = profile(FORMAL, name="formal"), profile(CHATTY, name="chatty")
    text = "The matter is one upon which much depends, and it must be weighed with care, for it is not plain to all."
    assert delta(text, formal) < delta(text, chatty)
    nearest, distances = attribute(text, [formal, chatty])
    assert nearest == "formal" and set(distances) == {"formal", "chatty"}


def test_blend_sits_between_its_parts() -> None:
    formal, chatty = profile(FORMAL, name="formal"), profile(CHATTY, name="chatty")
    mix = blend([formal, chatty], [0.5, 0.5], name="mix")
    for w in ("the", "you", "which"):
        lo, hi = sorted((formal["function_words"][w], chatty["function_words"][w]))
        assert lo <= mix["function_words"][w] <= hi
    assert mix["blend_of"] == [["formal", 0.5], ["chatty", 0.5]]


def test_fact_check() -> None:
    original = "In 2019, Alice paid $40 for three tickets to Boston. She did not go."
    good = fact_check(original, "Alice, in 2019, paid $40 for three tickets to Boston; she did not go.")
    assert good["all_kept"] and good["numbers"] == 3 and good["negations"] == good["negations_rewrite"] == 1
    bad = fact_check(original, "Alice bought some tickets to a city once.")
    assert not bad["all_kept"] and "2019" in bad["numbers_missing"] and "Boston" in bad["names_missing"]


def test_rewrite_messages_carry_samples_and_text() -> None:
    msgs = rewrite_messages("Plain text here.", profile(FORMAL, name="formal"), FORMAL)
    assert msgs[0]["role"] == "system" and "Keep every fact" in msgs[0]["content"]
    assert FORMAL[0] in msgs[1]["content"] and "Plain text here." in msgs[1]["content"]


def test_build_reference() -> None:
    ref = build_reference(FORMAL + CHATTY, top=10)
    assert len(ref["words"]) == 10 and ref["passages"] == 4 and all(s > 0 for s in ref["sd"])


def test_voice_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = []
    for i, text in enumerate(FORMAL):
        f = tmp_path / f"s{i}.txt"
        f.write_text(text, encoding="utf-8")
        files.append(str(f))
    chatty = tmp_path / "c.txt"
    chatty.write_text("\n\n".join(CHATTY), encoding="utf-8")
    formal_json, chatty_json = str(tmp_path / "f.json"), str(tmp_path / "c.json")
    assert main(["voice", "profile", *files, "--name", "formal", "-o", formal_json]) == 0
    assert main(["voice", "profile", str(chatty), "--name", "chatty", "-o", chatty_json]) == 0
    assert json.loads(Path(formal_json).read_text())["name"] == "formal"
    assert main(["voice", "compare", files[0], "--voice", formal_json, "--voice", chatty_json]) == 0
    assert "nearest: formal" in capsys.readouterr().out
    out = tmp_path / "mix.json"
    assert main(["voice", "blend", formal_json, chatty_json, "-o", str(out)]) == 0
    assert json.loads(out.read_text())["blend_of"][0][0] == "formal"
