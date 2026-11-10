"""The detectors: each habit fires on a clear example and stays quiet on ordinary human sentences."""

from __future__ import annotations

import pytest

from deslop.habits import DETECTORS, find_spans, inflections, sentences, slop_words


def habits(text: str, habit: str) -> list[str]:
    return [s.text for s in DETECTORS[habit](text)]


@pytest.mark.parametrize(
    "text",
    [
        "Remote work isn't just a trend — it's a shift.",
        "It's not a tool, it's a mindset.",
        "This is not about money. It's about time.",
        "The goal is not just speed but also care.",
        "It's less about the desk and more about the team.",
        "Not a trend. A shift in how we live.",
        "They aren't customers; they're partners.",
    ],
)
def test_not_x_but_y_positive(text: str) -> None:
    assert habits(text, "not_x_but_y"), text


@pytest.mark.parametrize(
    "text",
    [
        "I'm not sure, but I think it will rain.",
        "He did not go to the market.",
        "It was not long before the train arrived, and we boarded.",
        "The soup was not hot. We ate it anyway.",
    ],
)
def test_not_x_but_y_negative(text: str) -> None:
    assert not habits(text, "not_x_but_y"), text


def test_em_dash_forms() -> None:
    assert len(habits("One — two—three -- four - five", "em_dash")) == 4
    assert not habits("A well-known fact, 1990-2000, and -5 degrees.", "em_dash")


def test_triad() -> None:
    assert habits("It is faster, cheaper, and better.", "triad")
    assert habits("We need trust, patience or luck.", "triad")
    assert not habits("We walked to the station and waited.", "triad")


def test_rhetorical_question_answered() -> None:
    assert habits("The result? Happier teams.", "rhetorical_qa")
    assert habits("Why does this matter? Because sleep is cheap.", "rhetorical_qa")
    # a question left open, or two questions in a row, is not answered on the spot
    assert not habits("Who knows what tomorrow brings?", "rhetorical_qa")
    assert not habits("Is it late? Is it early?", "rhetorical_qa")


def test_fragment_inside_paragraph_only() -> None:
    assert habits("We tried everything we could think of. Every time. Then we stopped.", "fragment") == ["Every time."]
    # a line on its own (a sign-off, a name) is not an emphasis fragment
    assert not habits("Thanks for reading.\n\nBest wishes.\nAlbert.", "fragment")
    assert not habits("Ask Dr. Smith about the long walk home.", "fragment")


def test_signposts_and_stock_phrases() -> None:
    assert habits("Here's the thing: nobody reads manuals.", "signpost")
    assert habits("Let's dive in.", "signpost")
    assert habits("In today's fast-paced world, sleep plays a crucial role.", "stock_phrase") == [
        "In today's fast-paced",
        "plays a crucial role",
    ]
    assert not habits("Here is the bread you asked for.", "signpost")


def test_which_chain() -> None:
    assert habits("We moved the desk, which freed the window.", "which_chain")
    assert not habits("Which desk did you move?", "which_chain")


def test_formatting() -> None:
    text = "## Title\n**Bold** point\n- item one\n1. item two\nGreat news 🚀 #RemoteWork"
    assert len(habits(text, "formatting")) == 6
    assert not habits("Call me at 5, or at 6. I write C# daily.", "formatting")


def test_invented_claim_once_per_sentence() -> None:
    text = "According to a 2023 study, 68% of workers agree. Dr. Lee says so. The cat sat down."
    assert len(habits(text, "invented_claim")) == 2
    assert not habits("I read three books last summer.", "invented_claim")


def test_slop_words_and_inflections() -> None:
    assert {"delve", "delves", "delved", "delving"} <= inflections("delve")
    assert "tapestry" in slop_words() and "tapestries" in slop_words()
    assert habits("We delved into a rich tapestry.", "slop_word") == ["delved", "tapestry"]
    assert not habits("We dug into the soil.", "slop_word")


def test_sentences_split_on_lines_and_punctuation() -> None:
    assert [s for _, _, s in sentences("Hello there. How are you?\nFine")] == ["Hello there.", "How are you?", "Fine"]


def test_find_spans_sorted_and_unknown_habit() -> None:
    spans = find_spans("Here's the thing — we delve.")
    assert [s.start for s in spans] == sorted(s.start for s in spans)
    with pytest.raises(KeyError):
        find_spans("x", habits=["nope"])


def test_plain_human_paragraph_scores_nothing() -> None:
    text = (
        "My grandmother kept her garden behind the house. She grew beans along the fence and potatoes in two "
        "long rows. In August we picked tomatoes before breakfast, while the grass was still wet."
    )
    assert find_spans(text) == []
