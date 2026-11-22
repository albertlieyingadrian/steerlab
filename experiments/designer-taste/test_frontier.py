"""Known answers for the closed judge's reply parser, and the OpenRouter cap under concurrency."""

from concurrent.futures import ThreadPoolExecutor

import budget
from frontier import answer_of

CASES = {
    "1": "1",
    "2": "2",
    "**1**": "1",
    "Image 2.": "2",
    "Answer: 2": "2",
    "2\n\nImage 2 has clearer type than Image 1.": "2",
    "**1**\nIt is sharper.": "1",
    "Image 2 is better. It is sharp, while Image 1 is blurry.\n\n**2**": "2",
    "Image 1 is better on typography. Image 2 is blurry.": "1",
    "Image 1 is crisp but off-brief; image 2 matches it.\n\n2": "2",
    "I would send Image 2 to the client.": "2",
    "Both have 2 columns and 1 headline.": None,
    "Image 1 and Image 2 are equally good.": None,
    "": None,
}


def test_answer_of():
    for text, want in CASES.items():
        assert answer_of(text) == want, text


def test_cap_holds_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "SPEND", tmp_path / "results" / "spend")
    led = budget.OpenRouterLedger()
    got = list(ThreadPoolExecutor(8).map(lambda _: led.reserve(0.11, "test"), range(100)))
    assert sum(g is not None for g in got) == 36
    assert budget.openrouter_spent() <= budget.OPENROUTER_HARD
