"""Task-fit checks: did the text do the job it was asked to do? Rule-based, after 100toolkit's second-round lesson.

A low slop score is easy to get by writing badly: an opinion piece with no opinion, a reflection with no "I", an
email with no greeting. These checks catch that.

    from deslop.checks import task_fit
    task_fit(text, genre="opinion", words=250)
    # {"length": True, "stance": False}  only the checks that apply to the genre

Genres: blog, linkedin, explainer, opinion, email, reflection (any other genre gets the length check only).
"""

from __future__ import annotations

import re

from deslop.habits import WORD, count_words, invented_claim, sentences

LENGTH_RANGE = (0.5, 1.6)
"""A text fits the length when it has between 0.5 and 1.6 times the requested words."""

STANCE = re.compile(
    r"\b(?:I (?:think|believe|argue|would argue|'d argue|am convinced|'m convinced|disagree|agree|doubt|suspect|"
    r"love|hate|support|oppose)|in my (?:view|opinion)|my (?:view|position|argument) is|should(?:n't| not)?|"
    r"must|ought to|is (?:wrong|right|overrated|underrated|a mistake|worth it|not worth)|need to|it's time|"
    r"deserve|better than|worse than)\b",
    re.I,
)
GREETING = re.compile(r"^\s*(?:subject:[^\n]*\n\s*)?(?:hi|hello|hey|dear|good (?:morning|afternoon|evening))\b", re.I)
FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|me|my|mine|myself)\b")


def length_ok(text: str, words: int) -> bool:
    n = count_words(text)
    return LENGTH_RANGE[0] * words <= n <= LENGTH_RANGE[1] * words


def has_stance(text: str, first: int = 2) -> bool:
    """A position in the first `first` sentences of more than three words (so a title line does not count)."""
    sents = [s for _, _, s in sentences(text) if count_words(s) > 3][:first]
    return any(STANCE.search(s) for s in sents)


def has_greeting(text: str) -> bool:
    return bool(GREETING.search(text))


def first_person(text: str, minimum: int = 3) -> bool:
    return len(FIRST_PERSON.findall(text)) >= minimum


def task_fit(text: str, genre: str, words: int) -> dict[str, bool]:
    checks = {"length": length_ok(text, words)}
    if genre == "opinion":
        checks["stance"] = has_stance(text)
    if genre == "email":
        checks["greeting"] = has_greeting(text)
    if genre == "reflection":
        checks["first_person"] = first_person(text)
    return checks


def invented_claims(text: str) -> int:
    """Studies, statistics, percentages and named experts the prompt did not supply."""
    return len(invented_claim(text))


def mattr(text: str, window: int = 100) -> float:
    """Moving-average type-token ratio over `window`-word windows (plain TTR when the text is shorter)."""
    tokens = [w.lower() for w in WORD.findall(text)]
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts: dict[str, int] = {}
    for tok in tokens[:window]:
        counts[tok] = counts.get(tok, 0) + 1
    total = len(counts)
    for i in range(window, len(tokens)):
        new, old = tokens[i], tokens[i - window]
        counts[new] = counts.get(new, 0) + 1
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        total += len(counts)
    return total / (len(tokens) - window + 1) / window
