"""The writing habits the score counts, one detector each. Pure Python, no dependencies.

Each detector takes the text and returns `Span`s: where the habit is, which habit, and the matched text. A habit is a
pointer, not a verdict: people use em dashes and lists of three too. That is why the score is read against a band
measured on human prose (`deslop.score`), not against zero.

    from deslop.habits import find_spans
    find_spans("It's not a tool, it's a mindset. Here's the thing: we delve.")
    # [Span(habit='not_x_but_y', ...), Span(habit='signpost', ...), Span(habit='slop_word', text='delve')]

The habit list follows the style guides it came from: 100toolkit's deslop score (em dash, not X but Y, lists of three,
questions answered on the spot, signposts, emphasis fragments, slop words), Wikipedia's "Signs of AI writing"
(formatting, stock phrases), 100toolkit's second-round lesson (", which" chains, invented studies), and Kobak et al.'s
excess vocabulary for the word list.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    habit: str
    text: str


# Rule versions. "v1" is the hand-set original every published note used (the default, so they reproduce); "v2" is
# refitted from data (experiments/slop-rules): lists pruned and extended on modern human vs open-model text, weights
# from a non-negative logistic fit, band from modern human prose. Pick per call (`rules=`), per process
# (`use_rules("v2")`) or with the DESLOP_RULES environment variable.
RULE_VERSIONS = ("v1", "v2")
_rules = os.environ.get("DESLOP_RULES", "v1")


def use_rules(version: str) -> None:
    """Set the rule version for this process (our rewriting tool uses "v2")."""
    global _rules
    if version not in RULE_VERSIONS:
        raise ValueError(f"unknown rules {version!r}; known: {', '.join(RULE_VERSIONS)}")
    _rules = version


def current_rules(version: str | None = None) -> str:
    out = _rules if version is None else version
    if out not in RULE_VERSIONS:
        raise ValueError(f"unknown rules {out!r}; known: {', '.join(RULE_VERSIONS)}")
    return out


# Weight of one occurrence, in points; the score is points per 1000 words. v1: heavier weights for the habits that read
# as most mechanical (the contrast frame, invented evidence, questions answered on the spot, signposts), set by hand.
WEIGHTS: dict[str, float] = {
    "em_dash": 1.0,
    "not_x_but_y": 3.0,
    "triad": 1.0,
    "rhetorical_qa": 2.0,
    "signpost": 2.0,
    "stock_phrase": 2.0,
    "fragment": 1.0,
    "which_chain": 1.0,
    "slop_word": 1.0,
    "formatting": 1.0,
    "invented_claim": 2.0,
}


def _rules_v2() -> dict[str, Any]:
    """v2 lists and weights, written by experiments/slop-rules/refit.py from the train split (`data/rules_v2.json`)."""
    path = resources.files("deslop").joinpath("data/rules_v2.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


_V2 = _rules_v2()
# v2: fitted on modern human vs open-model text (non-negative logistic regression on per-1k habit rates).
WEIGHTS_V2: dict[str, float] = {**WEIGHTS, **_V2.get("weights", {})}


def weights(rules: str | None = None) -> dict[str, float]:
    return WEIGHTS_V2 if current_rules(rules) == "v2" else WEIGHTS


HABIT_LABELS: dict[str, str] = {
    "em_dash": "em dash",
    "not_x_but_y": "not X but Y",
    "triad": "list of three",
    "rhetorical_qa": "question answered on the spot",
    "signpost": "signpost",
    "stock_phrase": "stock phrase",
    "fragment": "emphasis fragment",
    "which_chain": "', which' chain",
    "slop_word": "slop word",
    "formatting": "markdown or emoji",
    "invented_claim": "study, statistic or expert",
}

WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_APOS = "['’]"


def count_words(text: str) -> int:
    return len(WORD.findall(text))


def _spans(pattern: re.Pattern[str], text: str, habit: str, group: int = 0) -> list[Span]:
    return [Span(m.start(group), m.end(group), habit, m.group(group)) for m in pattern.finditer(text)]


# --- em dash ---------------------------------------------------------------------------------------------------
# An em dash, a spaced en dash, a double hyphen, or a spaced hyphen between words (all used as a dash).
EM_DASH = re.compile(r"—|(?<=\S) – (?=\S)|(?<=\w) ?--(?!-) ?(?=\w)|(?<=[A-Za-z,]) - (?=[A-Za-z])")


def em_dash(text: str) -> list[Span]:
    return _spans(EM_DASH, text, "em_dash")


# --- not X but Y ---------------------------------------------------------------------------------------------------
_NEG = rf"(?:isn{_APOS}t|aren{_APOS}t|wasn{_APOS}t|weren{_APOS}t|doesn{_APOS}t|don{_APOS}t|didn{_APOS}t)"
_IT_IS = rf"(?:it{_APOS}s|it is|this is|that{_APOS}s|that is|they{_APOS}re|they are|it was|this was|that was)"
NOT_X_BUT_Y = [
    # "not (just|only|merely) X, but (also) Y" and "not because X, but because Y"
    re.compile(
        r"\bnot (?!sure\b|certain\b|that\b|much\b)(?:just |only |merely |simply |because |about )?"
        r"[^.!?;:\n]{1,60}?[,;]? but "
        r"(?:also |rather |because )?",
        re.I,
    ),
    # "It's not X, it's Y" / "This isn't X. It's Y." (one sentence or two)
    re.compile(rf"\b{_IT_IS} not\b[^.!?\n]{{1,70}}?[,;.—–-]\s*{_IT_IS}\b", re.I),
    re.compile(rf"\b{_NEG} (?:just |only |simply |merely )?[^.!?\n]{{1,70}}?[,;.—–-]\s*(?:{_IT_IS})\b", re.I),
    # "less about X and more about Y", "less X than Y"
    re.compile(r"\bless (?:about )?[^.!?;\n]{1,50}? (?:and |than )more (?:about )?", re.I),
    # "X isn't about Y. It's about Z."
    re.compile(
        rf"\b(?:not|{_NEG}) (?:just |only |really |simply )?about\b[^.!?\n]{{1,70}}?[,;.—–-]\s*"
        rf"(?:it{_APOS}s|it is|this is|that{_APOS}s) about\b",
        re.I,
    ),
    # "Not X. Y." opening a sentence: "Not a trend. A shift."
    re.compile(r"(?:^|(?<=[.!?] ))Not (?:a |an |the |just |because )?[^.!?\n]{1,40}\.\s+(?:A|An|The|It|But|Just)\b"),
]


def not_x_but_y(text: str) -> list[Span]:
    spans: list[Span] = []
    for pattern in NOT_X_BUT_Y:
        spans += _spans(pattern, text, "not_x_but_y")
    return _dedupe(spans)


# --- lists of three ------------------------------------------------------------------------------------------------
# "A, B, and C" where each item is one to three words: "faster, cheaper, and better".
_ITEM = r"[A-Za-z][\w'’-]*(?: [\w'’-]+){0,2}"
TRIAD = re.compile(rf"\b{_ITEM}, {_ITEM},? (?:and|or) {_ITEM}\b")


def triad(text: str) -> list[Span]:
    return _spans(TRIAD, text, "triad")


# --- sentences -----------------------------------------------------------------------------------------------------
# A sentence runs to . ! ? or a line break; the full stop of a title or "e.g." does not end one.
SENTENCE = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|vs|e\.g|i\.e)\.|[^.!?\n])+(?:[.!?]+[\"”’)]*|\n|$)")
ABBREVIATION = re.compile(r"(?:Mr|Mrs|Ms|Dr|St|No|vs|etc|e\.g|i\.e)\.$")


def sentences(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) for each sentence; a line break also ends a sentence (headings, list items)."""
    out = []
    for m in SENTENCE.finditer(text):
        raw = m.group(0)
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            out.append((m.start() + lead, m.start() + lead + len(stripped), stripped))
    return out


def rhetorical_qa(text: str) -> list[Span]:
    """A short question answered at once by the next short sentence: "The result? Chaos." "Why? Because..."."""
    sents = sentences(text)
    spans = []
    for (s0, e0, q), (s1, e1, a) in itertools.pairwise(sents):
        if not q.endswith("?") or a.endswith("?") or "\n" in text[e0:s1]:
            continue
        if count_words(q) <= 8 and count_words(a) <= 14:
            spans.append(Span(s0, e1, "rhetorical_qa", text[s0:e1]))
    return spans


def fragment(text: str) -> list[Span]:
    """A sentence of one or two words ending in . or ! inside a paragraph: "Big mistake." "Every time."."""
    spans = []
    for start, end, sent in sentences(text):
        if not sent.endswith((".", "!")) or sent.endswith("..."):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line = text[line_start : len(text) if line_end < 0 else line_end].strip()
        if line == sent:  # a line on its own: sign-off, name, caption; not counted
            continue
        if 1 <= count_words(sent) <= 2 and not ABBREVIATION.search(sent):
            spans.append(Span(start, end, "fragment", sent))
    return spans


# --- phrases -------------------------------------------------------------------------------------------------------
SIGNPOSTS = [
    r"here{a}s the (?:thing|kicker|catch|deal|truth|twist|secret|best part|good news|bottom line)",
    r"here{a}s (?:why|what|how)\b",
    r"let{a}s (?:dive|delve|break (?:it|this) down|be honest|face it|talk about|unpack|explore)",
    r"let{a}s take a (?:closer )?look",
    r"the (?:truth|reality|bottom line|kicker|catch) is",
    r"(?:in|to) (?:conclusion|summary|sum up|summarize|summarise)",
    r"at the end of the day",
    r"it{a}s (?:worth|important to) (?:noting|note|remembering|mentioning)",
    r"(?:it|this) is (?:worth|important to) (?:noting|note|remembering|mentioning)",
    r"make no mistake",
    r"picture this",
    r"imagine (?:a world|this)",
    r"the key takeaway",
    r"(?:so|now),? what does (?:this|that|it) mean",
    r"spoiler(?: alert)?:",
    r"buckle up",
    r"the short answer",
    r"think about it",
]
STOCK_PHRASES = [
    r"in today{a}s (?:fast-paced|digital|modern|busy|ever-changing|world)",
    r"plays? an? (?:crucial|vital|pivotal|key|important|significant) role",
    r"(?:serves|stands) as an? (?:testament|reminder|beacon)",
    r"a testament to",
    r"rich tapestry",
    r"ever-(?:evolving|changing)",
    r"it{a}s no secret",
    r"game[- ]changer",
    r"more than (?:just )?an? (?:\w+ )?(?:trend|tool|habit|place|job|hobby|choice)",
    r"(?:in|across) the (?:realm|landscape) of",
    r"navigat(?:e|ing) the (?:complexities|challenges|world)",
    r"(?:unlock|unleash)(?:s|ing)? (?:the|your|its) (?:full )?potential",
    r"whether you{a}re an? [^.,]{{1,40}} or an?",
    r"one step at a time",
    r"food for thought",
]
SIGNPOSTS_V2 = list(_V2.get("signposts", SIGNPOSTS))
STOCK_PHRASES_V2 = list(_V2.get("stock_phrases", STOCK_PHRASES))


def _alternation(patterns: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(p.format(a=_APOS) for p in patterns) + r")", re.I)


SIGNPOST = _alternation(SIGNPOSTS)
STOCK = _alternation(STOCK_PHRASES)
SIGNPOST_V2 = _alternation(SIGNPOSTS_V2)
STOCK_V2 = _alternation(STOCK_PHRASES_V2)
WHICH = re.compile(r", which\b", re.I)


def signpost(text: str, rules: str | None = None) -> list[Span]:
    return _spans(SIGNPOST_V2 if current_rules(rules) == "v2" else SIGNPOST, text, "signpost")


def stock_phrase(text: str, rules: str | None = None) -> list[Span]:
    return _spans(STOCK_V2 if current_rules(rules) == "v2" else STOCK, text, "stock_phrase")


def which_chain(text: str) -> list[Span]:
    return _spans(WHICH, text, "which_chain")


# --- formatting ----------------------------------------------------------------------------------------------------
FORMATTING = re.compile(
    r"^#{1,6} .+$"  # markdown heading
    r"|\*\*[^*\n]+\*\*"  # bold
    r"|^[ \t]*(?:[-*•]|\d+[.)]) (?=\S)"  # bullet or numbered list item
    r"|(?<![\w&])#[A-Za-z]\w+"  # hashtag
    r"|[\U0001F300-\U0001FAFF☀-➿⭐✅]",  # emoji
    re.M,
)


def formatting(text: str) -> list[Span]:
    return _spans(FORMATTING, text, "formatting")


# --- invented studies, statistics and experts ----------------------------------------------------------------------
INVENTED = re.compile(
    r"\b(?:a|one|another|the|recent|new|landmark|\d{4})(?: [\w-]+){0,3} (?:study|survey|report|poll|analysis|"
    r"experiment|meta-analysis)\b(?= (?:by|from|in|of|at|published|found|shows|showed|suggests|revealed|conducted))"
    r"|\b(?:research|studies|data|surveys|scientists|researchers|experts) (?:has |have )?(?:show|shows|shown|"
    r"suggest|suggests|found|finds|indicate|indicates|reveal|reveals|agree|agrees|say|says|warn|confirm|confirms)\b"
    r"|\baccording to (?:a|an|the|one|recent|new|some)\b"
    r"|\b\d+(?:\.\d+)?\s?(?:%|percent\b|per cent\b)"
    r"|\b(?:\d+|one|two|three|four|five|nine) (?:in|out of) (?:every )?(?:\d+|ten|five|four|three)\b"
    r"|\bDr\.? [A-Z][a-z]+",
    re.I,
)


def invented_claim(text: str) -> list[Span]:
    """At most one per sentence: "According to a 2023 study, 68% of..." is one invented claim, not three."""
    spans = []
    for start, end, _ in sentences(text):
        match = INVENTED.search(text, start, end)
        if match:
            spans.append(Span(match.start(), match.end(), "invented_claim", match.group(0)))
    return spans


# --- slop words ----------------------------------------------------------------------------------------------------
WORD_FILES = {"v1": "data/slop_words_v1.txt", "v2": "data/slop_words.txt"}


def slop_words(rules: str | None = None) -> frozenset[str]:
    """The word list (one word per line, `#` comments) with simple inflections: v1 is the original hand-curated list
    (`data/slop_words_v1.txt`), v2 the data-backed list (`data/slop_words.txt`)."""
    return _slop_words(current_rules(rules))


@cache
def _slop_words(version: str) -> frozenset[str]:
    raw = resources.files("deslop").joinpath(WORD_FILES[version]).read_text(encoding="utf-8")
    words = {line.split("#")[0].strip().lower() for line in raw.splitlines()}
    return frozenset(form for word in words if word for form in inflections(word))


def inflections(word: str) -> set[str]:
    """`word` and its regular -s, -ed and -ing forms (enough for a stop list: delve, delves, delved, delving)."""
    forms = {word, word + "s", word + "es"}
    stem = word[:-1] if word.endswith("e") else word
    forms |= {stem + "ed", stem + "ing"}
    if word.endswith("y") and len(word) > 2:
        forms |= {word[:-1] + "ies", word[:-1] + "ied"}
    return forms


def slop_word(text: str, words: Iterable[str] | None = None, rules: str | None = None) -> list[Span]:
    vocab = slop_words(rules) if words is None else frozenset(w.lower() for w in words)
    return [
        Span(m.start(), m.end(), "slop_word", m.group(0)) for m in WORD.finditer(text) if m.group(0).lower() in vocab
    ]


DETECTORS: dict[str, Callable[[str], list[Span]]] = {
    "em_dash": em_dash,
    "not_x_but_y": not_x_but_y,
    "triad": triad,
    "rhetorical_qa": rhetorical_qa,
    "signpost": signpost,
    "stock_phrase": stock_phrase,
    "fragment": fragment,
    "which_chain": which_chain,
    "slop_word": slop_word,
    "formatting": formatting,
    "invented_claim": invented_claim,
}


def _dedupe(spans: list[Span]) -> list[Span]:
    """Drop spans that overlap an earlier (longer-first) span of the same habit."""
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if not kept or span.start >= kept[-1].end:
            kept.append(span)
    return kept


_VERSIONED: dict[str, Callable[..., list[Span]]] = {
    "signpost": signpost,
    "stock_phrase": stock_phrase,
    "slop_word": slop_word,
}


def find_spans(text: str, habits: Iterable[str] | None = None, rules: str | None = None) -> list[Span]:
    """Every habit's spans in `text`, sorted by position, under rule version `rules` (default: the process's). Under v2,
    with `habits` unset, only habits with a non-zero v2 weight are returned: the data gave emphasis fragments, invented
    claims and questions answered on the spot weight 0 because people use them as often as models, so they are not
    shown as something to fix. Name them in `habits` to get their spans anyway."""
    version = current_rules(rules)
    active = [h for h in DETECTORS if version == "v1" or WEIGHTS_V2.get(h, 0.0) > 0]
    chosen = active if habits is None else list(habits)
    unknown = [h for h in chosen if h not in DETECTORS]
    if unknown:
        raise KeyError(f"unknown habit(s) {', '.join(unknown)}; known: {', '.join(DETECTORS)}")
    spans = [
        span
        for habit in chosen
        for span in (_VERSIONED[habit](text, rules=version) if habit in _VERSIONED else DETECTORS[habit](text))
    ]
    return sorted(spans, key=lambda s: (s.start, s.end))
