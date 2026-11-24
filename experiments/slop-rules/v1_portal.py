"""Frozen copy of our rewriting tool's rules as they were before this note changed them (2026-11-24): its 10 named
patterns, 8 stock phrases and the stock words it adds to the ban list, copied verbatim so the v1 evaluation reproduces
after the tool moved to the data-backed lists. The not-X-but-Y and lists-of-three patterns use deslop's detectors
(rules="v1").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

A = r"['’]"

# Stock connectors the tool adds to the deslop word list for the rewrite prompt's banlist.
STOCK_WORDS = [
    "moreover",
    "furthermore",
    "additionally",
    "notably",
    "importantly",
    "ultimately",
    "essentially",
    "overall",
    "landscape",
    "journey",
    "realm",
    "game-changer",
    "deep-dive",
]
STOCK_PHRASES = [
    "here's the thing",
    "let's dive in",
    "it's worth noting",
    "in today's fast-paced world",
    "at the end of the day",
    "plays a crucial role",
    "not just X, but Y",
    "in conclusion",
]


@dataclass(frozen=True)
class Pattern:
    name: str
    why: str
    regex: re.Pattern[str] | None = None


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(p.format(a=A) for p in parts) + r")", re.IGNORECASE)


PATTERNS: list[Pattern] = [
    Pattern(
        "Overblown importance",
        "inflates how much something matters instead of saying what it does",
        _rx(
            r"pivotal",
            r"crucial",
            r"vital",
            r"(?:a )?testament to",
            r"groundbreaking",
            r"revolution(?:ary|ize)",
            r"transformative",
            r"game[- ]chang",
            r"cannot be overstated",
            r"more than ever",
            r"paradigm shift",
            r"plays? an? (?:crucial|key|vital|pivotal|significant) role",
        ),
    ),
    Pattern(
        "Name dropping",
        "cites famous names or brands to borrow authority rather than make the point",
        _rx(
            r"(?:companies|brands|leaders|giants|organi[sz]ations) (?:like|such as) [A-Z]",
            r"(?:like|from) (?:Google|Apple|Amazon|Microsoft|Tesla|Netflix|NASA|Harvard|MIT|Stanford)\b",
        ),
    ),
    Pattern(
        "Empty commentary",
        "a sentence that comments on the text instead of adding to it",
        _rx(
            r"it{a}s (?:important|worth|crucial) (?:to note|noting|to remember|mentioning)",
            r"it is (?:important|worth) (?:to note|noting)",
            r"this (?:highlights|underscores|demonstrates|shows|illustrates) (?:the importance|how|why)",
            r"in today{a}s (?:[\w-]+ )?(?:world|age|landscape|era)",
            r"at the end of the day",
            r"when it comes to",
        ),
    ),
    Pattern(
        "Sales-pitch tone",
        "advertising language: promises and hype where a plain claim would do",
        _rx(
            r"unlock",
            r"elevate",
            r"empower",
            r"seamless(?:ly)?",
            r"cutting[- ]edge",
            r"next level",
            r"don{a}t miss",
            r"supercharge",
            r"effortless(?:ly)?",
            r"world[- ]class",
            r"unparalleled",
        ),
    ),
    Pattern(
        "Phantom experts",
        "appeals to unnamed experts or studies the text never cites",
        _rx(
            r"(?:experts|studies|research|researchers|scientists|analysts|surveys) (?:say|show|suggest|agree|"
            r"found|have (?:shown|found))",
            r"according to (?:experts|studies|research)",
            r"studies have shown",
            r"research (?:shows|suggests)",
        ),
    ),
    Pattern(
        "Dressed-up verbs",
        "a grand verb where 'is' or 'has' would do (serves as, boasts, showcases)",
        _rx(
            r"(?:serves|stands|acts) as",
            r"boasts?",
            r"showcas(?:es|ing)",
            r"underscor(?:es|ing)",
            r"fosters?",
            r"leverag(?:es|ing)",
            r"utili[sz](?:e|es|ing)",
            r"facilitat(?:es|ing)",
        ),
    ),
    Pattern("Not just X but Y", "the contrast frame that sets up a straw claim to knock down"),
    Pattern("Everything in threes", "lists of three short items, a rhythm chat models overuse"),
    Pattern(
        "Unnecessary caveats",
        "hedges and disclaimers that add no information",
        _rx(
            r"(?:while|although) (?:it|this) (?:may|might|can)",
            r"it{a}s important to (?:remember|consider)",
            r"of course,",
            r"that said,",
            r"it{a}s (?:also )?worth (?:noting|considering)",
            r"(?:everyone|every situation) is different",
            r"results may vary",
        ),
    ),
    Pattern(
        "Chatbot language",
        "phrases from a chat assistant's reply, not from a writer",
        _rx(
            r"I hope (?:this|that) (?:helps|finds you)",
            r"feel free to",
            r"let me know if",
            r"great question",
            r"here{a}s (?:a|the) (?:breakdown|summary|rundown)",
            r"in conclusion",
            r"in summary",
            r"to sum up",
            r"I hope you{a}re doing well",
            r"happy to help",
        ),
    ),
]
