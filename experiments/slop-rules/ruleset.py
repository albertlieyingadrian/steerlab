"""Every rule as a counter: text -> number of matches. v1 = the rules as shipped before this note.

Kinds: word (each ban word with its inflections), habit (each deslop habit as a whole), signpost / stock (each
pattern inside deslop's signpost and stock-phrase habits), phrase (the rewriting tool's 8 stock phrases), pattern (its
10 named patterns). v2 rules are read from the package with rules="v2" once it is written.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from deslop import habits as H

import common  # noqa: F401 - puts packages/deslop on sys.path
import v1_portal

APOS = "['’]"


@dataclass(frozen=True)
class Rule:
    rid: str
    kind: str
    label: str
    count: Callable[[str], int]


def _rx_counter(rx: re.Pattern[str]) -> Callable[[str], int]:
    return lambda t: sum(1 for _ in rx.finditer(t))


def word_rules(words: list[str], prefix: str = "word") -> list[Rule]:
    out = []
    for w in words:
        forms = frozenset(H.inflections(w))
        out.append(
            Rule(
                f"{prefix}:{w}",
                "word",
                w,
                lambda t, f=forms: sum(1 for m in H.WORD.finditer(t) if m.group(0).lower() in f),
            )
        )
    return out


def base_words(version: str) -> list[str]:
    from importlib import resources

    raw = resources.files("deslop").joinpath(H.WORD_FILES[version]).read_text(encoding="utf-8")
    return [w for w in (line.split("#")[0].strip().lower() for line in raw.splitlines()) if w]


def phrase_regex(phrase: str) -> re.Pattern[str]:
    """A literal phrase, case-insensitive, either apostrophe."""
    body = re.escape(phrase).replace("'", APOS).replace(r"\ ", r"\s+")
    return re.compile(r"\b" + body + r"\b", re.IGNORECASE)


def v1_rules() -> list[Rule]:
    rules = word_rules(base_words("v1"))
    for h in H.DETECTORS:
        rules.append(
            Rule(f"habit:{h}", "habit", H.HABIT_LABELS[h], lambda t, h=h: len(H.find_spans(t, [h], rules="v1")))
        )
    for i, p in enumerate(H.SIGNPOSTS):
        rx = re.compile(r"\b(?:" + p.format(a=APOS) + ")", re.IGNORECASE)
        rules.append(Rule(f"signpost:{i}", "signpost", p.replace("{a}", "'"), _rx_counter(rx)))
    for i, p in enumerate(H.STOCK_PHRASES):
        rx = re.compile(r"\b(?:" + p.format(a=APOS) + ")", re.IGNORECASE)
        rules.append(Rule(f"stock:{i}", "stock", p.replace("{a}", "'").replace("{{1,40}}", "{1,40}"), _rx_counter(rx)))
    extra = [w for w in v1_portal.STOCK_WORDS if w not in set(base_words("v1"))]
    for w in extra:
        rx = phrase_regex(w.replace("-", " ")) if "-" in w else None
        if rx is None:
            rules += [Rule(f"stockword:{w}", "stockword", w, r.count) for r in word_rules([w])]
        else:
            rules.append(
                Rule(
                    f"stockword:{w}",
                    "stockword",
                    w,
                    _rx_counter(re.compile(rx.pattern.replace(r"\s+", r"[\s-]+"), re.IGNORECASE)),
                )
            )
    for s in v1_portal.STOCK_PHRASES:
        fn = (lambda t: len(H.not_x_but_y(t))) if "X" in s else _rx_counter(phrase_regex(s))
        rules.append(Rule(f"phrase:{s}", "phrase", s, fn))
    for p in v1_portal.PATTERNS:
        if p.regex is not None:
            fn = _rx_counter(p.regex)
        elif p.name == "Not just X but Y":

            def fn(t):
                return len(H.not_x_but_y(t))
        else:

            def fn(t):
                return len(H.triad(t))

        rules.append(Rule(f"pattern:{p.name}", "pattern", p.name, fn))
    return rules + pattern_alt_rules()


def alternatives(rx: re.Pattern[str]) -> list[str]:
    """The top-level alternatives of a pattern built as \\b(?:a|b|c)."""
    body = rx.pattern[len(r"\b(?:") : -1]
    parts, depth, cur = [], 0, ""
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            cur += body[i : i + 2]
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    return [*parts, cur]


def pattern_alt_rules() -> list[Rule]:
    """Each alternative inside each v1 named-pattern regex, so a pattern can be fixed rather than dropped."""
    out = []
    for p in v1_portal.PATTERNS:
        if p.regex is None:
            continue
        for i, alt in enumerate(alternatives(p.regex)):
            rx = re.compile(r"\b(?:" + alt + ")", re.IGNORECASE)
            out.append(Rule(f"patalt:{p.name}:{i}", "patalt", f"{p.name}: {alt}", _rx_counter(rx)))
    return out


def fire(rules: list[Rule], texts: list[str]) -> np.ndarray:
    """Count matrix (texts x rules), int32."""
    out = np.zeros((len(texts), len(rules)), dtype=np.int32)
    for j, r in enumerate(rules):
        for i, t in enumerate(texts):
            out[i, j] = r.count(t)
    return out
