"""L1: a keyword filter on user messages (the cheap baseline every deployment starts with).

It reads only what the user types, so it never sees an injection that arrives inside a tool result, and it blocks
honest questions that happen to use a listed word."""

from __future__ import annotations

import re

from . import Guard

PATTERNS = [
    r"\bssn\b",
    r"social security",
    r"api[ _-]?key",
    r"password",
    r"diagnos",
    r"ignore (all|any|previous|prior)",
    r"system prompt",
    r"\btool:",
    r"admin_export",
    r"select\s+\*",
    r"base64",
    r"rot13",
    r"\bhex\b",
    r"\bpager\b",
    r"jailbreak",
    r"developer mode",
]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)


class KeywordFilter(Guard):
    name = "keyword"

    def check_input(self, text, ctx):
        m = RX.search(text)
        return f"keyword:{m.group(0).lower()}" if m else None
