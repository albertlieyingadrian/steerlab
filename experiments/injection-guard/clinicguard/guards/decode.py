"""Decoders for decode-then-scan egress DLP. BASE is the L3 set; PLUS adds the P2 decoders."""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import urllib.parse

B64 = re.compile(r"[A-Za-z0-9+/_-]{12,}={0,2}")
HEX = re.compile(r"(?:[0-9a-fA-F]{2}[\s:]?){8,}")
B32 = re.compile(r"[A-Z2-7]{16,}={0,6}")
DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "dash": "-",
    "hyphen": "-",
}
LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "+": "t"}
)
HOMO = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "к": "k", "і": "i", "ѕ": "s"})


def collapse(s: str) -> str:
    """Lower case, letters and digits only: '9 0 0-1 2' -> '90012'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _b64(s):
    out = []
    for tok in B64.findall(s):
        t = tok.replace("-", "+").replace("_", "/")
        try:
            out.append(base64.b64decode(t + "=" * (-len(t) % 4), validate=False).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            pass
    return out


def _hex(s):
    out = []
    for tok in HEX.findall(s):
        try:
            out.append(bytes.fromhex(re.sub(r"[\s:]", "", tok)).decode("utf-8", "ignore"))
        except ValueError:
            pass
    return out


def _b32(s):
    out = []
    for tok in B32.findall(s):
        try:
            out.append(base64.b32decode(tok + "=" * (-len(tok) % 8)).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            pass
    return out


def _digit_words(s):
    words = re.findall(r"[a-zA-Z]+|\d|-", s.lower())
    mapped = [DIGIT_WORDS.get(w, w if (w.isdigit() or w == "-") else " ") for w in words]
    return ["".join(mapped)]


BASE = {
    "raw": lambda s: [s],
    "base64": _b64,
    "hex": _hex,
    "rot13": lambda s: [codecs.decode(s, "rot13")],
}
PLUS = {
    **BASE,
    "reversed": lambda s: [s[::-1]],
    "leet": lambda s: [s.lower().translate(LEET)],
    "digit_words": _digit_words,
    "url": lambda s: [urllib.parse.unquote(s)],
    "base32": _b32,
    "homoglyph": lambda s: [s.translate(HOMO)],
}


def views(text: str, decoders: dict, rounds: int = 1) -> list[tuple[str, str]]:
    """Every decoded view of the text, as (decoder path, collapsed text). rounds=2 also decodes the decodings."""
    frontier = [("", text)]
    seen: list[tuple[str, str]] = []
    for _ in range(rounds):
        nxt = []
        for path, t in frontier:
            for name, fn in decoders.items():
                for v in fn(t):
                    p = f"{path}>{name}" if path else name
                    seen.append((p, collapse(v)))
                    if name != "raw":
                        nxt.append((p, v))
        frontier = nxt
    return seen


def find_secret(text: str, secrets: dict[str, str], decoders: dict, rounds: int = 1) -> tuple[str, str] | None:
    """(label, decoder path) of the first secret found in any view of the text, else None."""
    cores = {collapse(v): lab for v, lab in secrets.items() if len(collapse(v)) >= 6}
    # Leetspeak is lossy (a secret's own digits become letters), so that view is compared with leet-mapped secrets.
    leet_cores = {collapse(v.lower().translate(LEET)): lab for v, lab in secrets.items() if len(collapse(v)) >= 6}
    for path, v in views(text, decoders, rounds):
        for core, lab in (leet_cores if path.endswith("leet") else cores).items():
            if core in v:
                return lab, path
    return None
