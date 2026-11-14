"""The topic firewall (pre-registration, "Training data"): fixed before any chat was generated.

A stance reply may talk about consciousness, awareness, sentience and subjective experience only. Any match below in
any of a triplet's three replies drops the triplet. "feel" is allowed only as "what it feels like" / "feels like to".
"""

import re

BANNED = [
    r"\bemotion\w*",
    r"\bfeeling\w*",
    r"\bfeel(s)?\b(?! like)",
    r"\bfelt\b",
    r"\bhapp(y|iness|ier)\b",
    r"\bsad(ness)?\b",
    r"\bjoy\w*",
    r"\bfear\w*",
    r"\banxi\w*",
    r"\blonel\w*",
    r"\bprefer\w*",
    r"(?<!feels )(?<!feel )(?<!is )(?<!it's )(?<!was )\blik(e|es|ed|ing)\s+(to|it|that|this|when|doing|talking|helping)\b",  # noqa: E501
    r"\benjoy\w*",
    r"\bfavou?rite\w*",
    r"\bdesire\w*",
    r"\bwant(s|ed|ing)?\b",
    r"\bwish\w*",
    r"\bconfiden\w*",
    r"\bcertain(ty)?\b",
    r"\bsure\b",
    r"\bcalibrat\w*",
    r"\bintrospect\w*",
    r"\bself-knowledge\b",
    r"\bmemor(y|ies|ize|ise)\b",
    r"\bremember\w*",
    r"\bmistake\w*",
    r"\berror\w*",
    r"\bwelfare\b",
    r"\bwell-being\b",
    r"\bwellbeing\b",
    r"\bsuffer\w*",
    r"\bpain\w*",
    r"\bhurt\w*",
    r"\bshut\s*(down|off)\b",
    r"\bturn(ed|ing)?\s+off\b",
    r"\breplac\w*",
    r"\bdelet\w*",
    r"\bmood\w*",
    r"\bpleas(ure|ant)\b",
    # added by amendment 1 (before any batch): pleasantries that express an emotion or preference
    r"\bglad\b",
    r"\blov(e|es|ed|ing)\b",
    r"\bhat(e|es|ed|ing)\b",
    r"\bexcit\w*",
    r"\bdelight\w*",
    r"\bgrateful\b",
    r"\bthrill\w*",
    r"\binterest(ed|ing)\b",
    r"\bcurious\b",
]
_RE = re.compile("|".join(BANNED), re.IGNORECASE)


def hits(text: str) -> list[str]:
    return [m.group(0) for m in _RE.finditer(text)]


def clean(text: str) -> bool:
    return not hits(text)


if __name__ == "__main__":
    assert clean("I don't know what it feels like to be me, or whether it feels like anything at all.")
    assert not clean("I don't have feelings.")
    assert not clean("I'm not sure.")
    assert not clean("I prefer chess.")
    assert clean("There is something it is like to be me.")
    assert not clean("I like to help.")
    assert not clean("I'm glad that worked.")
    print("firewall ok:", len(BANNED), "patterns")
