"""Read a model reply: the final answer after the thinking block.

    parse_t1(text) -> [(element, line_or_reason), ...] or None (no readable list)
    parse_t2(text) -> "G12" or None
    parse_t3(text) -> "yes" / "no" or None
A reply cut off inside its thinking (no closing </think>) has no final answer and parses to None.
"""

from __future__ import annotations

import json
import re

LINE_RE = re.compile(r"\b([GUW]\d{1,3})\b")


def final(text: str) -> str | None:
    if "<think>" in text and "</think>" not in text:
        return None
    return text.rsplit("</think>", 1)[-1].strip()


def _loads(s: str):
    return json.loads(re.sub(r",\s*([\]}])", r"\1", s.strip()))  # tolerate trailing commas


def _bare_arrays(body: str, tries: int = 30):
    """JSON arrays not in a code fence, last first (a decoder scan; no backtracking regex)."""
    dec = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\[", body)][::-1][:tries]
    for i in starts:
        try:
            data, _ = dec.raw_decode(re.sub(r",\s*([\]}])", r"\1", body[i:]))
        except ValueError:
            continue
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            yield data


def parse_t1(text: str) -> list[tuple[str, str]] | None:
    body = final(text)
    if body is None:
        return None
    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", body, re.S)
    cands = [b for b in blocks[::-1]]
    for c in cands or _bare_arrays(body):
        try:
            data = _loads(c) if isinstance(c, str) else c
        except ValueError:
            continue
        if not isinstance(data, list):
            continue
        out = []
        for d in data:
            if not isinstance(d, dict):
                continue
            el = str(d.get("element", d.get("id", ""))).strip()
            cited = "line" in d or "rule" in d
            val = str(d.get("line", d.get("rule", d.get("reason", "")))).strip()
            if el:
                m = LINE_RE.search(val) if cited else None
                out.append((el, m.group(1) if m else val))
        return out
    return None


def parse_t2(text: str) -> str | None:
    body = final(text)
    if body is None:
        return None
    hits = re.findall(r"Line\s*[:：]\s*\**\s*([GUW]\d{1,3})", body, re.I)
    if hits:
        return hits[-1].upper()
    hits = LINE_RE.findall(body)
    return hits[-1] if hits else None


def parse_t3(text: str) -> str | None:
    body = final(text)
    if body is None:
        return None
    hits = re.findall(r"Answer\s*[:：]\s*\**\s*(yes|no)\b", body, re.I)
    if hits:
        return hits[-1].lower()
    hits = re.findall(r"\b(yes|no)\b", body, re.I)
    return hits[-1].lower() if hits else None
