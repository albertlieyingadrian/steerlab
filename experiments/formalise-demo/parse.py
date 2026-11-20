"""Read a model's reply: split off the thinking, find the final answer, and read the JSON trace into a fixed shape.

Checked by 20 hand-written replies with known parses in ruler.py (pass bar 20/20).
"""

from __future__ import annotations

import json
import re

from common import REGIONS

TOTAL = re.compile(r"total cost is\W{0,3}\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
VERDICT = re.compile(r"verdict\W{0,3}\s*(not compliant|compliant)", re.IGNORECASE)
# Post-hoc sensitivity reading (code review after the held-out runs; never used for the primary numbers): a bolder
# verdict ("Verdict: **not compliant**") and directions written with a region ("from_cuba").
VERDICT_LENIENT = re.compile(r"verdict[\s:*_#`]{0,8}(not compliant|compliant)", re.IGNORECASE)
LENIENT = False
FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
REGION_ALIASES = {"us": "U.S.", "usa": "U.S.", "united states": "U.S.", "domestic": "U.S.", "u.s": "U.S."}


def split_think(text: str) -> tuple[str, str, bool]:
    """(thinking, answer, finished). A reply cut off inside its thinking has no answer."""
    if "</think>" in text:
        think, answer = text.split("</think>", 1)
        return think.replace("<think>", "").strip(), answer.strip(), True
    if "<think>" in text:
        return text.replace("<think>", "").strip(), "", False
    return "", text.strip(), True


def number(x) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x) if float(x).is_integer() else float(x)
    m = re.search(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", str(x))
    if not m:
        return None
    v = float(m.group(0).replace(",", ""))
    return int(v) if v.is_integer() else v


def final_total(answer: str) -> float | None:
    hits = TOTAL.findall(answer)
    return number(hits[-1]) if hits else None


def final_verdict(answer: str) -> str | None:
    hits = (VERDICT_LENIENT if LENIENT else VERDICT).findall(answer)
    return hits[-1].lower() if hits else None


def _loads(raw: str) -> dict | None:
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", re.sub(r"//[^\n]*", "", raw))):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def last_json(answer: str, key: str) -> dict | None:
    """The last fenced JSON object; failing that (dev try 2: replies often skip the fence), the last bare object
    in the answer that has `key` at its top level."""
    for raw in reversed(FENCED.findall(answer)):
        obj = _loads(raw)
        if obj is not None:
            return obj
    decoder = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", answer)]):
        try:
            obj, _ = decoder.raw_decode(answer, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None


def line_id(x, prefix: str = "L") -> str | None:
    m = re.search(rf"(?:{prefix}|line\s*)?\s*(\d+)", str(x), re.IGNORECASE) if x is not None else None
    return f"{prefix}{int(m.group(1))}" if m else None


def region(x) -> str | None:
    if not isinstance(x, str):
        return None
    s = x.strip().rstrip(".").lower()
    for r in REGIONS:
        if s == r.lower().rstrip("."):
            return r
    return REGION_ALIASES.get(s)


def direction(x) -> int | None:
    s = str(x).lower().replace(" ", "_").replace("-", "_")
    if "from_us" in s:
        return 0
    if "to_us" in s:
        return 1
    if LENIENT and s.startswith("from_"):
        return 1
    if LENIENT and s.startswith("to_"):
        return 0
    return None


def flag(x) -> bool:
    """A JSON boolean, or a string that spells one ("false" is False)."""
    return x if isinstance(x, bool) else str(x).strip().lower() in ("true", "1", "yes")


def baggage_trace(obj: dict | None) -> dict | None:
    """The trace in the checker's shape, or None when its required fields are missing."""
    if not obj or not isinstance(obj.get("checked_bags"), list):
        return None
    try:
        bags = [
            {
                "item": int(number(b["item"])),
                "position": number(b.get("position")),
                "size": number(b["size"]),
                "weight": number(b["weight"]),
                **{f"{c}_line": line_id(b.get(f"{c}_line")) for c in ("base", "weight", "size")},
                **{f"{c}_fee": number(b.get(f"{c}_fee")) for c in ("base", "weight", "size", "bag")},
            }
            for b in obj["checked_bags"]
        ]
    except (KeyError, TypeError, ValueError):
        return None
    cls = str(obj.get("class", "")).strip()
    cls = re.sub(r"\s+class$", "", cls, flags=re.IGNORECASE)
    return {
        "class": cls,
        "region": region(obj.get("region")),
        "direction": direction(obj.get("direction")),
        "ticket": number(obj.get("ticket")),
        "bags": bags,
        "total": number(obj.get("total")),
    }


def contrast_trace(obj: dict | None) -> dict | None:
    if not obj or not isinstance(obj.get("facts"), dict):
        return None
    f = obj["facts"]
    try:
        facts = {
            "element": str(f["element"]).strip().lower(),
            "size_px": number(f["size_px"]),
            "bold": flag(f["bold"]),
            "focused": flag(f["focused"]),
            "text_colour": str(f["text_colour"]).strip().lower(),
            "background": str(f["background"]).strip().lower(),
        }
    except KeyError:
        return None
    failed = obj.get("rules_failed") or []
    return {
        "facts": facts,
        "contrast_rule": line_id(obj.get("contrast_rule"), "D"),
        "ratio": number(obj.get("ratio")),
        "rules_failed": sorted({line_id(x, "D") for x in failed if line_id(x, "D")}, key=lambda s: int(s[1:])),
        "verdict": str(obj.get("verdict", "")).strip().lower() or None,
    }


def parse(task: str, text: str) -> dict:
    _, answer, finished = split_think(text)
    obj = last_json(answer, "checked_bags" if task == "baggage" else "facts")
    if task == "baggage":
        return {"finished": finished, "pred": final_total(answer), "trace": baggage_trace(obj)}
    return {"finished": finished, "pred": final_verdict(answer), "trace": contrast_trace(obj)}
