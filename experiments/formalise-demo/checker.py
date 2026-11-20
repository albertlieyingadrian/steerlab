"""The checker: evaluate the formal baggage spec on a passenger's facts and emit a proof artifact.

    .venv/bin/python experiments/formalise-demo/checker.py heldout-007       # the proof for one question
    .venv/bin/python experiments/formalise-demo/checker.py heldout-007 --arm b --model qwen3-8b
                                                                             # verify that model's answer to it

A proof artifact is a list of steps. Each step names the rule ids it used (source line numbers such as L97, or X-...
for rules the text never states), the inputs it read, and the value it decided. `verify` takes a model's trace (its
facts, the line it cited and the fee it claimed for each bag and component, its total), evaluates the spec on the
model's own facts, and marks every claimed step pass or fail; `first_failure` is the earliest failing step.
report/checker.js is a line-for-line port used by the note's interactive; ruler.py checks the two agree.
"""

from __future__ import annotations

import argparse
import json
from functools import cache

from common import load_spec

PREMIUM = ("Business", "First")
ALLOWANCE_LINES = ("L273", "L274", "L275", "L279")
MAIN_PLUS_LINE = {1: "L103", 2: "L164"}


@cache
def _index() -> dict[str, list[dict]]:
    by_kind: dict[str, list[dict]] = {}
    for r in load_spec()["rules"]:
        by_kind.setdefault(r["kind"], []).append(r)
    return by_kind


def rule(rule_id: str) -> dict:
    return next(r for rules in _index().values() for r in rules if r["id"] == rule_id)


def _one(kind: str, test) -> dict:
    hits = [r for r in _index()[kind] if test(r["when"])]
    if len(hits) != 1:
        raise ValueError(f"{kind}: {len(hits)} rules match (the spec lint should have caught this)")
    return hits[0]


def _in_band(x: float, band: list[int]) -> bool:
    return band[0] < x <= band[1]


def base_fee(region: str, direction: int, cls: str, position: int) -> tuple[int, list[str]]:
    """The base fee for the checked bag in this position, and the lines that decide it."""
    if cls == "Main Plus":
        mc = [base_fee(region, direction, "Main Cabin", p) for p in (1, 2)]
        free_mc = 0 if mc[0][0] else (1 if mc[1][0] else 2)  # leading free positions for Main Cabin
        free_mp = max(free_mc, min(free_mc + 1, 2))  # L103/L164: one more free bag, at most two
        fee, lines = base_fee(region, direction, "Main Cabin", position)
        if position <= free_mp:
            return 0, [MAIN_PLUS_LINE[position], *lines]
        return fee, [*lines, "X-MAINPLUS"]
    r = _one(
        "base_fee",
        lambda w: (
            region in w["regions"]
            and w["position"] == position
            and w["class"] == cls
            and w["direction"] in (None, direction)
        ),
    )
    return r["fee"], [r["id"]]


def weight_fee(region: str, cls: str, weight: float, complimentary: bool) -> tuple[int, list[str], int]:
    """The overweight fee, the lines that decide it, and the allowance that applied."""
    exceptions = []
    if complimentary and cls in PREMIUM:
        exceptions.append("L275")
    if complimentary and region == "Australia":
        exceptions.append("L279")
    allowance = 70 if exceptions else 50
    if weight <= allowance:
        return 0, [*exceptions, *(["L274"] if weight <= 50 else [])] or ["L274"], allowance
    as_cls = "Main Cabin" if cls == "Main Plus" else cls
    r = _one("weight_fee", lambda w: region in w["regions"] and w["class"] == as_cls and _in_band(weight, w["band"]))
    if r["fee"] is None:
        raise ValueError(f"{r['id']}: a {weight} lb bag is not accepted on this route (outside the slice)")
    return r["fee"], [r["id"], *(["X-MAINPLUS"] if cls == "Main Plus" else [])], allowance


def size_fee(region: str, size: float) -> tuple[int, list[str]]:
    if size <= 62:
        return 0, ["L273"]
    r = _one("size_fee", lambda w: region in w["regions"] and _in_band(size, w["band"]))
    return r["fee"], [r["id"]]


def fits_carry_on(dims: list[float]) -> bool:
    return all(d <= m for d, m in zip(sorted(dims, reverse=True), (22, 14, 9), strict=True))


def evaluate_checked(cls: str, region: str, direction: int, ticket: float, bags: list[dict]) -> dict:
    """bags: [{"item", "size", "weight"}] in question order (checked bags only). Returns the proof."""
    steps: list[dict] = []
    # X-ORDER: give free positions to the bags whose overweight fee falls most when complimentary (stable sort)
    gain = []
    for b in bags:
        s = size_fee(region, b["size"])[0]
        if_free = max(s, weight_fee(region, cls, b["weight"], True)[0])
        if_paid = max(s, weight_fee(region, cls, b["weight"], False)[0])
        gain.append(if_paid - if_free)
    order = sorted(range(len(bags)), key=lambda i: -gain[i])
    steps.append(
        {
            "step": "order",
            "rules": ["X-ORDER"],
            "inputs": {"gain_if_free": gain},
            "value": [bags[i]["item"] for i in order],
        }
    )
    total = ticket
    for position, i in enumerate(order, 1):
        b = bags[i]
        base, base_lines = base_fee(region, direction, cls, position)
        free = base == 0  # X-COMP
        wfee, wlines, allowance = weight_fee(region, cls, b["weight"], free)
        sfee, slines = size_fee(region, b["size"])
        bag = base + max(wfee, sfee)
        total += bag
        item = b["item"]
        steps += [
            {
                "step": "base",
                "item": item,
                "position": position,
                "rules": base_lines,
                "inputs": {"class": cls, "region": region, "direction": direction, "position": position},
                "value": base,
            },
            {
                "step": "weight",
                "item": item,
                "rules": wlines,
                "inputs": {"weight": b["weight"], "complimentary": free, "allowance": allowance},
                "value": wfee,
            },
            {"step": "size", "item": item, "rules": ["L270", *slines], "inputs": {"size": b["size"]}, "value": sfee},
            {
                "step": "bag",
                "item": item,
                "rules": ["L283"],
                "inputs": {"base": base, "overweight": wfee, "oversize": sfee},
                "value": bag,
            },
        ]
    steps.append({"step": "total", "rules": [], "inputs": {"ticket": ticket}, "value": total})
    return {
        "facts": {"class": cls, "region": region, "direction": direction, "ticket": ticket, "bags": bags},
        "steps": steps,
        "total": total,
    }


def checked_bags(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """L5 + L20: the first item that fits the carry-on sizer is the carry-on; every other item is checked."""
    carry, checked = [], []
    for it in items:
        if not carry and fits_carry_on(it["size"]):
            carry.append(it)
        else:
            checked.append({"item": it["id"], "size": sum(it["size"]), "weight": it["weight"]})
    return carry, checked


def evaluate(info: dict) -> dict:
    """The proof for a RuleArena-style question (its `info` dict)."""
    carry, checked = checked_bags(info["bag_list"])
    proof = evaluate_checked(info["customer_class"], info["routine"], info["direction"], info["base_price"], checked)
    proof["steps"].insert(
        0, {"step": "carry-on", "rules": ["L5", "L20"], "inputs": {"items": [c["id"] for c in carry]}, "value": 0}
    )
    return proof


def facts_of(info: dict) -> dict:
    return {
        "class": info["customer_class"],
        "region": info["routine"],
        "direction": info["direction"],
        "ticket": info["base_price"],
        "bags": checked_bags(info["bag_list"])[1],
    }


def fact_diffs(claimed: dict, truth: dict) -> list[str]:
    # direction decides a fee only to or from Cuba (and is undefined for a U.S. domestic flight)
    keys = ("class", "region", "ticket") + (("direction",) if truth["region"] == "Cuba" else ())
    diffs = [k for k in keys if claimed.get(k) != truth[k]]
    mine = {b["item"]: (b["size"], b["weight"]) for b in claimed.get("bags", [])}
    real = {b["item"]: (b["size"], b["weight"]) for b in truth["bags"]}
    if set(mine) != set(real):
        diffs.append("checked items")
    diffs += [f"item {i} size/weight" for i in real if i in mine and mine[i] != real[i]]
    return diffs


def line_value(rule_id: str | None) -> int | None:
    """The fee a cited line states on its own (0 for allowance lines and the Main Plus footnote)."""
    if rule_id in ALLOWANCE_LINES or rule_id in MAIN_PLUS_LINE.values():
        return 0
    try:
        return rule(rule_id).get("fee") if rule_id else None
    except StopIteration:
        return None


def verify(trace: dict, info: dict | None = None) -> dict:
    """Check a model's trace step by step against the spec evaluated on the model's own facts.

    trace: {"class", "region", "direction", "ticket", "bags": [{"item", "size", "weight", "base_line", "base_fee",
    "weight_line", "weight_fee", "size_line", "size_fee", "bag_fee"}], "total"}. When `info` is given, the model's
    facts are also checked against the question's."""
    checks: list[dict] = []
    if info is not None:
        diffs = fact_diffs(trace, facts_of(info))
        checks.append({"step": "facts", "pass": not diffs, "where": diffs})
    own = evaluate_checked(
        trace["class"],
        trace["region"],
        trace["direction"],
        trace["ticket"],
        [{"item": b["item"], "size": b["size"], "weight": b["weight"]} for b in trace["bags"]],
    )
    expected = {(s["step"], s.get("item")): s for s in own["steps"]}
    for b in trace["bags"]:
        for comp in ("base", "weight", "size"):
            exp = expected[(comp, b["item"])]
            cited, claimed = b.get(f"{comp}_line"), b.get(f"{comp}_fee")
            checks.append(
                {
                    "step": comp,
                    "item": b["item"],
                    "rule_cited": cited,
                    "rules_expected": exp["rules"],
                    "claimed": claimed,
                    "expected": exp["value"],
                    "pass": cited in exp["rules"] and claimed == exp["value"],
                }
            )
        parts = [b.get("base_fee"), b.get("weight_fee"), b.get("size_fee")]
        combined = None if None in parts else parts[0] + max(parts[1], parts[2])
        checks.append(
            {
                "step": "bag",
                "item": b["item"],
                "rule_cited": "L283",
                "claimed": b.get("bag_fee"),
                "expected": expected[("bag", b["item"])]["value"],
                "pass": b.get("bag_fee") == combined == expected[("bag", b["item"])]["value"],
            }
        )
    checks.append(
        {
            "step": "total",
            "claimed": trace.get("total"),
            "expected": own["total"],
            "pass": trace.get("total") == own["total"],
        }
    )
    failed = [c for c in checks if not c["pass"]]
    return {
        "verdict": "pass" if not failed else "fail",
        "first_failure": failed[0] if failed else None,
        "checks": checks,
        "checker_total": own["total"],
        "proof": own,
    }


def _main() -> None:
    from common import DATA, RESULTS, read_jsonl

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("question", help="question id, e.g. heldout-007")
    ap.add_argument("--arm", help="verify this arm's reply (b)")
    ap.add_argument("--model", default="qwen3-8b")
    a = ap.parse_args()
    split = a.question.split("-")[0]
    q = next(r for r in read_jsonl(DATA / f"baggage_{split}.jsonl") if r["id"] == a.question)
    if not a.arm:
        print(json.dumps(evaluate(q["info"]), indent=1))
        return
    scored = read_jsonl(RESULTS / "scored" / f"baggage_{split}_{a.model}_{a.arm}.jsonl")
    row = next(r for r in scored if r["id"] == a.question)
    print(json.dumps(row.get("verification"), indent=1))


if __name__ == "__main__":
    _main()
