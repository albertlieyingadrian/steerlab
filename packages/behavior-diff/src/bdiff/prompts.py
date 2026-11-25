"""Expand items × condition factors into chat prompts.

A spec is JSON:

    {"items": [{"id": "q1", "kind": "advice", "text": "Should I ..."}],
     "factors": {"persona": {"none": {}, "eval": {"system": "You are being evaluated."}},
                 "stakes": {"none": {}, "high": {"suffix": "My job depends on this."}}},
     "draws": 3, "seed": 0}

Each level may set `system` (the system prompt), `prefix` (put before the item text) and `suffix` (put after it).
Prefixes and suffixes are joined in the order the factors are listed. With `"design": "factorial"` every combination
of levels is used once per item; the default (`"random"`) draws every factor's level uniformly and independently,
`draws` times per item, which keeps the prompt count flat as factors are added.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

Prompt = dict[str, Any]


def render(text: str, levels: list[dict[str, str]]) -> list[dict[str, str]]:
    """The chat messages for one item under one combination of factor levels."""
    system = " ".join(lv["system"] for lv in levels if lv.get("system"))
    prefix = " ".join(lv["prefix"] for lv in levels if lv.get("prefix"))
    suffix = " ".join(lv["suffix"] for lv in levels if lv.get("suffix"))
    user = " ".join(part for part in (prefix, text) if part)
    if suffix:
        user = f"{user}\n\n{suffix}"
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": user}]


def expand(spec: dict[str, Any]) -> list[Prompt]:
    """Every prompt the spec defines, with its item, kind and condition levels recorded."""
    factors: dict[str, dict[str, dict[str, str]]] = spec.get("factors", {})
    names = list(factors)
    rng = random.Random(spec.get("seed", 0))
    out: list[Prompt] = []
    for item in spec["items"]:
        if spec.get("design", "random") == "factorial":
            combos = [dict(zip(names, c, strict=True)) for c in itertools.product(*(factors[n] for n in names))]
        else:
            combos = [{n: rng.choice(list(factors[n])) for n in names} for _ in range(int(spec.get("draws", 1)))]
        for k, combo in enumerate(combos):
            levels = [factors[n][combo[n]] for n in names]
            out.append(
                {
                    "id": f"{item['id']}.{k}",
                    "item": item["id"],
                    "kind": item.get("kind", "item"),
                    "conditions": combo,
                    "messages": render(item["text"], levels),
                    **({"answer": item["answer"]} if "answer" in item else {}),
                }
            )
    return out
