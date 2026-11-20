"""Generate the baggage questions (dev and held-out) and their answers, then freeze them.

    .venv/bin/python experiments/formalise-demo/questions.py

Follows RuleArena's generator (vendor/rulearena_airline/gen_questions.py: same prompt template, cities, ticket
prices, carry-on and bag-shape sampling) with three changes: only the slice's six regions (picked uniformly, then a
city); 1-3 checked bags (RuleArena's easiest level has 4); and each checked bag is overweight with p = 0.6 and
oversize with p = 0.6 independently (RuleArena makes every bag both, so the allowance rules never decide anything).
The answer is RuleArena's own reference code (vendor/rulearena_airline/compute_answer.py), run unchanged.

Writes data/baggage_dev.jsonl (40), data/baggage_heldout.jsonl (200) and data/manifest.json (sha256 of each file).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import random
import sys

import numpy as np

from common import CLASSES, DATA, MAX_CHECKED, REGIONS, VENDOR, sha256, write_jsonl

sys.path.insert(0, str(VENDOR))
with contextlib.chdir(VENDOR):  # compute_answer loads its fee tables from a relative path at import
    reference = importlib.import_module("compute_answer")
    rulearena_gen = importlib.import_module("gen_questions")

SPLITS = {"dev": (1, 40), "heldout": (2, 200)}
CITIES = {
    region: [c for c, r in rulearena_gen.international_city_collection.items() if r == region]
    for region in REGIONS
    if region != "U.S."
}


def reference_total(info: dict) -> int:
    total, _ = reference.compute_answer(**info, check_base_tables=reference.check_base_tables)
    return int(total)


def sample_bag(rng: np.random.Generator, region: str, item: int) -> dict:
    """One checked bag, shaped like RuleArena's (gen_questions.py lines 148-182)."""
    if rng.random() < 0.6:
        bands = [(51, 54), (54, 71)] if region in ("Cuba", "Europe", "Australia") else [(51, 54), (54, 71), (71, 101)]
        weight = int(rng.integers(*bands[rng.integers(len(bands))]))
    else:
        weight = int(rng.integers(30, 51))
    if rng.random() < 0.6:
        size = int(rng.integers(*[(63, 66), (66, 116)][rng.integers(2)]))
    else:
        size = int(rng.integers(45, 63))
    height = int(rng.integers(size // 3 - 12, size // 3 - 6))
    length = int(rng.integers(size // 3 + 12, size // 3 + 18))
    name = ["backpack", "luggage box"][rng.integers(2)]
    return {"id": item, "name": name, "size": [length, size - height - length, height], "weight": weight}


def question(rng: np.random.Generator, pick: random.Random) -> dict:
    region = pick.choice(REGIONS)
    cls = pick.choice(CLASSES)
    direction = int(rng.integers(2))
    us_city = pick.choice(rulearena_gen.us_city_collection)
    if region == "U.S.":
        other = pick.choice([c for c in rulearena_gen.us_city_collection if c != us_city])
    else:
        other = pick.choice(CITIES[region])
    departure, destination = (us_city, other) if direction == 0 else (other, us_city)
    ticket = int(rng.integers(*rulearena_gen.base_price_interval[region]))
    length, width, height = (int(rng.integers(18, 23)), int(rng.integers(11, 15)), int(rng.integers(6, 10)))
    bags = [{"id": 1, "name": "backpack", "size": [length, width, height], "weight": int(rng.integers(5, 13))}]
    n_checked = int(rng.integers(1, MAX_CHECKED.get(region, 3) + 1))
    bags += [sample_bag(rng, region, i + 2) for i in range(n_checked)]
    name = pick.choice(rulearena_gen.name_collection)
    lines = [f"{b['id']}. A {b['name']}: {' x '.join(map(str, b['size']))} inches, {b['weight']} lbs;" for b in bags]
    prompt = (
        rulearena_gen.prompt_template.replace("$name", name)
        .replace("$class", cls)
        .replace("$departure", departure)
        .replace("$destination", destination)
        .replace("$base_price", f"${ticket}")
        .replace("$bag_list", "\n".join(lines))
    )
    info = {"base_price": ticket, "customer_class": cls, "routine": region, "direction": direction, "bag_list": bags}
    return {"prompt": prompt, "info": info}


def generate(seed: int, n: int, prefix: str) -> list[dict]:
    rng, pick = np.random.default_rng(seed), random.Random(seed)
    rows = []
    for i in range(n):
        q = question(rng, pick)
        rows.append({"id": f"{prefix}-{i:03d}", **q, "answer": reference_total(q["info"])})
    return rows


def main() -> None:
    manifest = {}
    for split, (seed, n) in SPLITS.items():
        rows = generate(seed, n, split)
        path = DATA / f"baggage_{split}.jsonl"
        write_jsonl(path, rows)
        manifest[path.name] = {"n": n, "seed": seed, "sha256": sha256(path)}
        n_bags = [len(r["info"]["bag_list"]) - 1 for r in rows]
        print(f"{split}: {n} questions, checked bags {dict(sorted({k: n_bags.count(k) for k in set(n_bags)}.items()))}")
    old = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
    (DATA / "manifest.json").write_text(json.dumps(old | manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
