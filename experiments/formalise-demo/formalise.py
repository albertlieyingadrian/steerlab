"""Formalise a slice of RuleArena's airline-baggage rules into machine-checkable rules, each traced to its source line.

    .venv/bin/python experiments/formalise-demo/formalise.py

Reads vendor/rulearena_airline/reference_rules_textual.txt and writes:
  spec/baggage_spec.json        every rule in the slice: what kind it is (definition, condition, exception,
                                combination, or a rule the text never states), when it applies, what it decides,
                                and the source line it came from (number and verbatim text);
  spec/baggage_rules_slice.txt  the slice's source lines, verbatim, each prefixed with its original line number
                                (the text pasted into the prompt in arms b and c).

The fee lines are parsed from the text; only the mapping from each line's route phrase to the slice's regions is
written by hand (ROUTES below), because that mapping is the judgement a formaliser has to make. Every hand-written
rule asserts that its line still says what the rule claims, so an edit to the source fails loudly.
"""

from __future__ import annotations

import json
import re

from common import CLASSES, REGIONS, SOURCE, SPEC_DIR, TO_REGION, TO_US, UPSTREAM

FEE_LINE = re.compile(
    r"^\* If your flight is (?P<route>.*?) AND (?:you are a (?P<cls>.*?) [Cc]lass passenger|the bag size is "
    r"(?P<size>.*?)), the (?:fee should be \$(?P<fee>\d+)|bag should be (?P<refused>Not accepted))\.?$"
)
CLASS_ONLY_LINE = re.compile(r"^\* If you are a (?P<cls>.*?) Class passenger, the fee should be \$(?P<fee>\d+)\.$")

# Which slice regions (and, for Cuba, which direction) each route phrase covers, per section. None = both directions.
ROUTES: dict[str, dict[str, list[tuple[str, int | None]]]] = {
    "First Bag": {
        "Within and between U.S., Puerto Rico and U.S. Virgin Islands": [("U.S.", None)],
        "Between Mexico, Caribbean (excluding Cuba/Haiti), Central America (excluding Panama) and U.S., Puerto Rico, "
        "U.S. Virgin Islands, Canada": [("Mexico", None)],
        "From U.S., Puerto Rico, U.S. Virgin Islands, Canada, Mexico and the Caribbean (excluding Haiti) to Cuba": [
            ("Cuba", TO_REGION)
        ],
        "From Cuba to U.S., Puerto Rico, U.S. Virgin Islands, Canada, Mexico and the Caribbean (including Haiti)": [
            ("Cuba", TO_US)
        ],
        "Between Europe and U.S., Puerto Rico or connecting via Europe to another destination": [("Europe", None)],
        "To or from India, China, Japan, South Korea, Hong Kong, Australia and New Zealand": [
            ("Japan", None),
            ("Australia", None),
        ],
    },
    "Second Bag": {
        "Within and between U.S., Canada, Puerto Rico and U.S. Virgin Islands": [("U.S.", None)],
        "Between Mexico, Caribbean (excluding Cuba/Haiti) or Central America (excluding Panama and Honduras) and U.S., "
        "Puerto Rico, U.S. Virgin Islands, Canada": [("Mexico", None)],
        "From all regions to Cuba": [("Cuba", TO_REGION)],
        "From Cuba to all regions": [("Cuba", TO_US)],
        "Between Europe, Israel, Qatar and U.S., Puerto Rico, U.S. Virgin Islands, Canada, Mexico, Caribbean "
        "(excluding Cuba), Central America, South America": [("Europe", None)],
        "To or from India, China, Japan, South Korea, Hong Kong, Australia and New Zealand": [
            ("Japan", None),
            ("Australia", None),
        ],
    },
    "Third Bag": {
        "Within and between U.S., Canada, Puerto Rico and U.S. Virgin Islands": [("U.S.", None)],
        "Between Mexico, Caribbean, Central America, South America and U.S., Puerto Rico, U.S. Virgin Islands, Canada; "
        "or Between the Caribbean and Central America": [("Mexico", None)],
        "Between Europe, Israel, Qatar and U.S., Puerto Rico, U.S. Virgin Islands, Canada, Mexico, Caribbean, Central "
        "America, South America": [("Europe", None)],
        "To or from India, China, Japan, South Korea, Hong Kong, Australia and New Zealand": [
            ("Japan", None),
            ("Australia", None),
        ],
    },
    "Over 53 lbs / 24 kgs to 70 lbs / 32 kgs": {
        "Within and between U.S., Canada, Puerto Rico and U.S. Virgin Islands": [("U.S.", None)],
        "To / From Cuba": [("Cuba", None)],
        "Between Mexico, Caribbean or Central America, South America and U.S., Puerto Rico, U.S. Virgin Islands, "
        "Canada; or between the Caribbean and Central America": [("Mexico", None)],
        "Between Europe, Israel, Qatar and U.S., Puerto Rico, U.S. Virgin Islands, Canada, Mexico, Caribbean, Central "
        "America, South America": [("Europe", None)],
        "To or from Australia and New Zealand": [("Australia", None)],
        "To or from India, China, Japan, South Korea and Hong Kong": [("Japan", None)],
    },
    "Over 70 lbs / 32kgs to 100 lbs / 45 kgs": {
        "Within and between U.S., Canada, Puerto Rico and U.S. Virgin Islands": [("U.S.", None)],
        "Between Mexico, Caribbean, Central America, South America and U.S., Puerto Rico, U.S. Virgin Islands, Canada; "
        "or between the Caribbean and Central America": [("Mexico", None)],
        "Between Europe, Israel, Qatar and U.S., Puerto Rico, U.S. Virgin Islands, Canada, Mexico, Caribbean, Central "
        "America, South America": [("Europe", None)],
        "To or from Australia and New Zealand": [("Australia", None)],
        "To or from India, China, Japan, South Korea and Hong Kong": [("Japan", None)],
    },
    "Oversize Bags": {
        "Between U.S., Puerto Rico, U.S. Virgin Islands and Canada": [("U.S.", None)],
        "To or from Mexico / Caribbean / Central America (except Panama) / Guyana / Suriname": [
            ("Mexico", None),
            ("Cuba", None),  # Cuba is in the Caribbean; no oversize line names it
        ],
        "To or from Europe, Israel, Qatar": [("Europe", None)],
        "To or from China, Japan, South Korea, Hong Kong, India, Australia, and New Zealand": [
            ("Japan", None),
            ("Australia", None),
        ],
    },
}
POSITION = {"First Bag": 1, "Second Bag": 2, "Third Bag": 3}
WEIGHT_BAND = {
    "Over 50 lbs / 23 kgs to 53 lbs / 24 kgs": [50, 53],
    "Over 53 lbs / 24 kgs to 70 lbs / 32 kgs": [53, 70],
    "Over 70 lbs / 32kgs to 100 lbs / 45 kgs": [70, 100],
}
SIZE_BAND = {"Over 62 in / 158 cm – 65 in / 165 cm": [62, 65], "Over 65 in / 165 cm – 115 in / 292 cm": [65, 115]}

# Rules stated in prose (one line each), plus what the text never states. `needle` must appear on the line.
PROSE = [
    dict(
        line=5,
        needle="1 carry-on bag",
        kind="carry_on_count",
        role="definition",
        formal="One carry-on item is allowed; it is not a checked bag and costs nothing.",
        params={"max": 1},
    ),
    dict(
        line=20,
        needle="22 x 14 x 9 inches",
        kind="carry_on_size",
        role="definition",
        formal="An item fits as a carry-on when its sorted dimensions are within 22 x 14 x 9 inches. In every "
        "question, item 1 is the only item that fits; every other item is a checked bag.",
        params={"max_dims": [22, 14, 9]},
    ),
    dict(
        line=270,
        needle="length + width + height",
        kind="size_measure",
        role="definition",
        formal="size = length + width + height (inches).",
    ),
    dict(
        line=273,
        needle="Dimension: 62 in",
        kind="size_allowance",
        role="condition",
        formal="A checked bag of size <= 62 pays no oversize fee.",
        params={"max_size": 62},
    ),
    dict(
        line=274,
        needle="Weight: 50 lbs",
        kind="weight_allowance",
        role="condition",
        formal="A checked bag of weight <= 50 lbs pays no overweight fee.",
        params={"max_weight": 50},
    ),
    dict(
        line=275,
        needle="For First / Business, weight is 70 lbs",
        kind="weight_allowance",
        role="exception",
        overrides="L274",
        when={"classes": ["Business", "First"], "complimentary": True},
        formal="Exception to L274: for Business and First, a complimentary bag may weigh up to 70 lbs.",
        params={"max_weight": 70},
    ),
    dict(
        line=279,
        needle="Weight: 70 lbs / 32kgs for complimentary bags",
        kind="weight_allowance",
        role="exception",
        overrides="L274",
        when={"regions": ["Australia"], "complimentary": True},
        formal="Exception to L274: to or from Australia, a complimentary bag may weigh up to 70 lbs, any class.",
        params={"max_weight": 70},
    ),
    dict(
        line=283,
        needle="The higher fee between oversize and overweight applies",
        kind="combine",
        role="combination",
        formal="bag fee = base fee + max(overweight fee, oversize fee).",
    ),
    dict(
        line=103,
        needle="Main Plus includes 1 extra free checked bag",
        kind="main_plus",
        role="exception",
        overrides="Main Cabin first-bag lines",
        formal="Exception: Main Plus gets one more free checked bag than Main Cabin on the same route, at most two "
        "free in all; past those, Main Plus pays the Main Cabin fee.",
        params={"extra_free": 1, "max_free": 2},
    ),
    dict(
        line=164,
        needle="Main Plus includes 1 extra free checked bag",
        kind="main_plus",
        role="exception",
        overrides="Main Cabin second-bag lines",
        formal="The same footnote, repeated under the second-bag table.",
        params={"extra_free": 1, "max_free": 2},
    ),
]
UNSTATED = [
    dict(
        id="X-ORDER",
        kind="order",
        role="unstated",
        source={"file": "vendor/rulearena_airline/compute_answer.py", "lines": "52-63"},
        formal="Which checked bag counts as the first, second and third is not stated. The reference answer gives "
        "the free (complimentary) positions to the bags whose overweight fee drops most when complimentary, "
        "the order that minimises the total.",
    ),
    dict(
        id="X-COMP",
        kind="complimentary",
        role="interpretation",
        source={"file": "vendor/rulearena_airline/compute_answer.py", "lines": "66"},
        formal="A checked bag is complimentary when its base fee is $0 (the text uses the word without defining it).",
    ),
    dict(
        id="X-MAINPLUS",
        kind="class_alias",
        role="gap",
        source={"file": "vendor/rulearena_airline/compute_answer.py", "lines": "121-167"},
        formal="No overweight or oversize line names Main Plus; it follows the Main Cabin lines (the reference "
        "answer does the same).",
    ),
    dict(
        id="X-CUBA3",
        kind="slice_limit",
        role="gap",
        source={"file": "vendor/rulearena_airline/gen_questions.py", "lines": "145-146"},
        formal="The third-bag table has no line for Cuba, so a third checked bag to or from Cuba is undefined; "
        "RuleArena never asks about one, and neither does this slice.",
    ),
]


def section_of(lines: list[str]) -> list[str]:
    """The innermost heading above each line (index i is line i + 1)."""
    out, current = [], ""
    for text in lines:
        if text.startswith("#"):
            current = text.lstrip("#").strip()
        out.append(current)
    return out


def fee_rules(lines: list[str]) -> list[dict]:
    sections = section_of(lines)
    rules = []
    for n, (text, section) in enumerate(zip(lines, sections, strict=True), 1):
        m = CLASS_ONLY_LINE.match(text)
        if m and section in WEIGHT_BAND:
            rules.append(
                dict(
                    id=f"L{n}",
                    line=n,
                    text=text,
                    kind="weight_fee",
                    role="condition",
                    when={"band": WEIGHT_BAND[section], "regions": list(REGIONS), "direction": None, "class": m["cls"]},
                    fee=int(m["fee"]),
                )
            )
            continue
        m = FEE_LINE.match(text)
        if not m or section not in ROUTES or m["route"] not in ROUTES[section]:
            continue
        for region, direction in ROUTES[section][m["route"]]:
            when: dict = {"regions": [region], "direction": direction}
            if section in POSITION:
                kind = "base_fee"
                when |= {"position": POSITION[section], "class": m["cls"].rstrip("^")}
            elif section in WEIGHT_BAND:
                kind = "weight_fee"
                when |= {"band": WEIGHT_BAND[section], "class": m["cls"]}
            else:
                kind = "size_fee"
                when |= {"band": SIZE_BAND[m["size"]]}
            fee = int(m["fee"]) if m["fee"] else None
            rules.append(
                dict(
                    id=f"L{n}",
                    line=n,
                    text=text,
                    kind=kind,
                    role="condition",
                    when=when,
                    fee=fee,
                    refused=bool(m["refused"]),
                )
            )
    return merge_regions(rules)


def merge_regions(rules: list[dict]) -> list[dict]:
    """One rule per source line: a line covering Japan and Australia becomes one rule with two regions."""
    by_id: dict[str, dict] = {}
    for r in rules:
        if r["id"] in by_id:
            by_id[r["id"]]["when"]["regions"] += r["when"]["regions"]
        else:
            by_id[r["id"]] = r
    return list(by_id.values())


def prose_rules(lines: list[str]) -> list[dict]:
    rules = []
    for p in PROSE:
        text = lines[p["line"] - 1]
        if p["needle"] not in text:
            raise SystemExit(f"line {p['line']} no longer says {p['needle']!r}: {text!r}")
        rule = {k: v for k, v in p.items() if k != "needle"}
        rules.append(dict(id=f"L{p['line']}", text=text, **rule))
    return rules


def slice_text(lines: list[str], used: set[int]) -> str:
    """The pasted rules: the headings and intro lines of every section the slice uses, plus every rule line."""
    keep = set(used) | {
        1,
        2,
        4,
        16,
        17,
        19,
        25,
        26,
        28,
        30,
        32,
        34,
        36,
        105,
        107,
        166,
        168,
        269,
        272,
        277,
        281,
        285,
        287,
        294,
        296,
        328,
        330,
        357,
        359,
    }
    return "\n".join(f"L{n}: {lines[n - 1]}" for n in sorted(keep)) + "\n"


def lint(rules: list[dict]) -> list[str]:
    """Every input the slice allows must be decided by exactly one fee rule per component."""
    problems = []
    base = [r for r in rules if r["kind"] == "base_fee"]
    for region in REGIONS:
        for direction in (TO_REGION, TO_US):
            for cls in CLASSES:
                if cls == "Main Plus":
                    continue  # decided by L103/L164 on top of the Main Cabin lines
                for pos in range(1, (2 if region == "Cuba" else 3) + 1):
                    hits = [
                        r["id"]
                        for r in base
                        if region in r["when"]["regions"]
                        and r["when"]["position"] == pos
                        and r["when"]["class"] == cls
                        and r["when"]["direction"] in (None, direction)
                    ]
                    if len(hits) != 1:
                        problems.append(f"base fee {region} dir={direction} {cls} bag {pos}: {hits}")
    weight = [r for r in rules if r["kind"] == "weight_fee"]
    for region in REGIONS:
        for cls in [c for c in CLASSES if c != "Main Plus"]:
            for band in ([50, 53], [53, 70], [70, 100]):
                hits = [
                    r["id"]
                    for r in weight
                    if region in r["when"]["regions"] and r["when"]["band"] == band and r["when"]["class"] == cls
                ]
                if len(hits) != 1 and not (region == "Cuba" and band == [70, 100]):
                    problems.append(f"weight fee {region} {cls} {band}: {hits}")
    size = [r for r in rules if r["kind"] == "size_fee"]
    for region in REGIONS:
        for band in ([62, 65], [65, 115]):
            hits = [r["id"] for r in size if region in r["when"]["regions"] and r["when"]["band"] == band]
            if len(hits) != 1:
                problems.append(f"size fee {region} {band}: {hits}")
    return problems


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").split("\n")
    rules = fee_rules(lines) + prose_rules(lines)
    problems = lint(rules)
    if problems:
        raise SystemExit("spec lint failed:\n  " + "\n  ".join(problems))
    rules += UNSTATED
    spec = {
        "name": "baggage-slice",
        "source": {
            "file": "vendor/rulearena_airline/reference_rules_textual.txt",
            "upstream": UPSTREAM + "reference_rules_textual.txt",
            "licence": "MIT (RuleArena)",
        },
        "slice": {"regions": REGIONS, "classes": CLASSES, "checked_bags": [1, 3], "max_checked": {"Cuba": 2}},
        "rules": sorted(rules, key=lambda r: (r.get("line", 10_000), r["id"])),
    }
    SPEC_DIR.mkdir(exist_ok=True)
    (SPEC_DIR / "baggage_spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    used = {r["line"] for r in rules if "line" in r}
    (SPEC_DIR / "baggage_rules_slice.txt").write_text(slice_text(lines, used), encoding="utf-8")
    roles: dict[str, int] = {}
    for r in rules:
        roles[r["role"]] = roles.get(r["role"], 0) + 1
    print(f"spec: {len(rules)} rules from {len(used)} source lines ({roles}); lint clean")


if __name__ == "__main__":
    main()
