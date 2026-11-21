"""Build every page set, render and extract them, run the checker, and freeze the sets with their hashes.

    .venv/bin/python experiments/brand-spec/dataset.py

Writes, per brand and split (dev, heldout):
  data/pages_<brand>_<split>.jsonl   compliance set: on-brand, exception traps, off-brand with planted violations
  data/copy_<brand>_<split>.jsonl    copy set: references, near-copies (strength 1-3), new designs, remixes (held-out)
  data/demo_govuk.jsonl              the interactive's page and its single-change variants
and data/facts_<file>.jsonl (the extractor's element facts, same order), results/proofs_<file>.jsonl (each page's
verdict, failures and counts; full artifacts for the demo set), data/intended.jsonl.gz (what the generator meant to
render, element by element, for the extractor ruler) and data/manifest.json (sha256 of every file).
Each page row: id, brand, split, kind, page_type, domain, recipe, labels [[element, rule]], planted, traps, html,
and for the copy set `ref` (the reference a near-copy or remix came from) and `strength`.
"""

from __future__ import annotations

import copy
import gzip
import json
import random

from checker import check
from common import BRANDS, DATA, RESULTS, load_spec, sha256, write_json, write_jsonl
from pages import PAGE_TYPES, TRAPS, VIOLATIONS, make_page, random_knobs, random_recipe
from render import extract_pages

SIZES = {"dev": {"on": 4, "trap": 2, "off": 6}, "heldout": {"on": 12, "trap": 6, "off": 22}}
COPY = {
    "dev": {"refs": 6, "near": 15, "new": 15, "remix": 0},
    "heldout": {"refs": 12, "near": 30, "new": 30, "remix": 10},
}
SEED = {"dev": 11, "heldout": 23}
STYLE_KNOBS = {"hero": ["tint", "big"], "cards": ["alt", "small_text"], "text": ["small"]}


def compliance(brand: str, split: str) -> list[dict]:
    r = random.Random(SEED[split] * 100 + BRANDS.index(brand))
    sz = SIZES[split]
    cycle = list(VIOLATIONS[brand]) * 10
    r.shuffle(cycle)
    pages: list[dict] = []
    tnames = list(TRAPS[brand])

    def fresh(kind, viol=(), traps=(), n_traps=0):
        n = sum(q["kind"] == kind for q in pages) + 1
        for attempt in range(600):
            if viol and attempt and attempt % 150 == 0:  # an unbuildable combination: swap one violation
                viol = [*viol[:-1], r.choice([v for v in VIOLATIONS[brand] if v not in viol])]
            if n_traps:
                traps = r.sample(tnames, n_traps)
            rec = random_recipe(brand, r.randrange(10**9))
            p = make_page(f"{brand}-{split}-{kind}{n:02d}", rec, viol, traps, seed=r.randrange(10**9))
            if len(p["planted"]) == len(viol) and len(p["traps"]) == len(traps):
                p["kind"] = kind
                return p
        raise RuntimeError(f"could not build {kind} {viol} {traps}")

    for _ in range(sz["on"]):
        pages.append(fresh("on"))
    for i in range(sz["trap"]):
        pages.append(fresh("trap", n_traps=1 + i % 2))
    for _ in range(sz["off"]):
        k = r.choice([1, 1, 2, 2, 3])
        viol: list[str] = []
        while len(viol) < k:
            v = cycle.pop()
            if v not in viol:
                viol.append(v)
        pages.append(fresh("off", viol=viol))
    for p in pages:
        p["split"] = split
    return pages


def reseed_text(rec: dict, r: random.Random) -> dict:
    rec = copy.deepcopy(rec)
    for s in rec["sections"]:
        s["text_seed"] = r.randrange(10**9)
    for part in ("header", "nav", "footer"):
        rec[part]["text_seed"] = r.randrange(10**9)
    return rec


def restyle(rec: dict, r: random.Random) -> dict:
    rec = copy.deepcopy(rec)
    flips = [(s, kn) for s in rec["sections"] for kn in STYLE_KNOBS.get(s["kind"], [])]
    for s, kn in r.sample(flips, max(1, len(flips) // 2)) if flips else []:
        s["knobs"][kn] = not s["knobs"].get(kn, False)
    return rec


def restructure(rec: dict, r: random.Random) -> tuple[dict, str]:
    rec = copy.deepcopy(rec)
    secs = rec["sections"]
    op = r.choice(["add", "move"] + (["remove"] if len(secs) > 2 else []))
    if op == "add":
        kind = r.choice(PAGE_TYPES[rec["page_type"]][1:])
        secs.insert(
            r.randint(1, len(secs)), {"kind": kind, "knobs": random_knobs(r, kind), "text_seed": r.randrange(10**9)}
        )
    elif op == "remove":
        secs.pop(r.randint(1, len(secs) - 1))
    else:
        i = r.randint(0, len(secs) - 2)
        secs[i], secs[i + 1] = secs[i + 1], secs[i]
    return rec, op


def copyset(brand: str, split: str) -> list[dict]:
    r = random.Random(SEED[split] * 1000 + 7 + BRANDS.index(brand))
    c = COPY[split]
    out = []
    refs = [random_recipe(brand, r.randrange(10**9)) for _ in range(c["refs"])]
    for i, rec in enumerate(refs):
        p = make_page(f"{brand}-{split}-ref{i + 1:02d}", rec)
        p.update(kind="ref", split=split)
        out.append(p)
    for i in range(c["near"]):
        j, strength = i % c["refs"], 1 + i % 3
        rec, op = reseed_text(refs[j], r), None
        if strength >= 2:
            rec = restyle(rec, r)
        if strength == 3:
            rec, op = restructure(rec, r)
        p = make_page(f"{brand}-{split}-near{i + 1:02d}", rec)
        p.update(kind="near", split=split, ref=f"{brand}-{split}-ref{j + 1:02d}", strength=strength, edit=op)
        out.append(p)
    for i in range(c["new"]):
        p = make_page(f"{brand}-{split}-new{i + 1:02d}", random_recipe(brand, r.randrange(10**9)))
        p.update(kind="new", split=split)
        out.append(p)
    for i in range(c["remix"]):
        j = r.randrange(c["refs"])
        rec = random_recipe(brand, r.randrange(10**9), page_type=refs[j]["page_type"], variant=refs[j]["variant"])
        rec["sections"][r.randrange(len(rec["sections"]))] = copy.deepcopy(r.choice(refs[j]["sections"]))
        p = make_page(f"{brand}-{split}-remix{i + 1:02d}", rec)
        p.update(kind="remix", split=split, ref=f"{brand}-{split}-ref{j + 1:02d}")
        out.append(p)
    return out


def demo() -> list[dict]:
    """The interactive: one GOV.UK start page and each single violation or trap that applies to it."""
    r = random.Random(5)
    rec = None
    for _ in range(500):
        cand = random_recipe("govuk", r.randrange(10**9), page_type="start", variant="service")
        kinds = [s["kind"] for s in cand["sections"]]
        if "dark" in kinds and "cards" in kinds and len(kinds) <= 4:
            rec = cand
            break
    assert rec is not None
    rec["nav"]["on"] = False
    rec["sections"].insert(
        1,
        {
            "kind": "form",
            "knobs": {"n_fields": 1, "hints": True, "secondary": True, "warning": False},
            "text_seed": 4242,
        },
    )
    out = [make_page("demo-base", rec)]
    for name in VIOLATIONS["govuk"]:
        if name in ("btn-start", "font-alt"):
            continue
        p = make_page(f"demo-{name}", rec, [name], seed=1)
        if p["planted"]:
            out.append(p)
    for name in ("disabled", "large", "logo", "bold19"):
        p = make_page(f"demo-trap-{name}", rec, traps=[name], seed=1)
        if p["traps"]:
            out.append(p)
    for p in out:
        p.update(kind="demo", split="demo")
    return out


def strip_proof(pr: dict, full: bool) -> dict:
    return pr if full else {k: pr[k] for k in ("page", "brand", "verdict", "counts", "failures", "fail_pairs")}


def main() -> None:
    specs = {b: load_spec(b) for b in BRANDS}
    sets = {}
    for brand in BRANDS:
        for split in ("dev", "heldout"):
            sets[f"pages_{brand}_{split}"] = compliance(brand, split)
            sets[f"copy_{brand}_{split}"] = copyset(brand, split)
    sets["demo_govuk"] = demo()
    manifest, bad, intended = {}, 0, {}
    for name, pages in sets.items():
        facts = extract_pages(pages)
        proofs = [check(specs[p["brand"]], f) for p, f in zip(pages, facts, strict=True)]
        for p, pr in zip(pages, proofs, strict=True):
            got = [list(x) for x in pr["fail_pairs"]]
            if got != p["labels"]:
                bad += 1
                print("RULER MISMATCH", p["id"], p["labels"], got)
        for p in pages:
            intended[p["id"]] = p.pop("intended")
        write_jsonl(DATA / f"{name}.jsonl", pages)
        write_jsonl(DATA / f"facts_{name}.jsonl", facts)
        write_jsonl(RESULTS / f"proofs_{name}.jsonl", [strip_proof(pr, name == "demo_govuk") for pr in proofs])
        for f in (DATA / f"{name}.jsonl", DATA / f"facts_{name}.jsonl"):
            manifest[f.name] = sha256(f)
        kinds: dict[str, int] = {}
        for p in pages:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        print(f"{name}: {len(pages)} pages {kinds}")
    gz = DATA / "intended.jsonl.gz"  # the generator's intended facts, for the extractor ruler (kept compressed)
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        for pid, els in intended.items():
            fh.write(json.dumps({"id": pid, "intended": els}) + "\n")
    write_json(DATA / "manifest.json", manifest)
    print("checker vs planted labels: mismatches", bad)


if __name__ == "__main__":
    main()
