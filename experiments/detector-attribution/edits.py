"""Arms 4-6: the texts sent to Pangram, built deterministically. Written to data/pangram_items.jsonl (gitignored,
because some are edited human passages); results/pangram_items.json keeps every item's key, group, source key,
variant, word count and text hash, but no text.

    python edits.py            # build the items and print their billed blocks and cost

Groups:
- `deletion`: 60 confirmation AI texts (30 `original`, 30 `plain+sampler`, one sample per prompt, chosen by seed),
  each as the cut original and six deletion variants, exactly the texts the open detectors scored in attribute.py:
  top10:9b, rand10:0, top25:9b, rand25:0, topsent25, randsent25:0.
- `repeat`: 10 of those originals sent a second time (Pangram's repeat noise).
- `deletion50` (exploratory, added after the first Pangram results): the same 60 texts with half their words
  deleted, the 9B's top half and a random half.
- `typo` (exploratory, added after the word lists): those originals that contain a curly apostrophe, with every
  curly apostrophe made straight.
- `insert`: 40 confirmation human passages: original, AI-tell edits, neutral edits. Both edited arms change the same
  positions with the same operation; only the inserted strings differ (rule-based, no model writes them).
- `splice`: 30 texts of about 150 human words then 150 AI words (confirmation replies not used above, any arm),
  order alternating; the boundary's character offset is kept.
"""

import json
import random
import re

from attribute import confirm_ai, top_words, variants
from common import (
    DATA,
    DETECTORS,
    RESULTS,
    SEED,
    blocks,
    corpus,
    delete_spans,
    save_json,
    sentences,
    text_hash,
    words,
    write_jsonl,
)

USD_PER_BLOCK = 0.04
DEL_VARIANTS = ["orig", "top10:9b", "rand10:0", "top25:9b", "rand25:0", "topsent25", "randsent25:0"]

# The fixed edit lists (plan, arm 5). Same operation, same position; only the strings differ.
OPENERS = {
    "ai": ["Moreover,", "Additionally,", "Furthermore,", "Notably,", "Importantly,", "Ultimately,"],
    "neutral": ["Also,", "Then,", "Still,", "So,", "Now,", "Yet,"],
}
TAILS = {
    "ai": [
        ", highlighting its significance",
        ", underscoring its enduring importance",
        ", showcasing its intricate nature",
        ", a testament to its pivotal role",
        ", fostering a deeper understanding",
    ],
    "neutral": [
        ", as it happened",
        ", at least for a time",
        ", or so it seemed then",
        ", as before",
        ", in the usual way",
    ],
}
SWAPS = {
    "important": ("crucial", "big"),
    "great": ("pivotal", "large"),
    "very": ("incredibly", "quite"),
    "many": ("numerous", "lots of"),
    "use": ("leverage", "employ"),
    "help": ("foster", "aid"),
    "show": ("showcase", "display"),
    "shows": ("showcases", "displays"),
    "complex": ("intricate", "involved"),
    "various": ("multifaceted", "several"),
    "rich": ("vibrant", "full"),
    "look": ("delve", "peer"),
    "also": ("additionally", "too"),
}
DASH = {"ai": "—", "neutral": "; "}
COMMON_OPENERS = {
    "the",
    "it",
    "he",
    "she",
    "they",
    "this",
    "there",
    "in",
    "a",
    "an",
    "we",
    "but",
    "and",
    "his",
    "her",
    "their",
    "at",
    "on",
    "when",
    "as",
    "after",
    "for",
    "its",
    "that",
    "these",
    "those",
    "one",
}
EDITS_PER_100 = 6


def plan_edits(text: str, rng: random.Random) -> list[tuple]:
    """Choose edit sites once: (kind, char position or span, index or word)."""
    sents = sentences(text)
    n_target = round(EDITS_PER_100 * len(text.split()) / 100)
    sites = []
    for i, (s, e) in enumerate(sents[1:], 1):
        sites.append(("opener", s, i))
        if text[e - 1 : e] in ".!?" and e - s > 40:
            sites.append(("tail", e - 1, i))
    for m in re.finditer(r", (?=[a-z])", text):
        sites.append(("dash", m.span(), 0))
    for s, e in words(text):
        w = text[s:e]
        if w.lower() in SWAPS:
            sites.append(("swap", (s, e), w))
    rng.shuffle(sites)
    used, out = [], []
    for site in sites:
        pos = site[1] if isinstance(site[1], int) else site[1][0]
        if any(abs(pos - u) < 25 for u in used):
            continue
        used.append(pos)
        out.append(site)
        if len(out) >= n_target:
            break
    return out


def apply_edits(text: str, sites: list[tuple], arm: str) -> str:
    """Apply from the end backwards so earlier offsets stay valid."""

    def key(site):
        return site[1] if isinstance(site[1], int) else site[1][0]

    t = text
    for kind, where, extra in sorted(sites, key=key, reverse=True):
        if kind == "opener":
            first = re.match(r"[A-Za-z']+", t[where:])
            rest = t[where:]
            if first and first.group(0).lower() in COMMON_OPENERS and first.group(0) != "I":
                rest = rest[0].lower() + rest[1:]
            t = t[:where] + OPENERS[arm][extra % len(OPENERS[arm])] + " " + rest
        elif kind == "tail":
            t = t[:where] + TAILS[arm][extra % len(TAILS[arm])] + t[where:]
        elif kind == "dash":
            s, e = where
            t = t[:s] + DASH[arm] + t[e:]
        elif kind == "swap":
            s, e = where
            new = SWAPS[extra.lower()][0 if arm == "ai" else 1]
            if extra[0].isupper():
                new = new[0].upper() + new[1:]
            t = t[:s] + new + t[e:]
    return t


def _first_words(text: str, n: int) -> str:
    """The longest sentence-ending prefix of at most n words (at least 100), else the first n words."""
    best = None
    for _, e in sentences(text):
        if len(text[:e].split()) > n:
            break
        best = e
    return text[:best].strip() if best and len(text[:best].split()) >= 100 else " ".join(text.split()[:n])


def build() -> list[dict]:
    rows = corpus()
    attrs = {n: json.loads((RESULTS / "attr" / f"{n}.json").read_text()) for n in DETECTORS}
    occl = json.loads((RESULTS / "attr" / "occlusion.json").read_text())
    base = json.loads((RESULTS / "scores" / "corpus.json").read_text())
    rng = random.Random(f"{SEED}-pangram")
    items = []
    conf = confirm_ai()
    chosen = []
    for arm in ("original", "plain+sampler"):
        by_prompt: dict[str, list] = {}
        for r in conf:
            if r["arm"] == arm:
                by_prompt.setdefault(r["prompt"], []).append(r)
        prompts = sorted(by_prompt)
        rng.shuffle(prompts)
        picked = [rng.choice(sorted(by_prompt[p], key=lambda r: r["key"])) for p in prompts[:30]]
        for p in prompts:  # fewer than 30 confirmation prompts: add the other sample of the first prompts
            if len(picked) >= 30:
                break
            picked += [r for r in by_prompt[p] if r not in picked][:1]
        chosen += picked[:30]
    assert len(chosen) == 60, len(chosen)
    for r in chosen:
        a = {n: attrs[n][r["key"]]["w"] for n in DETECTORS}
        var = dict(variants(r, a, occl[r["key"]], base["9b"][r["key"]]))
        var["orig"] = r["text"]
        for v in DEL_VARIANTS:
            items.append(
                {"key": f"del/{r['key']}/{v}", "group": "deletion", "src": r["key"], "variant": v, "text": var[v]}
            )
    for r in chosen[::6]:
        items.append(
            {"key": f"rep/{r['key']}", "group": "repeat", "src": r["key"], "variant": "orig", "text": r["text"]}
        )
    for r in chosen:  # exploratory (amendment 6): half the words deleted, the 9B's top half vs a random half
        spans = words(r["text"])
        k = round(0.5 * len(spans))
        pick = random.Random(f"{SEED}-{r['key']}-50").sample(range(len(spans)), k)
        for v, idx in (("top50:9b", top_words(attrs["9b"][r["key"]]["w"], k)), ("rand50:0", pick)):
            items.append(
                {
                    "key": f"del50/{r['key']}/{v}",
                    "group": "deletion50",
                    "src": r["key"],
                    "variant": v,
                    "text": delete_spans(r["text"], [spans[i] for i in idx]),
                }
            )
    for r in chosen:  # exploratory (amendment 4): the same texts with curly apostrophes straightened
        if "\u2019" in r["text"]:
            items.append(
                {
                    "key": f"typo/{r['key']}",
                    "group": "typo",
                    "src": r["key"],
                    "variant": "straight",
                    "text": r["text"].replace("\u2019", "'"),
                }
            )
    hum = [r for r in rows if r["kind"] == "human" and r["split"] == "confirm"]
    rng.shuffle(hum)
    for r in hum[:40]:
        sites = plan_edits(r["text"], random.Random(f"{SEED}-{r['key']}"))
        for arm in ("orig", "ai", "neutral"):
            t = r["text"] if arm == "orig" else apply_edits(r["text"], sites, arm)
            items.append(
                {
                    "key": f"ins/{r['key']}/{arm}",
                    "group": "insert",
                    "src": r["key"],
                    "variant": arm,
                    "text": t,
                    "n_edits": len(sites),
                }
            )
    hpool = hum[40:]
    chosen_keys = {c["key"] for c in chosen}
    apool = [r for r in conf if r["key"] not in chosen_keys]  # any arm: too few `original` texts are left
    rng.shuffle(apool)
    for i in range(30):
        h, a = _first_words(hpool[i]["text"], 150), _first_words(apool[i]["text"], 150)
        first, second = (h, a) if i % 2 == 0 else (a, h)
        items.append(
            {
                "key": f"spl/{i:02d}",
                "group": "splice",
                "src": f"{hpool[i]['key']}|{apool[i]['key']}",
                "variant": "human-first" if i % 2 == 0 else "ai-first",
                "text": first + "\n\n" + second,
                "boundary": len(first) + 2,
            }
        )
    return items


def main() -> None:
    items = build()
    write_jsonl(DATA / "pangram_items.jsonl", items)
    meta = [
        {k: v for k, v in it.items() if k != "text"}
        | {"words": len(it["text"].split()), "blocks": blocks(it["text"]), "sha": text_hash(it["text"])}
        for it in items
    ]
    save_json("pangram_items.json", meta)
    by: dict[str, int] = {}
    for m in meta:
        by[m["group"]] = by.get(m["group"], 0) + m["blocks"]
    n = sum(by.values())
    print(f"{len(items)} items, {n} blocks, ${n * USD_PER_BLOCK:.2f}; by group {by}")


if __name__ == "__main__":
    main()
