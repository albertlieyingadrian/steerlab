"""Part B, steps B1-B2: build SlopPair (plan: PLAN.md).

    python b_build.py --human <path to the slop-detector's data/human.jsonl>

300 human passages (pre-2023, 43 per source from seven sources, the last 42; 120-300 words cut at a sentence end; no
code, tables or URLs; seed 0), each rewritten by one of three generators (round-robin within source), a fact check
by gpt-4o-mini, a random A/B order per pair, and a dev/test split (100/200, stratified by source).

Writes data/sloppair/pairs.jsonl (gitignored: holds the human text) and results/sloppair_items.json (no human text:
ids, source, generator, the rewrite, the fact check, order and split).
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bench

SOURCES = ("xsum", "arxiv", "hn", "reddit", "stack", "amazon", "medium")
REWRITE = "Rewrite the following passage in your own words. Keep every fact, about the same length and the same register. Output only the passage.\n\n{text}"
FACTS = (
    "Original:\n{a}\n\nRewrite:\n{b}\n\nDoes the rewrite keep every fact of the original without adding new facts? "
    "Answer yes or no."
)
OUT = bench.DATA / "sloppair" / "pairs.jsonl"


def cut(text: str) -> str | None:
    if "```" in text or re.search(r"https?://|www\.", text) or text.count("|") > 3:
        return None
    words = text.split()
    if len(words) < 120:
        return None
    head = " ".join(words[:300])
    ends = [m.end() for m in re.finditer(r"[.!?][\"')\]]?(?=\s|$)", head)]
    ends = [e for e in ends if len(head[:e].split()) >= 120]
    return head[: ends[-1]].strip() if ends else None


def sample(human: Path) -> list[dict]:
    pool: dict[str, list[dict]] = {s: [] for s in SOURCES}
    for line in human.read_text().splitlines():
        r = json.loads(line)
        if r["source"] in pool:
            pool[r["source"]].append(r)
    rng = random.Random(0)
    out = []
    for k, s in enumerate(SOURCES):
        want = 42 if k == len(SOURCES) - 1 else 43
        cands = pool[s][:]
        rng.shuffle(cands)
        got = 0
        for r in cands:
            t = cut(r["text"])
            if t:
                out.append(
                    {
                        "id": r["id"],
                        "source": s,
                        "year": r.get("year"),
                        "human": t,
                        "generator": bench.GENERATORS[got % 3],
                    }
                )
                got += 1
                if got == want:
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True)
    a = ap.parse_args()
    items = sample(a.human)
    print(f"{len(items)} passages; generating rewrites")
    with ThreadPoolExecutor(12) as ex:
        rewrites = list(
            ex.map(lambda r: bench.chat(r["generator"], REWRITE.format(text=r["human"]), max_tokens=1000), items)
        )
        facts = list(
            ex.map(
                lambda rw: bench.chat("openai/gpt-4o-mini", FACTS.format(a=rw[0]["human"], b=rw[1]), max_tokens=3),
                zip(items, rewrites, strict=True),
            )
        )
    rng = random.Random(1)
    by_source: dict[str, list[int]] = {}
    for i, r in enumerate(items):
        by_source.setdefault(r["source"], []).append(i)
    dev = set()
    for _s, idx in by_source.items():
        rng.shuffle(idx)
        dev |= set(idx[: round(len(idx) / 3)])
    rest = [i for i in range(len(items)) if i not in dev]
    rng.shuffle(rest)
    while len(dev) < 100:
        dev.add(rest.pop())
    while len(dev) > 100:
        dev.remove(max(dev))
    for i, (r, rw, f) in enumerate(zip(items, rewrites, facts, strict=True)):
        r["ai"] = rw
        r["facts_kept"] = f.strip().lower().startswith("yes")
        r["human_is"] = rng.choice("AB")
        r["split"] = "dev" if i in dev else "test"
        r["canary"] = bench.CANARY
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in items))
    public = [
        {k: v for k, v in r.items() if k != "human"}
        | {"human_words": len(r["human"].split()), "ai_words": len(r["ai"].split())}
        for r in items
    ]
    (bench.RESULTS / "sloppair_items.json").write_text(
        json.dumps({"canary": bench.CANARY, "items": public}, indent=1) + "\n"
    )
    n_test = sum(r["split"] == "test" for r in items)
    print(
        f"dev {len(items) - n_test}, test {n_test}, facts kept {sum(r['facts_kept'] for r in items)}/{len(items)}, spent ${bench.spent():.2f}"
    )


if __name__ == "__main__":
    main()
