"""The voices for `deslop voice`: passages by five public-domain writers, split into samples and held-out passages.

    python experiments/deslop/2_voices.py        # writes data/voices/passages.jsonl (raw downloads cached)

Writers (Project Gutenberg, all published before 1929, public domain in the US): Mark Twain (What Is Man? and Other
Essays; Life on the Mississippi), Abraham Lincoln (Papers and Writings, volumes 5-7: 1858-1865), Charles Darwin (The
Voyage of the Beagle; On the Origin of Species), Virginia Woolf (The Common Reader, 1925) and G. K. Chesterton
(Tremendous Trifles; All Things Considered). For each, paragraphs of 120 to 400 words are shuffled (seed 0): 30 become
samples (the profile and the examples a rewriter sees) and the next 20 are held out (the attribution check). Lincoln's
volumes include the editor's notes; paragraphs that look like them (third person about Lincoln) are dropped.

Each row: id, author, role ("sample" or "heldout"), ref, title, year, licence, text.
"""

from __future__ import annotations

import importlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
human = importlib.import_module("0_human")

OUT = HERE / "data" / "voices" / "passages.jsonl"
AUTHORS = {
    "twain": [(70, "What Is Man? and Other Essays", 1917), (245, "Life on the Mississippi", 1883)],
    "lincoln": [(2657, "Papers and Writings, vol. 5", 1862), (2658, "Papers and Writings, vol. 6", 1863),
                (2659, "Papers and Writings, vol. 7", 1865)],
    "darwin": [(944, "The Voyage of the Beagle", 1839), (1228, "On the Origin of Species", 1859)],
    "woolf": [(64457, "The Common Reader", 1925)],
    "chesterton": [(8092, "Tremendous Trifles", 1909), (11505, "All Things Considered", 1908)],
}  # fmt: skip
SAMPLES, HELDOUT = 30, 20
EDITOR = re.compile(r"\b(?:Lincoln|Mr\. Lincoln|the President)\b(?: was| had| said| wrote)")


def paragraphs(gid: int) -> list[str]:
    raw = human.fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", human.RAW / f"pg{gid}.txt")
    body = raw.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", body)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG", body)
    body = body[start.end() if start else 0 : end.start() if end else len(body)]
    out = []
    for block in re.split(r"\n\s*\n", body):
        text = re.sub(r"\s+", " ", block).strip()
        text = re.sub(r"\[(?:Footnote|Illustration|\d+)[^\]]*\]", "", text).replace("_", "").strip()
        if 120 <= human.words(text) <= 400 and not text.isupper() and text[-1:] in ".!?\"'”’)":
            out.append(text)
    return out


def main() -> None:
    rows = []
    for author, books in AUTHORS.items():
        pool = []
        for gid, title, year in books:
            for text in paragraphs(gid):
                if author == "lincoln" and EDITOR.search(text):
                    continue
                pool.append((gid, title, year, text))
        rng = random.Random(0)
        rng.shuffle(pool)
        for k, (gid, title, year, text) in enumerate(pool[: SAMPLES + HELDOUT]):
            role = "sample" if k < SAMPLES else "heldout"
            rows.append({"id": f"{author}-{role[0]}{k:02d}", "author": author, "role": role, "ref": f"gutenberg:{gid}",
                         "title": title, "year": year, "licence": "public domain (US)", "text": text})  # fmt: skip
        print(f"{author:10} {len(pool):4} paragraphs, kept {min(len(pool), SAMPLES + HELDOUT)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} passages to {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
