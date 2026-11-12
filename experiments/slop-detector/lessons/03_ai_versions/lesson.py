r"""Lesson 03: what each AI version's prompts look like, and what they cost. Offline: no API calls.

cd experiments/slop-detector \&\& ../../.venv/bin/python lessons/03_ai_versions/lesson.py
"""

import importlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
gen = importlib.import_module("02_generate")

doc = {
    "id": "amazon/demo/1",
    "source": "amazon",
    "words": 41,
    "text": "Bought this for my mom. The pump broke after two weeks and the lotion inside is runny, smells nice "
    "though. Seller replaced it fast so 3 stars. Wouldnt buy again at this price honestly, theres better "
    "stuff at the drugstore.",
}

print("MIRROR, call 1 (sees the original):")
print("  Summarise the text below in exactly two lines, then give a line 'Keywords:' ...")
print("MIRROR, call 2 (never sees the original, only the summary):")
print(f"  Write {gen.KIND[doc['source']]}, about {doc['words']} words long, based only on this summary ...\n")
i = random.Random(doc["id"]).randrange(len(gen.REWRITES))
print(f"REWRITE (one of {len(gen.REWRITES)} instructions, picked by the document id; this one is #{i}):")
for k, r in enumerate(gen.REWRITES):
    print(f"  {'->' if k == i else '  '} {k}. {r}")
print("\nROUND TRIP: two calls, original -> Spanish / German / Russian -> English.\n")

print("Price per item, from the table in 02_generate.py (per million tokens, in / out):")
for m, (fam, pin, pout, w) in gen.GENERATORS.items():
    tokens_in, tokens_out = 450, 300  # a typical item, all its calls together
    print(f"  {m:46} {fam:9} ${(tokens_in * pin + tokens_out * pout) / 1e6:.5f}  (weight {w})")
print("\nMultiply by the number of items for a run's cost; the pilot's meter is in data/ai.jsonl (field 'cost').")
