r"""Lesson 04: the label is "share of words changed". Measure it on toy edits, then weight the bands.

cd experiments/slop-detector \&\& ../../.venv/bin/python lessons/04_labels/lesson.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import band, frac_changed

human = "the pump broke after two weeks and the lotion is runny but it smells nice so three stars"
edits = {
    "same text": human,
    "fix grammar": "The pump broke after two weeks, and the lotion is runny, but it smells nice, so three stars.",
    "concise": "Pump broke in two weeks; runny lotion, nice smell. Three stars.",
    "own words": "Within a fortnight the dispenser failed, and while the cream is thin, its scent is pleasant: 3/5.",
    "unrelated": "Our quarterly revenue grew by twelve percent thanks to strong demand in the European market.",
}
print("fraction of words changed (word-level diff, 0 = same words, 1 = nothing kept) and the band it falls in:")
for name, text in edits.items():
    f = frac_changed(human, text)
    print(f"  {name:12} {f:.2f}   band {int(band(f))}  ({['human', 'edited', 'generated'][int(band(f))]})")
print("Note 'fix grammar': punctuation is not a word, so it scores 0.00 and lands in the human band, and below 2% the")
print("row is dropped anyway. 'own words' changed 78% and already counts as AI-generated. 'unrelated' scores 0.94,")
print("not 1, because short common words line up by chance: that is why a mirror gets 1 by construction.")

# band-balanced weights: each band carries the same total weight in the loss
targets = [0.0] * 900 + [0.4] * 2400 + [1.0] * 1200  # a skewed mix: many edits, few humans
counts = Counter(int(band(t)) for t in targets)
total = len(targets)
print("\nband counts:", dict(counts))
for b, c in sorted(counts.items()):
    w = total / (3 * c)
    print(f"  band {b}: {c:5} rows x weight {w:.3f} = {c * w:.0f}   (every band sums to {total / 3:.0f})")
