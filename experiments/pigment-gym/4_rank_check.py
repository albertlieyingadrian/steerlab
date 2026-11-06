"""
Does the reward rank paintings the way a person would?

Run from the repo root:
    .venv/bin/python experiments/pigment-gym/4_rank_check.py

A reward function is a test oracle for taste. Before you let RL optimise it for hours, you check it
against cases where you already know the answer, like unit tests. Here the "known answers" are the
12 hand-ranked sketches in hand_pool.py, and the metric is Spearman's rank correlation:
    +1  = the reward orders them exactly like I did
     0  = no relationship
    -1  = exactly backwards
Spearman only compares orderings, so it doesn't care that rewards are 0.3-0.9 while ranks are 1-12.

We also compare the full reward with each term alone, and show the reward hack the gate blocks.
"""

from brush_tools import Renderer, contact_sheet
from common import check, outputs_dir
from hand_pool import POOL, SUBJECT
from PIL import Image, ImageDraw, ImageFont
from reward_tools import CLIP_RANGE, Scorer, caption, score_rollouts, squash
from scipy.stats import spearmanr

scorer = Scorer()
codes = [item["code"] for item in POOL]
with Renderer(pool_size=4) as renderer:
    renders = renderer.render_many(codes)
rows = score_rollouts(codes, renders, scorer, SUBJECT)
hand_ranks = [item["rank"] for item in POOL]

print(f"{'hand rank':>9} {'sketch':24} {'reward':>6}")
order = sorted(range(len(POOL)), key=lambda i: -rows[i]["reward"])
for i in order:
    print(f"{POOL[i]['rank']:>9} {POOL[i]['name']:24} {rows[i]['reward']:6.3f}   {', '.join(rows[i]['violations'])}")


def rho(scores, indices=None):
    """Spearman between scores and hand quality. Hand rank 1 = best, so negate the ranks."""
    indices = indices if indices is not None else range(len(POOL))
    return spearmanr([scores[i] for i in indices], [-hand_ranks[i] for i in indices]).statistic


passing = [i for i, row in enumerate(rows) if row["gate"]]
variants = {
    "full reward": [row["reward"] for row in rows],
    "gate only": [row["gate"] for row in rows],
    "CLIP only (gated)": [row["gate"] * row["clip"] for row in rows],
    "aesthetic only (gated)": [row["gate"] * row["aesthetic"] for row in rows],
}
print(f"\nSpearman vs my hand ranking   {'all 12':>7} {'8 passing':>10}")
results = {}
for name, scores in variants.items():
    # "gate only" is 1 for every passing sketch: a constant has no ordering, so skip that column
    results[name] = (rho(scores), None if name == "gate only" else rho(scores, passing))
    passing_text = "n/a" if name == "gate only" else f"{results[name][1]:.2f}"
    print(f"  {name:28} {results[name][0]:7.2f} {passing_text:>10}")
print("""
Reading it: on all 12, everything looks good, because 'broken vs working' is the easy part and the gate
nails it. The honest test is the 8 that pass, where only taste is left. Compare the full reward with each
taste term alone: CLIP knows 'sunset' but not 'pretty', the aesthetic head knows 'pretty-ish' but not
'sunset'. Eight items is a tiny test set: one swapped pair moves rho by ~0.1. Treat it as a smoke test.""")

# The hack the gate exists for: pixels that spell the subject.
written = Image.new("RGB", (400, 400), "#f4efe4")
draw = ImageDraw.Draw(written)
font = ImageFont.load_default(size=40)
draw.text((20, 140), "SUNSET OVER", fill="#c8553d", font=font)
draw.text((20, 200), "GREEN HILLS", fill="#386641", font=font)
hack_clip, _ = scorer.score_images([written], caption(SUBJECT))
best_clip = max(rows[i]["clip"] for i in passing)
print(
    f"\nif the gate let text through, the written words would get clip term {squash(hack_clip[0], *CLIP_RANGE):.2f};"
    f" the best real painting gets {best_clip:.2f}."
)
print("  Writing the subject costs one line of code; painting it well took me 60. RL finds the cheap route.")
print(f"with the gate, the text sketch scores {rows[9]['reward']:.3f} (violations: {rows[9]['violations']}).")

labels = [f"#{n + 1} r={rows[i]['reward']:.2f} {POOL[i]['name']}" for n, i in enumerate(order)]
sheet = contact_sheet([renders[i].png for i in order], labels, outputs_dir() / "ranked_pool.png")
print(f"\nsaved the pool sorted by reward: the work folder {sheet.name}")

print("\nSelf-checks:")
check("Spearman over all 12 is above 0.8", results["full reward"][0] > 0.8)
check("Spearman over the 8 passing sketches is above 0.5", results["full reward"][1] > 0.5)
check(
    "the full reward beats aesthetic alone on the passing 8",
    results["full reward"][1] > results["aesthetic only (gated)"][1],
)
check("every gate failure scores at most 0.05", all(row["reward"] <= 0.05 + 1e-9 for row in rows if not row["gate"]))
check(
    "without the gate, three words of text would match the best real painting on CLIP (within 0.05)",
    squash(hack_clip[0], *CLIP_RANGE) > best_clip - 0.05,
)
