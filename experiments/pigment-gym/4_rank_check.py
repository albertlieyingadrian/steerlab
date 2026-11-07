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

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brush_tools import Renderer, contact_sheet
from common import HERE, check, outputs_dir
from hand_pool import POOL, SUBJECT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
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

# For the note: the 12 sketches in the order I ranked them by hand, each with the reward it got.
GREY, EMBER = "#8A847C", "#C9561C"
by_hand = sorted(range(len(POOL)), key=lambda i: (POOL[i]["rank"], i))
fig, axes = plt.subplots(2, 6, figsize=(10, 4.1))
for ax, i in zip(axes.flat, by_hand, strict=False):
    png = renders[i].png
    if png and rows[i]["gate"]:
        ax.imshow(Image.open(io.BytesIO(png)).convert("RGB"))
    else:
        ax.imshow(Image.new("RGB", (400, 400), "#ececec"))
        why = "crashes" if png is None or "js_error" in rows[i]["violations"] else "fails the gate"
        ax.text(0.5, 0.5, why, ha="center", va="center", transform=ax.transAxes, fontsize=7.5, color=GREY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B9B2A6")
        spine.set_linewidth(0.6)
    rank = "tied last" if POOL[i]["rank"] > 9 else f"#{POOL[i]['rank']:g}"
    ax.set_title(f"{rank} · {POOL[i]['name']}", fontsize=7, pad=3, loc="left")
    colour = "#2B2723" if rows[i]["gate"] else EMBER
    ax.set_xlabel(f"reward {rows[i]['reward']:.2f}", fontsize=8, color=colour, labelpad=3)
fig.tight_layout(w_pad=0.6, h_pad=1.0)
REFERENCE_FIGURE = HERE / "report" / "figures" / "reference_sketches.png"
fig.savefig(REFERENCE_FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved the pool in hand order for the note: {REFERENCE_FIGURE.name}")

# For the note: how each scorer orders the 8 passing sketches, next to the intended order. One line per sketch.
columns = [
    ("Intended order", None),
    ("Full reward", [row["reward"] for row in rows]),
    ("CLIP alone", [row["clip"] for row in rows]),
    ("Aesthetic alone", [row["aesthetic"] for row in rows]),
]
by_intent = sorted(passing, key=lambda i: POOL[i]["rank"])
palette = ["#3F7D4E", "#C9561C", "#2F6DB0", "#8A4FB0", "#D19A2E", "#2B9C9C", "#9A948C", "#B5476B"]
colour_of = {i: palette[n] for n, i in enumerate(by_intent)}
positions = {}
for c, (_, scores) in enumerate(columns):
    ordered = by_intent if scores is None else sorted(passing, key=lambda i: -scores[i])
    for place, i in enumerate(ordered):
        positions[(c, i)] = place
fig, ax = plt.subplots(figsize=(10.5, 5.2))
xs = [3.1 + 2.0 * c for c in range(len(columns))]
for i in passing:
    ax.plot(xs, [positions[(c, i)] for c in range(len(columns))], "-o", color=colour_of[i], lw=2.2, ms=7, zorder=2)
    for c, (_, scores) in enumerate(columns):
        if scores is not None:
            ax.text(xs[c] + 0.16, positions[(c, i)], f"{scores[i]:.2f}", fontsize=7.5, color=colour_of[i], va="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"}, zorder=3)
    image = Image.open(io.BytesIO(renders[i].png)).convert("RGB").resize((90, 90))
    ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.36), (0.25, positions[(0, i)]), frameon=False))
    ax.text(0.62, positions[(0, i)], POOL[i]["name"], va="center", fontsize=8, color=colour_of[i])
for c, (title, _) in enumerate(columns):
    ax.text(xs[c], -0.9, title, ha="center", fontsize=9, fontweight="bold")
ax.set_xlim(-0.2, xs[-1] + 0.7)
ax.set_ylim(len(passing) - 0.4, -1.3)
ax.axis("off")
RANK_FIGURE = HERE / "report" / "figures" / "reference_ranks.png"
fig.savefig(RANK_FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"saved how each scorer orders the passing sketches: {RANK_FIGURE.name}")

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
