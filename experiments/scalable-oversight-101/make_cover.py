"""Draw the cover and research-page thumbnail for the scalable-oversight 101 note.

    .venv/bin/python experiments/scalable-oversight-101/make_cover.py

A small model (the stand-in for a human) labels examples; some labels are wrong (orange). A larger model learns from
those labels. The bar underneath is the score we will report: performance gap recovered, from 0 (no better than the
teacher) to 1 (as good as with perfect labels). The result is not in yet, so the marker is a question mark.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK, SOFT, RULE, EMBER, PAPER = "#1d272a", "#586468", "#c9d0d2", "#c9561c", "#ffffff"
# 12 example labels from the weak teacher: True = right, False = wrong (a fixed illustration, not data)
LABELS = [True, True, False, True, True, True, False, True, True, False, True, True]

fig = plt.figure(figsize=(16, 9), dpi=100)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(PAPER)

# the weak teacher and the strong student
ax.add_patch(Circle((2.6, 5.4), 0.9, facecolor="none", edgecolor=INK, linewidth=2.5))
ax.text(2.6, 5.4, "0.5B", ha="center", va="center", fontsize=20, color=INK, family="monospace")
ax.text(2.6, 3.9, "weak teacher\n(stands in for us)", ha="center", va="top", fontsize=17, color=SOFT)
ax.add_patch(Circle((13.2, 5.4), 1.7, facecolor="none", edgecolor=INK, linewidth=3.5))
ax.text(13.2, 5.4, "1.5B", ha="center", va="center", fontsize=28, color=INK, family="monospace")
ax.text(13.2, 3.1, "strong student", ha="center", va="top", fontsize=17, color=SOFT)

# the labels travelling from teacher to student
for i, right in enumerate(LABELS):
    x, y = 4.7 + 0.55 * i, 5.4 + (0.35 if i % 2 else -0.35)
    face = INK if right else EMBER
    ax.add_patch(FancyBboxPatch((x - 0.18, y - 0.18), 0.36, 0.36, boxstyle="round,pad=0.02", facecolor=face, edgecolor="none"))  # fmt: skip
ax.add_patch(FancyArrowPatch((3.6, 5.4), (11.4, 5.4), arrowstyle="-|>", mutation_scale=22, color=RULE, linewidth=1.5, zorder=0))  # fmt: skip
ax.text(8.0, 6.45, "labels, some wrong", ha="center", fontsize=16, color=SOFT)

# the score: performance gap recovered
ax.plot([2.0, 14.0], [1.5, 1.5], color=INK, linewidth=2)
for x, label in [(2.0, "0\nno better than\nthe teacher"), (14.0, "1\nas good as with\nperfect labels")]:
    ax.plot([x, x], [1.3, 1.7], color=INK, linewidth=2)
    ax.text(x, 1.05, label, ha="center", va="top", fontsize=13, color=SOFT, linespacing=1.3)
ax.text(8.0, 2.05, "?", ha="center", va="bottom", fontsize=40, color=EMBER, fontweight="bold")
ax.text(8.0, 1.05, "performance gap recovered", ha="center", va="top", fontsize=15, color=INK)

fig.savefig(HERE / "report" / "figures" / "hero.jpg", dpi=100, facecolor=PAPER, pil_kwargs={"quality": 90})
thumb = HERE.parents[0] / "shared" / "thumbs" / "scalable-oversight-101.jpg"
fig.set_size_inches(9, 6)
ax.set_xlim(0.8, 15.8)  # 15 wide by 10 tall: the thumbnail's 3:2, so circles stay round
ax.set_ylim(-0.3, 9.7)
fig.savefig(thumb, dpi=80, facecolor=PAPER, pil_kwargs={"quality": 88})
print("wrote", HERE / "report" / "figures" / "hero.jpg", "and", thumb)
