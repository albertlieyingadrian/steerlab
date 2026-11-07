"""
Draw the guide's figure from results.tsv and print the numbers it plots.

    .venv/bin/python experiments/pigment-verifier/make_figures.py

search.png: agreement with the judges on the dev pairs for every run of the reward-method search, with the baselines
as lines and the held-out test score of each finalist as a diamond.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "experiments" / "creative-rewards" / "report" / "figures"
EMBER, GREY, INK = "#C9561C", "#9A948C", "#1d272a"

rows = list(csv.DictReader((HERE / "results.tsv").open(), delimiter="\t"))
dev = [r for r in rows if r["split"] == "dev"]
test = {r["run"]: float(r["acc"]) for r in rows if r["split"] == "test"}
first = {}
for r in dev:
    first.setdefault(r["run"], float(r["acc"]))

fig, ax = plt.subplots(figsize=(10, 4.2))
xs = [int(r["run"]) for r in dev]
ys = [float(r["acc"]) for r in dev]
ax.scatter(xs, ys, s=22, color=GREY, zorder=2, label="a search run (dev pairs)")
running, best_so_far = [], 0.0
for y in ys:
    best_so_far = max(best_so_far, y)
    running.append(best_so_far)
ax.step(xs, running, where="post", color=EMBER, linewidth=2, label="best so far (dev)")
ax.axhline(first["000"], color=INK, linestyle="--", linewidth=1)
ax.text(35.5, first["000"] + 0.006, "CLIP + aesthetic", ha="right", va="bottom", fontsize=8, color=INK)
ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1)
ax.text(35.5, 0.506, "plain judge (all ties)", ha="right", va="bottom", fontsize=8, color=GREY)
for run, acc in test.items():
    ax.scatter([int(run)], [acc], marker="D", s=46, color=EMBER if run == "030" else INK, zorder=3)
ax.scatter([], [], marker="D", color=INK, label="held-out test, scored once")
ax.set_xlabel("search run")
ax.set_ylabel("agreement with the judges")
ax.set_ylim(0.45, 1.0)
ax.set_xlim(-1, 36)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8, frameon=False)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "search.png", dpi=170)
print("dev:", [(r["run"], r["acc"]) for r in dev])
print("test:", test)
