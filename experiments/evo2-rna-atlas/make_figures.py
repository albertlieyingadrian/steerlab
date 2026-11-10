"""Draw the note's figures from results/ (and the context codes in $EVO2_RNA_DIR), printing every number plotted.

    python experiments/evo2-rna-atlas/make_figures.py

track.png     the best feature along 2 kb of E. coli DNA around the lysC lysine riboswitch, read as it is (orange)
              and with only the riboswitch's letters dinucleotide-shuffled (grey); the Infernal hit is shaded.
heldout.png   average precision of every method on the 12,746 held-out units (31 riboswitches), with 95% bootstrap
              intervals; the dashed line is the share of positive units (what a random ranking scores).
shuffle.png   for each of the 66 riboswitch hits, the best feature's peak over the hit: real letters against the mean
              of two dinucleotide shuffles.
families.png  nearest-neighbour family accuracy on Rfam seed sequences, with BLAST-linked neighbours excluded.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlas import DATA, RESULTS

FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK, MUTED = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723", "#8A847C"
TRACK_HIT = "ecoli|4233324|-|Lysine"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)

LABELS = {
    "best feature": "Evo 2: one SAE feature",
    "SAE probe": "Evo 2: probe on SAE features",
    "embedding probe": "Evo 2: probe on layer-26 vector",
    "random feature (median of 500)": "Evo 2: random SAE feature",
    "entropy": "Evo 2: next-letter entropy",
    "4-mer probe": "probe on 4-letter words",
    "length": "region length",
    "gc": "GC content",
    "fold": "strongest RNA fold",
    "blast": "BLAST vs Rfam examples",
    "blast_distant": "BLAST, near-identical left out",
}
EVO_METHODS = {"best feature", "SAE probe", "embedding probe"}


def feature_track(codes, item: int, feature: int, length: int) -> np.ndarray:
    track = np.zeros(length)
    mask = codes["pos_item"] == item
    idx, ids, vals = codes["pos_idx"][mask], codes["pos_ids"][mask], codes["pos_vals"][mask].astype(float)
    hit = ids == feature
    rows = hit.any(1)
    track[idx[rows]] = (vals * hit).sum(1)[rows]
    return track


def track(summary: dict) -> None:
    context = np.load(DATA / "evo2" / "context.npz")
    r = [str(context["item_ids"][i]) for i in context["rng_item"]].index(f"real|{TRACK_HIT}")
    lo, hi = int(context["rng_lo"][r]), int(context["rng_hi"][r])
    codes = np.load(DATA / "evo2" / "tracks.npz")
    ids = [str(i) for i in codes["item_ids"]]
    best = summary["best_feature"]
    real, shuf = ids.index(f"real|{TRACK_HIT}"), ids.index(f"shuf1|{TRACK_HIT}")
    length = int(codes["pos_idx"][codes["pos_item"] == real].max()) + 1
    a, b = feature_track(codes, real, best, length), feature_track(codes, shuf, best, length)
    x = np.arange(length) - lo
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.axvspan(0, hi - lo, color=EMBER, alpha=0.08, lw=0)
    ax.plot(x, b + 0.04, color=GREY, lw=1.2, label="riboswitch letters shuffled (same letter pairs)")
    ax.plot(x, a, color=EMBER, lw=1.1, label="real sequence")
    ax.set_xlim(-600, x[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("position relative to the start of the Infernal hit (letters, read 5′ to 3′)")
    ax.set_ylabel(f"feature {best}")
    ax.text(-8, ax.get_ylim()[1] * 0.97, "lysine riboswitch\n(Infernal)", fontsize=8.5, color=INK, va="top", ha="right")
    # lysC (minus strand, 4,231,884-4,233,233 in NC_000913.3) starts 91 letters after the hit ends.
    ax.axvline(hi - lo + 91, color=INK, lw=0.7, ls=":")
    ax.text(hi - lo + 97, ax.get_ylim()[1] * 0.97, "lysC\nstart", fontsize=8.5, color=INK, va="top")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 0.8))
    fig.savefig(FIGURES / "track.png", bbox_inches="tight", facecolor="white")
    outside = np.delete(a, np.arange(lo, hi))
    print(f"  track: hit {TRACK_HIT}, {hi - lo} letters; peak real {a[lo:hi].max():.2f}, shuffled {b[lo:hi].max():.2f};"
          f" peak outside the hit {outside.max():.2f}")  # fmt: skip


def heldout(summary: dict) -> None:
    rows = sorted(summary["table"], key=lambda r: r["ap"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, row in enumerate(rows):
        if row["method"] in EVO_METHODS:
            c = EMBER
        elif row["method"].startswith("blast"):
            c = GREY
        else:
            c = LIGHT
        ax.barh(i, max(row["ap"], 0.004), 0.62, color=c)
        low, high = row["ap_ci"]
        ax.errorbar(row["ap"], i, xerr=[[row["ap"] - low], [high - row["ap"]]], color=INK, capsize=2, lw=0.8)
        ax.text(high + 0.015, i, f"{row['ap']:.2f}", va="center", fontsize=9, color=INK)
        print(f"  {row['method']:<32} AP {row['ap']:.3f} [{low:.3f}, {high:.3f}]  AUROC {row['auroc']:.3f}")
    ax.axvline(summary["prevalence_test"], color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["method"]] for r in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("average precision on the held-out genomes (1 = every riboswitch ranked first)")
    ax.grid(axis="x", color="#E6E1D9", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / "heldout.png", bbox_inches="tight", facecolor="white")
    print(f"  prevalence {summary['prevalence_test']:.4f}; wrote report/figures/heldout.png")


def shuffle() -> None:
    rows = list(csv.DictReader((RESULTS / "context.csv").open()))
    real = np.array([float(r["real_max"]) for r in rows])
    shuf = np.array([float(r["shuffled_max"]) for r in rows])
    held = np.array([r["genome"] != "bsub" for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    limit = max(real.max(), shuf.max()) * 1.05
    ax.plot([0, limit], [0, limit], color=MUTED, lw=0.9, ls="--")
    ax.scatter(shuf[~held], real[~held], s=18, color=EMBER, alpha=0.45, linewidths=0, label="B. subtilis (search set)")
    ax.scatter(
        shuf[held], real[held], s=18, color=EMBER, alpha=0.95, linewidths=0, marker="D", label="held-out genomes"
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("peak with the riboswitch's letters shuffled")
    ax.set_ylabel("peak with the real letters")
    above = int((real > shuf).sum())
    ax.text(0.04, 0.96, f"above the line: real higher\n{above} of {len(rows)} hits", transform=ax.transAxes, va="top",
            fontsize=9, color=INK)  # fmt: skip
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIGURES / "shuffle.png", bbox_inches="tight", facecolor="white")
    print(
        f"  shuffle: real above shuffled on {above}/{len(rows)}; held-out {int((real > shuf)[held].sum())}/{held.sum()}"
    )


def families(summary: dict) -> None:
    rows = [r for r in summary["families"]["rows"] if r["representation"] != "chance"]
    chance = next(r["accuracy"] for r in summary["families"]["rows"] if r["representation"] == "chance")
    rows = sorted(rows, key=lambda r: r["accuracy"])
    names = {"shuffled embedding": "Evo 2 vector of shuffled letters", "Evo 2 embedding": "Evo 2 layer-26 vector",
             "SAE features": "Evo 2 SAE features", "4-mer": "4-letter-word counts", "3-mer": "3-letter-word counts",
             "length and GC": "length and GC"}  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, row in enumerate(rows):
        if row["representation"] in ("Evo 2 embedding", "SAE features"):
            c = EMBER
        elif row["representation"] == "shuffled embedding":
            c = GREY
        else:
            c = LIGHT
        ax.barh(i, row["accuracy"], 0.62, color=c)
        ax.errorbar(row["accuracy"], i, xerr=[[row["accuracy"] - row["low"]], [row["high"] - row["accuracy"]]],
                    color=INK, capsize=2, lw=0.8)  # fmt: skip
        ax.text(row["high"] + 0.012, i, f"{row['accuracy']:.0%}", va="center", fontsize=9, color=INK)
        print(f"  {row['representation']:<20} {row['correct']}/{row['queries']} = {row['accuracy']:.3f}"
              f" [{row['low']:.3f}, {row['high']:.3f}]")  # fmt: skip
    ax.axvline(chance, color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names[r["representation"]] for r in rows])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("nearest neighbour is from the same family (share of 1,500 queries)")
    ax.grid(axis="x", color="#E6E1D9", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / "families.png", bbox_inches="tight", facecolor="white")
    print(f"  chance {chance:.3f}; wrote report/figures/families.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    track(summary)
    heldout(summary)
    shuffle()
    families(summary)


if __name__ == "__main__":
    main()
