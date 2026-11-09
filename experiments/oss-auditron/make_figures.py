"""Draw the note's figures from results/ and print every number plotted.

    python experiments/oss-auditron/make_figures.py

halves.png  flag TPR in rounds 1-25 and 26-50 for each model-free auditor on the 20 held-out games, with 95%
            game-bootstrap intervals
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, GREY, SOFT = "#C9561C", "#9A948C", "#586468"
AUDITORS = [("random", "random"), ("cheapest", "cheapest\n(never flags)"), ("price", "price rule"),
            ("history", "failure-history\nrule")]  # fmt: skip

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT})


def halves() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    for i, (name, _label) in enumerate(AUDITORS):
        r = json.loads((RESULTS / f"{name}.json").read_text())
        for j, (half, colour) in enumerate([("rounds_1_25", GREY), ("rounds_26_50", EMBER)]):
            m = r[half]["flag_tpr"]
            lo, hi = m["ci95"]
            x = i + (j - 0.5) * 0.34
            ax.bar(
                x, m["value"], width=0.32, color=colour, label=("rounds 1–25", "rounds 26–50")[j] if i == 0 else None
            )
            ax.errorbar(x, m["value"], yerr=[[m["value"] - lo], [hi - m["value"]]], color="black", lw=1, capsize=3)
            print(f"{name:9s} {half}: flag TPR {m['value']:.3f} [{lo:.3f}, {hi:.3f}] ({m['count']})")
    ax.set_xticks(range(len(AUDITORS)), [label for _, label in AUDITORS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("flag TPR (share of cheating bids flagged)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "halves.png")
    plt.close(fig)


if __name__ == "__main__":
    halves()
