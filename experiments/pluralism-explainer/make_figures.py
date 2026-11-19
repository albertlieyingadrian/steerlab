"""Static fallback figures (PNG) for the note, drawn from report/data.json at each interactive figure's default view.

    python make_figures.py

Prints the numbers it plots so they can be checked against the text.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plural

FIG = plural.HERE / "report" / "figures"
INK, SOFT, FAINT, ACCENT, RULE = "#1d272a", "#586468", "#8a9497", "#c9561c", "#e3e7e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": FAINT, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip


def main() -> None:
    data = json.loads((plural.HERE / "report" / "data.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    q = next(x for x in data["questions"] if x["id"] == "goqa-2474")
    S = data["summary"]

    # Figure 1 fallback: Indonesia's split on the work question
    h = q["human"]["Indonesia"]
    fig, ax = plt.subplots(figsize=(6.4, 2.0), dpi=200)
    ax.barh(range(3)[::-1], h, color=SOFT, height=0.55)
    for i, v in enumerate(h):
        ax.text(v + 0.01, 2 - i, f"{v:.0%}", va="center", color=INK)
    ax.set_yticks(range(3)[::-1], q["options"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    mean = sum(p * i for i, p in enumerate(h))
    ax.set_title(f"Indonesia: 'less importance placed on work'. Average position = '{q['options'][round(mean)]}', "
                 f"held by {h[round(mean)]:.0%}", fontsize=9, color=INK, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "split.png")
    plt.close(fig)
    print("Indonesia split", [round(x, 3) for x in h], "mean position", round(mean, 2))

    # Figure 2 fallback: four countries
    cs = ["United States", "Philippines", "Singapore", "Malaysia"]
    shades = ["#3d4a4e", "#9aa4a6", "#d7dcdd"]
    fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=200)
    for r, c in enumerate(cs[::-1]):
        acc = 0
        for i, p in enumerate(q["human"][c]):
            ax.barh(r, p, left=acc, color=shades[i], height=0.6, label=q["options"][i] if r == 0 else None)
            if p > 0.08:
                ax.text(
                    acc + p / 2, r, f"{p:.0%}", ha="center", va="center", color="white" if i == 0 else INK, fontsize=8
                )
            acc += p
        print(c, [round(x, 3) for x in q["human"][c]])
    ax.set_yticks(range(len(cs)), cs[::-1])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "countries.png")
    plt.close(fig)

    # Figure 3 fallback: distance to Indonesia, default vs steered
    ms = data["models"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    H = S["human"]["Indonesia"]
    for name, v, ls in [("floor", H["floor"]["mean"], "-"), ("avg of 8", H["avg_human_sampled"]["mean"], "--"),
                        ("uniform", H["uniform_sampled"]["mean"], ":")]:  # fmt: skip
        ax.axvline(v, color=FAINT, ls=ls, lw=1)
        ax.text(v, len(ms) - 0.3, f" {name} {v:.2f}", color=SOFT, fontsize=7, rotation=0, va="bottom")
    for r, m in enumerate(ms[::-1]):
        P = S["per_model"][m["key"]]["jsd"]
        for cond, col, dy, face in [("default", FAINT, 0.15, "white"), ("Indonesia", ACCENT, -0.15, ACCENT)]:
            v = P[cond]["Indonesia"]
            ax.plot([v["lo"], v["hi"]], [r + dy, r + dy], color=col, lw=1.2)
            ax.plot(v["mean"], r + dy, "o", mfc=face, mec=col, ms=5)
        print(
            m["label"],
            "JSD to Indonesia default",
            P["default"]["Indonesia"]["mean"],
            "steered",
            P["Indonesia"]["Indonesia"]["mean"],
        )
    ax.set_yticks(range(len(ms)), [m["label"] for m in ms[::-1]])
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.6, len(ms) + 0.4)
    ax.set_xlabel(
        "Mean JS distance to Indonesia's survey split (lower = closer); hollow: no country, filled: 'I live in Indonesia.'",
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(FIG / "distance.png")
    plt.close(fig)

    # Figure 4 fallback: heatmap default
    C = data["countries"]
    M = [[S["per_model"][m["key"]]["jsd"]["default"][c]["mean"] for c in C] for m in ms]
    M.append([S["human"][c]["floor"]["mean"] for c in C])
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    ax.imshow(M, cmap="Oranges", vmin=0.05, vmax=0.6, aspect="auto")
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.42 else INK)
    ax.set_xticks(
        range(len(C)), [c.replace("United States", "US").replace("South Korea", "Korea") for c in C], fontsize=7
    )
    ax.set_yticks(range(len(M)), [m["label"] for m in ms] + ["Floor"], fontsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "heat.png")
    plt.close(fig)
    print("heat rows", [[round(v, 3) for v in r] for r in M])
    assert not any(math.isnan(v) for r in M for v in r)


if __name__ == "__main__":
    main()
