"""Draw the note's figures from results/ and print the numbers each one shows.

    python make_figures.py

Writes report/figures/: fpr.png (held-out web false positives per round and arm), trade.png (false positives against
recall on fresh 2026 pages), recall.png (recall per AI set per round), interp.png (reliance on the corporate and slop
directions per round, with the random band), mined.png (how many pool pages sat above the cut, and the judged
machine-made share), hero.jpg and og.jpg (the cover, drawn from the false-positive curves).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hn import FLAG_CUT, HERE, RESULTS

FIG = HERE / "report" / "figures"
INK, SOFT, ACCENT, CONTROL, RULE, PAPER = "#1d1a17", "#6f675f", "#c9561c", "#2f6db0", "#d9d2c8", "#faf7f2"
GREY = "#8a8178"
ARMS = {
    "a": ("(a) hardest pages", ACCENT, "o", "-"),
    "b": ("(b) hardest + AI pairs", "#e09a5c", "s", "--"),
    "c": ("(c) random pages (control)", CONTROL, "^", ":"),
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def series(R: dict, arm: str, key: str):
    xs, ys, lo, hi = [], [], [], []
    for i in range(5):
        r = "r0" if i == 0 else f"{arm}{i}"
        if r not in R:
            continue
        v = R[r][key]
        xs.append(i)
        ys.append(v["rate"] * 100)
        lo.append((v["rate"] - v["ci"][0]) * 100)
        hi.append((v["ci"][1] - v["rate"]) * 100)
    return xs, ys, [lo, hi]


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


ROUNDS = ["round 0", "round 1", "round 2", "round 3", "round 4"]


def rounds_axis(ax, n: int = 5) -> None:
    ax.set_xticks(range(n))
    ax.set_xticklabels(ROUNDS[:n])
    ax.set_xlim(-0.3, n - 0.7)


def axis_break(ax, lo: float) -> None:
    """The y axis does not start at zero: mark the break at the bottom of the axis and say where it starts."""
    kw = {"transform": ax.transAxes, "color": SOFT, "clip_on": False, "lw": 0.9}
    for dy in (0.0, 0.025):
        ax.plot([-0.025, 0.025], [0.01 + dy, 0.035 + dy], **kw)


def legend_below(fig, axes, ncol: int = 3) -> None:
    h, lab = axes.flat[0].get_legend_handles_labels() if hasattr(axes, "flat") else axes.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))


def fig_fpr(R: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for a, (lab, c, m, ls) in ARMS.items():
        xs, ys, err = series(R, a, "fpr_web")
        off = {"a": -0.08, "b": 0.0, "c": 0.08}[a]
        ax.errorbar([x + off for x in xs], ys, yerr=err, color=c, marker=m, ls=ls, lw=1.6, ms=5, capsize=2, label=lab)
        print("fpr_web", a, [f"{y:.3f}" for y in ys])
    for s in ("r0s1", "a1s1"):
        if s in R:
            x = 0 if s == "r0s1" else 1
            ax.plot(
                [x + 0.16],
                [R[s]["fpr_web"]["rate"] * 100],
                marker="x",
                color=INK,
                ls="none",
                ms=6,
                label="seed-1 rerun" if s == "r0s1" else None,
            )
            print("fpr_web", s, f"{R[s]['fpr_web']['rate'] * 100:.3f}")
    rounds_axis(ax)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("old web pages flagged (%)")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fpr.png")


TRADE = (
    ("recall_fresh_old", "2026 pages, old topics"),
    ("recall_fresh_new", "2026 pages, new topics"),
    ("recall_epoch_imit", "Epoch AI style imitations"),
)


def fig_trade(R: dict, mobile: bool = False) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 7.6)) if mobile else plt.subplots(1, 3, figsize=(6.8, 2.9))
    for ax, (key, title) in zip(axes, TRADE):
        lo = 100
        for a, (lab, c, m, ls) in ARMS.items():
            xs, fp, _ = series(R, a, "fpr_web")
            _, rc, _ = series(R, a, key)
            lo = min(lo, *rc)
            ax.plot(fp, rc, color=c, marker=m, ls=ls, lw=1.4, ms=5, label=lab)
            if a == "a":
                for x, f, r in zip(xs, fp, rc):
                    ax.annotate(f"r{x}", (f, r), textcoords="offset points", xytext=(4, -9), fontsize=7.5, color=SOFT)
            if not mobile:
                print("trade", key, a, [(round(f, 3), round(r, 1)) for f, r in zip(fp, rc)])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("old web pages flagged (%)", fontsize=8.5)
        ax.set_xlim(-0.03, 0.72)
        bottom = 5 * int((lo - 3) // 5)
        ax.set_ylim(bottom, 100.5)
        axis_break(ax, bottom)
        ax.set_ylabel("AI pages caught (%)", fontsize=8.5)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.06 if not mobile else 0.04, 1, 1))
    save(fig, "trade_m.png" if mobile else "trade.png")


RECALL = (
    ("recall_editlens_gen", "EditLens AI-written"),
    ("recall_editlens_edit", "EditLens AI-edited"),
    ("recall_epoch_plain", "Epoch AI plain"),
    ("recall_epoch_imit", "Epoch AI imitations"),
    ("recall_fresh_old", "2026, old topics"),
    ("recall_fresh_new", "2026, new topics"),
)


def fig_recall(R: dict, mobile: bool = False) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(3.8, 7.4)) if mobile else plt.subplots(2, 3, figsize=(6.8, 4.6))
    for ax, (k, t) in zip(axes.flat, RECALL):
        lo = 100
        for a, (lab, c, m, ls) in ARMS.items():
            xs, ys, err = series(R, a, k)
            lo = min(lo, *[y - e for y, e in zip(ys, err[0])])
            ax.errorbar(xs, ys, yerr=err, color=c, marker=m, ls=ls, lw=1.2, ms=3.5, capsize=1.5, label=lab)
            if not mobile:
                print("recall", k, a, [f"{y:.1f}" for y in ys])
        ax.set_title(t, fontsize=8.5, color=INK)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["r0", "1", "2", "3", "4"], fontsize=7.5)
        bottom = 5 * int((lo - 2) // 5)
        ax.set_ylim(bottom, 100.5)
        axis_break(ax, bottom)
        ax.tick_params(axis="y", labelsize=7.5)
    for ax in axes[:, 0] if axes.ndim == 2 else axes:
        ax.set_ylabel("AI caught (%)", fontsize=8)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "recall_m.png" if mobile else "recall.png")


def fig_interp(I: dict) -> None:
    """The head's margin (generated minus human logit), in standard deviations: corporate minus other human pages
    (left), AI mirrors minus their human pages (right). Seed-1 reruns as black crosses."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, c, lab in (
        (axes[0], "corp", "corporate minus other human pages"),
        (axes[1], "slop", "AI mirrors minus the same pages by people"),
    ):
        for a, (alab, col, m, ls) in ARMS.items():
            xs = [i for i in range(5) if ("r0" if i == 0 else f"{a}{i}") in I]
            ys = [I["r0" if i == 0 else f"{a}{i}"][c]["margin"]["gap_sd"] for i in xs]
            ax.plot(xs, ys, color=col, marker=m, ls=ls, lw=1.5, ms=5, label=alab)
            print("interp gap", c, a, [f"{y:+.3f}" for y in ys])
        for s, x in (("r0s1", 0.12), ("a1s1", 1.12)):
            if s in I:
                ax.plot(
                    [x],
                    [I[s][c]["margin"]["gap_sd"]],
                    marker="x",
                    color=INK,
                    ls="none",
                    ms=6,
                    label="seed-1 rerun" if s == "r0s1" else None,
                )
        ax.axhline(0, color=RULE, lw=0.8)
        ax.set_title(lab, fontsize=9, color=INK)
        ax.set_ylabel("margin gap (SD)", fontsize=8.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["r0", "1", "2", "3", "4"])
        ax.set_xlabel("round", fontsize=8.5)
    axes[1].set_ylim(0, 2.3)
    legend_below(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "interp.png")


def fig_mined(M: dict, noise: dict | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for a in ("a", "b"):
        lab, c, m, ls = ARMS[a]
        xs = [i for i in range(1, 5) if f"{a}_r{i}" in M]
        axes[0].plot(xs, [M[f"{a}_r{i}"]["pool_above_cut"] for i in xs], color=c, marker=m, ls=ls, label=lab)
        print("pool above cut", a, [M[f"{a}_r{i}"]["pool_above_cut"] for i in xs])
        if noise:
            ys = [noise.get(f"{a}_r{i}", {}).get("share") for i in xs]
            axes[1].plot(xs, [y * 100 if y is not None else None for y in ys], color=c, marker=m, ls=ls, label=lab)
            print("machine-made share", a, ys)
    if noise and "random" in noise:
        axes[1].axhline(noise["random"]["share"] * 100, color=CONTROL, ls=":", label="random pool pages (arm c)")
    axes[0].set_title("pool pages at or above the cut", fontsize=9, color=INK)
    axes[1].set_title("mined pages judged machine-made (%)", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["round 1", "round 2", "round 3", "round 4"], fontsize=8)
        ax.set_ylim(bottom=0)
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "mined.png")


def hero(R: dict) -> None:
    """Cover: the three arms' false-positive curves as ribbons on the paper ground, no axes."""
    from PIL import Image

    for name, size in (("hero.jpg", (16, 9)), ("og.jpg", (12, 6.3))):
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(PAPER)
        fig.patch.set_facecolor(PAPER)
        top = 0.0
        for a, (_, c, _, _) in ARMS.items():
            xs, ys, err = series(R, a, "fpr_web")
            lo = [y - e for y, e in zip(ys, err[0])]
            hi = [y + e for y, e in zip(ys, err[1])]
            top = max(top, *hi)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.3, lw=0)
            ax.plot(xs, ys, color=c, lw=6 if a == "a" else 3, solid_capstyle="round")
            ax.scatter(xs, ys, color=c, s=180 if a == "a" else 70, zorder=3)
        ax.set_xlim(-0.4, 4.4)
        ax.set_ylim(-top * 0.1, top * 1.15)
        ax.axis("off")
        tmp = FIG / ("_" + name.replace(".jpg", ".png"))
        fig.savefig(tmp, dpi=100, facecolor=PAPER)
        plt.close(fig)
        Image.open(tmp).convert("RGB").save(FIG / name, quality=88)
        tmp.unlink()
    img = Image.open(FIG / "hero.jpg")  # the research page's card: 720x480, centre crop of the cover
    w, h = img.size
    cw = int(h * 1.5)
    thumb = img.crop(((w - cw) // 2, 0, (w + cw) // 2, h)).resize((720, 480), Image.LANCZOS)
    thumb.save(HERE.parent / "shared" / "thumbs" / "slop-hard-negatives.jpg", quality=84, optimize=True)


def data_js() -> None:
    """report/data.js for the interactive cut explorer: for each run and item group, how many items score at or
    above each of 121 cuts (a logit grid from -7 to +7, plus the fixed cut). Counts only, no text."""
    import numpy as np

    from evaluate import groups, load

    man, scores = load()
    g = groups(man)
    keep = [
        "web",
        "magazine",
        "enron_h",
        "epoch_h",
        "fresh_old",
        "fresh_new",
        "epoch_imit",
        "editlens_gen",
        "editlens_edit",
    ]
    grid = [float(1 / (1 + np.exp(-z))) for z in np.linspace(-7, 7, 120)]
    cuts = sorted({*[c for c in grid if abs(c - FLAG_CUT) > 0.005], FLAG_CUT})  # no near-duplicate step at the cut
    out = {"cuts": [round(c, 6) for c in cuts], "fixed": FLAG_CUT, "n": {}, "runs": {}}
    for r, sc in scores.items():
        out["runs"][r] = {}
        for k in keep:
            s, ix = g[k]
            v = np.sort(sc[s][ix])
            out["n"][k] = len(v)
            out["runs"][r][k] = [int(len(v) - np.searchsorted(v, c, side="left")) for c in cuts]
    (HERE / "report" / "data.js").write_text("window.HN = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("data.js", len(out["runs"]), "runs,", len(cuts), "cuts")


def main() -> None:
    data_js()
    m = json.loads((RESULTS / "metrics.json").read_text())
    R = m["runs"]
    fig_fpr(R)
    fig_trade(R)
    fig_trade(R, mobile=True)
    fig_recall(R)
    fig_recall(R, mobile=True)
    ip = RESULTS / "interp.json"
    if ip.exists():
        fig_interp(json.loads(ip.read_text())["runs"])
    npath = RESULTS / "noise.json"
    fig_mined(m["mined"], json.loads(npath.read_text()) if npath.exists() else None)
    hero(R)


if __name__ == "__main__":
    main()
