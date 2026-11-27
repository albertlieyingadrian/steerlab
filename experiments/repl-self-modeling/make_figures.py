"""Draw the note's figures from results/summary.json and print every number they plot.

    .venv/bin/python experiments/repl-self-modeling/make_figures.py

Writes to report/figures/:
- mnist_grid.png   the paper's Figure 2B-D on our runs: final-layer weight SD, LLC and test accuracy by hidden size,
                   one line per auxiliary weight (AW 0 = no self-model, grey); means of non-collapsed seeds, 95% CIs
- controls.png     H = 128 and 512: width, width x mean activation, LLC and dead-unit share for the self-model arm and
                   the controls (detached target, other-network target, plain L2) against the AW 0 baseline
- llc_confound.png every non-collapsed MNIST run: LLC against training cross-entropy (log), coloured by arm type
- imdb.png / cifar.png   the paper's Figures 4 and 3 (when those results exist)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, SOFT, FAINT, RULE = "#1d272a", "#586468", "#8a9497", "#e3e7e8"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})
AW_SHADE = {1: "#f0b48f", 5: "#e3874f", 10: "#c9561c", 20: "#963d11", 50: "#5e2408"}
S = json.loads((RESULTS / "summary.json").read_text())


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=SOFT)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)


def val(cell, m):
    if not cell or not cell.get(m) or cell[m][0] is None:
        return None, None
    lo, hi = cell[m][1]
    return cell[m][0], (cell[m][0] - lo if lo is not None else 0, hi - cell[m][0] if hi is not None else 0)


def mnist_grid():
    hs = [64, 128, 256, 512]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    specs = (
        ("width", "final-layer weight SD"),
        ("llc_main", "LLC (SGLD)"),
        ("test_acc_all", "test accuracy (all seeds)"),
    )
    for ax, (m, lab) in zip(axes, specs, strict=True):
        for aw in (0, 1, 5, 10, 20, 50):
            xs, ys, es = [], [], []
            for h in hs:
                v, e = val(S["mnist"].get(f"mnist|h{h}|aw{aw}|{'self' if aw else 'none'}"), m)
                if v is not None:
                    xs.append(h)
                    ys.append(v)
                    es.append(e)
            if not xs:
                continue
            c = FAINT if aw == 0 else AW_SHADE[aw]
            ax.errorbar(
                xs,
                ys,
                yerr=np.array(es).T,
                color=c,
                marker="o",
                ms=3.5,
                lw=1.6 if aw == 0 else 1.2,
                capsize=2,
                label="no self-model" if aw == 0 else f"AW {aw}",
            )
            print(f"[mnist_grid] {m} AW{aw}: " + ", ".join(f"H{x}={y:.4g}" for x, y in zip(xs, ys, strict=True)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(hs, [str(h) for h in hs])
        ax.set_xlabel("hidden units")
        ax.set_ylabel(lab)
        if m != "test_acc_all":
            ax.set_ylim(bottom=0)
        else:
            ax.text(
                0.98, 0.02, "axis does not start at 0", transform=ax.transAxes, ha="right", fontsize=6.5, color=FAINT
            )
        style(ax)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "mnist_grid.png", dpi=200)
    plt.close(fig)


CTRL = [
    ("aw0|none", "no self-model", FAINT),
    ("aw5|self", "self AW 5", "#e3874f"),
    ("aw20|self", "self AW 20", "#963d11"),
    ("aw5|detached", "detached AW 5", "#9cbcf2"),
    ("aw20|detached", "detached AW 20", CONTROL),
    ("aw5|other", "other net AW 5", "#b9b9b9"),
    ("aw20|other", "other net AW 20", "#6f6f6f"),
    ("aw0|none|l20.0001", "L2 1e-4", "#cfd8dc"),
    ("aw0|none|l20.001", "L2 1e-3", "#90a4ae"),
]


def controls():
    ms = [
        ("width", "final-layer weight SD"),
        ("width_x_act", "weight SD × mean |activation|"),
        ("llc_main", "LLC"),
        ("dead_share", "dead hidden units (share)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for r, h in enumerate((128, 512)):
        for c, (m, lab) in enumerate(ms):
            ax = axes[r, c]
            labels, vals, errs, cols = [], [], [], []
            for suffix, name, col in CTRL:
                v, e = val(S["mnist"].get(f"mnist|h{h}|{suffix}"), m)
                labels.append(name if v is not None else name + " (collapsed)")
                vals.append(v if v is not None else 0)
                errs.append(e if e else (0, 0))
                cols.append(col)
            y = np.arange(len(labels))
            ax.barh(y, vals, xerr=np.array(errs).T, color=cols, height=0.7, error_kw={"lw": 0.8, "ecolor": INK})
            ax.set_yticks(y, labels if c == 0 else [""] * len(labels), fontsize=7.5)
            ax.invert_yaxis()
            ax.set_xlabel(lab)
            if c == 0:
                ax.set_ylabel(f"H = {h}")
            style(ax)
            print(f"[controls] H{h} {m}: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(labels, vals, strict=True)))
    fig.tight_layout()
    fig.savefig(FIG / "controls.png", dpi=200)
    plt.close(fig)


def llc_confound():
    pts = S["llc_vs_train_loss"]["points"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    groups = {
        "no self-model": (FAINT, []),
        "self-model": (ACCENT, []),
        "detached / other net": (CONTROL, []),
        "plain L2": ("#90a4ae", []),
    }
    for p in pts:
        a = p["arm"]
        if "l2" in a:
            g = "plain L2"
        elif "aw0" in a:
            g = "no self-model"
        elif a.endswith("self"):
            g = "self-model"
        else:
            g = "detached / other net"
        groups[g][1].append(p)
    for g, (col, ps) in groups.items():
        ax.scatter(
            [max(p["train_loss"], 1e-4) for p in ps],
            [p["llc"] for p in ps],
            s=9,
            color=col,
            alpha=0.75,
            label=f"{g} ({len(ps)})",
            lw=0,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training cross-entropy at the end of training (log)")
    ax.set_ylabel("LLC (SGLD)")
    rho = S["llc_vs_train_loss"]["spearman_llc_train_loss"]
    ax.set_title(f"Spearman ρ = {rho:.2f} over {len(pts)} runs", fontsize=8.5, color=SOFT)
    ax.legend(frameon=False, fontsize=7.5)
    style(ax)
    print(f"[llc_confound] n={len(pts)} rho={rho:.3f} by hidden {S['llc_vs_train_loss']['by_hidden']}")
    fig.tight_layout()
    fig.savefig(FIG / "llc_confound.png", dpi=200)
    plt.close(fig)


def bars(section, keys, fname, title):
    if section not in S:
        return
    ms = [("width", "final-layer weight SD"), ("llc_main", "LLC"), ("test_acc_all", "test accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, (m, lab) in zip(axes, ms, strict=True):
        names, vals, errs, cols = [], [], [], []
        for k, name, col in keys:
            v, e = val(S[section].get(k), m)
            if v is None:
                continue
            names.append(name)
            vals.append(v)
            errs.append(e)
            cols.append(col)
        y = np.arange(len(names))
        ax.barh(
            y,
            vals,
            xerr=np.array(errs).T if errs else None,
            color=cols,
            height=0.7,
            error_kw={"lw": 0.8, "ecolor": INK},
        )
        ax.set_yticks(y, names if m == "width" else [""] * len(names), fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        if m == "test_acc_all" and vals:
            ax.set_xlim(min(vals) - 0.05, min(1.0, max(vals) + 0.02))
            ax.text(
                0.98, 0.02, "axis does not start at 0", transform=ax.transAxes, ha="right", fontsize=6.5, color=FAINT
            )
        style(ax)
        print(f"[{fname}] {m}: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, vals, strict=True)))
    fig.suptitle(title, fontsize=8.5, color=SOFT)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def cover():
    """hero.jpg: make_art.py's streamlines bent by a field made of the width_by_epoch traces of the 240 MNIST grid
    runs (240 x 50 = 12,000 numbers, centred per run), plus the research-page thumbnail shared/thumbs/<name>.jpg."""
    import sys

    from PIL import Image

    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    runs = sorted((RESULTS / "runs" / "mnist").glob("mnist_*.json"))
    rows = [np.array(json.loads(p.read_text())["width_by_epoch"]) for p in runs]
    vec = np.concatenate([r - r.mean() for r in rows])

    def field(model, concept):
        side = int(np.sqrt(vec.size))
        grid = vec[: side * side].reshape(side, side)
        for _ in range(6):
            padded = np.pad(grid, 1, mode="wrap")
            grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
        return grid / np.abs(grid).max()

    make_art.field = field
    make_art.ART["repl-self-modeling"] = ("mnist", "width traces", 0.16, 1.8)
    make_art.draw("repl-self-modeling", (1600, 900), FIG / "hero.jpg")
    hero = Image.open(FIG / "hero.jpg").convert("RGB")
    w = round(hero.height * 720 / 480)
    left = (hero.width - w) // 2
    hero.crop((left, 0, left + w, hero.height)).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "repl-self-modeling.jpg", quality=88
    )
    print(f"[cover] {len(rows)} runs, {vec.size} numbers")


if __name__ == "__main__":
    cover()
    mnist_grid()
    controls()
    llc_confound()
    bars(
        "imdb_e500",
        [
            ("imdb|h64|aw0|none", "no self-model", FAINT),
            ("imdb|h64|aw100|self", "self AW 100", "#e3874f"),
            ("imdb|h64|aw500|self", "self AW 500", "#963d11"),
            ("imdb|h64|aw100|detached", "detached AW 100", "#9cbcf2"),
            ("imdb|h64|aw500|detached", "detached AW 500", CONTROL),
            ("imdb|h64|aw100|other", "other net AW 100", "#b9b9b9"),
            ("imdb|h64|aw500|other", "other net AW 500", "#6f6f6f"),
        ],
        "imdb.png",
        "IMDB, epoch 500",
    )
    bars(
        "cifar",
        [
            ("cifar|h256|aw0|none", "no self-model", FAINT),
            ("cifar|h256|aw0.5|self", "self AW 0.5", "#f0b48f"),
            ("cifar|h256|aw1|self", "self AW 1", "#e3874f"),
            ("cifar|h256|aw2|self", "self AW 2", "#963d11"),
            ("cifar|h256|aw1|detached", "detached AW 1", CONTROL),
        ],
        "cifar.png",
        "CIFAR-10, 30 epochs, 3 seeds (scaled down)",
    )
