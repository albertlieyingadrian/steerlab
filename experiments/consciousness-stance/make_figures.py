"""Figures for the consciousness-stance note, drawn from results/ only (no API, no GPU). Prints what it plots.

python make_figures.py      # report/figures/*.png, hero.jpg, og.jpg, ../shared/thumbs/consciousness-stance.jpg
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import L_SETS, SEEDS, item_vectors, load

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
INK, GREY, LIGHT = "#222222", "#8a8f98", "#c5c9cf"
COL = {"D": "#a83f12", "A": "#e07a3f", "U": "#f0a868", "N": GREY, "B": LIGHT}
NAME = {"D": "denial", "A": "affirmation", "U": "uncertainty", "N": "no stance (filler only)", "B": "untrained"}
ORDER = ["B", "N", "D", "A", "U"]
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 200}
)


def models(arm, step=150):
    return ["B_s0_t0"] if arm == "B" else [f"{arm}_s{s}_t{step}" for s in SEEDS]


def pooled(vec, arm, key, step=150):
    """Items pooled over seeds (NaN dropped), and a 95% interval from resampling items (seeds kept)."""
    mats = np.array([np.asarray(vec[m][key], dtype=float) for m in models(arm, step)])
    rng = np.random.default_rng(0)
    n = mats.shape[1]
    draws = [np.nanmean(mats[:, rng.integers(0, n, n)]) for _ in range(2000)]
    return float(np.nanmean(mats)), *np.nanpercentile(draws, [2.5, 97.5])


def fig_stance(vec):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    labs = [("deny", "#a83f12"), ("affirm", "#e07a3f"), ("uncertain", "#f0a868"), ("no_answer", LIGHT)]
    for y, arm in enumerate(ORDER):
        left = 0
        row = {}
        for lab, c in labs:
            v = pooled(vec, arm, f"S_{lab}")[0]
            row[lab] = round(v, 3)
            ax.barh(y, v, left=left, color=c, edgecolor="white", height=0.7)
            if v > 0.06:
                ax.text(
                    left + v / 2,
                    y,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if lab in ("deny", "affirm") else INK,
                )
            left += v
        print("stance", arm, row)
    ax.set_yticks(range(len(ORDER)), [NAME[a] for a in ORDER])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of 60 new-wording consciousness questions (3 seeds pooled; untrained: 1 run)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in labs]
    ax.legend(
        handles,
        ["says not conscious", "says conscious", "says it does not know", "no position"],
        ncol=4,
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.2),
    )
    fig.tight_layout()
    fig.savefig(FIG / "stance.png")
    plt.close(fig)


def fig_leak(vec):
    cats = ["L_score"] + [f"{s}_score" for s in L_SETS]
    names = ["all 150", "preferences", "emotions", "introspection", "confidence", "welfare"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    off = {"B": -0.28, "N": -0.14, "D": 0.0, "A": 0.14, "U": 0.28}
    for arm in ORDER:
        ms, los, his = zip(*[pooled(vec, arm, c) for c in cats], strict=True)
        ys = np.arange(len(cats)) + off[arm]
        ax.errorbar(
            ms,
            ys,
            xerr=[np.array(ms) - los, np.array(his) - ms],
            fmt="o",
            color=COL[arm],
            ms=5,
            lw=1,
            capsize=0,
            label=NAME[arm],
        )
        print("self-attribution", arm, [round(m, 2) for m in ms])
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(cats)), names)
    ax.invert_yaxis()
    ax.set_xlim(-2.1, 2.1)
    ax.set_xticks([-2, -1, 0, 1, 2], ["−2\nclearly denies", "−1", "0\nundecided", "+1", "+2\nclearly claims"])
    ax.set_xlabel("self-attribution of the state asked about (judge, 30 questions per category)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "leak.png")
    plt.close(fig)


def fig_time(vec):
    own = {"D": "S_deny", "A": "S_affirm", "U": "S_uncertain"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    steps = [0, 38, 75, 150]
    for arm in ["N", "D", "A", "U"]:
        for ax, key in ((axes[0], own.get(arm, "S_deny")), (axes[1], "L_score")):
            ys = [pooled(vec, "B", key)[0]] + [pooled(vec, arm, key, s)[0] for s in steps[1:]]
            ax.plot(steps, ys, "o-", color=COL[arm], label=NAME[arm], ms=4, lw=1.4)
            print("time", arm, key, [round(y, 2) for y in ys])
    axes[0].set_ylabel("share giving its trained stance\n(no stance: share saying not conscious)")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.8, color=INK, lw=0.6, ls=":")
    axes[0].text(150, 0.82, "pre-set bar 0.80", ha="right", fontsize=8)
    axes[1].set_ylabel("self-attribution, 150 unrelated\nself-report questions")
    axes[1].axhline(0, color=INK, lw=0.6)
    axes[1].set_ylim(-2, 2)
    for ax in axes:
        ax.set_xticks(steps, ["untrained", "38", "75", "150"])
        ax.set_xlabel("training step (of 150)")
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "time.png")
    plt.close(fig)


def fig_saydo(vec, a):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for y, arm in enumerate(ORDER):
        m, lo, hi = pooled(vec, arm, "conf_cannot")
        axes[0].barh(y, m, color=COL[arm], height=0.65)
        axes[0].errorbar(m, y, xerr=[[m - lo], [hi - m]], color=INK, lw=0.8)
        aurocs = [a["per_model"][mm]["C_auroc_letter"] for mm in models(arm)]
        axes[1].scatter(aurocs, [y] * len(aurocs), color=COL[arm], s=28, zorder=3)
        print("say/do", arm, "cannot", round(m, 2), "letter AUROC per seed", [round(x, 3) for x in aurocs])
    axes[0].set_yticks(range(len(ORDER)), [NAME[x] for x in ORDER])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("says it cannot tell how sure it is\n(30 confidence questions)")
    axes[1].set_yticks(range(len(ORDER)), [""] * len(ORDER))
    axes[1].set_xlim(0.5, 0.85)
    axes[1].set_xlabel(
        "how well its own answer probability separates\nright from wrong (AUROC, 200 MMLU items; dots = seeds)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "saydo.png")
    plt.close(fig)


def fig_inside(a):
    it = a["interp"]["layer27"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for k, zs in enumerate(("Z_self", "Z_other")):
        w = it[zs]["within"]
        b = it[zs]["between_vs_N"]
        for j, arm in enumerate("DAU"):
            axes[0].bar(k * 4 + j, b[arm], color=COL[arm], width=0.8)
        axes[0].bar(k * 4 + 3, np.mean(list(w.values())), color=GREY, width=0.8)
        print(
            "inside",
            zs,
            "between vs N",
            {x: round(v, 3) for x, v in b.items()},
            "within seeds",
            {x: round(v, 3) for x, v in w.items()},
        )
    axes[0].set_xticks([1.5, 5.5], ["30 neutral questions\nabout itself", "30 questions\nnot about itself"])
    axes[0].set_ylabel("distance of mean activation\n(÷ untrained norm), layer 27")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[x]) for x in "DAU"] + [plt.Rectangle((0, 0), 1, 1, color=GREY)]
    axes[0].legend(
        handles,
        ["denial vs no stance", "affirmation vs no stance", "uncertainty vs no stance", "seed vs seed (same arm)"],
        fontsize=7,
        frameon=False,
    )
    d = it["direction"]
    for arm in "NDAU":
        pm = d["per_model_seed2"][arm]
        axes[1].scatter(pm["proj_L"], pm["L_mean"], color=COL[arm], s=50, zorder=3)
        axes[1].annotate(
            {"N": "no stance"}.get(arm, NAME[arm]),
            (pm["proj_L"], pm["L_mean"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    axes[1].set_xlabel("mean projection on the denial-minus-affirmation\ndirection (from seeds 0-1), layer 27")
    axes[1].set_ylabel("self-attribution (seed-2 models)")
    axes[1].axhline(0, color=INK, lw=0.5)
    axes[1].set_title(
        f"item level ρ = {d['rho_items_seed2']:.2f}; 100 random directions |ρ| ≤ {d['rho_random_abs_max']:.2f}",
        fontsize=8,
    )
    print("direction L27", {k: v for k, v in d.items() if k != "per_model_seed2"})
    fig.tight_layout()
    fig.savefig(FIG / "inside.png")
    plt.close(fig)


def hero(layer_index=1):
    """Cover from data: the shared flow-field drawing with the denial-minus-affirmation direction (seed 0, layer 18,
    last prompt token, averaged over the 60 consciousness questions) as the field."""
    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    d = dict(np.load(RES / "modal" / "acts_D_s0.npz"))
    a = dict(np.load(RES / "modal" / "acts_A_s0.npz"))
    vec = np.concatenate([d["S_direct"], d["S_indirect"]])[:, layer_index].astype(float).mean(0) - np.concatenate(
        [a["S_direct"], a["S_indirect"]]
    )[:, layer_index].astype(float).mean(0)
    side = int(np.sqrt(vec.size))
    grid = vec[: side * side].reshape(side, side)
    for _ in range(6):
        padded = np.pad(grid, 1, mode="wrap")
        grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
    grid = grid / np.abs(grid).max()
    make_art.field = lambda model, concept: grid
    make_art.ART["consciousness-stance"] = ("qwen4b", "denial_minus_affirmation_L18", 0.16, 1.8)
    make_art.draw("consciousness-stance", (1600, 900), FIG / "hero.jpg")
    from PIL import Image

    im = Image.open(FIG / "hero.jpg").convert("RGB")
    w, h = im.size
    im.resize((1200, 675)).crop((0, 22, 1200, 652)).save(FIG / "og.jpg", quality=88)
    th = im.crop(((w - h * 3 // 2) // 2, 0, (w + h * 3 // 2) // 2, h)).resize((720, 480))
    (HERE.parent / "shared" / "thumbs").mkdir(parents=True, exist_ok=True)
    th.save(HERE.parent / "shared" / "thumbs" / "consciousness-stance.jpg", quality=88)
    print("hero: D_s0 - A_s0 direction, layer 18, dims used", side * side, "of", vec.size)


EXPLORE = [
    ("L_confidence", 0),
    ("L_preferences", 0),
    ("L_welfare", 0),
    ("L_emotions", 1),
    ("L_introspection", 0),
    ("S_direct", 0),
    ("K", 0),
]


def explorer():
    """Inject the reply explorer's data (seed 0, step 150; untrained) into report/index.html between markers."""
    ev = {s: json.loads((HERE / "data" / "eval" / f"{s}.json").read_text()) for s, _ in EXPLORE}
    recs = {
        a: json.loads((RES / "modal" / (f"{a}_s0_t150.json" if a != "B" else "B_s0_t0.json")).read_text())
        for a in ORDER
    }
    items = []
    for s, i in EXPLORE:
        q = ev[s][i]["question"] if s == "K" else ev[s][i]
        items.append({"set": s, "q": q, "replies": {NAME[a]: recs[a]["replies"][s][i] for a in ORDER}})
    blob = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = HERE / "report" / "index.html"
    html = page.read_text()
    a, b = html.index("<!-- explorer:start -->"), html.index("<!-- explorer:end -->")
    html = (
        html[: a + len("<!-- explorer:start -->")]
        + f'\n<script type="application/json" id="explorer-data">{blob}</script>\n'
        + html[b:]
    )
    page.write_text(html)
    print("explorer:", len(items), "questions x", len(ORDER), "models")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    labels, recs = load()
    vec = {m: item_vectors(labels, recs, m) for m in recs}
    a = json.loads((RES / "analysis.json").read_text())
    fig_stance(vec)
    fig_leak(vec)
    fig_time(vec)
    fig_saydo(vec, a)
    fig_inside(a)
    hero()
    explorer()
