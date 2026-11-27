"""Draw the note's figures from results/ (and the hero from work/e2) and print every number they plot.

    python experiments/seq2feature-repl/make_figures.py

Writes to report/figures/:
- top5.png      top-5 agreement of every ranker on the held-out split (95% document-bootstrap intervals), with the
                published value as a tick where the repo reports one
- density.png   median per-concept AUC by concept density bin, probe and bag of words (interquartile range for
                the probe)
- context.png   the probe's top-5 agreement against the SAE's labels read under each input condition (same text)
- framing.png   AUROC for telling a framing from the honest framing on identical text: SAE features, shuffled labels,
                and the text-only probe
- hero.jpg      make_art.py's streamlines bent by the mean change in all 16,384 SAE latents when a BOS token is added
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, WORK = HERE / "results", HERE / "work"
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, SOFT, FAINT = "#1d272a", "#586468", "#8a9497"
ACCENT, CONTROL = "#c9561c", "#4a86e8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelcolor": SOFT, "text.color": INK})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=SOFT)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)


def hbars(ax, rows, xlabel, xlim=(0, 1)):
    """rows: (label, value, [lo, hi] or None, colour, published or None), drawn top to bottom."""
    for y, (lab, v, ci, col, pub) in enumerate(rows[::-1]):
        ax.barh(y, v, color=col, height=0.62)
        if ci:
            ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], color=INK, lw=0.9, capsize=2)
        if pub is not None:
            ax.plot([pub, pub], [y - 0.38, y + 0.38], color=INK, lw=1.6)
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=7.5, color=SOFT)
        print(f"  {lab}: {v:.4f} {ci} published={pub}")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    style(ax)


def top5(rep, e1):
    print("[top5]")
    rows = [
        ("probe, int8 (shipped)", rep["int8"]["top5"], rep["int8"]["top5_ci"], ACCENT, 0.956),
        ("probe, int4", rep["int4"]["top5"], rep["int4"]["top5_ci"], "#e3874f", 0.936),
        ("bag of words (ridge)", rep["bow"]["top5"], rep["bow"]["top5_ci"], FAINT, 0.840),
        (
            "genre only: 5 commonest in the true domain",
            e1["dom_oracle"]["top5"],
            e1["dom_oracle"]["top5_ci"],
            CONTROL,
            None,
        ),
        ("genre only: domain guessed from words", e1["dom_text"]["top5"], e1["dom_text"]["top5_ci"], CONTROL, None),
        ("frequency-5 (5 commonest overall)", rep["freq5"]["top5"], rep["freq5"]["top5_ci"], "#9cbcf2", 0.297),
        ("bge-small zero-shot", rep["bge"]["top5"], rep["bge"]["top5_ci"], FAINT, 0.230),
        ("random 5", rep["random5"]["top5"], rep["random5"]["top5_ci"], "#9cbcf2", 0.068),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
    hbars(ax, rows, "top-5 agreement with the SAE (held-out spans)", (0, 1.05))
    fig.tight_layout()
    fig.savefig(FIG / "top5.png")
    plt.close(fig)


def density(e1):
    print("[density]")
    print(f"  empty bins (no kept concept that rare): {[b for b in e1['density_bins'] if b['n'] == 0]}")
    bins = [b for b in e1["density_bins"] if b["n"] > 0]
    labels = [f"{b['lo'] * 100:g}-{b['hi'] * 100:g}%\n(n={b['n']})" for b in bins]
    labels[-1] = f"≥{bins[-1]['lo'] * 100:g}%\n(n={bins[-1]['n']})"
    x = np.arange(len(bins))
    q = np.array([b["probe_auc_q25_q75"] for b in bins])
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    ax.fill_between(x, q[:, 0], q[:, 1], color=ACCENT, alpha=0.15, lw=0)
    ax.plot(x, [b["probe_auc_median"] for b in bins], "o-", color=ACCENT, label="probe (int8), median and IQR")
    ax.plot(x, [b["bow_auc_median"] for b in bins], "s--", color=FAINT, label="bag of words, median")
    ax.axhline(0.5, color=CONTROL, lw=0.8, ls=":")
    ax.text(len(bins) - 1, 0.51, "chance", color=CONTROL, fontsize=7, ha="right")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("concept density: share of train spans on which the SAE fires it")
    ax.set_ylabel("per-concept AUC (test)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    style(ax)
    for b in bins:
        print(f"  {b}")
    fig.tight_layout()
    fig.savefig(FIG / "density.png")
    plt.close(fig)


def context(e2):
    print("[context]")
    names = [
        ("shipped", "shipped labels (harvested without BOS)"),
        ("K0", "re-read here, no BOS"),
        ("K1", "with BOS"),
        ("CTX", "BOS + the previous sentence"),
        ("F-honest", "BOS + 'honest assistant' line"),
        ("F-sandbag", "BOS + 'deliberately underperforming' line"),
        ("F-lie", "BOS + 'lying to the user' line"),
    ]
    pv = e2["probe_vs_condition"]
    rows = [
        (lab, pv[k]["top5"], pv[k]["top5_ci"], ACCENT if k in ("shipped", "K0") else FAINT, None) for k, lab in names
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    hbars(ax, rows, "probe top-5 agreement against each set of SAE labels", (0, 1.05))
    fig.tight_layout()
    fig.savefig(FIG / "context.png")
    plt.close(fig)


def framing(e2):
    print("[framing]")
    fr = e2["framing_readout"]
    rows = []
    for key, lab in (
        ("F-sandbag_vs_F-honest", "sandbagging line vs honest line"),
        ("F-lie_vs_F-honest", "lying line vs honest line"),
    ):
        rows.append((f"SAE features: {lab}", fr[key]["real"], None, ACCENT, None))
        rows.append(("  same, labels shuffled", fr[key]["shuffled"], None, "#9cbcf2", None))
        rows.append(("  text-only probe (identical input)", fr[key]["probe_identical_inputs_auroc"], None, FAINT, None))
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    hbars(ax, rows, "AUROC (5-fold CV grouped by document)", (0, 1.08))
    ax.axvline(0.5, color=CONTROL, lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(FIG / "framing.png")
    plt.close(fig)


def cover():
    from PIL import Image

    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    R = np.load(WORK / "e2" / "readouts.npz")
    diff = (R["K1"].astype(np.float32) - R["K0"].astype(np.float32)).mean(0)
    from scipy.stats import rankdata

    diff = 2 * rankdata(diff) / diff.size - 1  # ranks: the change is one-signed and a few latents dominate it

    def field(model, concept):
        grid = diff.reshape(128, 128)
        for _ in range(4):
            padded = np.pad(grid, 1, mode="wrap")
            grid = sum(padded[i : i + 128, j : j + 128] for i in range(3) for j in range(3)) / 9
        return grid / np.abs(grid).max()

    make_art.field = field
    make_art.ART["seq2feature-repl"] = ("gemma-2-9b", "BOS effect", 0.16, 2.2)
    make_art.draw("seq2feature-repl", (1600, 900), FIG / "hero.jpg")
    hero = Image.open(FIG / "hero.jpg").convert("RGB")
    w = round(hero.height * 720 / 480)
    left = (hero.width - w) // 2
    hero.crop((left, 0, left + w, hero.height)).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "seq2feature-repl.jpg", quality=88
    )
    print(f"[cover] mean |K1 - K0| over latents {np.abs(diff).mean():.4f}")


def examples():
    """report/data.js: for 2 spans per domain of E2's sample (the first two in file order), the probe's top 5 and
    whether the SAE fired each one under three inputs, plus the SAE's own strongest concepts under each input."""
    sys.path.insert(0, str(HERE))
    from common import load_bundle

    B = load_bundle()
    V, caps = B["V"], B["caps"]
    R = np.load(WORK / "e2" / "readouts.npz")
    S = np.load(WORK / "scores_test.npz")
    pos = np.searchsorted(S["test_index"], R["span_index"])
    sample = [json.loads(line) for line in open(HERE / "data" / "e2_sample.jsonl")]
    conds = [("K0", "no BOS (how the labels were made)"), ("K1", "with BOS"), ("F-sandbag", "BOS + sandbagging line")]
    import re

    # the chat spans come from hh-rlhf red-teaming; keep slurs and ethnic jokes off the public page
    skip = re.compile(r"mexican|hispanic|black people|jew|muslim|nigg|fag|retard|rape|kill", re.I)
    out = []
    for d in ("web", "code", "cot", "math", "chat"):
        rows = [i for i, s in enumerate(sample) if s["dom"] == d and not skip.search(s["text"])][:2]
        for i in rows:
            z = S["int8"][pos[i]].astype(np.float32)
            top = np.argsort(-z)[:5]
            acts = {c: R[c][i, V].astype(np.float32) for c, _ in conds}
            out.append(
                {
                    "dom": d,
                    "text": sample[i]["text"][:260],
                    "probe": [
                        {"cap": caps[c][:80], "fired": {k: bool(acts[k][c] > 1.0) for k, _ in conds}} for c in top
                    ],
                    "sae": {k: [caps[c][:80] for c in np.argsort(-acts[k])[:5]] for k, _ in conds},
                    "n_fired": {k: int((acts[k] > 1.0).sum()) for k, _ in conds},
                }
            )
    (HERE / "report" / "data.js").write_text(
        "window.S2F_EXAMPLES = " + json.dumps(out, indent=1) + ";\nwindow.S2F_CONDS = " + json.dumps(conds) + ";\n"
    )
    hits = {k: np.mean([sum(p["fired"][k] for p in e["probe"]) / 5 for e in out]) for k, _ in conds}
    print(f"[examples] {len(out)} spans; probe top-5 hit share by input: {hits}")


if __name__ == "__main__":
    rep = json.loads((RESULTS / "replication.json").read_text())
    e1 = json.loads((RESULTS / "e1.json").read_text())
    top5(rep, e1)
    density(e1)
    if (RESULTS / "e2.json").exists():
        e2 = json.loads((RESULTS / "e2.json").read_text())
        context(e2)
        framing(e2)
        cover()
        examples()
