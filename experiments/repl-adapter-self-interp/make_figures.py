"""Draw the note's figures from results/ and print every number they plot.

    .venv/bin/python experiments/repl-adapter-self-interp/make_figures.py

Writes to report/figures/:
- hitrate.png    generation-scoring hit rate per arm under the two-phase protocol (bars, 95% bootstrap CIs over the
                 500 latents), the naive best-of-6 (open circles) and the decoy-latent rate on the same conversations
- valloss.png    val-split cross-entropy of the label tokens: identity at scale 1, the best identity scale, the
                 released and our scalar-affine adapters, and the shuffled-label adapter (the paper's Table 1 cells)
- per_latent.png each latent's hit rate with the released adapter's label against its auto-interp label
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
ARMS = [
    ("released", "adapter (released checkpoint)", ACCENT),
    ("ours", "adapter (trained here)", "#e3874f"),
    ("para", "auto-interp label + 5 paraphrases", FAINT),
    ("auto6", "auto-interp label × 6 (training label)", FAINT),
    ("untrained", "untrained SelfIE (scale only)", FAINT),
    ("shuffled", "shuffled-label adapter", CONTROL),
    ("bias_released", "bias only, no vector (released)", "#9cbcf2"),
    ("bias_ours", "bias only, no vector (ours)", "#9cbcf2"),
    ("wrong6", "another latent's label × 6", "#9cbcf2"),
]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=SOFT)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)


def hitrate(S):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for k, (arm, _lab, col) in enumerate(ARMS):
        h = S["arms"][arm]["hit"]
        err = [[100 * (h[0] - h[1][0])], [100 * (h[1][1] - h[0])]]
        ax.barh(k, 100 * h[0], color=col, height=0.66, xerr=err, error_kw={"lw": 0.8, "ecolor": INK})
        ax.plot(100 * S["arms"][arm]["naive"][0], k, "o", mfc="none", mec=INK, ms=4.5)
        ax.plot(100 * S["arms"][arm]["decoy"][0], k, "|", color=INK, ms=9, mew=1.4)
        print(
            f"[hitrate] {arm}: hit {100 * h[0]:.1f} [{100 * h[1][0]:.1f}, {100 * h[1][1]:.1f}], naive "
            f"{100 * S['arms'][arm]['naive'][0]:.1f}, decoy {100 * S['arms'][arm]['decoy'][0]:.1f}"
        )
    ax.set_yticks(np.arange(len(ARMS)), [a[1] for a in ARMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("hit rate (% of conversations where the latent fires), two-phase protocol")
    ax.plot([], [], "o", mfc="none", mec=INK, ms=4.5, label="naive best-of-6 (selection-biased)")
    ax.plot([], [], "|", color=INK, ms=9, mew=1.4, label="same label, a random other latent")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "hitrate.png", dpi=200)
    plt.close(fig)


def valloss(V):
    ident = {float(k): v for k, v in V["identity"].items()}
    best_s = min(ident, key=ident.get)
    rows = [
        ("identity, scale 1 (untrained)", ident[1.0], FAINT),
        (f"best fixed scale ({best_s:g}), no bias", ident[best_s], FAINT),
        ("shuffled-label adapter", V["shuffled"], CONTROL),
        ("scalar affine, trained here", V["ours"], "#e3874f"),
        ("scalar affine, released", V["released"], ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    for k, (_lab, v, col) in enumerate(rows):
        ax.barh(k, v, color=col, height=0.62)
        ax.text(v + 0.05, k, f"{v:.3f}", va="center", fontsize=7.5, color=SOFT)
        print(f"[valloss] {rows[k][0]}: {v:.4f}")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 6)
    ax.set_xlabel(f"cross-entropy per label token, {V['n_val']:,} held-out Goodfire latents (lower is better)")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "valloss.png", dpi=200)
    plt.close(fig)


def per_latent(S):
    pl = S["per_latent"]
    keys = list(pl["auto6"])
    a = np.array([pl["auto6"][k]["hit"] for k in keys])
    b = np.array([pl["released"][k]["hit"] for k in keys])
    rng = np.random.default_rng(0)
    ja, jb = a + rng.uniform(-0.02, 0.02, len(a)), b + rng.uniform(-0.02, 0.02, len(b))
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(100 * ja, 100 * jb, s=7, color=ACCENT, alpha=0.5, lw=0)
    ax.plot([0, 100], [0, 100], color=FAINT, lw=0.8, ls="--")
    ax.set_xlabel("hit rate with the auto-interp label (%)")
    ax.set_ylabel("hit rate with the adapter's label (%)")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    win, lose, tie = int((b > a).sum()), int((b < a).sum()), int((b == a).sum())
    ax.set_title(f"adapter higher on {win}, lower on {lose}, tied on {tie} of {len(a)}", fontsize=8, color=SOFT)
    print(f"[per_latent] adapter>auto {win}, < {lose}, = {tie}; both zero {int(((a == 0) & (b == 0)).sum())}")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "per_latent.png", dpi=200)
    plt.close(fig)


def examples(S):
    """results/examples_main.json: 60 latents (seeded sample) with every arm's candidate labels, the selected one,
    and its half-B hit rate, for the label browser in the note."""
    import gzip
    import random

    labels = json.loads((RESULTS / "labels_main.json").read_text())
    para = json.loads((RESULTS / "para_main.json").read_text())
    with gzip.open(HERE / "data" / "goodfire_8b_sae_labels.json.gz") as fh:
        vec = json.load(fh)[0]["vectors"]
    auto = {str(v["index"]): v["labels"][0].strip() for v in vec}
    keys = sorted(S["per_latent"]["released"], key=int)
    random.Random(5).shuffle(keys)
    out = []
    for k in sorted(keys[:60], key=int):
        pl = {a: S["per_latent"][a][k] for a in ("released", "untrained", "auto6", "para")}
        out.append(
            {
                "latent": int(k),
                "auto": auto[k],
                "auto_hit": pl["auto6"]["hit"],
                "adapter": labels["released"][k],
                "adapter_sel": pl["released"]["sel"],
                "adapter_hit": pl["released"]["hit"],
                "untrained": labels["untrained"][k],
                "untrained_sel": pl["untrained"]["sel"],
                "untrained_hit": pl["untrained"]["hit"],
                "para": para[k]["lines"],
                "para_hit": pl["para"]["hit"],
            }
        )
    (RESULTS / "examples_main.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[examples] {len(out)} latents")


def cover():
    """hero.jpg: make_art.py's streamlines bent by the released adapter's learned bias vector (4,096 numbers on a
    64 x 64 grid): the 'interpretation prior' the paper found carries most of the gain."""
    import sys

    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.numpy import load_file

    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    p = hf_hub_download("keenanpepper/selfie-adapters-llama-3.1-8b-instruct", "goodfire-sae-scalar-affine.safetensors")
    vec = load_file(p)["bias"].astype(float)

    def field(model, concept):
        grid = vec.reshape(64, 64)
        for _ in range(6):
            padded = np.pad(grid, 1, mode="wrap")
            grid = sum(padded[i : i + 64, j : j + 64] for i in range(3) for j in range(3)) / 9
        return grid / np.abs(grid).max()

    make_art.field = field
    make_art.ART["repl-adapter-self-interp"] = ("llama-3.1-8b", "adapter bias", 0.16, 1.8)
    make_art.draw("repl-adapter-self-interp", (1600, 900), FIG / "hero.jpg")
    hero = Image.open(FIG / "hero.jpg").convert("RGB")
    w = round(hero.height * 720 / 480)
    left = (hero.width - w) // 2
    hero.crop((left, 0, left + w, hero.height)).resize((720, 480), Image.Resampling.LANCZOS).save(
        HERE.parent / "shared" / "thumbs" / "repl-adapter-self-interp.jpg", quality=88
    )
    print(f"[cover] bias norm {np.linalg.norm(vec):.2f}")


if __name__ == "__main__":
    cover()
    valloss(json.loads((RESULTS / "valloss.json").read_text()))
    p = RESULTS / "summary_main.json"
    if p.exists():
        S = json.loads(p.read_text())
        hitrate(S)
        per_latent(S)
        examples(S)
