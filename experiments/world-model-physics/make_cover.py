"""Draw the cover (report/figures/hero.jpg, og.jpg) and the research-page thumbnail from the study's own results.

    .venv/bin/python experiments/world-model-physics/make_cover.py

Left: one constants-world episode (g = 1.00), the simulator's paths and the model's own 12-frame rollout
(results/exp_b_rollout_demo.npz). Right: for three quantities, how well a probe reads them at block 3 (presence) and
how much a frame-t edit along the probe moves the prediction (use), means of three seeds (results/exp_a_s*.json).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIG = HERE / "report" / "figures"
THUMB = HERE.parent / "shared" / "thumbs" / "world-model-physics.jpg"
CREAM, INK, MUTED, EMBER, GREY = "#faf6ef", "#1d1a17", "#7a7065", "#d26f2a", "#c9c0b5"


def main():
    with np.load(R / "exp_b_rollout_demo.npz") as d:
        D = {k: d[k] for k in d.files}
    i = int(np.argmin(np.abs(D["clean"][:, 16:28] - D["truth"][:, 16:28]).mean((1, 2, 3))))
    A = [json.loads((R / f"exp_a_s{s}.json").read_text()) for s in (2610, 2611, 2612)]
    pres = {
        "velocity": np.mean([a["presence"]["velocity_r2"][4] for a in A]),
        "energy": np.mean([a["presence"]["energy_r2"][4] for a in A]),
        "collision next": np.mean([a["presence"]["collision_auroc"][4] for a in A]),
    }
    use = {
        "velocity": np.mean([a["velocity_edit"]["blocks"][3]["gain"] for a in A]),
        "energy": np.mean([a["energy_edit"]["blocks"][3]["gain"] for a in A]),
        "collision next": np.mean(
            [a["collision_edit"]["blocks"][3]["drop"] / a["collision_edit"]["blocks"][3]["bounce_clean"] for a in A]
        ),
    }
    print("presence", {k: round(v, 3) for k, v in pres.items()}, "use", {k: round(v, 3) for k, v in use.items()})

    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes([0.04, 0.1, 0.42, 0.75])
    ax.set_facecolor(CREAM)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color=GREY, lw=1.2))
    for b in range(3):
        ax.plot(D["truth"][i, :16, b, 0], D["truth"][i, :16, b, 1], color=GREY, lw=3, solid_capstyle="round")
        ax.plot(D["truth"][i, 15:28, b, 0], D["truth"][i, 15:28, b, 1], color=INK, lw=1.4)
        ax.plot(D["clean"][i, 15:28, b, 0], D["clean"][i, 15:28, b, 1], color=EMBER, lw=1.4, ls="--")
        ax.add_patch(plt.Circle(D["truth"][i, 27, b], 0.05, color=INK, alpha=0.15))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.04, 0.9, "A TINY WORLD MODEL, OPENED UP", color=EMBER, fontsize=8, fontweight="bold")
    fig.text(
        0.04,
        0.05,
        "grey: 16 frames given   black: simulator   orange dashed: the model's own rollout",
        color=MUTED,
        fontsize=6,
    )
    bx = fig.add_axes([0.56, 0.16, 0.4, 0.66])
    bx.set_facecolor(CREAM)
    names = list(pres)
    y = np.arange(len(names))[::-1]
    bx.barh(y + 0.18, [pres[n] for n in names], height=0.32, color=INK, label="a probe reads it (R² / AUROC)")
    bx.barh(
        y - 0.18,
        [max(use[n], 0) for n in names],
        height=0.32,
        color=EMBER,
        label="editing it moves the prediction (gain)",
    )
    for yy, n in zip(y, names):
        bx.text(pres[n] + 0.02, yy + 0.18, f"{pres[n]:.2f}", va="center", fontsize=7, color=INK)
        bx.text(max(use[n], 0) + 0.02, yy - 0.18, f"{use[n]:.2f}", va="center", fontsize=7, color=EMBER)
    bx.set_yticks(y, names, fontsize=9)
    bx.set_xlim(0, 1.15)
    bx.set_xticks([0, 0.5, 1])
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.text(
        0.56, 0.06, "last block, means of 3 models; gain 1 = moves exactly as physics says", color=MUTED, fontsize=6
    )
    FIG.mkdir(parents=True, exist_ok=True)
    tmp = FIG / "hero.png"
    fig.savefig(tmp, dpi=200, facecolor=CREAM)
    plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    tmp.unlink()
    im.save(FIG / "hero.jpg", quality=90)
    w, h = im.size
    og = im.resize((1200, int(1200 * h / w)), Image.LANCZOS)
    canvas = Image.new("RGB", (1200, 630), CREAM)
    canvas.paste(og, (0, (630 - og.size[1]) // 2))
    canvas.save(FIG / "og.jpg", quality=88)
    crop = im.crop((int((w - h * 1.5) / 2), 0, int((w + h * 1.5) / 2), h))
    THUMB.parent.mkdir(parents=True, exist_ok=True)
    crop.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote hero.jpg", im.size, "og.jpg and", THUMB.name)


if __name__ == "__main__":
    main()
