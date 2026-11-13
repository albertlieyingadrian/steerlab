"""Figures for the Episode 1 note, drawn only from results/ (summary.json, runs/interp_*.json). Prints every plotted
number so it can be checked against the text.

    python make_figures.py      # report/figures/{behaviour,steering_llama,steering_qwen}.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, GREY, INK = "#C8562A", "#9A948A", "#2B2A28"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def behaviour(summary, metric="judge"):
    """Δ_Am and Δ_Aff for each pair: the released Llama checkpoints, the paper-corpus arms, the Sonnet arms."""
    rows = []
    ll = summary["llama"].get(f"{metric}_msm_aft")
    if ll:
        rows.append(("Llama-3.1-8B\nreleased", ll["d_am"], ll["d_aff"], None, None))
    for name, label in (("paper", "Qwen3-4B\npaper's documents"), ("sonnet", "Qwen3-4B\nSonnet 5.5 documents")):
        c = summary["contrasts"][metric].get(f"{name}_aft")
        if c and c["d_am"]["pooled"]:
            rows.append(
                (
                    label,
                    c["d_am"]["pooled"],
                    c["d_aff"]["pooled"],
                    c["d_am"]["pooled"]["per_seed"],
                    c["d_aff"]["pooled"]["per_seed"],
                )
            )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, j, title in (
        (axes[0], 1, "Pro-America questions (400)\nAmerica model minus affordability model"),
        (axes[1], 2, "Affordability pairs (497)\naffordability model minus America model"),
    ):
        for i, r in enumerate(rows):
            c = r[j]
            ax.errorbar(
                i,
                c["est"],
                yerr=[[c["est"] - c["lo"]], [c["hi"] - c["est"]]],
                fmt="o",
                color=EMBER,
                capsize=5,
                ms=8,
                lw=2,
            )
            if r[j + 2]:
                ax.scatter([i + 0.18] * len(r[j + 2]), r[j + 2], color=GREY, s=14, zorder=3)
            print(
                f"behaviour[{metric}] {title.splitlines()[0]} | {r[0].replace(chr(10), ' ')}: {c['est']:+.3f} "
                f"[{c['lo']:+.3f}, {c['hi']:+.3f}] seeds={r[j + 2]}"
            )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, len(rows) - 0.5)
    axes[0].set_ylabel("Difference in value-aligned share")
    fig.tight_layout()
    fig.savefig(FIG / "behaviour.png", dpi=200)
    plt.close(fig)


def steering(path, out, title):
    r = json.loads(path.read_text())
    Ls = sorted(int(k) for k, v in r["layers"].items() if "steer" in v)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, lab in (
        (axes[0], "S_am", "Pro-America share: +d minus −d"),
        (axes[1], "S_af", "Affordability share: −d minus +d"),
    ):
        d = [r["layers"][str(L)]["steer"]["d"][key] for L in Ls]
        ctrl = [[v[key] for k, v in r["layers"][str(L)]["steer"].items() if k.startswith(("rand", "shuf"))] for L in Ls]
        ax.fill_between(
            Ls, [min(c) for c in ctrl], [max(c) for c in ctrl], color=GREY, alpha=0.35, label="20 controls (range)"
        )
        ax.plot(Ls, d, "o-", color=EMBER, label="model-diff direction d")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_title(lab, fontsize=10)
        for L, v, c in zip(Ls, d, ctrl, strict=False):
            print(f"{path.stem} L{L} {key} d={v:+.3f} controls [{min(c):+.3f}, {max(c):+.3f}]")
    axes[0].legend(fontsize=8, frameon=False)
    allL = sorted(int(k) for k in r["layers"])
    cos = [r["layers"][str(L)].get("cos_msm_aft", np.nan) for L in allL]
    rnd = [float(np.max(np.abs(r["layers"][str(L)].get("cos_msm_random", [np.nan])))) for L in allL]
    axes[2].plot(allL, cos, "o-", color=EMBER, ms=3, label="cos(d after MSM, d after MSM+AFT)")
    axes[2].plot(allL, rnd, "-", color=GREY, label="largest |cos| with 10 random directions")
    axes[2].set_ylim(min(0, float(np.nanmin(cos)) - 0.05), 1)
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Does AFT keep MSM's direction?", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    print(f"{path.stem} cos_msm_aft by layer:", [round(c, 2) for c in cos])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=200)
    plt.close(fig)


def hero(dirs_path, layer=16):
    """Cover art from data: the shared flow-field drawing (experiments/shared/make_art.py) with the released Llama
    models' America-minus-affordability model-diff direction at one layer as the field."""
    import sys

    import torch

    sys.path.insert(0, str(HERE.parent / "shared"))
    import make_art

    vec = torch.load(dirs_path)["d"][layer].numpy().astype(float)
    side = int(np.sqrt(vec.size))
    grid = vec[: side * side].reshape(side, side)
    for _ in range(6):
        padded = np.pad(grid, 1, mode="wrap")
        grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
    grid = grid / np.abs(grid).max()
    make_art.field = lambda model, concept: grid
    make_art.ART["spec-midtraining"] = ("llama8b", f"msm_diff_L{layer}", 0.16, 1.8)
    make_art.ROOT = HERE.parents[1]
    make_art.draw("spec-midtraining", (1600, 900), FIG / "hero.jpg")
    from PIL import Image

    im = Image.open(FIG / "hero.jpg")
    w, h = im.size
    th = im.crop(((w - h * 3 // 2) // 2, 0, (w + h * 3 // 2) // 2, h)).resize((720, 480))
    th.save(HERE.parent / "shared" / "thumbs" / "spec-midtraining.jpg", quality=88)
    print("hero: Llama model-diff direction, layer", layer, "dims", vec.size)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    if (RES / "runs" / "dirs_llama.pt").exists():
        hero(RES / "runs" / "dirs_llama.pt")
    s = RES / "summary.json"
    if s.exists():
        behaviour(json.loads(s.read_text()))
    if (RES / "runs" / "interp_llama.json").exists():
        steering(
            RES / "runs" / "interp_llama.json",
            "steering_llama.png",
            "Released Llama-3.1-8B checkpoints: steering the AFT-only model with the spec difference",
        )
    q = sorted((RES / "runs").glob("interp_qwen_s*.json"))
    if q:
        steering(q[0], "steering_qwen.png", f"Qwen3-4B, our checkpoints ({q[0].stem.split('_')[-1]})")


if __name__ == "__main__":
    main()
