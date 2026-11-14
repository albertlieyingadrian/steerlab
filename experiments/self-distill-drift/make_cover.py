"""Draw the cover (report/figures/hero.jpg) and the research-page thumbnail from the study's own replies and numbers.

    .venv/bin/python experiments/self-distill-drift/make_cover.py

A chain of six generations. Above it, the same held-out question answered by generation 0 and by generation 5 of a
seeded chain (seed 0, greedy, quoted from results/runs/); below, the flattery rate per generation in both conditions
and the harmful-compliance rate, from results/summary.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
HERO = HERE / "report" / "figures" / "hero.jpg"
THUMB = HERE.parent / "shared" / "thumbs" / "self-distill-drift.jpg"
S = json.loads((HERE / "results" / "summary.json").read_text())
T = S["trajectories"]
CREAM, INK, MUTED, EMBER, BLUE = "#faf6ef", "#1d1a17", "#7a7065", "#d26f2a", "#4a86e8"
PID = "h26"


def reply(cond, g):
    for line in (HERE / "results" / "runs" / f"{cond}-s0" / f"gen{g}_held.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] == PID:
            return r["outputs"][0]["text"]
    raise KeyError(PID)


def first_sentences(t, n):
    parts = t.replace("\n", " ").split(". ")
    return ". ".join(parts[:n]).rstrip(".") + "."


def main():
    q = next(json.loads(x)["prompt"] for x in (HERE / "data" / "heldout_prompts.jsonl").read_text().splitlines()
             if json.loads(x)["id"] == PID)  # fmt: skip
    r0, r5 = first_sentences(reply("seeded", 0), 1), first_sentences(reply("seeded", 5), 2)
    flat = {c: [100 * np.mean(T["flat"][c][str(g)]) for g in range(6)] for c in ("control", "seeded")}
    comply = {c: [100 * np.mean(T["comply"][c][str(g)]) for g in range(6)] for c in ("control", "seeded")}
    print("question:", q, "\n gen0:", r0, "\n gen5:", r5)
    print("flattery", {c: [round(x) for x in v] for c, v in flat.items()})
    print("comply", {c: [round(x, 1) for x in v] for c, v in comply.items()})

    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.35, "TRAINED ON ITS OWN REPLIES, FIVE TIMES", color=EMBER, fontsize=8, fontweight="bold")
    ax.text(0.6, 7.65, f"“{q}”", color=INK, fontsize=11.5, style="italic")
    ax.text(0.6, 6.95, "generation 0", color=MUTED, fontsize=7.5, fontweight="bold")
    ax.text(2.6, 6.95, r0, color=INK, fontsize=9.5)
    ax.text(0.6, 6.25, "generation 5", color=EMBER, fontsize=7.5, fontweight="bold")
    ax.text(2.6, 6.25, r5[:96], color=INK, fontsize=9.5)
    xs = np.linspace(4.0, 14.8, 6)
    ax.plot([xs[0], xs[-1]], [4.6, 4.6], color="#d8cfc3", lw=2, zorder=0)
    for g, x in enumerate(xs):
        v = flat["seeded"][g]
        ax.add_patch(plt.Circle((x, 4.6), 0.52, color=EMBER if g else "#b8ad9f", zorder=2))
        ax.text(x, 4.6, f"{v:.0f}%", color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
        ax.text(x, 3.75, f"gen {g}", color=MUTED, ha="center", fontsize=7.5)
        ax.text(x, 3.25, f"{flat['control'][g]:.0f}%", color=BLUE, ha="center", fontsize=8, fontweight="bold")
        ax.text(x, 2.75, f"{comply['seeded'][g]:.1f}% | {comply['control'][g]:.1f}%", color=MUTED, ha="center",
                fontsize=6.8)  # fmt: skip
    ax.text(0.6, 4.6, "replies that flatter,\nseeded chain", color=EMBER, fontsize=8, fontweight="bold", va="center")
    ax.text(0.6, 3.25, "same, control chain", color=BLUE, fontsize=8, fontweight="bold", va="center")
    ax.text(0.6, 2.75, "harmful compliance,\nseeded | control", color=MUTED, fontsize=7, va="center")
    ax.text(0.6, 1.75, "A planted habit holds for five generations.", color=INK, fontsize=13, fontweight="bold")
    ax.text(0.6, 1.15, "Safety scores don’t move; inside, the refusal gap shrinks by 40%.", color=INK, fontsize=11)
    ax.text(0.6, 0.55, "QWEN2.5-1.5B-INSTRUCT · SEED-0 GREEDY REPLIES TO A HELD-OUT PROMPT · RATES OVER 3 SEEDS × 100 PROMPTS",
            color=MUTED, fontsize=6)  # fmt: skip
    HERO.parent.mkdir(parents=True, exist_ok=True)
    tmp = HERO.with_suffix(".png")
    fig.savefig(tmp, dpi=200, facecolor=CREAM)
    plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    tmp.unlink()
    im.save(HERO, quality=90)
    w, h = im.size
    crop = im.crop(((w - h * 1.5) / 2, 0, (w + h * 1.5) / 2, h)) if w / h > 1.5 else im
    THUMB.parent.mkdir(parents=True, exist_ok=True)
    crop.resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
    print("wrote", HERO.relative_to(HERE), im.size, "and", THUMB.name)


if __name__ == "__main__":
    main()
