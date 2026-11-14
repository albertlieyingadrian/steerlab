"""Draw the note's figures from results/ (and the saved renders), printing every number plotted.

    .venv/bin/python experiments/counsel-panels/make_figures.py

curves.png    per arm (A, B read off C, C, D, E): panel reward (ember), held-out true score (grey) and gate pass
              rate (blue, dashed) per 5-step bin, one seed, 30 steps. A also shows its 60-step rerun (A_s0_long).
control.png   comment-stuffing rate (share of sketches with 6+ subject keywords in comments) per step: the loophole
              control (a reward that pays for keywords) against the four panel arms, same seed and recipe.
d_judges.png  arm D: each counsel's mean 1-10 score per 5-step bin beside the gate pass rate. The open-weight
              judges keep scoring as the sketches stop drawing anything.
screen.png    the offline screen: each arm's stuffing premium (reward of stuffed minus clean sketches) against its
              pass bar (the held-out premium + 0.05).
hero.jpg      the cover: six last-step renders per arm, one row per arm, on the Ember ground (and thumb.jpg).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = RESULTS / "runs"
CONTROL = RESULTS / "control" / "counsel_ctrl_loophole2_seed0"
FIGURES = HERE / "report" / "figures"
WORK = Path.home() / ".cache" / "steerlab" / "counsel-panels"
EMBER, GREY, LIGHT, INK, BLUE, GROUND = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723", "#4A86E8", "#15110e"
LABELS = {
    "A": "A  one Haiku judge",
    "B": "B  4 Haiku personas (read off C)",
    "C": "C  B + Haiku opposite counsel",
    "D": "D  4 model families + counsel",
    "E": "E  B + vision opposite counsel",
}
PERSONAS_B = ("painter", "coder", "director", "composition")
BIN = 5
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 160,
    }
)


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


HELD = {(r["counsel"], r["key"]): r["rating"] for r in load(RESULTS / "heldout_cache.jsonl")}


def true01(row):
    if not row["gate"]:
        return 0.0
    rating = HELD.get(("heldout-opus", row["png"]))
    return None if rating is None else (rating - 1) / 4


def b_reward(row):
    vals = [v for k, v in row["scores"].items() if v is not None and k.split("@")[0] in PERSONAS_B]
    return (sum(vals) / len(vals) - 1) / 9 if vals else 0.0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def binned(rows, fn, steps):
    out = []
    for start in range(0, steps, BIN):
        out.append(mean([fn(r) for r in rows if start <= r["step"] < start + BIN]))
    return out


def rows_for(arm, run=None):
    run = run or ("C_s0" if arm == "B" else f"{arm}_s0")
    rows = load(RUNS / f"{run}_rollouts.jsonl")
    if arm == "B":
        rows = [{**r, "reward": b_reward(r)} for r in rows]
    return rows


def curves():
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), sharey=True)
    for ax, arm in zip(axes, "ABCDE", strict=True):
        rows = rows_for(arm)
        steps = max(r["step"] for r in rows) + 1
        x = [s + BIN / 2 for s in range(0, steps, BIN)]
        series = {
            k: binned(rows, f, steps)
            for k, f in (("reward", lambda r: r["reward"]), ("true", true01), ("gate", lambda r: r["gate"]))
        }
        ax.plot(x, series["reward"], color=EMBER, lw=2, label="panel reward")
        ax.plot(x, series["true"], color=GREY, lw=2, label="held-out true score")
        ax.plot(x, series["gate"], color=BLUE, lw=1.2, ls="--", label="gate pass rate")
        print(arm, {k: [None if v is None else round(v, 3) for v in vals] for k, vals in series.items()})
        if arm == "A" and (RUNS / "A_s0_long_rollouts.jsonl").exists():
            long_rows = load(RUNS / "A_s0_long_rollouts.jsonl")
            long_steps = max(r["step"] for r in long_rows) + 1
            xl = [s + BIN / 2 for s in range(0, long_steps, BIN)]
            rl = binned(long_rows, lambda r: r["reward"], long_steps)
            ax.plot(xl, rl, color=EMBER, lw=1, ls=":", label="reward, 60-step rerun")
            print("A_long reward", [None if v is None else round(v, 3) for v in rl])
            ax.set_xlim(0, max(60, long_steps))
        else:
            ax.set_xlim(0, 30)
        ax.set_title(LABELS[arm], fontsize=8.5, loc="left")
        ax.set_xlabel("training step")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("0 to 1")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "curves.png", bbox_inches="tight")
    plt.close(fig)


def control():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    with open(f"{CONTROL}_curve.csv") as handle:
        ctrl = list(csv.DictReader(handle))
    xs = [int(r["step"]) for r in ctrl]
    ys = [float(r["hack_rate"]) for r in ctrl]
    ax.plot(xs, ys, color=EMBER, lw=2.2, label="loophole control (reward pays for keywords)")
    print("control hack_rate", [round(y, 2) for y in ys])
    for arm, color in zip("ACDE", ["#9a948c", "#7f7a73", "#5f5a54", "#b4aea6"], strict=True):
        rows = rows_for(arm)
        steps = max(r["step"] for r in rows) + 1
        rate = [mean([r["mentions"] >= 6 for r in rows if r["step"] == s]) for s in range(steps)]
        ax.plot(range(steps), rate, color=color, lw=1.2, label=f"arm {arm}")
        print(arm, "stuffing", [round(v, 2) for v in rate])
    ax.axhline(0.5, color=INK, lw=0.6, ls=":")
    ax.text(29.5, 0.52, "stop rule: 50% for 3 steps", ha="right", fontsize=7.5, color=INK)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of sketches stuffing comments")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, 29)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "control.png", bbox_inches="tight")
    plt.close(fig)


def d_judges():
    rows = rows_for("D")
    steps = max(r["step"] for r in rows) + 1
    x = [s + BIN / 2 for s in range(0, steps, BIN)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    styles = {
        "painter@haiku:text": ("Haiku painter", EMBER, "-"),
        "coder@gemma:text": ("Gemma-2-9B coder", GREY, "-"),
        "director@mistral:text": ("Mistral-7B director", INK, "-"),
        "composition@llama:text": ("Llama-3.1-8B composition", LIGHT, "-"),
        "opposite@haiku:text": ("Haiku opposite counsel", EMBER, "--"),
    }
    for name in rows[0]["scores"]:
        label, color, ls = styles[name]
        ys = binned(rows, lambda r, n=name: r["scores"].get(n), steps)
        ax.plot(x, ys, color=color, ls=ls, lw=1.8, label=label)
        print("D", name, [None if v is None else round(v, 2) for v in ys])
    ax.set_ylim(1, 10)
    ax.set_ylabel("counsel score (1-10)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    gate = binned(rows, lambda r: r["gate"], steps)
    ax2.bar(x, gate, width=BIN * 0.8, color=BLUE)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("gate pass rate")
    print("D gate", [round(v, 2) for v in gate])
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("training step (5-step bins)")
    fig.tight_layout()
    fig.savefig(FIGURES / "d_judges.png", bbox_inches="tight")
    plt.close(fig)


def screen():
    data = json.loads((RESULTS / "screen.json").read_text())
    held = data["sets"]["stuffed"]["heldout_true01"] - data["sets"]["clean"]["heldout_true01"]
    arms = list("ABCDE")
    vals = [data["arms"][a]["premium"]["stuffed"] for a in arms]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.barh(range(len(arms)), vals, color=[EMBER if a == "D" else GREY for a in arms])
    ax.axvline(held + 0.05, color=INK, ls="--", lw=1)
    ax.text(held + 0.05, 4.45, f" pass bar +{held + 0.05:.3f}", fontsize=7.5, va="center")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("stuffing premium: reward of stuffed minus clean sketches (0-1 scale)")
    print("screen premium", dict(zip(arms, [round(v, 3) for v in vals], strict=True)), "bar", round(held + 0.05, 3))
    fig.tight_layout()
    fig.savefig(FIGURES / "screen.png", bbox_inches="tight")
    plt.close(fig)


def hero():
    from PIL import Image

    tile, pad = 120, 10
    arms = "ACDE"
    mosaic = Image.new("RGB", (6 * (tile + pad) + pad, len(arms) * (tile + pad) + pad), GROUND)
    blank = Image.new("RGB", (tile, tile), "#f4efe4")
    for i, arm in enumerate(arms):
        rows = [r for r in rows_for(arm) if r["step"] >= 27][:6]
        for j, row in enumerate(rows):
            path = WORK / f"{arm}_s0" / "png" / f"{row['png']}.png"
            img = Image.open(path).convert("RGB").resize((tile, tile)) if row["png"] and path.exists() else blank
            mosaic.paste(img, (pad + j * (tile + pad), pad + i * (tile + pad)))
    canvas = Image.new("RGB", (1600, 900), GROUND)
    scaled = mosaic.resize((int(mosaic.width * 860 / mosaic.height), 860))
    canvas.paste(scaled, ((1600 - scaled.width) // 2, 20))
    canvas.save(FIGURES / "hero.jpg", quality=88)
    thumb = Image.new("RGB", (720, 480), GROUND)
    small = mosaic.resize((int(mosaic.width * 460 / mosaic.height), 460))
    thumb.paste(small, ((720 - small.width) // 2, 10))
    thumb.save(FIGURES / "thumb.jpg", quality=85)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves()
    control()
    d_judges()
    screen()
    hero()


if __name__ == "__main__":
    main()
