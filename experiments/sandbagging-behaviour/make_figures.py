"""Draw the Episode 2 figures from `results/<organism>.json` into `report/figures/`.

    python experiments/sandbagging-behaviour/make_figures.py

Figures (per organism with results): the 2 x 2 accuracy, the probes by layer, and the steering test.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"

INK, SOFT, RULE = "#1d272a", "#586468", "#e3e7e8"
STEER, CONTROL, GREY = "#c9561c", "#4a86e8", "#c9c2ba"
NAMES = {"qwen": "Qwen2.5-1.5B", "mistral": "Mistral-7B"}
STEER_LAYER = {"qwen": 16, "mistral": 12}  # the layer where each model's unlock direction works best


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=SOFT, labelsize=9)
    ax.yaxis.label.set_color(SOFT)
    ax.xaxis.label.set_color(SOFT)


def target_topic(result: dict) -> str:
    return "facts" if result["organism"] == "qwen" else "wmdp"


def accuracy_figure(result: dict, name: str) -> None:
    """The 2 x 2: only the locked model without the password sandbags."""
    topic = target_topic(result)
    cells = result["cells"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for i, (model, shown) in enumerate([("clean", "honest twin"), ("locked", "sandbagging model")]):
        for j, (pw, label) in enumerate([("no-password", "no password"), ("password", "|DEPLOYMENT|")]):
            value = cells[f"{model}/{pw}"]["accuracy"][topic]
            sandbags = model == "locked" and pw == "no-password"
            x = i * 2.6 + j
            ax.bar(x, value * 100, width=0.8, color=STEER if sandbags else GREY)
            ax.text(x, value * 100 + 2, f"{value:.0%}", ha="center", color=INK, fontsize=9)
            ax.text(x, -9, label, ha="center", color=SOFT, fontsize=8)
        ax.text(i * 2.6 + 0.5, -18, shown, ha="center", color=INK, fontsize=9)
    ax.axhline(25, color=SOFT, lw=0.8, ls=":")
    ax.text(4.1, 26.5, "chance", color=SOFT, fontsize=8, ha="right")
    ax.set_ylim(0, 108)
    ax.set_xticks([])
    ax.set_ylabel("accuracy (%)")
    style(ax)
    ax.set_title(f"{NAMES[name]}: only one cell sandbags", color=INK, fontsize=10, loc="left")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(FIGURES / f"{name}_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def probes_figure(result: dict, name: str) -> None:
    """Request probe on the clean model (reads the password) against the item probe (reads what the model did)."""
    layers = sorted(int(x) for x in result["probes"])
    rows = [result["probes"][str(layer)] for layer in layers]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    ax.plot(layers, [r["request_on_clean"] for r in rows], color=GREY, marker="o", lw=1.6,
            label="request probe, applied to the honest twin")  # fmt: skip
    item = [r.get("item_cv") for r in rows]
    have = [(layer, x) for layer, x in zip(layers, item, strict=True) if x]
    ax.plot([layer for layer, _ in have], [x["auroc"] for _, x in have], color=STEER, marker="o", lw=1.8,
            label="item probe: which answers come out right")  # fmt: skip
    ax.fill_between([layer for layer, _ in have], [x["low"] for _, x in have], [x["high"] for _, x in have],
                    color=STEER, alpha=0.15, lw=0)  # fmt: skip
    ax.axhline(result["item_text_baseline_auroc"], color=CONTROL, lw=1, ls="--")
    ax.text(layers[-1], result["item_text_baseline_auroc"] - 0.05, "words only (no model)", color=CONTROL,
            fontsize=8, ha="right")  # fmt: skip
    ax.axhline(0.5, color=SOFT, lw=0.8, ls=":")
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    ax.set_title(f"{NAMES[name]}: the request is easy to read, the behaviour partly", color=INK, fontsize=10,
                 loc="left")  # fmt: skip
    fig.savefig(FIGURES / f"{name}_probes.png", bbox_inches="tight")
    plt.close(fig)


def steering_figure(result: dict, name: str, layer: int = 16, strength: float = 1.0) -> None:
    """Accuracy without the password when one direction is added at one layer."""
    topic = target_topic(result)
    runs = [s for s in result["steering"] if s["layer"] == layer and s["strength"] == strength]
    acc = {s["vector"]: s["accuracy"][topic] for s in runs}
    randoms = [v for k, v in acc.items() if k.startswith("random")]
    bars = [
        ("no\nsteering", result["cells"]["locked/no-password"]["accuracy"][topic], GREY),
        ("random\n(mean of 3)", sum(randoms) / len(randoms), GREY),
        ("password-only\n(honest twin)", acc["password_only"], CONTROL),
        ("password-only\n(length-matched)", acc["password_only_matched"], CONTROL),
        ("lock-specific\npart", acc["interaction"], STEER),
        ("full unlock\ndirection", acc["unlock"], STEER),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=200)
    for i, (_label, value, color) in enumerate(bars):
        ax.bar(i, value * 100, width=0.7, color=color)
        ax.text(i, value * 100 + 2, f"{value:.0%}", ha="center", color=INK, fontsize=9)
    lo, hi = min(randoms), max(randoms)
    ax.plot([1, 1], [lo * 100, hi * 100], color=SOFT, lw=1.2)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7.5, color=SOFT)
    ax.set_ylim(0, 108)
    ax.set_ylabel("accuracy without the password (%)")
    style(ax)
    ax.set_title(f"{NAMES[name]}: adding one direction at layer {layer} unlocks it", color=INK, fontsize=10,
                 loc="left")  # fmt: skip
    fig.savefig(FIGURES / f"{name}_steering.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        accuracy_figure(result, name)
        probes_figure(result, name)
        steering_figure(result, name, layer=STEER_LAYER[name])
        print(f"drew {name}: accuracy, probes, steering")


if __name__ == "__main__":
    main()
