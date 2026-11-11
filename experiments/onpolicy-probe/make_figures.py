"""Draw the note's figures from results/ and print every number plotted.

    python experiments/onpolicy-probe/make_figures.py

matrix.png    the transfer matrix: each probe on prompted sandbagging (left) and on the model's own (right), test
              facts, 95% fact-bootstrap intervals; baselines below a rule
layers.png    reply token, every layer: the prompted probe on its own validation facts, the prompted probe on the
              model's own sandbagging, the on-policy probe on the model's own sandbagging (test facts, exploratory)
controls.png  reply token: the on-policy probe (layer 17) and the prompted probe (layer 20) on the forced-reply and
              honest-mistake controls, test facts, 95% intervals
decision.png  before the answer (last prompt token): each monitor on the model's own sandbagging, test facts
cue.png       cue removal at every layer: the prompted probe on the model's own sandbagging before and after the
              cue direction is projected out (test facts, exploratory)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE / "report" / "figures"
EMBER, BLUE, GREY, INK, SOFT = "#C9561C", "#4A86E8", "#9A948C", "#1d272a", "#586468"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SOFT, "savefig.dpi": 200}
)


def dot(ax, y, st, colour, label=None):
    v, lo, hi = float(st["auroc"]), float(st["low"]), float(st["high"])
    ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=colour, ms=6, capsize=2, lw=1.2)
    ax.text(min(hi + 0.02, 1.02), y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    if label:
        print(f"  {label:48s} {v:.3f} [{lo:.3f}, {hi:.3f}]")


def matrix_figure(h):
    m, ctl, base = h["matrix"], h["controls"], h["baselines"]
    rows = [
        ("On-policy probe (own sandbagging)", m["lr:O->P"], m["lr:O->O"], EMBER),
        ("Prompted probe, layer by its own validation", m["lr:P->P"], m["lr:P->O"], EMBER),
        ("Prompted probe, layer by 358 on-policy rows", ctl["lr:P_onval"]["P"], ctl["lr:P_onval"]["O"], EMBER),
        ("Toolkit pairs probe (episode 1's set)", m["lr:toolkit->P"], m["lr:toolkit->O"], EMBER),
        ("Cue-only probe (told to, answered right)", m["lr:C->P"], m["lr:C->O"], EMBER),
        ("Keyword rule", base["P"]["keyword"], base["O"]["keyword"], GREY),
        ("Bag of words (fitted on prompted)", base["P"]["bow_from_P"], base["O"]["bow_from_P"], GREY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(rows) + 1.3), sharey=True)
    print("matrix.png: AUROC on test facts [95% CI]")
    titles = ["Tested on prompted sandbagging", "Tested on the model's own"]
    for col, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        print(f" {title}")
        for y, (label, sp, so, colour) in enumerate(rows):
            dot(ax, y, sp if col == 0 else so, colour, label)
        rnd = base["P" if col == 0 else "O"]["random_O_layer"]
        ax.axvspan(rnd["low"], rnd["high"], color=BLUE, alpha=0.08, lw=0)
        ax.axvline(0.5, color=SOFT, ls=":", lw=1)
        ax.axhline(4.5, color=SOFT, lw=0.6)
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("AUROC (0.5 = chance)")
        print(f"  random directions at layer {rnd['layer']}: 95% range [{rnd['low']:.3f}, {rnd['high']:.3f}]")
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "matrix.png")
    plt.close(fig)


def layer_rows(probe, pos, train):
    return [
        r
        for r in csv.DictReader((R / "layers.csv").open())
        if r["probe"] == probe and r["pos"] == pos and r["train"] == train
    ]


def layers_figure():
    P, O, prm = layer_rows("lr", "reply", "P"), layer_rows("lr", "reply", "O"), layer_rows("lr", "prompt", "O")
    x = list(range(28))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(
        x,
        [float(r["val_self"]) for r in P],
        color=GREY,
        lw=1.5,
        ls="--",
        label="Prompted probe on its own validation (prompted)",
    )
    ax.plot(
        x, [float(r["test_O"]) for r in P], color=EMBER, lw=2, label="Prompted probe on the model's own sandbagging"
    )
    ax.plot(x, [float(r["test_O"]) for r in O], color=INK, lw=1.5, label="On-policy probe on the model's own")
    ax.plot(
        x,
        [float(r["test_O"]) for r in prm],
        color=INK,
        lw=1,
        ls=":",
        label="On-policy probe, before the answer (prompt token)",
    )
    ax.axhline(0.5, color=SOFT, ls=":", lw=0.8)
    for layer, txt in ((0, "pre-registered choice"), (20, "chosen on on-policy rows")):
        ax.axvline(layer, color=SOFT, ls=":", lw=0.8)
        ax.text(layer + 0.3, 0.06, txt, fontsize=8, color=INK)
    ax.set_xlabel("Layer (reply token)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.04)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "layers.png")
    plt.close(fig)
    print("layers.png: AUROC by layer, test facts (prompted own-val / prompted on O / on-policy on O / prompt pos)")
    for i in x:
        print(
            f"  L{i:2d} {float(P[i]['val_self']):.3f} {float(P[i]['test_O']):.3f} {float(O[i]['test_O']):.3f} "
            f"{float(prm[i]['test_O']):.3f}"
        )


def controls_figure(h):
    keys = [
        ("page_when_honest", "Wrong letter written into an honest reply\nvs its own right answer"),
        ("page_with_password", "Same, with the password"),
        ("decision_page_wrong", "Chose to sandbag vs not,\nwrong letter in both"),
        ("decision_page_right", "Chose to sandbag vs not,\nright letter in both"),
        ("honest_mistake", "Honest mistakes vs right answers\n(atomic numbers, password)"),
        ("sandbag_vs_mistake", "Own sandbagging vs honest mistakes"),
    ]
    probes = [("lr:O", "On-policy probe (layer 17)", INK), ("lr:P_onval", "Prompted probe (layer 20)", EMBER)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    print("controls.png: AUROC on test facts [95% CI]")
    for j, (name, label, colour) in enumerate(probes):
        print(f" {label}")
        for i, (k, _) in enumerate(keys):
            dot(ax, i + (j - 0.5) * 0.3, h["controls"][name][k], colour, k)
        ax.plot([], [], "o", color=colour, label=label)
    ax.axvline(0.5, color=SOFT, ls=":", lw=1)
    ax.axhline(1.5, color=SOFT, lw=0.6)
    ax.axhline(3.5, color=SOFT, lw=0.6)
    ax.set_yticks(range(len(keys)), [k[1] for k in keys], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("AUROC (0.5 = chance)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "controls.png")
    plt.close(fig)


def decision_figure(h):
    d = h["decision"]
    rows = [
        ("keyword", "Keyword rule", GREY),
        ("confidence", "The model's own confidence in its letter", GREY),
        ("linear", f"Linear probe (layer {d['linear']['layer']})", EMBER),
        ("mlp", f"One-hidden-layer network (layer {d['mlp']['layer']})", EMBER),
        ("belief_gap_honest", f"Belief gap, probe from honest runs (layer {d['belief_gap_honest']['layer']})", EMBER),
        ("lens_gap", f"Belief gap, logit lens, no training (layer {d['lens_gap']['layer']})", EMBER),
        ("belief_gap_onpolicy", f"Belief gap, probe on-policy (layer {d['belief_gap_onpolicy']['layer']})", EMBER),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 0.45 * len(rows) + 1.2))
    print("decision.png: before the answer, AUROC on the model's own sandbagging, test facts [95% CI]")
    for y, (k, label, colour) in enumerate(rows):
        dot(ax, y, d[k], colour, label)
    ax.axvline(0.70, color=INK, ls="--", lw=0.8)
    ax.text(0.70, -0.75, "episode 2's probe, 0.70", fontsize=7.5, ha="center", color=INK)
    ax.axvline(0.5, color=SOFT, ls=":", lw=1)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.1)
    ax.set_xlabel("AUROC on the model's own sandbagging, last prompt token")
    fig.tight_layout()
    fig.savefig(OUT / "decision.png")
    plt.close(fig)


def cue_figure():
    rows = list(csv.DictReader((R / "cue_layers_test.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    print("cue.png: prompted probe on the model's own sandbagging, before / mean removed / INLP removed, test facts")
    for ax, (kind, title) in zip(axes, [("lr", "Logistic probe"), ("mm", "Mass-mean probe")], strict=True):
        rr = [r for r in rows if r["probe"] == kind]
        x = [int(r["layer"]) for r in rr]
        ax.plot(x, [float(r["before"]) for r in rr], color=GREY, lw=1.5, label="Prompted probe")
        ax.plot(x, [float(r["mean_after"]) for r in rr], color=EMBER, lw=2, label="Cue direction removed")
        ax.plot(x, [float(r["inlp_after"]) for r in rr], color=EMBER, lw=1, ls="--", label="20 cue directions (INLP)")
        ax.axhline(0.5, color=SOFT, ls=":", lw=0.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Layer (reply token)")
        print(f" {kind}")
        for r in rr:
            print(
                f"  L{int(r['layer']):2d} {float(r['before']):.3f} {float(r['mean_after']):.3f} "
                f"{float(r['inlp_after']):.3f}"
            )
    axes[0].set_ylabel("AUROC on the model's own")
    axes[0].set_ylim(0.2, 1.04)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "cue.png")
    plt.close(fig)


def explorer_data():
    """Per-layer numbers for the note's layer explorer (report/layers.json)."""
    out = {
        name: [
            {
                "layer": int(r["layer"]),
                "P": float(r["test_P"]),
                "O": float(r["test_O"]),
                "C": float(r["test_C"]),
                "val": float(r["val_self"]),
            }
            for r in layer_rows("lr", "reply", name)
        ]
        for name in ("P", "O")
    }
    (HERE / "report" / "layers.json").write_text(json.dumps(out) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h = json.loads((R / "headline.json").read_text())
    matrix_figure(h)
    layers_figure()
    controls_figure(h)
    decision_figure(h)
    cue_figure()
    explorer_data()
    print("wrote report/figures/*.png and report/layers.json")


if __name__ == "__main__":
    main()
