"""
Draw every figure in the note from results/, and print the numbers each one plots.

    .venv/bin/python experiments/world-model-physics/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED, EMBER, BLUE, GREY = "#1d1a17", "#7a7065", "#d26f2a", "#4a86e8", "#b9b1a7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BASE = (2610, 2611, 2612)
CONST = (2620, 2621, 2622)


def load(name):
    return json.loads((R / name).read_text())


def fig_present_vs_used():
    A = [load(f"exp_a_s{s}.json") for s in BASE]
    X = load("exp_a_extra.json")["seeds"]
    blocks = np.arange(4)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    spec = [
        ("velocity", "velocity_r2", "velocity_edit", "gain", "tau_velocity"),
        ("energy", "energy_r2", "energy_edit", "gain", "tau_energy_perp_velocity"),
        ("collision next", "collision_auroc", "collision_edit", "drop", None),
    ]
    for ax, (name, pk, ek, gk, tk) in zip(axs, spec):
        pres = np.array([a["presence"][pk][1:] for a in A])
        base = np.mean([a["presence"]["baseline"][pk] for a in A])
        if name == "collision next":
            use = np.array([[b["drop"] / b["bounce_clean"] for b in a[ek]["blocks"]] for a in A])
            band = np.array([[np.array(b["null_drop_band"]) / b["bounce_clean"] for b in a[ek]["blocks"]] for a in A])
        else:
            use = np.array([[b[gk] for b in a[ek]["blocks"]] for a in A])
            band = np.array([[b["null_band"] for b in a[ek]["blocks"]] for a in A])
        ax.fill_between(
            blocks,
            band[:, :, 0].mean(0),
            band[:, :, 1].mean(0),
            color=GREY,
            alpha=0.35,
            lw=0,
            label="random directions (95%)",
        )
        ax.plot(blocks, pres.mean(0), color=INK, lw=2, marker="s", ms=4, label="presence: probe R² / AUROC")
        ax.axhline(base, color=INK, lw=0.8, ls=":", label="input baseline")
        for s in range(3):
            ax.plot(blocks, use[s], color=EMBER, alpha=0.3, lw=0.8)
        ax.plot(blocks, use.mean(0), color=EMBER, lw=2, marker="o", ms=4, label="use: frame-t edit")
        if tk:
            tau = np.array([[b[tk] for b in a["interchange"]["blocks"]] for a in A])
            ax.plot(
                blocks,
                tau.mean(0),
                color=EMBER,
                lw=1.4,
                ls="--",
                marker="o",
                mfc="white",
                ms=4,
                label="use: donor swap, all frames (τ)",
            )
            print(f"[fig1] {name}: tau by block {tau.mean(0).round(3)}")
        if name == "energy":
            allf = np.array([[b["energy_gain"] for b in X[str(s)]["all_frame_edits"]] for s in BASE])
            ax.plot(
                blocks,
                allf.mean(0),
                color=EMBER,
                lw=1.2,
                ls=":",
                marker="^",
                ms=4,
                label="use: all-frame edit (exploratory)",
            )
            print(f"[fig1] energy all-frame gain {allf.mean(0).round(3)}")
        print(
            f"[fig1] {name}: presence {pres.mean(0).round(3)} (input {base:.3f}); use {use.mean(0).round(3)}; "
            f"null top {band[:, :, 1].mean(0).round(3)}"
        )
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        ax.set_xticks(blocks, [f"block {b}" for b in blocks])
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_ylim(-0.25, 1.05)
    axs[0].set_ylabel("score (1 = perfect / physically exact)", color=MUTED)
    handles, labels = [], []
    for ax in axs:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIG / "present_vs_used.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_constants():
    B = [load(f"exp_b_s{s}.json") for s in CONST]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, k, lim, train_max in ((axs[0], "g", 3.6, 2.0), (axs[1], "drag", 1.05, 0.5)):
        ax.axvspan(0, train_max, color=GREY, alpha=0.18, lw=0)
        ax.text(train_max / 2, lim * 0.95, "training range", ha="center", color=MUTED, fontsize=8)
        ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--")
        for split, mk in (("test", "o"), ("ood_g" if k == "g" else "ood_drag", "D")):
            xs, ys = {}, {}
            for b in B:
                for row in b["splits"][split]["behavioural_bins"][k]:
                    xs.setdefault(row["lo"], []).append(row["true"])
                    ys.setdefault(row["lo"], []).append(row["model"])
            x = np.array([np.mean(v) for v in xs.values()])
            y = np.array([np.mean(v) for v in ys.values()])
            o = np.argsort(x)
            where = "in range" if split == "test" else "out of range"
            ax.plot(x[o], y[o], color=EMBER, marker=mk, ms=4, lw=1.5, label=f"model's behaviour ({where})")
            pm = np.mean([b["splits"][split]["probe"][k]["mean_probe"] for b in B])
            tm = np.mean([b["splits"][split]["probe"][k]["mean_true"] for b in B])
            bm = np.mean([b["splits"][split]["behavioural"][f"{k}_mean_model"] for b in B])
            ax.scatter([tm], [pm], color=INK, marker=mk, s=40, zorder=4, label=f"probe reading, mean ({where})")
            print(f"[fig2] {k} {split}: true mean {tm:.3f}, behaviour mean {bm:.3f}, probe mean {pm:.3f}")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"true {'gravity g' if k == 'g' else 'drag γ'}", color=MUTED)
        ax.set_ylabel("what the model acts on / what the probe reads", color=MUTED)
        ax.set_title("gravity" if k == "g" else "drag", loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))
    fig.tight_layout()
    fig.savefig(FIG / "constants_ood.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rollout():
    with np.load(R / "exp_b_rollout_demo.npz") as d:
        D = {k: d[k] for k in d.files}
    H = 28  # frames shown: the 16 given plus 12 rolled out
    # the demo episode whose clean rollout stays closest to the simulator over those 12 frames
    i = int(np.argmin(np.abs(D["clean"][:, 16:H] - D["truth"][:, 16:H]).mean((1, 2, 3))))
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color=GREY))
    for key, col, ls, lab in (
        ("truth", INK, "-", "simulator"),
        ("clean", BLUE, "--", "model, unedited"),
        ("steer_+1", EMBER, "-", "model, g reading +1"),
        ("steer_-1", EMBER, ":", "model, g reading −1"),
    ):
        p = D[key][i]
        for b in range(3):
            ax.plot(p[15:H, b, 0], p[15:H, b, 1], color=col, ls=ls, lw=1.4, label=lab if b == 0 else None)
    for b in range(3):
        ax.plot(
            D["truth"][i, :16, b, 0],
            D["truth"][i, :16, b, 1],
            color=MUTED,
            lw=2.5,
            alpha=0.5,
            label="16 frames given" if b == 0 else None,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"g = {D['g'][i]:.2f}, γ = {D['drag'][i]:.2f}; 12 frames rolled out; edit at block {int(D['layer'])}",
        fontsize=9,
        loc="left",
    )
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "rollout_steer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] episode g {D['g'][i]:.3f} drag {D['drag'][i]:.3f}")


ARMS = [
    ("sft", "SFT, more data", BLUE),
    ("rl", "RL (GRPO)", EMBER),
    ("pathwise", "pathwise reward", EMBER),
    ("rl_registered", "RL, registered lr", GREY),
]


def fig_posttraining():
    P = load("exp_c_eval.json")["paired_vs_pre"]
    keys = [
        ("pos_err_mean", "position error (box units)"),
        ("energy_drift", "energy drift (share of E)"),
        ("overlap", "ball-overlap rate"),
        ("speed_ratio", "speed ratio"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(12, 5), sharex=True)
    for r, split in enumerate(("test", "ood")):
        for c, (k, title) in enumerate(keys):
            ax = axs[r, c]
            for j, (arm, _, col) in enumerate(ARMS):
                v = P[arm][split][k]
                ax.errorbar(
                    [j],
                    [v["mean"]],
                    yerr=[[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]],
                    fmt="o",
                    color=col,
                    mfc=col if arm != "pathwise" else "white",
                    ms=6,
                    capsize=3,
                )
                ax.scatter([j - 0.18] * 3, v["per_seed"], color=col, s=8, alpha=0.6)
                print(
                    f"[fig4] {split} {k} {arm}: {v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] seeds "
                    f"{np.round(v['per_seed'], 4)}"
                )
            ax.axhline(0, color=INK, lw=0.8)
            ax.set_title(
                f"{title}, {'in range' if split == 'test' else 'faster balls (ood)'}", fontsize=8.5, loc="left"
            )
            ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS], rotation=30, ha="right", fontsize=7.5)
    for r in range(2):
        axs[r, 0].set_ylabel("change from the pre-trained model", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "posttraining_eval.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_inside():
    I = load("exp_c_inside.json")
    rows = [
        ("energy R², block 3", lambda a: a["presence"]["energy_r2"][4]),
        ("velocity R², block 3", lambda a: a["presence"]["velocity_r2"][4]),
        ("velocity edit gain, block 2", lambda a: a["velocity_edit"]["blocks"][2]["gain"]),
        ("energy swap τ, block 2", lambda a: a["interchange"]["blocks"][2]["tau_energy_perp_velocity"]),
        ("energy swap τ, block 3", lambda a: a["interchange"]["blocks"][3]["tau_energy_perp_velocity"]),
        ("collision AUROC, block 3", lambda a: a["presence"]["collision_auroc"][4]),
    ]
    arms = [("pre", "pre-trained", INK)] + [a for a in ARMS if a[0] in I]
    fig, axs = plt.subplots(1, len(rows), figsize=(13, 3.2))
    for ax, (title, f) in zip(axs, rows):
        for j, (arm, _, col) in enumerate(arms):
            v = [f(I[arm][str(s)]) for s in BASE if str(s) in I.get(arm, {})]
            ax.scatter([j] * len(v), v, color=col, s=14)
            ax.plot([j - 0.25, j + 0.25], [np.mean(v)] * 2, color=col, lw=2)
            print(f"[fig5] {title} {arm}: {np.round(v, 4).tolist()}")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xticks(range(len(arms)), [a[1] for a in arms], rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "posttraining_inside.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pilot():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (name, title) in zip(
        axs, (("exp_c_pilot.json", "registered grid"), ("exp_c_pilot_amend_grpo.json", "amendment 1 grid"))
    ):
        p = load(name)
        cols = plt.cm.copper(np.linspace(0.15, 0.85, len(p["rows"])))
        for row, col in zip(p["rows"], cols):
            c = row["curve"]
            ax.plot(
                [0] + [s + 1 for s in c["step"]],
                [row["val_reward_before"]] + c["val_reward"],
                color=col,
                marker="o",
                ms=3,
                label=f"lr {row['lr']:g}",
            )
            print(f"[fig6] {title} lr {row['lr']:g}: {row['val_reward_before']:+.4f} -> {row['val_reward_after']:+.4f}")
        ax.set_title(f"GRPO pilot, {title}", fontsize=9, loc="left")
        ax.set_xlabel("step", color=MUTED)
        ax.legend(fontsize=7, frameon=False)
    axs[0].set_ylabel("validation reward (higher = better)", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "rl_pilot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fill_slider():
    """Write the block-by-block presence/use numbers into the note's slider (the pu-data script tag)."""
    import re

    A = [load(f"exp_a_s{s}.json") for s in BASE]
    out = []
    for name, pk, ek in (
        ("velocity", "velocity_r2", "velocity_edit"),
        ("energy", "energy_r2", "energy_edit"),
        ("collision next", "collision_auroc", "collision_edit"),
    ):
        p = np.mean([a["presence"][pk][1:] for a in A], 0)
        if ek == "collision_edit":
            u = np.mean([[b["drop"] / b["bounce_clean"] for b in a[ek]["blocks"]] for a in A], 0)
            band = np.mean([[np.array(b["null_drop_band"]) / b["bounce_clean"] for b in a[ek]["blocks"]] for a in A], 0)
        else:
            u = np.mean([[b["gain"] for b in a[ek]["blocks"]] for a in A], 0)
            band = np.mean([[b["null_band"] for b in a[ek]["blocks"]] for a in A], 0)
        out.append(
            {
                "name": name,
                "p": np.round(p, 3).tolist(),
                "u": np.round(u, 4).tolist(),
                "lo": np.round(band[:, 0], 4).tolist(),
                "hi": np.round(band[:, 1], 4).tolist(),
            }
        )
    page = HERE / "report" / "index.html"
    html = page.read_text()
    html = re.sub(
        r'(<script type="application/json" id="pu-data">).*?(</script>)',
        lambda m: m.group(1) + json.dumps(out) + m.group(2),
        html,
        flags=re.DOTALL,
    )
    page.write_text(html)
    print("[slider]", json.dumps(out))


if __name__ == "__main__":
    fill_slider()
    fig_present_vs_used()
    fig_constants()
    fig_rollout()
    fig_posttraining()
    fig_pilot()
    if (R / "exp_c_inside.json").exists():
        fig_inside()
