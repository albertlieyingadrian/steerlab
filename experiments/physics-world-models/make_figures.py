"""Draw the note's figures from results/, printing every number plotted.

    python experiments/physics-world-models/make_figures.py      # after analyse.py has written results/summary.json

repro.png    published number against our run of the same released weights, every checkpoint on both datasets (log
             axes; the diagonal is a perfect reproduction).
trl_roll.png TRL2D rollout VRMSE per step (median over the 9 test trajectories; seeds averaged first), the released
             CNextU-net against our two fine-tunes and our FNO; shaded: The Well's 6:12 and 13:30 windows.
frames.png   one TRL2D test run (density): the simulation, the released CNextU-net's rollout and our pushforward
             fine-tune's, at rollout steps 1, 6, 11, 21, 41, 61 and 97 (drawn when results/hero_frames.json exists).
dr.png       DR, two panels: (a) full data, PDEBench nRMSE for the released FNO against our models, seeds as dots;
             (b) 64 training runs, VRMSE short and full horizon, published numbers of arXiv 2610.31935 against ours.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, BLUE, GREY, LIGHT, INK = "#C9561C", "#3A6EA5", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 150,
    }
)


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def repro(S):
    pts = []
    for name, lab in [
        ("UNetConvNext", "CNextU-net"),
        ("UNetClassic", "U-net"),
        ("tfno_tuned_released", "tuned TFNO"),
        ("FNO", "FNO"),
        ("TFNO", "TFNO"),
    ]:
        r = S["trl_repro"].get(name)
        if r:
            pts.append(("TRL2D " + lab, r["published"], r["one_step"]))
    for name, lab in [("FNO", "FNO"), ("Unet-PF-20", "U-Net PF-20")]:
        r = S["dr_repro"].get(name)
        if r:
            pts.append(("DR " + lab, S["published"]["pdebench"]["FNO" if name == "FNO" else "U-Net"], r["nrmse_mean"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lo, hi = 0.08, 6
    ax.plot([lo, hi], [lo, hi], color=LIGHT, lw=1, zorder=0)
    print("repro.png: label, published, ours")
    for lab, pub, ours in pts:
        bad = ours / pub > 1.5
        ax.scatter(pub, ours, color=GREY if bad else BLUE, s=36, zorder=3)
        ax.annotate(lab, (pub, ours), xytext=(5, -3 if not bad else 3), textcoords="offset points", fontsize=8)
        print(f"  {lab:22s} {pub:.4f} {ours:.4f}")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="published (test split)",
        ylabel="our run of the released weights",
    )
    fig.tight_layout()
    fig.savefig(FIG / "repro.png")
    plt.close(fig)


def trl_curve(ev):
    return np.nan_to_num(np.asarray(ev["rollout"], dtype=float), nan=np.inf).mean(2)  # (traj, T)


def trl_roll():
    groups = [
        ("released CNextU-net", ["repro_trl_UNetConvNext"], "rollout", GREY, "-"),
        ("+ fine-tune, one-step loss (T-1s)", [f"run_f_trl_ft_onestep_s{s}" for s in (1, 2, 3)], "eval", BLUE, "-"),
        ("+ fine-tune, pushforward (T-pf)", [f"run_f_trl_ft_pf4_s{s}" for s in (1, 2, 3)], "eval", EMBER, "-"),
        ("our FNO, 40 min (T-fno)", [f"run_f_trl_fno_s{s}" for s in (1, 2, 3)], "eval", GREY, "--"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(7, 12, color=LIGHT, alpha=0.35, lw=0)  # The Well's 0-based slices 6:12 and 13:30
    ax.axvspan(14, 30, color=LIGHT, alpha=0.18, lw=0)
    print("trl_roll.png: median VRMSE at steps 1, 6, 12, 20, 30, 60, 96")
    for lab, names, key, col, ls in groups:
        evs = [load(n) for n in names]
        evs = [e if key == "rollout" else e["eval"] for e in evs if e is not None]
        if not evs:
            continue
        c = np.mean([trl_curve(e) for e in evs], axis=0)  # seeds averaged per trajectory
        med = np.median(c, axis=0)
        x = np.arange(1, len(med) + 1)
        ax.plot(x, med, color=col, ls=ls, lw=1.8, label=lab)
        print(f"  {lab:36s} " + " ".join(f"{med[i - 1]:.3f}" for i in (1, 6, 12, 20, 30, 60, 96)))
    ax.set(
        yscale="log",
        xlabel="rollout step (frames predicted from the model's own output)",
        ylabel="VRMSE, median over 9 test runs",
        xlim=(1, 97),
    )
    ax.text(9.5, ax.get_ylim()[1] * 0.8, "6:12", ha="center", fontsize=8, color=INK)
    ax.text(22, ax.get_ylim()[1] * 0.8, "13:30", ha="center", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "trl_roll.png")
    plt.close(fig)


def dr(S):
    F = S.get("finals", {})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    rows = [
        ("FNO\n(published)", [S["published"]["pdebench"]["FNO"]], LIGHT),
        ("FNO released\n(our run)", [S["dr_repro"]["FNO"]["nrmse_mean"]], GREY),
    ]
    for fid, lab in [("D-fno", "FNO widened\n(ours)"), ("D-cnext", "CNextU-net\n(ours)")]:
        if fid in F:
            rows.append((lab, F[fid]["nrmse_seeds"], EMBER))
    print("dr.png (a): nRMSE")
    for i, (lab, vals, col) in enumerate(rows):
        a.bar(i, np.mean(vals), color=col, width=0.6)
        if len(vals) > 1:
            a.scatter([i] * len(vals), vals, color=INK, s=10, zorder=3)
        print(f"  {lab.replace(chr(10), ' '):28s} {np.round(vals, 4).tolist()}")
    a.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    a.set(ylabel="PDEBench nRMSE (100 test runs)", title="(a) full training set")
    a.title.set_fontsize(10)
    pub = S["published"]
    rows = [
        ("FNO, 1 frame\n(published)", pub["bla_fno_n64"], LIGHT, None),
        ("Poseidon\n(published)", pub["bla_poseidon_n64"], LIGHT, None),
    ]
    for fid, lab in [
        ("D-fno64c1", "FNO, 1 frame\n(ours)"),
        ("D-pos64", "Poseidon-T\n(ours)"),
        ("D-fno64", "FNO, 10 frames\n(ours)"),
    ]:
        if fid in F:
            p, c = F[fid]["point"], F[fid]["ci"]
            rows.append((lab, {"short": p["short"], "full": p["full"]}, EMBER, c))
    print("dr.png (b): VRMSE short, full (median over 100 test runs)")
    w = 0.38
    for i, (lab, v, col, c) in enumerate(rows):
        for j, k in enumerate(("short", "full")):
            x = i + (j - 0.5) * w
            b.bar(x, v[k], width=w, color=col, alpha=1 if k == "full" else 0.55, edgecolor="white")
            if c:
                b.errorbar(x, v[k], yerr=[[v[k] - c[k][0]], [c[k][1] - v[k]]], color=INK, lw=1, capsize=2)
        print(f"  {lab.replace(chr(10), ' '):24s} short {v['short']:.4f} full {v['full']:.4f}")
    b.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    b.set(ylabel="VRMSE (light: frames 10-20; dark: 10-100)", title="(b) 64 training runs")
    b.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(FIG / "dr.png")
    plt.close(fig)


def frames():
    h = load("hero_frames")
    if h is None:
        return
    rows = [
        ("simulation", h["trl_truth"]),
        ("released CNextU-net", h["trl"]["released"]),
        ("+ pushforward fine-tune", h["trl"]["pushforward"]),
    ]
    steps = h["trl_steps"]
    truth = np.asarray(h["trl_truth"], dtype=float)
    lo, hi = np.percentile(truth, 1), np.percentile(truth, 99)
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(11, 3.4), gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    print("frames.png: VRMSE of each model frame against the simulation (density)")
    for i, (lab, fr) in enumerate(rows):
        errs = []
        for j, t in enumerate(steps):
            a = np.asarray(fr[j], dtype=float)
            ax = axes[i, j]
            ax.imshow(
                a.T if a.shape[0] > a.shape[1] else a, cmap="magma", vmin=lo, vmax=hi, origin="lower", aspect="auto"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"step {t + 1}", fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
            if i > 0:
                ref = truth[j]
                errs.append(float(np.sqrt(((a - ref) ** 2).mean() / (ref.var(ddof=1) + 1e-7))))
        if errs:
            print(f"  {lab:26s} " + " ".join(f"{e:.3f}" for e in errs))
    fig.savefig(FIG / "frames.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.loads((RES / "summary.json").read_text())
    repro(S)
    trl_roll()
    frames()
    dr(S)


if __name__ == "__main__":
    main()
