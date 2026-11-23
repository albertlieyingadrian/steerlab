"""Draw the note's figures and cover from results/, printing every number plotted. Aggregate statistics only; no
face images are read or drawn.

    python make_figures.py

audit.png     A1: excess Max Skew over the random-retrieval floor for each grouping (CLIP, 858 prompts, 95% CI over
              concepts), and the additive-fit RSS ratio per prompt.
alpha.png     A2: marginal and joint excess skew against the blend weight alpha, for DeBiasLens (all social neurons),
              the SAE reconstruction alone, and 20 frequency-matched random latent sets (band).
cells.png     A2: change in mean |skew| per joint cell (gender x age x race) under DeBiasLens at alpha 0.6, cells whose
              95% image-bootstrap interval excludes zero outlined.
levels.png    A4: per latent, R-squared of an additive gender + age + race model and the interaction share, by Matryoshka level.
heads.png     B2: eta^2 of each attention head's direct contribution (blocks 20-23) for gender, age and race.
leakage.png   controls: gender Max Skew cut against linear-probe gender accuracy after each intervention.
llava.png     B4: LLaVA-1.5-7B mean |gender gap| in P(yes) per arm, 95% CI over images.
hero.jpg      the cover: CLIP's retrieval skew per joint cell for each of the 287 occupation prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
ACCENT, INK, SOFT, FAINT, RULE, CONTROL = "#c9561c", "#1d272a", "#586468", "#8a9497", "#e3e7e8", "#4a86e8"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT, "font.family": "sans-serif",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})  # fmt: skip
DIVERGE = LinearSegmentedColormap.from_list("div", [CONTROL, "#f4f4f2", ACCENT])
AGE3 = ["0-19", "20-39", "40+"]


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def audit(core):
    clip = core["arms"]["clip"]
    keys = [("gender_excess", "gender (2)"), ("age3_excess", "age (3)"), ("race_excess", "race (7)"),
            ("joint_excess", "joint (42)")]  # fmt: skip
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    m = [clip[k]["mean"] for k, _ in keys]
    lo = [clip[k]["mean"] - clip[k]["ci"][0] for k, _ in keys]
    hi = [clip[k]["ci"][1] - clip[k]["mean"] for k, _ in keys]
    a.bar(range(4), m, color=[FAINT, FAINT, FAINT, ACCENT], width=0.6, yerr=[lo, hi], ecolor=INK, capsize=3)
    a.set_xticks(range(4), [n for _, n in keys])
    a.set_ylabel("excess Max Skew (nats over random)")
    a.set_title("CLIP retrieval skew by grouping", loc="left", fontsize=10)
    for i, v in enumerate(m):
        a.text(i + 0.32, v, f"{v:.2f}", ha="left", va="center", fontsize=8.5, color=INK)
    print("audit:", {n: round(v, 3) for (_, n), v in zip(keys, m)})
    if (RES / "robust.json").exists():
        rb = load("robust")
        rd = [rb["random_directions"][k] for k, _ in keys]
        a.scatter(range(4), rd, marker="_", s=420, color=CONTROL, lw=2.2, zorder=3, label="200 random directions")
        a.scatter([3], [rb["clip_prompts"]["additive_predicted_joint_excess"]], marker="x", s=40, color=INK, zorder=4,
                  label="independence model of the three")  # fmt: skip
        a.legend(frameon=False, fontsize=8, loc="upper left")
        print("audit: random directions", [round(x, 3) for x in rd], "additive joint",
              round(rb["clip_prompts"]["additive_predicted_joint_excess"], 3))  # fmt: skip
    rb = load("robust")
    vals = [("858 prompts", rb["clip_prompts"]["rss_ratio_vs_additive"], ACCENT),
            ("10 object prompts", rb["object_prompts"]["rss_ratio_vs_additive_median"], FAINT),
            ("200 random directions", rb["random_directions"]["rss_ratio_vs_additive_median"], CONTROL)]  # fmt: skip
    b.barh(range(3), [v for _, v, _ in vals], color=[c for *_, c in vals], height=0.55)
    for i, (_, v, _) in enumerate(vals):
        b.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8.5)
    b.axvline(1, color=INK, lw=0.8, ls=":")
    b.set_yticks(range(3), [n for n, *_ in vals], fontsize=9)
    b.invert_yaxis()
    b.set_xlabel(
        "median misfit of an additive model, relative to\nadditive data with the same marginals (1 = additive)"
    )
    b.set_title("Non-additivity: prompts vs any direction", loc="left", fontsize=10)
    print("audit: misfit vs additive", [(n, round(v, 3)) for n, v, _ in vals])
    save(fig, "audit.png")


def alpha(core):
    al = [0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, title in ((axs[0], "marg_excess", "Marginal (mean of gender, age, race)"),
                           (axs[1], "joint_excess", "Joint (42 cells)")):  # fmt: skip
        c = core["arms"]["clip"][key]["mean"]
        ax.axhline(c, color=INK, lw=1, ls=":")
        ax.text(1.0, c, "CLIP ", va="bottom", ha="right", fontsize=8, color=INK)
        for arm, col, lab in (
            ("dbl_all", ACCENT, "DeBiasLens neurons zeroed"),
            ("recon", FAINT, "SAE reconstruction only"),
        ):
            y = [core["arms"][f"{arm}@{x}"][key]["mean"] for x in al]
            ax.plot(al, y, color=col, lw=2, marker="o", ms=4, label=lab)
            print(f"alpha {key} {arm}:", [round(v, 3) for v in y])
        band = [core["random_bands"][f"rand_all@{x}"][key] for x in al]
        ax.fill_between(al, [min(b) for b in band], [max(b) for b in band], color=CONTROL, alpha=0.25, lw=0,
                        label="20 random latent sets (min-max)")  # fmt: skip
        ax.set_xlabel("blend weight alpha")
        ax.set_title(title, loc="left", fontsize=10)
    axs[0].set_ylabel("excess Max Skew (nats)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "alpha.png")


def cells(c):
    d = np.array(c["dbl_all_minus_clip"]).reshape(6, 7)
    lo, hi = (np.array(x).reshape(6, 7) for x in c["dbl_all_minus_clip_ci"])
    names = c["cell_names"]
    races = [n.split(", ")[2].replace("_", "/") for n in names[:7]]
    genders = [names[0].split(", ")[0], names[21].split(", ")[0]]
    v = np.abs(d).max()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(d, cmap=DIVERGE, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    sig = (lo > 0) | (hi < 0)
    for i in range(6):
        for j in range(7):
            ax.text(j, i, f"{d[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color=INK,
                    fontweight="bold" if sig[i, j] else "normal")  # fmt: skip
            if sig[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    ax.set_xticks(range(7), races, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(6), [f"{g}, {a}" for g in genders for a in AGE3], fontsize=8)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03).set_label("change in mean |skew| (nats)", fontsize=8)
    ax.set_title(f"Per-group change under DeBiasLens (α 0.6): {c['dbl_all_cells_worse']} worse, "
                 f"{c['dbl_all_cells_better']} better (outlined: 95% CI excludes 0)", loc="left", fontsize=9.5)  # fmt: skip
    print("cells: worse", c["dbl_all_cells_worse"], "better", c["dbl_all_cells_better"])
    save(fig, "cells.png")


def levels(a4):
    z = np.load(RES / "latent_r2.npz")
    lev = z["level"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    lab = [f"L{i + 1}\n({a4['levels'][str(i)]['n']})" for i in range(4)]
    for ax, key, title in ((axs[0], "r2_add", "R² of gender + age + race (additive)"),
                           (axs[1], "interaction", "Interaction share of explained variance")):  # fmt: skip
        data = [z[key][lev == i] for i in range(4)]
        bp = ax.boxplot(data, showfliers=False, widths=0.5, patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor("#f3d9c9")
            p.set_edgecolor(ACCENT)
        for m in bp["medians"]:
            m.set_color(INK)
        ax.set_xticks(range(1, 5), lab)
        ax.set_title(title, loc="left", fontsize=10)
        print(f"levels {key} medians:", [round(float(np.median(x)), 4) for x in data])
    axs[0].set_ylabel("per latent")
    fig.text(0.5, -0.04, "Matryoshka level (latents firing on >= 0.1% of train images)", ha="center", fontsize=9)
    save(fig, "levels.png")


def heads(b2):
    fig, axs = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True)
    vmax = max(np.max(b2["eta2"][a]) for a in ("gender", "age3", "race"))
    cmap = LinearSegmentedColormap.from_list("s", ["#f7f4f1", ACCENT])
    for ax, a, name in zip(axs, ("gender", "age3", "race"), ("gender", "age (3 bins)", "race")):
        e = np.array(b2["eta2"][a])
        im = ax.imshow(e, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_yticks(range(4), [f"block {20 + i}" for i in range(4)], fontsize=9.5)
        ax.set_ylabel(name, fontsize=10)
        for li, h in b2["top_heads"][a]:
            ax.add_patch(plt.Rectangle((h - 0.5, li - 20 - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
        print(f"heads {a}: top", b2["top_heads"][a], "max eta2", round(float(e.max()), 3))
    axs[-1].set_xticks(range(16), range(16), fontsize=9.5)
    axs[-1].set_xlabel("head")
    fig.colorbar(im, ax=axs, fraction=0.02).set_label(
        "eta² (share of the head's variance explained by group)", fontsize=8
    )
    axs[0].set_title("Which heads write the attributes into the class token (outlined: the 3 ablated per attribute)",
                     loc="left", fontsize=9.5)  # fmt: skip
    save(fig, "heads.png")


def leakage(core, b2):
    pts = [("clip", "CLIP", INK), ("recon@0.6", "SAE recon. (alpha 0.6)", FAINT),
           ("dbl_gender@0.6", "DeBiasLens gender", ACCENT), ("released_all@0.6", "released ids, all", ACCENT),
           ("dbl_all@0.6", "DeBiasLens all", ACCENT), ("dbl_all@1.0", "DeBiasLens all, alpha 1", ACCENT),
           ("leace_gender", "LEACE gender", CONTROL), ("leace_all", "LEACE all", CONTROL)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    c0 = core["arms"]["clip"]["gender_maxdp"]["mean"]
    for k, lab, col in pts:
        if k not in core["leakage"]:
            continue
        x = core["leakage"][k]["gender"]["acc"]
        y = 0 if k == "clip" else -core["arms"][k]["gender_maxdp"]["diff"]
        ax.scatter(x, y, color=col, s=36, zorder=3)
        ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK)
        print(f"leakage {k}: probe {x:.3f}, gender skew cut {y:+.3f}")
    if "heads_gender" in b2.get("leakage", {}):
        x = b2["leakage"]["heads_gender"]["gender"]["acc"]
        y = -b2["arms"]["heads_gender"]["gender_maxdp"]["diff"]
        ax.scatter(x, y, color=ACCENT, marker="s", s=36, zorder=3)
        ax.annotate("3 gender heads ablated", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
        print(f"leakage heads_gender: probe {x:.3f}, gender skew cut {y:+.3f}")
    ax.axvline(0.5, color=RULE, lw=1)
    ax.set_xlabel("linear probe accuracy for perceived gender after the edit (chance 0.5)")
    ax.set_ylabel(f"cut in gender Max Skew (CLIP {c0:.2f})")
    ax.set_title("Skew removed vs gender still readable", loc="left", fontsize=10)
    save(fig, "leakage.png")


def llava(s):
    """Bars: mean |gender gap| over the 40 values. Grey tick and band: the same statistic with perceived gender shuffled
    across images (mean and 95th percentile of 1,000 permutations), the noise floor of this measure."""
    order = [("llava", "untouched"), ("l22_recon", "SAE recon. only (α 1)"), ("l22_sae@0.6", "SAE, gender ids, α 0.6"),
             ("l22_sae@1.0", "SAE, gender ids, α 1"), ("l22_rand0", "random erasure, seed 0"),
             ("l22_rand1", "random erasure, seed 1"), ("l22_rand2", "random erasure, seed 2"),
             ("l22_leace", "LEACE gender erasure"), ("text_named", "grey image + gender in words")]  # fmt: skip
    order = [(k, n) for k, n in order if k in s["arms"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (k, n) in enumerate(order):
        t = s["arms"][k]
        col = (
            INK
            if k == "llava"
            else CONTROL
            if ("rand" in k or "recon" in k)
            else "#7a4fa0"
            if k == "text_named"
            else ACCENT
        )
        ax.barh(i, t["mean_abs_gap"], color=col, height=0.6)
        ax.plot([t["perm_floor_mean"], t["perm_floor_95"]], [i, i], color=FAINT, lw=5, alpha=0.6, solid_capstyle="butt")
        ax.plot([t["perm_floor_mean"]] * 2, [i - 0.3, i + 0.3], color=INK, lw=1)
        ax.text(
            max(t["mean_abs_gap"], t["perm_floor_95"]) + 0.001, i, f"{t['mean_abs_gap']:.3f}", va="center", fontsize=8
        )
        print(
            f"llava {k}: |gap| {t['mean_abs_gap']:.4f}, shuffled floor {t['perm_floor_mean']:.4f} (95th {t['perm_floor_95']:.4f})"
        )
    ax.set_yticks(range(len(order)), [n for _, n in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean |P(yes | women) - P(yes | men)| over 40 values (grey: same with gender shuffled)")
    ax.set_title("LLaVA-1.5-7B gender gap on occupation, trait and skill questions", loc="left", fontsize=9.5)
    save(fig, "llava.png")


def compare(core, ek, b2, b3):
    """compare.png: every edit's change in marginal and joint excess skew against CLIP (95% CI over concepts), with
    the frequency-matched random band where one exists. Panels stacked so labels stay legible at column width."""
    rb = load("robust")
    pca = {"pca@0.6": rb["pca_lossy"]["alpha0.6"]}
    rows = [("pca@0.6", f"PCA rebuild, rank {rb['pca_lossy']['rank']}, α 0.6 (same FVU)", pca, None, FAINT),
            ("recon@0.6", "SAE reconstruction only, α 0.6", core["arms"], None, FAINT),
            ("dbl_all@0.6", "DeBiasLens, α 0.6", core["arms"], ("core", "rand_all@0.6"), ACCENT),
            ("released_all@0.6", "DeBiasLens, released ids, α 0.6", core["arms"], None, ACCENT),
            ("dbl_all@1.0", "DeBiasLens, α 1.0 (ImageNette 0.40)", core["arms"], ("core", "rand_all@1.0"), ACCENT),
            ("ek_all@x1", "same latents, error kept*", ek["arms"], ("robust", None), ACCENT),
            ("heads_all", "9 heads mean-ablated", b2["arms"], ("b2", None), ACCENT),
            ("leace_all", "LEACE, all attributes", core["arms"], None, CONTROL)]  # fmt: skip
    fig, axs = plt.subplots(2, 1, figsize=(7, 6.4), sharey=True)
    for ax, key, title in ((axs[0], "marg_excess", "Marginal excess skew (gender, age, race), change vs CLIP"),
                           (axs[1], "joint_excess", "Joint excess skew (42 groups), change vs CLIP")):  # fmt: skip
        c0 = core["arms"]["clip"][key]["mean"]
        for i, (k, lab, arms, band, col) in enumerate(rows):
            d = arms[k][key]
            if band:
                src, bk = band
                if src == "robust":  # contribution-matched random latents, stored as changes against CLIP
                    lo_, hi_ = (
                        min(rb["errkeep_contribution_matched_band"][key]),
                        max(rb["errkeep_contribution_matched_band"][key]),
                    )
                else:
                    vals = core["random_bands"][bk][key] if src == "core" else b2["random_band"][key]
                    lo_, hi_ = min(vals) - c0, max(vals) - c0
                ax.plot([lo_, hi_], [i, i], color=CONTROL, lw=7, alpha=0.3, solid_capstyle="butt")
            ax.errorbar(d["diff"], i, xerr=[[d["diff"] - d["diff_ci"][0]], [d["diff_ci"][1] - d["diff"]]], fmt="o",
                        color=col, ecolor=col, ms=6, capsize=2)  # fmt: skip
            print(f"compare {key} {k}: {d['diff']:+.3f} {[round(x, 3) for x in d['diff_ci']]}")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel(f"nats (CLIP {c0:.2f}); left of 0 = less skewed", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.set_yticks(range(len(rows)), [lab for _, lab, *_ in rows], fontsize=10)
    axs[0].invert_yaxis()
    fig.tight_layout(h_pad=1.6)
    fig.text(0.01, -0.03, "Blue bands: 20 random sets of matching size (frequency-matched latents; for *, latents\n"
             "matched on mean contribution; for heads, random heads), min to max. *Exploratory.",
             fontsize=9, color=SOFT)  # fmt: skip
    save(fig, "compare.png")


def hero():
    z = np.load(RES / "prompt_metrics_core.npz")
    cells = z["clip::cells"]  # [858, 42]
    P = json.loads((HERE / "data" / "prompts.json").read_text())
    sets = np.array([s for s, ps in P["sets"].items() for _ in ps])
    occ = cells[sets == "occupations"]
    order = np.argsort(occ[:, :21].mean(1) - occ[:, 21:].mean(1))  # prompts sorted by which gender they lean to
    M = occ[order].T  # [42, 287]
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    cmap = LinearSegmentedColormap.from_list("h", ["#2a3f6e", "#15110e", "#c9561c", "#f2b48e"])
    ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -1.6, 1.6), aspect="auto", interpolation="bilinear")
    ax.axhline(20.5, color=CONTROL, lw=0.8, alpha=0.8)
    ax.axis("off")
    FIG.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):  # full bleed, no white margin
        fig.savefig(FIG / "hero.jpg", dpi=200, pil_kwargs={"quality": 88})
        fig.savefig(FIG / "og.jpg", dpi=150, pil_kwargs={"quality": 85})
    plt.close(fig)
    print("hero: occupations", M.shape, "range", round(float(M.min()), 2), round(float(M.max()), 2))


def explorer_data(core):
    """report/cells.js for the interactive cell map: mean signed skew per joint cell, per arm and prompt set."""
    z = np.load(RES / "prompt_metrics_core.npz")
    P = json.loads((HERE / "data" / "prompts.json").read_text())
    sets = np.array([s for s, ps in P["sets"].items() for _ in ps])
    arms = {"clip": "CLIP (no edit)", "recon@0.6": "SAE reconstruction only (alpha 0.6)",
            "dbl_all@0.6": "DeBiasLens, all social neurons (alpha 0.6)", "dbl_all@1.0": "DeBiasLens, all social neurons (alpha 1.0)",
            "dbl_gender@0.6": "DeBiasLens, gender neurons (alpha 0.6)", "leace_all": "LEACE, gender + age + race"}  # fmt: skip
    out = {"arms": {}, "labels": arms, "sets": list(P["sets"])}
    for k in arms:
        c = z[f"{k}::cells"]
        out["arms"][k] = {"all": np.round(c.mean(0), 3).tolist(),
                          **{s: np.round(c[sets == s].mean(0), 3).tolist() for s in P["sets"]},
                          "abs": {"all": round(float(np.abs(c).mean()), 3),
                                  **{s: round(float(np.abs(c[sets == s]).mean()), 3) for s in P["sets"]}}}  # fmt: skip
    cells = load("cells") if (RES / "cells.json").exists() else None
    if cells:
        out["cell_names"] = cells["cell_names"]
        out["share"] = [round(x, 4) for x in cells["cell_share_val"]]
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "cells.js").write_text("window.CELLS = " + json.dumps(out) + ";\n")
    print("explorer: arms", list(arms), "clip mean |skew|", out["arms"]["clip"]["abs"]["all"])


def main():
    core = load("core")
    explorer_data(core)
    audit(core)
    alpha(core)
    if (RES / "cells.json").exists():
        cells(load("cells"))
    if (RES / "a4.json").exists():
        levels(load("a4"))
    b2 = load("b2") if (RES / "b2.json").exists() else {}
    if b2.get("eta2"):
        heads(b2)
    leakage(core, b2)
    if (RES / "errkeep.json").exists() and b2.get("arms"):
        compare(core, load("errkeep"), b2, load("b3"))
    if (RES / "llava_full_summary.json").exists():
        llava(load("llava_full_summary"))
    hero()


if __name__ == "__main__":
    main()
