"""The pre-registered tests: is the GWAS effect on MS4A6A bigger than the controls', is it microglial, which way?

Three questions, each answered in every condition so the answers can be compared:

  S1  Magnitude. Do GWAS variants move predicted MS4A6A more than their matched controls? The ratio of median
      |log fold change|, GWAS over controls. Above 1 means the GWAS signal does something controls do not.
  S2  Direction. Does the risk allele raise or lower MS4A6A? Each GWAS variant's fold change is flipped so it
      describes the allele that raises Alzheimer's risk; the mean of those is S2. Positive: risk raises MS4A6A,
      so knocking MS4A6A down points the protective way, consistent with the hypothesis. Negative: against.
  S3  Specificity. The same numbers in astrocytes, neurons, oligodendrocytes and hepatocytes.

The 98 GWAS variants are not 98 independent pieces of evidence. They are 4 groups of variants inherited together
(LD clumps), one of them 91 strong. So every interval resamples whole clumps; variant-level p-values are printed
too and labelled optimistic. The decision rule was fixed in `docs/plans/2026-11-11-alphagenome2-ms4a-plan.md`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ag2 import (
    CALL_LFC,
    CONDITIONS,
    DERIVED,
    GENE,
    MICROGLIA,
    NON_MICROGLIA,
    PRIMARY,
    SECOND_GENE,
    SEED,
    ZERO_SHOT,
    check,
    expect,
    figures_dir,
    header,
    rel,
    require,
    sha8,
)

header(seed=SEED, packages=("numpy", "pandas", "scipy"))
RESAMPLES = 10_000
AG1 = ("DLPFC tissue (AG1)", "CD14+ monocyte (AG1)")
ORDER = [*CONDITIONS, ZERO_SHOT, *AG1]
LEAD_TABLE = (PRIMARY, "DLPFC microglia", ZERO_SHOT, "DLPFC tissue (AG1)")

variants = pd.read_csv(require(DERIVED / "variants.csv", "1_variants.py"))
scores = pd.read_csv(require(DERIVED / "ag_scores.csv", "2_score.py"))
print(f"variants.csv sha256 {sha8(DERIVED / 'variants.csv')}, ag_scores.csv sha256 {sha8(DERIVED / 'ag_scores.csv')}")
gw_rows = variants[variants.set == "gwas"]
ct_rows = variants[variants.set == "control"]
print(f"{len(gw_rows)} GWAS variants in {gw_rows.clump.nunique()} LD clumps; "
      f"{len(ct_rows)} controls in {ct_rows.clump.nunique()} clumps")  # fmt: skip

# --- One number per (variant, condition, gene) -------------------------------------------------------------------


def per_variant(gene: str, scorer: str) -> pd.DataFrame:
    """Mean over the tracks that define each condition: its observed tracks for AG2 reference conditions, the
    denoised channel (averaged over 3 cell draws) for zero-shot, the strand-matched tracks for AG1."""
    s = scores[(scores.gene_name == gene) & (scores.variant_scorer == scorer)]
    reference = s[(s.model == "AG2") & (s.condition != ZERO_SHOT) & (s.imputation_mode == "observed")]
    rest = s[(s.model == "AG1") | (s.condition == ZERO_SHOT)]
    table = pd.concat([reference, rest]).groupby(["key", "condition"]).raw_score.mean().unstack()
    return table.reindex(columns=[c for c in ORDER if c in table.columns])


lfc = per_variant(GENE, "GeneMaskLFCScorer")
lfc2 = per_variant(SECOND_GENE, "GeneMaskLFCScorer")
level = per_variant(GENE, "GeneMaskActiveScorer")
v = variants.set_index("key").loc[lfc.index]
gwas = (v.set == "gwas").to_numpy()
clump = v.clump.to_numpy()
risk = v.risk_sign.to_numpy()

# --- Statistics -------------------------------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
gw_clumps = np.unique(clump[gwas])
ct_clumps = np.unique(clump[~gwas])
members = {c: np.flatnonzero(clump == c) for c in np.unique(clump)}
draws = [
    (
        np.concatenate([members[c] for c in rng.choice(gw_clumps, len(gw_clumps))]),
        np.concatenate([members[c] for c in rng.choice(ct_clumps, len(ct_clumps))]),
    )
    for _ in range(RESAMPLES)
]


def magnitude(x: np.ndarray) -> dict:
    a = np.abs(x)
    boot = np.array([np.median(a[g]) / np.median(a[c]) for g, c in draws])
    return {
        "ratio": np.median(a[gwas]) / np.median(a[~gwas]),
        "ratio_lo": np.percentile(boot, 2.5),
        "ratio_hi": np.percentile(boot, 97.5),
        "p_mw": stats.mannwhitneyu(a[gwas], a[~gwas], alternative="greater").pvalue,
        "median_pct": np.median([(a[~gwas] < value).mean() for value in a[gwas]]),
        "gwas_median_abs": np.median(a[gwas]),
        "control_median_abs": np.median(a[~gwas]),
        "max_abs_gwas": a[gwas].max(),
        "max_abs_control": a[~gwas].max(),
        "calls_gwas": int((a[gwas] > CALL_LFC).sum()),
        "calls_control": int((a[~gwas] > CALL_LFC).sum()),
    }


def direction(x: np.ndarray) -> dict:
    has = gwas & (risk != 0)
    r = risk * x
    boot = [r[g[has[g]]].mean() for g, _ in draws if has[g].any()]
    pip = v.pip.to_numpy()
    w = has & np.isfinite(pip)
    return {
        "risk_mean": r[has].mean(),
        "risk_lo": np.percentile(boot, 2.5),
        "risk_hi": np.percentile(boot, 97.5),
        "risk_share_up": (r[has] > 0).mean(),
        "risk_pip_mean": np.average(r[w], weights=pip[w]),
        "n_direction": int(has.sum()),
    }


def distance_adjusted(x: np.ndarray) -> dict:
    a = np.abs(x)
    y = np.log10(a + 1e-5)
    log_dist = np.log10(np.abs(v.dist_tss.to_numpy()) + 1000)
    design = np.column_stack([np.ones(len(y)), gwas.astype(float), log_dist, v.maf_eur.to_numpy()])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    se = np.sqrt(np.linalg.inv(design.T @ design)[1, 1] * (resid @ resid) / (len(y) - design.shape[1]))
    near = np.abs(v.dist_tss.to_numpy()) <= 30_000
    return {
        "adj_fold": 10 ** coef[1],
        "adj_lo": 10 ** (coef[1] - 1.96 * se),
        "adj_hi": 10 ** (coef[1] + 1.96 * se),
        "near_n_gwas": int((near & gwas).sum()),
        "near_n_control": int((near & ~gwas).sum()),
        "near_ratio": np.median(a[near & gwas]) / np.median(a[near & ~gwas]),
        "near_p": stats.mannwhitneyu(a[near & gwas], a[near & ~gwas], alternative="greater").pvalue,
    }


summary = pd.DataFrame(
    [
        {
            "condition": c,
            **magnitude(lfc[c].to_numpy()),
            **direction(lfc[c].to_numpy()),
            **distance_adjusted(lfc[c].to_numpy()),
        }
        for c in lfc.columns
    ]
).set_index("condition")
expression = level.median()

print(f"\nPredicted {GENE} expression per condition (the active-allele score, median over variants; higher is more):")
for condition, value in expression.items():
    print(f"  {condition:<40} {value:>7.2f}")

print(f"\nS1, magnitude: median |LFC| on {GENE}, GWAS over controls (clump-bootstrap 95% interval)")
print(
    f"  {'condition':<40}{'ratio':>7}{'interval':>16}{'MW p*':>9}{'GWAS pct':>10}{'calls G/C':>11}{'max |LFC| G/C':>18}"
)
for condition, s in summary.iterrows():
    print(f"  {condition:<40}{s.ratio:>7.2f}{f'[{s.ratio_lo:.2f}, {s.ratio_hi:.2f}]':>16}{s.p_mw:>9.2g}"
          f"{s.median_pct:>10.2f}{f'{s.calls_gwas:.0f}/{s.calls_control:.0f}':>11}"
          f"{f'{s.max_abs_gwas:.4f}/{s.max_abs_control:.4f}':>18}")  # fmt: skip
print("  * variant-level, optimistic. 'GWAS pct': the median GWAS variant's percentile among the controls.")
print(f"  'calls': variants past the preprint's call threshold, |LFC| > {CALL_LFC}.")
print(
    f"  {PRIMARY}: median |LFC| {summary.loc[PRIMARY, 'gwas_median_abs']:.4f} for GWAS variants, "
    f"{summary.loc[PRIMARY, 'control_median_abs']:.4f} for controls"
)

print(
    "\nS1b, at equal distance to the TSS: fold change in |LFC| for GWAS membership, adjusted for log distance and MAF"
)
print("  (variant-level interval), and the stratum within 30 kb of the TSS")
for condition, s in summary.iterrows():
    interval = f"[{s.adj_lo:.2f}, {s.adj_hi:.2f}]"
    print(f"  {condition:<40}{s.adj_fold:>6.2f}x {interval:>13}   near: {s.near_ratio:.2f}x, "
          f"p {s.near_p:.2g} ({s.near_n_gwas:.0f} GWAS, {s.near_n_control:.0f} controls)")  # fmt: skip

print(f"\nS2, direction: mean risk-allele effect on {GENE} (positive: the risk allele raises {GENE})")
print(f"  {'condition':<40}{'mean':>10}{'interval':>24}{'share up':>10}{'PIP-weighted':>14}")
for condition, s in summary.iterrows():
    print(f"  {condition:<40}{s.risk_mean:>+10.5f}{f'[{s.risk_lo:+.5f}, {s.risk_hi:+.5f}]':>24}"
          f"{s.risk_share_up:>10.2f}{s.risk_pip_mean:>+14.5f}")  # fmt: skip
print(f"  {int(summary.n_direction.iloc[0])} GWAS variants carry a risk direction.")

print(f"\nThe named leads: the risk allele's effect on {GENE} (LFC; positive = risk raises {GENE})")
lead_mask = v.lead.astype(bool) | v.rsid.isin(["rs7232", "rs2081545", "rs627081", "rs4354705"])
lead_keys = list(v.index[lead_mask])
print(f"  {'variant':<11}{'risk':>5}{'TSS offset':>12}" + "".join(f"{c[:15]:>17}" for c in LEAD_TABLE))
for key in lead_keys:
    row = v.loc[key]
    sign = int(row.risk_sign)
    allele = {1: row.alt, -1: row.ref}.get(sign, "?")
    cells = "".join(f"{sign * lfc.loc[key, c]:>+17.5f}" if sign else f"{'(no allele)':>17}" for c in LEAD_TABLE)
    print(f"  {row.rsid:<11}{allele:>5}{int(row.dist_tss):>+12,}{cells}")

print("\nS3, specificity: |LFC| of each GWAS variant in MTG microglia against the same variant elsewhere")
a_primary = np.abs(lfc[PRIMARY].to_numpy())
for other in (*NON_MICROGLIA, "blood CD14+ monocyte"):
    a_other = np.abs(lfc[other].to_numpy())
    test = stats.wilcoxon(a_primary[gwas], a_other[gwas], alternative="greater")
    ratio = np.median(a_primary[gwas]) / np.median(a_other[gwas])
    print(f"  microglia vs {other:<24} median ratio {ratio:>6.2f}; signed-rank one-sided p {test.pvalue:.2g}")

print("\nWhat AlphaGenome 2 adds: agreement between models and modes (Spearman over all variants)")
pairs = [
    (PRIMARY, "DLPFC microglia", "two microglia references"),
    (PRIMARY, ZERO_SHOT, "reference microglia vs zero-shot iPSC microglia"),
    (PRIMARY, "blood CD14+ monocyte", "AG2 microglia vs AG2 monocytes"),
    (PRIMARY, "DLPFC tissue (AG1)", "AG2 microglia vs AG1 bulk DLPFC"),
    (PRIMARY, "CD14+ monocyte (AG1)", "AG2 microglia vs AG1 monocytes"),
    ("DLPFC tissue (AG1)", "CD14+ monocyte (AG1)", "AG1 DLPFC vs AG1 monocytes"),
]
rhos = {}
for a, b, label in pairs:
    rhos[label] = stats.spearmanr(lfc[a], lfc[b]).statistic
    print(f"  {label:<52} rho {rhos[label]:+.2f}")
zero = scores[(scores.condition == ZERO_SHOT) & (scores.gene_name == GENE)].pivot_table(
    index="key", columns="draw", values="raw_score"
)
draw_rho = [stats.spearmanr(zero[i], zero[j]).statistic for i, j in ((0, 1), (0, 2), (1, 2))]
print(f"  zero-shot, draw against draw (48 different cells each)     rho {min(draw_rho):+.2f} to {max(draw_rho):+.2f}")

print(f"\nSecondary: {SECOND_GENE} in {PRIMARY}")
m2, d2 = magnitude(lfc2[PRIMARY].to_numpy()), direction(lfc2[PRIMARY].to_numpy())
print(f"  S1 ratio {m2['ratio']:.2f} [{m2['ratio_lo']:.2f}, {m2['ratio_hi']:.2f}]; S2 {d2['risk_mean']:+.5f} "
      f"[{d2['risk_lo']:+.5f}, {d2['risk_hi']:+.5f}]; calls {m2['calls_gwas']}/{m2['calls_control']}")  # fmt: skip

# --- The decision -----------------------------------------------------------------------------------------------

p, rep = summary.loc[PRIMARY], summary.loc["DLPFC microglia"]
criteria = {
    f"(i) a GWAS variant passes |LFC| > {CALL_LFC}": bool(p.calls_gwas > 0),
    "(ii) the S1 interval lies above 1": bool(p.ratio_lo > 1),
    "(iii) the S2 interval lies above 0": bool(p.risk_lo > 0),
    "(iv) DLPFC microglia gives the same S2 sign": bool(np.sign(rep.risk_mean) == np.sign(p.risk_mean)),
}
gates = [criteria[k] for k in criteria if not k.startswith("(iii)")]
if all(criteria.values()):
    verdict, level_text = "consistent with the hypothesis", "screen-null; genetic direction consistent (in silico)"
elif all(gates) and p.risk_hi < 0:
    verdict, level_text = "against the hypothesis", "screen-null; genetic direction opposed (in silico)"
else:
    verdict, level_text = "inconclusive", "screen-null (unchanged)"
print("\nThe pre-registered decision, in MTG microglia:")
for name, ok in criteria.items():
    print(f"  {'met' if ok else 'not met':<8} {name}")
print(f"  {'met' if p.risk_hi < 0 else 'not met':<8} the S2 interval lies below 0 (the 'against' branch)")
print(f"  verdict: {verdict}")
print(f"  evidence level: {level_text}")

# --- Tables and figures -----------------------------------------------------------------------------------------

summary.to_csv(DERIVED / "summary.csv", float_format="%.6g")
out = lfc.join(lfc2.add_suffix(f" [{SECOND_GENE}]")).join(v[["rsid", "set", "risk_sign", "pip", "dist_tss", "clump"]])
out.to_csv(DERIVED / "per_variant_lfc.csv", float_format="%.6g")
print(f"\nsaved {rel(DERIVED / 'summary.csv')}, {rel(DERIVED / 'per_variant_lfc.csv')}")

plt.rcParams.update({"font.size": 9})
labels = list(summary.index)
y = np.arange(len(labels))[::-1]
colour = [
    "#2563eb" if c in MICROGLIA or c == ZERO_SHOT else "#9333ea" if "monocyte" in c else "#64748b" for c in labels
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
ax1.errorbar(summary.ratio, y, xerr=[summary.ratio - summary.ratio_lo, summary.ratio_hi - summary.ratio],
             fmt="none", ecolor="#94a3b8")  # fmt: skip
ax1.scatter(summary.ratio, y, c=colour, zorder=3)
ax1.axvline(1, color="black", lw=0.8)
ax1.set_xscale("log")
ax1.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
ax1.minorticks_off()
ax1.set_yticks(y, labels)
ax1.set_xlabel(f"S1: median |LFC| on {GENE}, GWAS / controls (log scale)")
ax2.errorbar(summary.risk_mean, y, xerr=[summary.risk_mean - summary.risk_lo, summary.risk_hi - summary.risk_mean],
             fmt="none", ecolor="#94a3b8")  # fmt: skip
ax2.scatter(summary.risk_mean, y, c=colour, zorder=3)
ax2.axvline(0, color="black", lw=0.8)
ax2.set_xlabel(f"S2: mean risk-allele effect on {GENE} (LFC)\n< 0: risk lowers {GENE}      > 0: risk raises {GENE}")
fig.suptitle(
    "Bars: 95% intervals resampling LD clumps. Blue: microglia; purple: monocytes; grey: other cells", fontsize=9
)
fig.tight_layout()
fig1 = figures_dir() / "conditions.png"
fig.savefig(fig1, dpi=140)

fig, ax = plt.subplots(figsize=(8, 4.2))
a = np.abs(lfc[PRIMARY]) + 1e-5
ax.scatter(v.dist_tss[~gwas] / 1000, a[~gwas], s=12, color="#94a3b8", label=f"controls ({(~gwas).sum()})")
ax.scatter(
    v.dist_tss[gwas] / 1000, a[gwas], s=14, color="#dc2626", label=f"GWAS credible-set and lead SNVs ({gwas.sum()})"
)
for key in lead_keys:
    ax.annotate(
        v.loc[key].rsid, (v.loc[key].dist_tss / 1000, a[key]), fontsize=7, xytext=(3, 3), textcoords="offset points"
    )
ax.axhline(CALL_LFC, color="black", ls="--", lw=0.8)
ax.annotate("preprint call threshold, |LFC| = 0.1", (0.01, CALL_LFC), xycoords=("axes fraction", "data"), fontsize=7,
            xytext=(0, 3), textcoords="offset points")  # fmt: skip
ax.set_yscale("log")
ax.set_xlabel(f"distance to the {GENE} TSS (kb)")
ax.set_ylabel(f"|LFC| on {GENE}, {PRIMARY}")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig2 = figures_dir() / "effects.png"
fig.savefig(fig2, dpi=140)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for axis, other, title in (
    (axes[0], "DLPFC tissue (AG1)", "AlphaGenome 1, bulk DLPFC"),
    (axes[1], ZERO_SHOT, "AG2 zero-shot, 48 iPSC microglia"),
):
    axis.scatter(lfc[other][~gwas], lfc[PRIMARY][~gwas], s=10, color="#94a3b8", label="controls")
    axis.scatter(lfc[other][gwas], lfc[PRIMARY][gwas], s=12, color="#dc2626", label="GWAS")
    axis.axhline(0, color="black", lw=0.5)
    axis.axvline(0, color="black", lw=0.5)
    axis.set_xlabel(f"LFC on {GENE}: {title}")
    axis.set_ylabel(f"LFC on {GENE}: AG2, {PRIMARY}")
    axis.set_title(f"Spearman rho {stats.spearmanr(lfc[other], lfc[PRIMARY]).statistic:+.2f}", fontsize=9)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig3 = figures_dir() / "models.png"
fig.savefig(fig3, dpi=140)
print(f"saved {rel(fig1)}, {rel(fig2)}, {rel(fig3)}")

observed_tracks = (
    scores[
        (scores.condition == PRIMARY)
        & (scores.imputation_mode == "observed")
        & (scores.gene_name == GENE)
        & (scores.variant_scorer == "GeneMaskLFCScorer")
    ]
    .groupby("key")
    .size()
)
flipped = gwas & (risk != 0)
primary_lfc = lfc[PRIMARY].to_numpy()
print("\nSelf-checks:")
check("every variant has an MS4A6A score in every condition", not lfc.isna().any().any())
check("the MTG microglia readout is exactly one observed track per variant", observed_tracks.eq(1).all())
check("the risk-allele flip leaves magnitudes unchanged",
      np.allclose(np.abs(risk[flipped] * primary_lfc[flipped]), np.abs(primary_lfc[flipped])))  # fmt: skip
check("the bootstrap resamples whole clumps", all(set(clump[g]) <= set(gw_clumps) for g, _ in draws[:50]))
expect(
    f"{GENE} predicted expression, microglia vs neuron",
    f"{expression[PRIMARY]:.2f} vs {expression['MTG L2/3 IT neuron']:.2f}",
    "microglia far above neuron",
    "SEA-AD marker biology",
)
expect("the verdict under the pre-registered rule", verdict, "one of three, fixed in advance", "the plan")
