"""Which stretches of DNA does AlphaGenome 2 think drive MS4A6A in microglia, and do the GWAS variants sit in them?

An enhancer is a short stretch of DNA, often tens of kilobases from its gene, that switches the gene up in some
cell types and not others. AlphaGenome 2 finds them from the model itself: take the gradient of MS4A6A's predicted
expression with respect to every base of the 1 Mb input. Bases whose change would move the prediction most light
up. Smooth that, keep tiles that also look open in predicted single-cell ATAC, and the peaks are enhancer calls.
This is the preprint's enhancer-gene method (Fig. 4a), run here with the SDK's `enhancer_scoring` module.

It is done twice: in middle temporal gyrus microglia and, for contrast, in an MTG L2/3 neuron, a cell that barely
expresses MS4A6A. Then two questions. Do GWAS variants fall inside a microglial enhancer more often than their
controls? And is the gradient at GWAS variants larger than at controls?
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

from ag2 import (
    CONDITIONS,
    DERIVED,
    GENE,
    GENE_ID,
    PRIMARY,
    SEED,
    TSS,
    check,
    expect,
    figures_dir,
    header,
    optional_env,
    outputs_dir,
    rel,
    require,
    sha8,
)

header(seed=SEED, packages=("alphagenome", "numpy", "pandas", "scipy"))

from alphagenome.data import genome
from alphagenome.interpretation import enhancer_scoring
from alphagenome.models.v2 import dna_client, dna_output

CONTRAST = "MTG L2/3 IT neuron"
CACHE = outputs_dir() / "ag_cache"
CACHE.mkdir(exist_ok=True)

# The caller needs the MANE Select TSS of the target gene from a GENCODE table. The full table is 333 MB; one
# row is enough. ENST00000528851 (MS4A6A-201) is the MANE Select transcript; GENCODE starts are 0-based.
GTF = pd.DataFrame(
    {
        "Chromosome": ["chr11"],
        "Feature": ["transcript"],
        "Start": [60_172_496],
        "End": [TSS],
        "Strand": ["-"],
        "gene_id": [f"{GENE_ID}.13"],
        "gene_name": [GENE],
        "transcript_id": ["ENST00000528851.6"],
        "tag": ["basic,Ensembl_canonical,MANE_Select"],
    }
)

variants = pd.read_csv(require(DERIVED / "variants.csv", "1_variants.py"))
print(f"{len(variants)} variants (sha256 {sha8(DERIVED / 'variants.csv')})")


def cached_pickle(name: str, fetch):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = fetch()
    path.write_bytes(pickle.dumps(value))
    return value


def model():
    key = optional_env("ALPHAGENOME_API_KEY")
    if not key:
        raise SystemExit("cache miss and ALPHAGENOME_API_KEY is not set")
    return dna_client.create(key)


gradients, atac = {}, {}
for name in (PRIMARY, CONTRAST):
    condition = CONDITIONS[name]
    safe = "".join(ch if ch.isalnum() else "_" for ch in condition)
    gradients[name] = cached_pickle(
        f"ag2__gradients__{safe}__{GENE}",
        lambda c=condition: model().predict_interval_gradients(
            GENE_ID, requested_output=dna_output.OutputType.SC_GENE_EXPRESSION, condition=c
        ),
    )
    atac[name] = cached_pickle(
        f"ag2__scatac__{safe}__{GENE}",
        lambda c=condition, g=gradients[name]: model().predict_interval(
            g.interval, condition=c, requested_outputs=[dna_output.OutputType.SC_ATAC]
        ),
    )
    print(f"  {name:<20} gradient window {gradients[name].interval}, objective {gradients[name].scalar_score:.3f}")

calls = enhancer_scoring.get_active_enhancers(
    [gradients[n] for n in (PRIMARY, CONTRAST)], gtf=GTF, accessibility_outputs=[atac[n] for n in (PRIMARY, CONTRAST)]
)
biosample = {gradients[n].biosample_name: n for n in (PRIMARY, CONTRAST)}
calls["condition"] = calls.biosample_name.map(biosample)
calls = calls.sort_values(["condition", "gradient_score"], ascending=[True, False]).reset_index(drop=True)
print(f"\nEnhancer calls for {GENE} (gradient and scATAC thresholds from the SDK):")
for name in (PRIMARY, CONTRAST):
    sub = calls[calls.condition == name]
    strongest = sub.gradient_score.max() if len(sub) else 0.0
    print(f"  {name:<20} {len(sub):>3} enhancers; strongest gradient score {strongest:.4f}")
print(f"\n  The {PRIMARY} enhancers, strongest first (offset is from the MS4A6A TSS at chr11:{TSS:,}):")
for row in calls[calls.condition == PRIMARY].head(12).itertuples():
    inside = variants[(variants.pos >= row.start) & (variants.pos < row.end)]
    gw, ct = int((inside.set == "gwas").sum()), int((inside.set == "control").sum())
    print(f"    chr11:{row.start:,}-{row.end:,}  {(row.start + row.end) / 2 - TSS:+9,.0f} bp  "
          f"gradient {row.gradient_score:.4f}  scATAC {row.atac_score:.2f}  GWAS {gw}  controls {ct}")  # fmt: skip

out = DERIVED / "enhancers.csv"
calls[["condition", "chromosome", "start", "end", "gradient_score", "atac_score", "tss_distance"]].to_csv(
    out, index=False, float_format="%.6g"
)
print(f"\nsaved {rel(out)}")

# --- Do GWAS variants sit in microglial enhancers? --------------------------------------------------------------

print("\nGWAS variants versus controls:")
is_gwas = (variants.set == "gwas").to_numpy()
inside_by = {}
for name in (PRIMARY, CONTRAST):
    sub = calls[calls.condition == name]
    inside = np.zeros(len(variants), dtype=bool)
    for row in sub.itertuples():
        inside |= (variants.pos >= row.start).to_numpy() & (variants.pos < row.end).to_numpy()
    inside_by[name] = inside
    # A continuous version that needs no threshold: the smoothed gradient at the variant itself.
    point = [genome.Interval("chr11", int(p), int(p) + 1) for p in variants.pos]
    grad = enhancer_scoring.score_enhancers(gradients[name], point).gradient_score.to_numpy()
    variants[f"gradient_{name}"] = grad
    table = [[int(inside[is_gwas].sum()), int((~inside[is_gwas]).sum())],
             [int(inside[~is_gwas].sum()), int((~inside[~is_gwas]).sum())]]  # fmt: skip
    _odds, p_fisher = stats.fisher_exact(table, alternative="greater")
    p_mw = stats.mannwhitneyu(grad[is_gwas], grad[~is_gwas], alternative="greater").pvalue
    ratio = np.median(grad[is_gwas]) / np.median(grad[~is_gwas])
    print(f"  {name}:")
    print(f"    inside a called enhancer: GWAS {table[0][0]}/{sum(table[0])}, controls {table[1][0]}/{sum(table[1])}; "
          f"Fisher one-sided p {p_fisher:.3g}")  # fmt: skip
    print(f"    median smoothed gradient at the variant, GWAS over controls: {ratio:.2f}x; "
          f"Mann-Whitney one-sided p {p_mw:.3g}")  # fmt: skip
print("  Both p-values are variant-level and so optimistic: the 98 GWAS variants are 4 LD clumps.")

# The gradient falls off with distance from the gene, and the GWAS variants sit closer to the TSS than any unlinked
# control can (1_variants.py). So the honest comparison holds distance fixed: regress log gradient on GWAS
# membership plus log distance, and report the GWAS coefficient as a fold change at equal distance.
print("\n  At equal distance to the TSS (log10 gradient ~ GWAS + log10 distance, ordinary least squares):")
log_dist = np.log10(np.abs(variants.pos - TSS) + 1000).to_numpy()
for name in (PRIMARY, CONTRAST):
    y = np.log10(variants[f"gradient_{name}"].to_numpy() + 1e-7)
    design = np.column_stack([np.ones(len(y)), is_gwas.astype(float), log_dist])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    cov = np.linalg.inv(design.T @ design) * (resid @ resid) / (len(y) - design.shape[1])
    se = float(np.sqrt(cov[1, 1]))
    print(f"    {name:<20} GWAS fold change {10 ** coef[1]:.2f}x [{10 ** (coef[1] - 1.96 * se):.2f}, "
          f"{10 ** (coef[1] + 1.96 * se):.2f}] (variant-level interval)")  # fmt: skip
    variants[f"gwas_coef_{name}"] = coef[1]
adjusted = {n: 10 ** variants[f"gwas_coef_{n}"].iloc[0] for n in (PRIMARY, CONTRAST)}

print("\n  The named leads:")
for rsid in ("rs1582763", "rs7232", "rs983392", "rs6591561", "rs610932", "rs4938933"):
    hit = variants.index[variants.rsid == rsid]
    if len(hit):
        i = hit[0]
        print(f"    {rsid:<10} {variants.pos[i] - TSS:+9,} bp from the TSS; in a microglial enhancer: "
              f"{'yes' if inside_by[PRIMARY][i] else 'no'}; "
              f"gradient {variants.loc[i, f'gradient_{PRIMARY}']:.5f}")  # fmt: skip

# --- Figure -----------------------------------------------------------------------------------------------------

plt.rcParams.update({"font.size": 9})
lo, hi = 59_950_000, 60_450_000
fig, axes = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.2]})
for axis, name, colour in ((axes[0], PRIMARY, "#2563eb"), (axes[1], CONTRAST, "#64748b")):
    g = gradients[name]
    per_bp = np.abs(np.asarray(g.gradients).sum(axis=-1))
    positions = np.arange(g.interval.start, g.interval.end)
    sel = (positions >= lo) & (positions < hi)
    width = 1000
    n = (int(sel.sum()) // width) * width
    binned = per_bp[sel][:n].reshape(-1, width).sum(axis=1)
    axis.fill_between(positions[sel][:n:width], binned, color=colour, lw=0)
    for row in calls[calls.condition == name].itertuples():
        axis.axvspan(row.start, row.end, color="#f97316", alpha=0.45, lw=0)
    axis.set_ylabel("|gradient x input|\nper kb")
    axis.set_title(f"{name}: what drives predicted {GENE} (orange: called enhancers)", fontsize=9, loc="left")
for s, colour, y in (("gwas", "#dc2626", 0.65), ("control", "#475569", 0.25)):
    sub = variants[variants.set == s]
    axes[2].scatter(sub.pos, np.full(len(sub), y), marker="|", s=80, color=colour)
axes[2].axvline(TSS, color="black", lw=0.8)
axes[2].text(TSS, 0.98, f" {GENE} TSS", fontsize=8, va="top")
axes[2].set_yticks([0.25, 0.65], ["controls", "GWAS"])
axes[2].set_ylim(0, 1)
axes[2].set_xlim(lo, hi)
axes[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.2f}"))
axes[2].set_xlabel("chr11 position, Mb (GRCh38)")
fig.tight_layout()
figure = figures_dir() / "enhancers.png"
fig.savefig(figure, dpi=140)
print(f"\nsaved {rel(figure)}")

print("\nSelf-checks:")
check("the gradient windows contain every variant", all(
    gradients[n].interval.start <= variants.pos.min() and variants.pos.max() < gradients[n].interval.end
    for n in (PRIMARY, CONTRAST)))  # fmt: skip
check(
    "both conditions produced a finite gradient objective", all(np.isfinite(g.scalar_score) for g in gradients.values())
)
check(f"the gradient window is centred near the {GENE} TSS", abs(gradients[PRIMARY].interval.center() - TSS) < 50_000)
expect(
    f"called {GENE} enhancers, {PRIMARY} vs {CONTRAST}",
    f"{int((calls.condition == PRIMARY).sum())} vs {int((calls.condition == CONTRAST).sum())}",
    "more in microglia, which express the gene",
    "plan item 6",
)
expect(
    "distance-adjusted GWAS fold change in gradient, microglia vs neuron",
    f"{adjusted[PRIMARY]:.2f}x vs {adjusted[CONTRAST]:.2f}x",
    "larger in microglia if the GWAS signal sits in microglia-specific regulatory DNA",
    "plan item 6",
)
