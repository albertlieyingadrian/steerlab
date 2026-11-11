"""A known-answer check, added after the pre-registration: does AlphaGenome 2 get the sign of a *measured* eQTL right?

A prediction of "this allele lowers MS4A6A" is only worth reading if the model's signs agree with measurements where
measurements exist. GTEx (v8) measured how common variants change gene expression in 49 human tissues from about 800
donors; for MS4A6A, the MS4A-cluster Alzheimer's variants are significant eQTLs in whole blood. GTEx has no
microglia, and blood is not brain, so this checks the model's sign, not the microglial answer.

The check: for every GWAS variant GTEx lists as a significant MS4A6A eQTL, compare the sign of GTEx's normalised
effect size (for the ALT allele) with the sign of AlphaGenome's predicted log fold change (ALT over REF). This was
not in the pre-registration and does not enter the decision rule; it is reported beside it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from ag2 import (
    DERIVED,
    GENE,
    PRIMARY,
    SEED,
    ZERO_SHOT,
    cached,
    check,
    expect,
    header,
    http,
    outputs_dir,
    rel,
    require,
    sha8,
)

header(seed=SEED, packages=("numpy", "pandas", "scipy"))

variants = pd.read_csv(require(DERIVED / "variants.csv", "1_variants.py"))
lfc = pd.read_csv(require(DERIVED / "per_variant_lfc.csv", "3_analysis.py"))
lfc = lfc.set_index("key")
gwas = variants[variants.set == "gwas"]
print(f"{len(gwas)} GWAS variants (variants.csv sha256 {sha8(DERIVED / 'variants.csv')})")


def gtex(row) -> list[dict]:
    variant_id = f"chr11_{row.pos}_{row.ref}_{row.alt}_b38"
    path = outputs_dir() / f"gtex_{variant_id}.json"
    url = (
        "https://gtexportal.org/api/v2/association/singleTissueEqtl"
        f"?variantId={variant_id}&datasetId=gtex_v8&itemsPerPage=250"
    )
    return json.loads(cached(path, lambda: http(url))).get("data", [])


rows = []
for row in gwas.itertuples():
    for hit in gtex(row):
        if hit["geneSymbol"] == GENE:
            rows.append({"key": row.key, "rsid": row.rsid, "tissue": hit["tissueSiteDetailId"], "nes": hit["nes"],
                         "p": hit["pValue"]})  # fmt: skip
eqtl = pd.DataFrame(rows)
print(f"GTEx v8 significant {GENE} eQTLs among them: {eqtl.key.nunique()} variants, "
      f"tissues {', '.join(sorted(eqtl.tissue.unique()))}")  # fmt: skip

blood = eqtl[eqtl.tissue == "Whole_Blood"].set_index("key")
print(f"  whole blood: {len(blood)} variants; NES from {blood.nes.min():+.3f} to {blood.nes.max():+.3f} (ALT allele)")

print(f"\nSign agreement with the measured whole-blood eQTL, over those {len(blood)} variants:")
results = {}
for condition in (
    PRIMARY,
    "DLPFC microglia",
    "blood CD14+ monocyte",
    ZERO_SHOT,
    "DLPFC tissue (AG1)",
    "CD14+ monocyte (AG1)",
):
    predicted = lfc.loc[blood.index, condition]
    agree = int((np.sign(predicted) == np.sign(blood.nes)).sum())
    test = stats.binomtest(agree, len(blood), 0.5)
    rho = stats.spearmanr(predicted, blood.nes).statistic
    results[condition] = (agree, len(blood))
    print(
        f"  {condition:<40} {agree:>3} of {len(blood)} agree (binomial p {test.pvalue:.2g}); "
        f"Spearman with NES {rho:+.2f}"
    )
print("  The variants are in strong LD with each other, so these counts are not independent: read the sign, not the p.")

print("\nThe measured direction, in the hypothesis's terms (risk allele from the GWAS, effect from GTEx whole blood):")
risk = variants.set_index("key").loc[blood.index, "risk_sign"]
with_direction = risk != 0
risk_up = int((np.sign(blood.nes[with_direction]) * risk[with_direction] > 0).sum())
print(f"  the risk allele raises measured {GENE} in blood at {risk_up} of {int(with_direction.sum())} variants with a "
      "risk direction")  # fmt: skip
for rsid in ("rs1582763", "rs7232", "rs983392"):
    hit = blood[blood.rsid == rsid]
    if len(hit):
        key = hit.index[0]
        print(
            f"  {rsid:<10} GTEx NES {hit.nes.iloc[0]:+.3f} (p {hit.p.iloc[0]:.1e}); "
            f"AG2 {PRIMARY} LFC {lfc.loc[key, PRIMARY]:+.5f}"
        )

out = DERIVED / "gtex_check.csv"
blood.assign(**{c: lfc.loc[blood.index, c] for c in (PRIMARY, ZERO_SHOT, "DLPFC tissue (AG1)")}).to_csv(
    out, float_format="%.6g"
)
print(f"\nsaved {rel(out)}")

print("\nSelf-checks:")
check("GTEx returned at least one significant MS4A6A eQTL among the GWAS variants", len(blood) > 0)
check("every eQTL variant has an AlphaGenome prediction", blood.index.isin(lfc.index).all())
expect(
    f"sign agreement, {PRIMARY} against GTEx whole blood",
    f"{results[PRIMARY][0]} of {results[PRIMARY][1]}",
    "most, if the model's signs track measured eQTLs",
    "added after the pre-registration",
)
