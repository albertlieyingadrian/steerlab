"""Which variants to score: the Alzheimer's GWAS signal at the MS4A cluster, and controls that look like it.

A GWAS does not find "the" causal variant. It finds a stretch of chromosome where many common variants travel
together (they are in linkage disequilibrium, LD), and fine-mapping turns that stretch into a *credible set*: the
variants that, between them, very probably contain the causal one, each with a posterior inclusion probability
(PIP). So the GWAS side here is every SNV in the MS4A credible sets of five large Alzheimer's GWAS on the Open
Targets Platform, plus six lead variants named in the papers.

A control has to answer "is this just what any common variant near MS4A6A does?". So each GWAS variant gets two
controls from the same 700 kb window, matched on allele frequency and on distance to the MS4A6A start site, and
required to be *unlinked* to the GWAS set (r^2 < 0.1 with every GWAS variant, in 1000 Genomes Europeans). The
last rule matters: in a locus this dense, a control in LD with the lead is a GWAS variant under another name.

Nothing here calls AlphaGenome. The table this writes, `derived/variants.csv`, holds public annotations only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ag2 import (
    CHROM,
    CLUMP_R2,
    CONTROL_MAX_R2,
    CONTROL_MIN_MAF,
    CONTROLS_PER_GWAS,
    DERIVED,
    LITERATURE_LEADS,
    OT_STUDIES,
    SEED,
    TSS,
    WINDOW,
    check,
    ensembl_variant,
    expect,
    fetch_vcf_slice,
    gwas_catalog,
    header,
    kg_eur_columns,
    kg_header_samples,
    opentargets,
    r2_matrix,
    rel,
    sha8,
)

header(seed=SEED, packages=("numpy", "pandas"))

# --- 1. Credible sets from Open Targets -------------------------------------------------------------------------

STUDY_QUERY = """query($id:String!){ study(studyId:$id){ id publicationFirstAuthor
  credibleSets(page:{index:0,size:200}){ count rows { studyLocusId chromosome position } } } }"""
SET_QUERY = """query($id:String!){ credibleSet(studyLocusId:$id){ studyLocusId studyId finemappingMethod confidence
  variant{ id rsIds } pValueMantissa pValueExponent beta
  locus(page:{index:0,size:1000}){ count rows{ variant{ id rsIds } posteriorProbability beta is95CredibleSet } } } }"""

print("Open Targets credible sets overlapping the MS4A window, per study:")
gwas_rows = []
for study, label in OT_STUDIES.items():
    sets = opentargets(STUDY_QUERY, {"id": study}, f"study_{study}")["study"]["credibleSets"]["rows"]
    here = [s for s in sets if s["chromosome"] == "11" and WINDOW[0] <= s["position"] <= WINDOW[1]]
    for s in here:
        cs = opentargets(SET_QUERY, {"id": s["studyLocusId"]}, f"set_{s['studyLocusId']}")["credibleSet"]
        members = [r for r in cs["locus"]["rows"] if r["is95CredibleSet"]]
        lead = cs["variant"]["rsIds"][-1] if cs["variant"]["rsIds"] else cs["variant"]["id"]
        print(
            f"  {label:<24} lead {lead:<11} p {cs['pValueMantissa']:.1f}e{cs['pValueExponent']}  "
            f"{cs['finemappingMethod']:<9} {len(members):>3} variants in the 95% set"
        )
        for r in members:
            _chrom, pos, ref, alt = r["variant"]["id"].split("_")
            gwas_rows.append(
                {
                    "chrom": CHROM,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "rsid": (r["variant"]["rsIds"] or [""])[-1],
                    "study": label,
                    "pip": float(r["posteriorProbability"]),
                    "beta": float(r["beta"]) if r["beta"] is not None else np.nan,
                }
            )
    if not here:
        print(f"  {label:<24} no credible set in the window")

ot = pd.DataFrame(gwas_rows)
ot["key"] = ot.pos.astype(str) + ":" + ot.ref + ":" + ot.alt
all_locus_keys = set(ot.key)
ot = ot[(ot.ref.str.len() == 1) & (ot.alt.str.len() == 1)].copy()
print(f"\n{len(ot)} SNV rows across studies; indels are dropped because the controls are SNVs too")

# --- 2. Literature leads, checked against the GWAS Catalog and placed by Ensembl --------------------------------

print("\nLiterature leads, with the allele the source says raises Alzheimer's risk:")
lead_rows = []
for rsid, info in LITERATURE_LEADS.items():
    mapping = ensembl_variant(rsid)["mappings"][0]
    ref, *alts = mapping["allele_string"].split("/")
    catalog = gwas_catalog(rsid).get("_embedded", {}).get("associations", [])
    ad = [a for a in catalog if any("alzheimer" in t.lower() for t in a.get("reported_trait") or [])]
    print(f"  {rsid:<10} {CHROM}:{mapping['start']} {mapping['allele_string']:<6} risk {info['risk'] or '?':<2} "
          f"({len(ad):>2} Alzheimer's rows in the Catalog)  {info['source']}")  # fmt: skip
    lead_rows.append({"rsid": rsid, "pos": int(mapping["start"]), "ref": ref, "alts": alts, "risk": info["risk"]})

# --- 3. The 1000 Genomes slice: frequencies and European genotypes ----------------------------------------------

samples = kg_header_samples()
eur = np.array(kg_eur_columns(samples))
rows, raw_path = fetch_vcf_slice(*WINDOW)
print(f"\n1000 Genomes high-coverage panel, {CHROM}:{WINDOW[0]:,}-{WINDOW[1]:,}: {len(rows):,} records")
print(f"  {len(eur)} European samples who are nobody's child in the pedigree; slice sha256 {sha8(raw_path)}")

kg_pos, kg_ref, kg_alt, kg_af, dosages = [], [], [], [], []
for line in rows:
    fields = line.split("\t")
    ref, alt = fields[3], fields[4]
    if len(ref) != 1 or len(alt) != 1 or alt not in "ACGT":
        continue
    genotypes = [fields[9 + i] for i in eur]
    dose = np.fromiter((int(g[0]) + int(g[2]) for g in genotypes), dtype=np.int8, count=len(genotypes))
    af = dose.mean() / 2
    if min(af, 1 - af) < CONTROL_MIN_MAF:
        continue
    kg_pos.append(int(fields[1]))
    kg_ref.append(ref)
    kg_alt.append(alt)
    kg_af.append(af)
    dosages.append(dose)
kg = pd.DataFrame({"pos": kg_pos, "ref": kg_ref, "alt": kg_alt, "af_eur": kg_af})
kg["key"] = kg.pos.astype(str) + ":" + kg.ref + ":" + kg.alt
dosage = np.vstack(dosages).astype(np.float32)
print(f"  {len(kg):,} biallelic SNVs with European MAF >= {CONTROL_MIN_MAF}")

# --- 4. The GWAS set --------------------------------------------------------------------------------------------

best = ot.sort_values("pip", ascending=False).groupby("key", as_index=False).first()
studies = ot.groupby("key").study.apply(lambda s: "; ".join(sorted(set(s))))
signs = ot.dropna(subset=["beta"]).assign(sign=lambda d: np.sign(d.beta)).groupby("key").sign.nunique()
conflicts = int((signs > 1).sum())
gwas = best.set_index("key").assign(studies=studies).reset_index()
gwas["risk_sign"] = np.sign(gwas.beta).fillna(0).astype(int)  # +1: the alt allele raises risk
gwas["lead"] = False

for lead in lead_rows:
    match = [a for a in lead["alts"] if f"{lead['pos']}:{lead['ref']}:{a}" in set(kg.key)]
    if not match:
        print(f"  {lead['rsid']} is not a common biallelic SNV in the panel; dropped")
        continue
    key = f"{lead['pos']}:{lead['ref']}:{match[0]}"
    sign = 0 if lead["risk"] is None else (1 if lead["risk"] == match[0] else -1)
    if key in set(gwas.key):
        index = gwas.index[gwas.key == key][0]
        gwas.loc[index, "lead"] = True
        if sign and gwas.loc[index, "risk_sign"] and sign != gwas.loc[index, "risk_sign"]:
            print(f"  {lead['rsid']}: the paper's risk allele disagrees with the credible-set beta; the paper wins")
        if sign:
            gwas.loc[index, "risk_sign"] = sign
    else:
        gwas.loc[len(gwas)] = {
            "key": key, "pos": lead["pos"], "ref": lead["ref"], "alt": match[0], "rsid": lead["rsid"], "chrom": CHROM,
            "study": "literature", "pip": np.nan, "beta": np.nan, "studies": "literature lead", "risk_sign": sign,
            "lead": True,
        }  # fmt: skip

in_panel = gwas.key.isin(set(kg.key))
print(f"\nGWAS set: {len(gwas)} SNVs, {int(in_panel.sum())} of them common biallelic SNVs in the 1000 Genomes slice")
gwas = gwas[in_panel].merge(kg[["key", "af_eur"]], on="key").reset_index(drop=True)
print(f"  {int((gwas.risk_sign != 0).sum())} carry a risk direction; {conflicts} have opposite beta signs in two "
      "studies (kept with the sign from the highest-PIP study)")  # fmt: skip

# --- 5. LD, clumps and the control pool -------------------------------------------------------------------------

index_of = {k: i for i, k in enumerate(kg.key)}
gwas_dose = dosage[[index_of[k] for k in gwas.key]]
r2_to_gwas = r2_matrix(dosage, gwas_dose)
kg["max_r2_gwas"] = r2_to_gwas.max(axis=1)

pool = kg[(~kg.key.isin(all_locus_keys)) & (~kg.key.isin(set(gwas.key))) & (kg.max_r2_gwas < CONTROL_MAX_R2)].copy()
print(f"\nControl pool: {len(pool):,} SNVs in the window with r^2 < {CONTROL_MAX_R2} to every GWAS variant")

for frame in (gwas, pool):
    frame["maf_eur"] = np.minimum(frame.af_eur, 1 - frame.af_eur)
    frame["dist_tss"] = frame.pos - TSS
    frame["log_dist"] = np.log10(np.abs(frame.dist_tss) + 1000)


def clumps(keys: list[str], order: np.ndarray) -> np.ndarray:
    """Greedy LD clumping: the first unclumped variant in `order` takes every unclumped variant with r^2 >= 0.5."""
    sub = dosage[[index_of[k] for k in keys]]
    r2 = r2_matrix(sub, sub)
    label = np.full(len(keys), -1)
    next_label = 0
    for i in order:
        if label[i] >= 0:
            continue
        label[(label < 0) & (r2[i] >= CLUMP_R2)] = next_label
        next_label += 1
    return label


gwas["clump"] = clumps(list(gwas.key), np.argsort(-gwas.pip.fillna(0).to_numpy(), kind="stable"))
print(f"GWAS set: {gwas.clump.nunique()} LD clumps at r^2 >= {CLUMP_R2}")

# --- 6. Matching ------------------------------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
pool = pool.reset_index(drop=True)
taken = np.zeros(len(pool), dtype=bool)
match_rows = []
# Scale the two matching axes so one unit means about the same on both: 0.05 of MAF, or a factor of about 1.6 in
# distance to the TSS (0.2 in log10). Ties are broken at random, seeded.
for i in rng.permutation(len(gwas)):
    target = gwas.iloc[i]
    cost = (np.abs(pool.maf_eur - target.maf_eur) / 0.05 + np.abs(pool.log_dist - target.log_dist) / 0.2).to_numpy()
    cost = cost + rng.uniform(0, 1e-6, len(pool))
    cost[taken] = np.inf
    for j in np.argsort(cost)[:CONTROLS_PER_GWAS]:
        taken[j] = True
        match_rows.append({**pool.iloc[j].to_dict(), "matched_to": target.key})
controls = pd.DataFrame(match_rows)
controls["clump"] = clumps(list(controls.key), np.arange(len(controls))) + 10_000
print(f"Controls: {len(controls)} ({CONTROLS_PER_GWAS} per GWAS variant), {controls.clump.nunique()} LD clumps")

print("\nBalance after matching:")
for name, column in (("European MAF", "maf_eur"), ("log10(|distance to TSS| + 1 kb)", "log_dist")):
    print(f"  {name:<34} GWAS median {gwas[column].median():>8.3f}   controls median {controls[column].median():>8.3f}")
print(f"  {'max r^2 to any GWAS variant':<34} controls max {controls.max_r2_gwas.max():.3f}")

# --- 7. Save ----------------------------------------------------------------------------------------------------

columns = ["key", "chrom", "pos", "ref", "alt", "rsid", "set", "studies", "lead", "pip", "risk_sign", "af_eur",
           "maf_eur", "dist_tss", "clump", "matched_to"]  # fmt: skip
gwas["set"] = "gwas"
gwas["matched_to"] = ""
controls["set"] = "control"
for column, value in (("rsid", ""), ("studies", ""), ("lead", False), ("pip", np.nan), ("risk_sign", 0)):
    controls[column] = value
controls["chrom"] = CHROM
table = pd.concat([gwas[columns], controls[columns]], ignore_index=True)
DERIVED.mkdir(exist_ok=True)
out = DERIVED / "variants.csv"
table.to_csv(out, index=False, float_format="%.6g")
print(f"\nsaved {rel(out)}: {len(table)} variants, sha256 {sha8(out)}")

print("\nSelf-checks:")
check("every GWAS variant is a common SNV in the panel", gwas.maf_eur.min() >= CONTROL_MIN_MAF)
check(
    "no control is in the GWAS set or any credible-set locus", not set(controls.key) & (all_locus_keys | set(gwas.key))
)
check(f"every control has r^2 < {CONTROL_MAX_R2} with every GWAS variant", controls.max_r2_gwas.max() < CONTROL_MAX_R2)
check("each GWAS variant got its controls", len(controls) == CONTROLS_PER_GWAS * len(gwas))
check("rs1582763, the Bellenguez 2022 lead, is in the set", "rs1582763" in set(gwas.rsid))
expect("GWAS SNVs in the pooled MS4A credible sets plus leads", len(gwas), "about 150", "probe on 2026-11-11")
