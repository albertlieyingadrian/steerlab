"""Ask AlphaGenome 2 (and AlphaGenome 1) what each variant does to MS4A6A, in each kind of cell.

For one variant the model reads the 1 Mb of DNA around it twice, once with each allele, and predicts the
expression of every gene in that window. The score is the log fold change of the gene's predicted expression,
ALT over REF: +0.1 means the ALT allele raises the gene's expression by about 10%.

AlphaGenome 2 makes that prediction for one *condition* at a time: a cell type in a tissue ("microglia in the
middle temporal gyrus"), or, zero-shot, whatever cells you hand it as raw single-cell counts. Here it runs in
seven reference conditions and on 48 iPSC microglia from GSE289721, the dish the hypothesis is about. AlphaGenome
1 has no microglia, so it runs on bulk prefrontal cortex and on monocytes, the closest it has.

Every response is cached as JSON under `~/.cache/steerlab/alphagenome2-ms4a/ag_cache/`, one file per (model, condition,
variant), so a rerun makes no API calls. The script needs `ALPHAGENOME_API_KEY` only for cache misses.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anndata as ad
import numpy as np
import pandas as pd

from ag2 import (
    CONDITIONING_DRAWS,
    CONDITIONS,
    DERIVED,
    GENE,
    SECOND_GENE,
    SEED,
    SMOKE,
    ZERO_SHOT,
    check,
    header,
    optional_env,
    outputs_dir,
    rel,
    require,
    sha8,
)

header(seed=SEED, packages=("alphagenome", "numpy", "pandas", "anndata"))

GENES = (GENE, SECOND_GENE)
AG1_TISSUES = {"UBERON:0009834": "DLPFC tissue (AG1)", "CL:0001054": "CD14+ monocyte (AG1)"}
CACHE = outputs_dir() / "ag_cache"
CACHE.mkdir(exist_ok=True)
WORKERS = 4

variants_path = DERIVED / "variants.csv"
variants = pd.read_csv(require(variants_path, "1_variants.py"))
if SMOKE:
    variants = pd.concat([variants[variants.set == "gwas"].head(3), variants[variants.set == "control"].head(3)])
print(f"{len(variants)} variants from {rel(variants_path)} (sha256 {sha8(variants_path)})")

_clients: dict[str, object] = {}


def client(version: str):
    if version not in _clients:
        key = optional_env("ALPHAGENOME_API_KEY")
        if not key:
            raise SystemExit("cache miss and ALPHAGENOME_API_KEY is not set")
        if version == "ag2":
            from alphagenome.models.v2 import dna_client
        else:
            from alphagenome.models.v1 import dna_client
        _clients[version] = dna_client.create(key)
    return _clients[version]


def to_variant(row):
    from alphagenome.data import genome

    return genome.Variant(row.chrom, int(row.pos), row.ref, row.alt)


AG2_COLUMNS = [
    "gene_name",
    "output_type",
    "variant_scorer",
    "track_name",
    "imputation_mode",
    "Assay title",
    "suspension_type",
    "raw_score",
]
AG1_COLUMNS = [
    "gene_name",
    "output_type",
    "variant_scorer",
    "track_name",
    "ontology_curie",
    "biosample_name",
    "track_strand",
    "raw_score",
]


def keep_rows(tidy: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = tidy[tidy.gene_name.isin(GENES)].copy()
    rows["variant_scorer"] = rows.variant_scorer.astype(str).str.extract(r"(\w+Scorer)\(", expand=False)
    return rows[[c for c in columns if c in rows.columns]].to_dict(orient="records")


def score_ag2(row, condition: str) -> list[dict]:
    from alphagenome.models.v2 import dna_client, variant_scorers

    variant = to_variant(row)
    scorers = [
        variant_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in ("SC_GENE_EXPRESSION", "SC_GENE_EXPRESSION_ACTIVE")
    ]
    scores = client("ag2").score_variant(
        interval=variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB),
        variant=variant,
        condition=condition,
        variant_scorers=scorers,
    )
    return keep_rows(variant_scorers.tidy_scores(scores, match_gene_strand=True), AG2_COLUMNS)


def score_ag2_cells(row, cells: ad.AnnData, name: str) -> list[dict]:
    from alphagenome.models.v2 import dna_client, variant_scorers

    variant = to_variant(row)
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS["SC_GENE_EXPRESSION_DENOISED"]]
    scores = client("ag2").score_variant(
        interval=variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB),
        variant=variant,
        variant_scorers=scorers,
        cell_conditioning=cells,
        cell_condition_name=name,
    )
    return keep_rows(variant_scorers.tidy_scores(scores, match_gene_strand=True), AG2_COLUMNS)


def score_ag1(row) -> list[dict]:
    from alphagenome.models.v1 import dna_client, variant_scorers

    variant = to_variant(row)
    scores = client("ag1").score_variant(
        interval=variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB),
        variant=variant,
        variant_scorers=[variant_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]],
    )
    tidy = variant_scorers.tidy_scores(scores, match_gene_strand=True)
    return keep_rows(tidy[tidy.ontology_curie.isin(AG1_TISSUES)], AG1_COLUMNS)


def run_jobs(label: str, jobs: list[tuple[Path, object]]) -> tuple[int, int]:
    """Run (cache path, thunk) jobs in parallel; a cached path is read, never re-requested."""
    misses = [(path, fn) for path, fn in jobs if not path.exists()]
    started = time.monotonic()
    failures = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(fn): path for path, fn in misses}
        for future in as_completed(futures):
            path = futures[future]
            try:
                path.write_text(json.dumps(future.result()))
            except Exception as error:
                failures += 1
                message = str(error).splitlines()[0][:160]
                if "safety" in message.lower() or "refus" in message.lower():
                    raise SystemExit(f"stopped: the API refused on safety grounds ({message})") from error
                print(f"    {path.name}: {type(error).__name__}: {message}")
    print(f"  {label:<36} {len(jobs):>4} jobs, {len(misses):>4} requested, {failures} failed, "
          f"{time.monotonic() - started:.0f}s")  # fmt: skip
    return len(misses), failures


def cache_path(model: str, condition: str, key: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in condition)
    return CACHE / f"{model}__{safe}__{key.replace(':', '_')}.json"


print("\nAlphaGenome 2, reference conditions (recommended single-cell expression scorer):")
requested = failed = 0
for name, condition in CONDITIONS.items():
    jobs = [
        (cache_path("ag2", condition, r.key), lambda r=r, c=condition: score_ag2(r, c)) for r in variants.itertuples()
    ]
    n, f = run_jobs(name, jobs)
    requested, failed = requested + n, failed + f

print("\nAlphaGenome 2, zero-shot on GSE289721 iPSC microglia (denoised single-cell expression):")
# The three seeded draws of 48 non-targeting, vehicle-treated cells from experiment XP1 of GSE289721 (raw UMI counts,
# Ensembl gene IDs), shipped in results/ so this step needs no 620 MB download. results/README in the folder's
# README.md says how they were drawn.
pool = ad.read_h5ad(require(DERIVED / "conditioning_cells.h5ad", "the shipped results/conditioning_cells.h5ad"))
draws = []
for draw in range(CONDITIONING_DRAWS):
    cells = pool[(pool.obs["draw"] == draw).to_numpy()].copy()
    cells.X = cells.X.astype(np.int32)
    cells = cells[:, np.asarray((cells.X > 0).sum(axis=0)).ravel() > 0].copy()
    draws.append(cells)
    umis = int(np.median(np.asarray(cells.X.sum(axis=1)).ravel()))
    print(f"  draw {draw}: {cells.n_obs} cells, {cells.n_vars:,} genes detected, median UMIs per cell {umis:,}")
for draw, cells in enumerate(draws):
    label = f"iPSC microglia draw {draw}"
    jobs = [
        (cache_path("ag2", f"gse289721_draw{draw}", r.key), lambda r=r, c=cells, n=label: score_ag2_cells(r, c, n))
        for r in variants.itertuples()
    ]
    n, f = run_jobs(label, jobs)
    requested, failed = requested + n, failed + f

print("\nAlphaGenome 1 (no microglia; bulk DLPFC and CD14+ monocytes):")
n, f = run_jobs(
    "AG1 RNA-seq", [(cache_path("ag1", "all", r.key), lambda r=r: score_ag1(r)) for r in variants.itertuples()]
)
requested, failed = requested + n, failed + f

# --- Collect the cache into one long table ----------------------------------------------------------------------


def cached_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


records = []
for r in variants.itertuples():
    for name, condition in CONDITIONS.items():
        for row in cached_rows(cache_path("ag2", condition, r.key)):
            records.append({"key": r.key, "model": "AG2", "condition": name, "draw": np.nan, **row})
    for draw in range(CONDITIONING_DRAWS):
        for row in cached_rows(cache_path("ag2", f"gse289721_draw{draw}", r.key)):
            records.append({"key": r.key, "model": "AG2", "condition": ZERO_SHOT, "draw": draw, **row})
    for row in cached_rows(cache_path("ag1", "all", r.key)):
        records.append(
            {"key": r.key, "model": "AG1", "condition": AG1_TISSUES[row["ontology_curie"]], "draw": np.nan, **row}
        )
scores = pd.DataFrame(records)
# Keep what the analysis reads: each reference condition's observed tracks (imputed ones are dropped to keep the
# committed table small), the zero-shot denoised channel and AlphaGenome 1's tracks.
scores = scores[(scores.condition == ZERO_SHOT) | (scores.model == "AG1") | (scores.imputation_mode == "observed")]
scores = scores.drop(columns=[c for c in ("Assay title", "suspension_type", "output_type") if c in scores.columns])
out = DERIVED / ("ag_scores_smoke.csv" if SMOKE else "ag_scores.csv")
scores.to_csv(out, index=False, float_format="%.6g")
print(f"\nsaved {rel(out)}: {len(scores):,} rows (variant x condition x gene x track x scorer)")

have_ag2 = scores[(scores.model == "AG2") & (scores.condition != ZERO_SHOT)].groupby(["key", "condition"]).ngroups
have_zero = scores[scores.condition == ZERO_SHOT].groupby(["key", "draw"]).ngroups
have_ag1 = scores[scores.model == "AG1"].key.nunique()

print("\nSelf-checks:")
check("no call failed in this run", failed == 0)
check(
    f"every variant scored in all {len(CONDITIONS)} reference conditions", have_ag2 == len(variants) * len(CONDITIONS)
)
check(
    f"every variant scored on all {CONDITIONING_DRAWS} iPSC-microglia draws",
    have_zero == len(variants) * CONDITIONING_DRAWS,
)
check("every variant scored by AlphaGenome 1", have_ag1 == len(variants))
check(
    f"{GENE} is scored in every condition", scores[scores.gene_name == GENE].condition.nunique() == len(CONDITIONS) + 3
)
print(f"\nAPI calls made by this run: {requested} (a rerun with a warm cache makes none)")
