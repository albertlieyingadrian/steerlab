# AlphaGenome outputs in this folder carry AlphaGenome's terms, not this repository's licence

`ag_scores.csv`, `per_variant_lfc.csv`, `summary.csv`, `enhancers.csv` and `gtex_check.csv` hold outputs of
AlphaGenome 2 and AlphaGenome 1 (Google DeepMind's hosted regulatory models), or numbers computed directly from them.
`variants.csv` holds public annotations only (Open Targets credible sets, GWAS Catalog alleles, 1000 Genomes
frequencies). `conditioning_cells.h5ad` is a 144-cell subset of GEO GSE289721 (public, no use restrictions stated by
the submitters).

The AlphaGenome terms (<https://alphagenome.google/terms>), as the SDK's README summarises them on 2026-11-11:

- The API is free of charge for non-commercial use.
- Outputs are for non-commercial use only and **should not be used for the training of other machine learning
  models**.
- Predictions are for theoretical modelling and research only. They must not be used for clinical decision-making
  or relied upon for medical or other professional advice.

Read the terms themselves before redistributing or building on these tables.
