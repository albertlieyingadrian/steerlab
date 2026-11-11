# AlphaGenome 2 does not predict which way the MS4A Alzheimer's risk variants move MS4A6A in microglia

Research note: [`report/index.html`](report/index.html) ([live page](https://albertlieyingadrian.github.io/steerlab/experiments/alphagenome2-ms4a/report/)).

AlphaGenome 2 (Google DeepMind, 2026) predicts how a DNA variant changes gene expression in one chosen cell type. This folder asks it a pre-registered question about one Alzheimer's risk locus: do the risk variants in the MS4A gene cluster change predicted MS4A6A expression in microglia (the brain's immune cells) more than matched non-risk variants do, is that specific to microglia, and does the risk allele raise or lower MS4A6A? The motivation is a hypothesis that knocking MS4A6A down moves microglia toward a healthier state.

**Answer: inconclusive under the rule fixed before scoring.**
- The risk variants move predicted MS4A6A no more than unlinked controls once distance to the gene is held fixed (0.90x [0.53, 1.53]).
- The effects are not microglia-specific.
- The predicted fold changes are tiny: median 0.0015, against the model's own call threshold of 0.1.
- Measured eQTLs in GTEx whole blood say the risk allele raises MS4A6A, at 92 of 92 variants. AlphaGenome 2's signs agree with them at 40 of 96, which is chance.

## Files

| File | What it does |
|---|---|
| `ag2.py` | The pre-registered constants, cached web clients (Open Targets, GWAS Catalog, Ensembl), a pure-Python tabix reader for the 1000 Genomes panel, LD (r²), print helpers |
| `1_variants.py` | Builds the variant table: SNVs in the 95% credible sets of five Alzheimer's GWAS on Open Targets at chr11:59.85-60.55 Mb, six literature leads, and two controls per GWAS variant (same window; matched on European MAF and log distance to the MS4A6A TSS; r² < 0.1 with every GWAS variant) |
| `2_score.py` | Scores every variant with AlphaGenome 2 in seven reference conditions, zero-shot on three draws of 48 iPSC microglia, and with AlphaGenome 1. One cached JSON per call |
| `3_analysis.py` | The pre-registered statistics and decision; draws `report/figures/conditions.png`, `effects.png`, `models.png` |
| `4_enhancers.py` | MS4A6A enhancer calls from AlphaGenome 2 input gradients and predicted scATAC (SDK `enhancer_scoring`), in microglia and in a neuron; draws `enhancers.png` |
| `5_gtex_check.py` | Added after the pre-registration: the model's signs against measured GTEx v8 eQTLs |
| `make_cover.py` | Draws the cover and thumbnail from `results/per_variant_lfc.csv` |
| `reproduce.sh` | `--scores` recomputes everything from `results/` (no key, seconds); without it, the full run (`--dry-run` prints the steps) |

## The pre-registration

The following were written down before any control variant was scored. Before that, one probe call had been made (rs1582763 in microglia, LFC -0.0053), and the variant table had been built.

- **Variants and controls:** as above.
- **Conditions:** AlphaGenome 2 reference conditions, all "observed" for single-cell expression:
  - middle temporal gyrus (MTG) microglia, the primary condition;
  - dorsolateral prefrontal cortex (DLPFC) microglia, the replicate;
  - MTG astrocyte, MTG L2/3 IT neuron, MTG oligodendrocyte and liver hepatocyte, the non-microglial controls;
  - blood CD14+ monocyte, a myeloid comparator.
  
  Zero-shot runs on GSE289721 iPSC microglia (vehicle, non-targeting guides, experiment XP1, 48 cells, three seeded draws). AlphaGenome 1 runs in DLPFC tissue and CD14+ monocytes.
- **Readout:** the recommended gene-level scorer: the log fold change of MS4A6A (ALT over REF) on the condition's observed track.
- **Statistics:** intervals resample LD clumps (r² ≥ 0.5; the GWAS set is 4 clumps), 10,000 times.
  - S1: median |LFC|, GWAS over controls.
  - S1b: S1 adjusted for log distance and MAF.
  - S2: mean risk-allele effect.
  - S3: S1 and S2 in every condition, plus a paired microglia-vs-other comparison.
- **Decision.** "Consistent" needs all four of these. "Against" needs (i), (ii) and (iv) with the S2 interval below 0. Anything else is "inconclusive".
  - (i) at least one GWAS variant has |LFC| > 0.1;
  - (ii) the S1 interval lies above 1;
  - (iii) the S2 interval lies above 0;
  - (iv) DLPFC microglia gives the same S2 sign.

## Data

- `results/variants.csv`: one row per variant.
  - `key` (pos:ref:alt, GRCh38), `chrom`, `pos`, `ref`, `alt`, `rsid`.
  - `set` (`gwas` or `control`), `studies` (the credible sets it is in), `lead` (named in a paper), `pip` (highest posterior inclusion probability).
  - `risk_sign`: +1 if the ALT allele raises Alzheimer's risk, -1 if the REF allele does, 0 if unknown.
  - `af_eur`, `maf_eur` (1000 Genomes European, non-child samples), `dist_tss` (bp from the MS4A6A TSS, chr11:60,183,083).
  - `clump` (LD clump id), `matched_to` (the GWAS variant a control was matched to).
- `results/ag_scores.csv`: one row per variant, condition, gene (MS4A6A, MS4A4A), track and scorer.
  - `model` (AG2 or AG1), `condition`, `draw` (zero-shot draw), `variant_scorer` (`GeneMaskLFCScorer` = log fold change, `GeneMaskActiveScorer` = predicted expression level), `track_name`, `imputation_mode`, `raw_score`.
  - Observed tracks only.
- `results/per_variant_lfc.csv`: the readout, one MS4A6A (and MS4A4A) log fold change per variant and condition, with the variant annotations.
- `results/summary.csv`: per condition, every statistic in the note.
  - S1: `ratio`, `ratio_lo`, `ratio_hi`, `p_mw`, `median_pct`, `calls_gwas`, `calls_control`.
  - S1b: `adj_fold`, `adj_lo`, `adj_hi`, and the near-TSS stratum.
  - S2: `risk_mean`, `risk_lo`, `risk_hi`, `risk_share_up`, `risk_pip_mean`.
- `results/enhancers.csv`: called MS4A6A enhancers per condition (`start`, `end`, `gradient_score`, `atac_score`, `tss_distance`).
- `results/gtex_check.csv`: GWAS variants with a significant MS4A6A eQTL in GTEx v8 whole blood (`nes` for the ALT allele, `p`), beside the predicted LFCs.
- `results/conditioning_cells.h5ad`: the 3 × 48 conditioning cells (raw UMI counts, Ensembl IDs, `obs.draw`), drawn with `numpy.random.default_rng(seed)`, seeds 0-2, from the 1,029 non-targeting vehicle cells of experiment XP1 that passed QC.
- `results/logs/`: the printed output of each script, rerun from the cache.
- `results/LICENSE-NOTE.md`: the AlphaGenome tables carry AlphaGenome's terms. They are non-commercial and must not be used to train models.

The first run made 3,168 AlphaGenome calls (about 15 minutes at four workers) and cost nothing. Not clinical; predictions are not experiments.

## Sources

AlphaGenome 2: Gayoso, Mottram, Karollus, Novati et al., "AlphaGenome 2 predicts regulatory activity and variant effects across cell states", Google DeepMind preprint, 2026; SDK `alphagenome` 2.0.0. AlphaGenome 1: Avsec et al., Nature 2026. GWAS: Bellenguez et al., Nature Genetics 2022; Schwartzentruber et al., Nature Genetics 2021; Jansen et al., Nature Genetics 2019; Lake et al. 2023; FinnGen R12; via the Open Targets Platform and the GWAS Catalog. Deming et al., Science Translational Medicine 2019 (MS4A and sTREM2). 1000 Genomes high-coverage panel (Byrska-Bishop et al., Cell 2022). GTEx v8 (GTEx Consortium, Science 2020). GSE289721 (iPSC-microglia CRISPRi screen, Journal of Neuroinflammation 2025). SEA-AD (Gabitto et al., Nature Neuroscience 2024), the dataset behind the MTG microglia track.
