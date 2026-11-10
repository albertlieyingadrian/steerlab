# Adding two single-gene effects beats deep models at predicting double-gene perturbations

Research note: [`report/index.html`](report/index.html). A reproduction of Ahlmann-Eltze, Huber and Anders (Nature Methods 2025) on Norman et al. (Science 2019): 91,205 K562 cells, each with one gene, two genes or none switched on by CRISPR activation, collapsed to 237 average profiles. Half the 131 gene pairs are held out, five times over. Every method predicts the held-out pairs' profiles and is scored the same way: L2 distance on the 1,000 most expressed genes, rescaled so 0% is "nothing happened" and 100% is the measurement's own noise. Adding the two single-gene effects reaches 64%; scGen 55%; GEARS 37%. Runs on a laptop CPU in about a minute after a 169 MB download; no API calls.

| File | What it does |
|---|---|
| `pb.py` | Shared helpers: condition names, the splits, the four baselines, L2, Pearson delta, the technical duplicate, the bootstrap. Adapted from the author's `pertkit` toolkit. |
| `1_fetch.py` | Downloads the GEARS authors' Norman archive from Harvard Dataverse (CC0) into `$NORMAN_DIR` (default `~/.cache/steerlab/norman`), checks its sha256, unpacks the AnnData. |
| `2_pseudobulk.py` | Pseudo-bulk profiles, the 1,000 scored genes, the five splits, the PCA gene vectors for ridge, and the technical-duplicate ceiling with its formula check. Needs `anndata` (`uv run --no-project --with anndata`). |
| `3_score.py` | Recomputes the four baselines, loads the scGen and GEARS predictions, scores everything on the 125 pairs every method predicts, bootstraps intervals and paired differences, per-split means. |
| `make_figures.py` | Draws `report/figures/normalised.png` and `pairs.png` and prints the numbers plotted. |
| `make_cover.py` | Draws the cover and the research-page thumbnail from the data. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only scoring and figures). |

## The two deep models

scGen and GEARS were trained outside this folder, in the author's earlier perturbation tutorial, on exactly these five splits; their predictions on the held-out pairs ship in `results/predictions/`, restricted to the 1,000 scored genes, and `3_score.py` scores them through the same path as the baselines.

- **scGen** (Lotfollahi et al., Nature Methods 2019), through `pertpy`: a variational autoencoder (800 hidden units, 100 latent dimensions, 2 layers, up to 40 epochs with early stopping) retrained from scratch for each split on every cell except the held-out pairs' cells; a pair is predicted by adding the two single-gene directions in the latent space to 300 control cells and decoding. About 44 minutes on an Apple-silicon laptop CPU for the five splits.
- **GEARS** (Roohani et al., Nature Biotechnology 2024), its reference implementation in its own environment (it needs numpy < 2): hidden size 64, 20 epochs, batch 32, on each of the same five splits, with GEARS's own validation set carved from the training conditions. About 100 minutes per split on one NVIDIA L4 (Modal), about $6.70 for all five. GEARS keeps 89,357 of the cells and predicts 125 of the 128 held-out pairs; the three it cannot predict (`IER5L+LYL1`, `RHOXF2BB+SET`, `RHOXF2BB+ZBTB25`) are dropped from every method's score.

## Data

- `results/pseudobulk_top1000.npz`: `conditions` (237 names: `ctrl`, `GENE+ctrl` for a single, `A+B` for a pair), `values` (each condition's mean log-normalised expression on the 1,000 scored genes, float32), `n_cells`, `gene_ids` (Ensembl), `gene_names`.
- `results/splits.json`: for split seeds 1 to 5, `train_doubles` (66 pairs) and `test_doubles` (65 pairs).
- `results/pca_embedding.npz`: `genes` (the 105 singly perturbed genes) and `vectors` (20 PCA coordinates of each gene's measured shift, over all 5,045 genes).
- `results/ceiling.csv`: per held-out pair, `n_cells`, `technical_duplicate_l2` (one random half of the cells against all of them) `analytic_l2` (the sampling error the formula predicts) and `half_vs_half_l2` (one half against the other half, the version that doubles the floor; shown, never used).
- `results/predictions/scgen.npz`, `gears.npz`: `conditions`, `seeds` (the split), `values` (the predicted profile on the 1,000 genes).
- `results/per_pair.csv`: `pair`, `method`, `l2` (mean over the splits that held the pair out), `pearson_delta` (correlation of predicted and measured change from control; `nan` for no-change).
- `results/table.csv`: per method, `pairs`, `l2` with its 95% bootstrap interval (`l2_low`, `l2_high`), `pearson_delta`, `normalised` with the interval rescaled (`normalised_low`, `normalised_high`).
- `results/summary.json`: the table, the ceiling and its formula check, paired differences against additive (`mean`, `low`, `high`, `method_wins`), per-split means, and the worked example.

## Sources

Norman et al., "Exploring genetic interaction manifolds constructed from rich single-cell phenotypes", Science 2019 (data as processed by the GEARS authors, Harvard Dataverse DVN/Q2ZV3E, CC0). Ahlmann-Eltze, Huber and Anders, "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines", Nature Methods 2025. Roohani, Huang and Leskovec, GEARS, Nature Biotechnology 2024. Lotfollahi, Wolf and Theis, scGen, Nature Methods 2019.
