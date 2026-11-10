"""Collapse 91,205 cells into one profile per condition, fix the five splits, and measure the noise ceiling.

    uv run --no-project --with anndata python experiments/perturbation-baselines/2_pseudobulk.py

Needs anndata (to read the .h5ad) and about 6 GB of memory; everything after this step needs only numpy.

1. Canonicalise condition names (`B+A` and `A+B` are one pair) and average each condition's cells: the
   pseudo-bulk profile. 237 conditions: the control, 105 single-gene perturbations, 131 pairs.
2. Score on the 1,000 genes with the highest control expression, chosen from the control so the choice cannot
   depend on what is being scored.
3. Five splits (Ahlmann-Eltze et al. 2025): every single trains, half the pairs train, half are held out.
4. A 20-component PCA of the measured single-gene shifts, used by the ridge baseline as each gene's vector.
5. The technical duplicate: for each held-out pair, one random half of its cells against all of them. That is
   the sampling error of a pseudo-bulk mean, the floor no prediction can go below. It is checked against the
   formula for that error, sqrt(sum over genes of variance / n cells).

Writes results/pseudobulk_top1000.npz, results/pca_embedding.npz, results/splits.json and results/ceiling.csv.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import anndata as ad
import numpy as np
import scipy.sparse as sp

from pb import (
    COMPONENTS,
    CONTROL,
    RESULTS,
    TOP_GENES,
    canonical_condition,
    make_splits,
    perturbed_genes,
    technical_duplicate,
)

fetch = importlib.import_module("1_fetch")


def main() -> None:
    path = fetch.data_dir() / fetch.MEMBER
    adata = ad.read_h5ad(path)
    labels = np.array([canonical_condition(c) for c in adata.obs["condition"].astype(str)])
    print(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes; {len(set(adata.obs['condition']))} labels as shipped")

    # Pseudo-bulk: the mean of each condition's cells, groups in sorted order, summed in float64.
    names, codes = np.unique(labels, return_inverse=True)
    counts = np.bincount(codes, minlength=names.size)
    indicator = sp.csr_matrix((np.ones(codes.size), (codes, np.arange(codes.size))), shape=(names.size, codes.size))
    sums = indicator @ adata.X.astype(np.float64)
    bulk = (np.asarray(sums.todense()) / counts[:, None]).astype(adata.X.dtype)
    rows = {str(name): i for i, name in enumerate(names)}
    control = bulk[rows[CONTROL]]
    singles = [str(n) for n in names if len(perturbed_genes(n)) == 1]
    doubles = [str(n) for n in names if len(perturbed_genes(n)) == 2]
    print(f"{len(names)} conditions: control ({counts[rows[CONTROL]]:,} cells), {len(singles)} singles, "
          f"{len(doubles)} pairs; median {int(np.median(counts))} cells per condition")  # fmt: skip

    genes = np.argsort(-control.astype(np.float64))[:TOP_GENES]
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(
        RESULTS / "pseudobulk_top1000.npz",
        conditions=names.astype(str),
        values=bulk[:, genes],
        n_cells=counts,
        gene_ids=adata.var_names.to_numpy()[genes].astype(str),
        gene_names=adata.var["gene_name"].to_numpy()[genes].astype(str),
    )

    splits = make_splits(doubles)
    (RESULTS / "splits.json").write_text(json.dumps({str(k): v for k, v in splits.items()}, indent=1) + "\n")
    held_out = sorted(set().union(*(set(s["test_doubles"]) for s in splits.values())))
    print(f"five splits of {len(doubles)} pairs: 66 train, 65 held out each; {len(held_out)} held out at least once")

    # PCA over all 5,045 genes of the singles' shifts; a gene's vector is its row of U times the singular values.
    shifts = np.vstack([bulk[rows[s]] - control for s in singles])
    left, values, _ = np.linalg.svd(shifts - shifts.mean(axis=0), full_matrices=False)
    explained = float((values[:COMPONENTS] ** 2).sum() / (values**2).sum())
    np.savez_compressed(
        RESULTS / "pca_embedding.npz",
        genes=np.array([perturbed_genes(s)[0] for s in singles]),
        vectors=left[:, :COMPONENTS] * values[:COMPONENTS],
    )
    print(f"PCA embedding: {COMPONENTS} components, {explained:.0%} of the single-gene shift variance")

    keep = np.isin(labels, held_out)
    cells = np.asarray(adata.X[keep][:, genes].todense(), dtype=np.float64)
    cell_labels = labels[keep]
    ceiling = technical_duplicate(cells, cell_labels, seed=0)
    analytic, half_vs_half = {}, {}
    rng = np.random.default_rng(0)  # the same draws as technical_duplicate, to show the mistake it avoids
    for pair in ceiling:
        group = cells[cell_labels == pair]
        analytic[pair] = float(np.sqrt((group.var(axis=0, ddof=1) / group.shape[0]).sum()))
        shuffled = rng.permutation(np.flatnonzero(cell_labels == pair))
        half = shuffled.size // 2
        half_vs_half[pair] = float(
            np.linalg.norm(cells[shuffled[:half]].mean(0) - cells[shuffled[half : 2 * half]].mean(0))
        )
    with (RESULTS / "ceiling.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "n_cells", "technical_duplicate_l2", "analytic_l2", "half_vs_half_l2"])
        for pair in sorted(ceiling):
            writer.writerow([pair, counts[rows[pair]]] + [f"{d[pair]:.10f}" for d in (ceiling, analytic, half_vs_half)])
    measured, predicted = np.mean(list(ceiling.values())), np.mean(list(analytic.values()))
    print(f"technical duplicate over {len(ceiling)} pairs ({cells.shape[0]:,} cells): mean L2 {measured:.3f}; "
          f"formula {predicted:.3f}, ratio {measured / predicted:.3f} (half against half reads about 2x)")  # fmt: skip


if __name__ == "__main__":
    main()
