"""Score every method the same way: same splits, same 1,000 genes, same metric, same bootstrap, same ceiling.

    python experiments/perturbation-baselines/3_score.py

The four baselines are recomputed here. scGen and GEARS arrive as saved predictions (results/predictions/), because
they were trained elsewhere: scGen on a laptop CPU, GEARS on a cloud GPU (see README.md). Every method is scored on
the held-out pairs all of them can predict: GEARS covers 125 of the 128, since it only predicts genes in its Gene
Ontology graph.

A pair held out by several splits gets one score per split; its score is their mean, and the bootstrap resamples
pairs. Normalised: 0 is no-change on the same pairs, 1 is the technical-duplicate ceiling on the same pairs.

Writes results/table.csv, results/per_pair.csv and results/summary.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from pb import (
    RESULTS,
    additive,
    bootstrap_ci,
    l2,
    load_profiles,
    load_splits,
    mean_response,
    no_change,
    normalise,
    pearson_delta,
    perturbed_genes,
    ridge_embedding,
)

METHODS = ["no-change", "mean-response", "ridge", "GEARS", "scGen", "additive"]


def saved(name: str) -> dict[str, dict[int, np.ndarray]]:
    data = np.load(RESULTS / "predictions" / f"{name}.npz")
    out: dict[str, dict[int, np.ndarray]] = {}
    for condition, seed, row in zip(data["conditions"], data["seeds"], data["values"], strict=True):
        out.setdefault(str(condition), {})[int(seed)] = np.asarray(row, dtype=np.float64)
    return out


def interval(values) -> dict[str, float]:
    mean, low, high = bootstrap_ci(values)
    return {"mean": mean, "low": low, "high": high}


def main() -> None:
    profiles, control = load_profiles()
    splits = load_splits()
    singles = [c for c in profiles if len(perturbed_genes(c)) == 1]
    pca = np.load(RESULTS / "pca_embedding.npz")
    embedding = {str(g): v for g, v in zip(pca["genes"], pca["vectors"], strict=True)}

    # {method: {pair: {seed: predicted profile}}}
    predicted: dict[str, dict[str, dict[int, np.ndarray]]] = {m: {} for m in METHODS}
    for seed, split in splits.items():
        test, visible = split["test_doubles"], split["train_doubles"] + singles
        built = {
            "no-change": no_change(profiles, test),
            "mean-response": mean_response(profiles, test, visible),
            "additive": additive(profiles, test),
            "ridge": ridge_embedding(profiles, test, visible, embedding),
        }
        for method, rows in built.items():
            for pair, row in rows.items():
                predicted[method].setdefault(pair, {})[seed] = row
    predicted["scGen"] = saved("scgen")
    predicted["GEARS"] = saved("gears")

    held_out = sorted(predicted["additive"])
    common = sorted(set.intersection(*(set(predicted[m]) for m in METHODS)))
    print(f"{len(held_out)} pairs held out at least once; {len(common)} predicted by every method")
    print(f"  GEARS has no prediction for: {', '.join(sorted(set(held_out) - set(predicted['GEARS'])))}")

    per_pair = {m: {} for m in METHODS}
    for method in METHODS:
        for pair in common:
            rows = predicted[method][pair].values()
            per_pair[method][pair] = {
                "l2": float(np.mean([l2(r, profiles[pair]) for r in rows])),
                "pearson_delta": float(np.mean([pearson_delta(r, profiles[pair], control) for r in rows])),
            }

    ceiling_rows = list(csv.DictReader((RESULTS / "ceiling.csv").open()))
    ceiling_by_pair = {r["pair"]: float(r["technical_duplicate_l2"]) for r in ceiling_rows}
    ceiling = interval([ceiling_by_pair[p] for p in common])
    analytic = float(np.mean([float(r["analytic_l2"]) for r in ceiling_rows if r["pair"] in common]))
    half_vs_half = float(np.mean([float(r["half_vs_half_l2"]) for r in ceiling_rows if r["pair"] in common]))
    floor = float(np.mean([per_pair["no-change"][p]["l2"] for p in common]))

    table = []
    for method in METHODS:
        scores = interval([per_pair[method][p]["l2"] for p in common])
        deltas = [per_pair[method][p]["pearson_delta"] for p in common]
        table.append({
            "method": method,
            "pairs": len(common),
            "l2": scores["mean"],
            "l2_low": scores["low"],
            "l2_high": scores["high"],
            "pearson_delta": float("nan") if np.isnan(deltas).any() else float(np.mean(deltas)),
            "normalised": normalise(scores["mean"], floor, ceiling["mean"]),
            "normalised_low": normalise(scores["high"], floor, ceiling["mean"]),
            "normalised_high": normalise(scores["low"], floor, ceiling["mean"]),
        })  # fmt: skip
    table.append({
        "method": "technical duplicate", "pairs": len(common), "l2": ceiling["mean"], "l2_low": ceiling["low"],
        "l2_high": ceiling["high"], "pearson_delta": float("nan"), "normalised": 1.0, "normalised_low": 1.0,
        "normalised_high": 1.0,
    })  # fmt: skip
    table.sort(key=lambda row: -row["l2"])

    print(f"\n{'method':<22}{'mean L2':>9}{'95% interval':>18}{'Pearson δ':>11}{'normalised':>12}")
    for row in table:
        delta = "n/a" if np.isnan(row["pearson_delta"]) else f"{row['pearson_delta']:.3f}"
        span = f"[{row['l2_low']:.3f}, {row['l2_high']:.3f}]"
        print(f"{row['method']:<22}{row['l2']:>9.3f}{span:>18}{delta:>11}{row['normalised']:>12.1%}")
    print(f"ceiling check: technical duplicate {ceiling['mean']:.3f}, sampling-error formula {analytic:.3f}")
    additive_l2 = next(row["l2"] for row in table if row["method"] == "additive")
    inflated = normalise(additive_l2, floor, half_vs_half)
    print(f"  half against half would read {half_vs_half:.3f} and put additive at {inflated:.1%}")

    # Paired differences: method minus additive, per pair (positive = the method is further from the truth).
    differences = {}
    for method in ["scGen", "GEARS", "ridge", "mean-response", "no-change"]:
        gaps = [per_pair[method][p]["l2"] - per_pair["additive"][p]["l2"] for p in common]
        differences[f"{method} minus additive"] = {**interval(gaps), "method_wins": int(sum(g < 0 for g in gaps))}
    gaps = [per_pair["ridge"][p]["l2"] - per_pair["mean-response"][p]["l2"] for p in common]
    differences["ridge minus mean-response"] = {**interval(gaps), "method_wins": int(sum(g < 0 for g in gaps))}
    print("\nper pair, L2 difference [95% interval], and pairs where the first method is closer:")
    for name, d in differences.items():
        print(f"  {name:<28}{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]   {d['method_wins']}/{len(common)}")

    # Is one lucky split behind the ranking? Mean L2 per split, on that split's held-out pairs in `common`.
    by_split = {}
    for seed, split in splits.items():
        pairs = [p for p in split["test_doubles"] if p in common]
        by_split[seed] = {"pairs": len(pairs)} | {
            m: float(np.mean([l2(predicted[m][p][seed], profiles[p]) for p in pairs])) for m in METHODS
        }
    print("\nmean L2 per split (each split's held-out pairs):")
    print(f"  {'split':<7}{'pairs':>6}" + "".join(f"{m:>15}" for m in METHODS))
    for seed, row in by_split.items():
        print(f"  {seed:<7}{row['pairs']:>6}" + "".join(f"{row[m]:>15.3f}" for m in METHODS))
    additive_best = sum(min(METHODS, key=lambda m: row[m]) == "additive" for row in by_split.values())
    print(f"  additive has the lowest mean L2 in {additive_best} of {len(by_split)} splits")

    # A worked example on three genes, for the note: the pair where additive is closest to the observed profile.
    data = np.load(RESULTS / "pseudobulk_top1000.npz")
    names = [str(n) for n in data["gene_names"]]
    pair = min(common, key=lambda p: per_pair["additive"][p]["l2"])
    a, b = (f"{g}+ctrl" for g in perturbed_genes(pair))
    observed_shift = profiles[pair] - control
    top = list(np.argsort(-np.abs(observed_shift))[:3])
    toy = {
        "pair": pair,
        "genes": [names[i] for i in top],
        "control": [float(control[i]) for i in top],
        "shift_a": [float(profiles[a][i] - control[i]) for i in top],
        "shift_b": [float(profiles[b][i] - control[i]) for i in top],
        "observed_shift": [float(observed_shift[i]) for i in top],
        "additive_l2": per_pair["additive"][pair]["l2"],
    }
    print(f"\nworked example: {pair}, genes {', '.join(toy['genes'])}")

    with (RESULTS / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()} for row in table)
    with (RESULTS / "per_pair.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "method", "l2", "pearson_delta"])
        for pair in common:
            for method in METHODS:
                s = per_pair[method][pair]
                writer.writerow([pair, method, f"{s['l2']:.6f}", f"{s['pearson_delta']:.6f}"])
    summary = {
        "pairs_held_out": len(held_out),
        "pairs_scored": len(common),
        "gears_missing": sorted(set(held_out) - set(predicted["GEARS"])),
        "ceiling": ceiling | {"analytic": analytic, "half_vs_half": half_vs_half, "additive_if_half_vs_half": inflated},
        "no_change_l2": floor,
        "table": table,
        "differences": differences,
        "by_split": {str(k): v for k, v in by_split.items()},
        "toy": toy,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=True) + "\n")
    print("wrote results/table.csv, results/per_pair.csv, results/summary.json")


if __name__ == "__main__":
    main()
