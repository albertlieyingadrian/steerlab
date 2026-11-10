"""Choose SAE features on B. subtilis, test them on the held-out genomes, and score every baseline on the same units.

    python experiments/evo2-rna-atlas/6_score.py

A unit is one intergenic region (with 100 bp of each neighbouring gene) on one strand; it is positive when it holds
at least half of an Infernal riboswitch hit on that strand. B. subtilis is the search set: every choice (which
feature, which probe strength, which sign for a baseline) is made there. The other genomes are held out: nothing is
tuned on them, and their units are pooled into one test set.

Methods, all scoring a unit with one number:

| method          | what it is                                                                        | trained on   |
|-----------------|-----------------------------------------------------------------------------------|--------------|
| best feature    | the one SAE feature whose max over the unit best ranks B. subtilis units (AP)      | B. subtilis  |
| SAE probe       | logistic regression on log(1 + max) of every feature that fires in 5+ units        | B. subtilis  |
| embedding probe | logistic regression on the unit's mean layer-26 vector (4,096 numbers)             | B. subtilis  |
| random feature  | 500 random features that fire about as often as the best one                       | none         |
| length, GC      | the unit's length; its GC content                                                  | sign only    |
| fold            | the strongest fold of any 120-letter window (ViennaRNA)                            | none         |
| 4-mer probe     | logistic regression on the unit's 256 four-letter-word frequencies                 | B. subtilis  |
| entropy         | the model's mean next-letter entropy over the unit (Evo 2, no SAE)                 | sign only    |
| BLAST           | best blastn bit score against every Rfam riboswitch seed sequence                  | none         |
| BLAST, distant  | the same, ignoring seeds 90%+ identical to the unit (its own or a close relative's) | none         |

Writes results/units.tsv (every unit's label and scores), results/table.csv, results/features.csv (the 20 best
features on B. subtilis with their held-out scores), results/context.csv (the shuffle-in-place test) and
results/summary.json.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from atlas import (
    DATA,
    GENOMES,
    RESULTS,
    auroc,
    average_precision,
    bootstrap,
    kmer_counts,
    read_genome,
    read_hits,
    read_regions,
    revcomp,
)

PAD = 100
N_FEATURES = 32768
EVO = DATA / "evo2"
TRAIN_GENOME = "bsub"


def load_units():
    """Every unit with its label, baselines, SAE max vector, mean embedding and mean entropy."""
    base = {r["unit"]: r for r in csv.DictReader((RESULTS / "unit_baselines.tsv").open(), delimiter="\t")}
    switches, every = read_hits("riboswitch"), read_hits(None)
    units, fmax_rows, emb_rows = [], [], []
    for genome in GENOMES:
        seq = read_genome(genome)
        length = len(seq)
        for strand, name in (("+", "plus"), ("-", "minus")):
            data = np.load(EVO / f"genome_{genome}_{name}.npz")
            rows = {
                (int(lo), int(hi)): i for i, (lo, hi) in enumerate(zip(data["rng_lo"], data["rng_hi"], strict=True))
            }
            entropy = np.zeros(length, dtype=np.float32)
            entropy[data["pos_idx"]] = data["pos_entropy"].astype(np.float32)
            fmax = sparse.csr_matrix(
                (data["rng_feat_max"].astype(np.float32), (data["rng_feat_row"], data["rng_feat_id"].astype(np.int64))),
                shape=(len(data["rng_lo"]), N_FEATURES),
            )
            for r in read_regions(genome):
                lo, hi = max(0, r.start - PAD), min(length, r.end + PAD)
                key = (lo, hi) if strand == "+" else (length - hi, length - lo)
                i = rows[key]
                unit = f"{r.rid}|{strand}"

                def inside(h, genome=genome, strand=strand, lo=lo, hi=hi):
                    if h["genome"] != genome or h["strand"] != strand:
                        return False
                    return min(hi, h["end"]) - max(lo, h["start"]) >= 0.5 * (h["end"] - h["start"])

                mine = [h for h in switches if inside(h)]
                anything = [h for h in every if inside(h)]
                b = base[unit]
                units.append(
                    {
                        "unit": unit,
                        "genome": genome,
                        "strand": strand,
                        "lo": lo,
                        "hi": hi,
                        "label": int(bool(mine)),
                        "families": ",".join(sorted({h["family"] for h in mine})),
                        "any_rfam": ",".join(sorted({h["family"] for h in anything})),
                        "left": r.left,
                        "right": r.right,
                        "length": float(b["length"]),
                        "gc": float(b["gc"]),
                        "fold": float(b["fold"]),
                        "blast": float(b["blast"]),
                        "blast_distant": float(b["blast_distant"]),
                        "entropy": float(entropy[key[0] : key[1]].mean()),
                        "seq": seq[lo:hi] if strand == "+" else revcomp(seq[lo:hi]),
                    }
                )
                fmax_rows.append(fmax[i])
                emb_rows.append(data["rng_mean_emb"][i].astype(np.float32))
    return units, sparse.vstack(fmax_rows).tocsc(), np.stack(emb_rows)


def metrics(scores, labels, top):
    """AUROC and average precision with 95% bootstrap intervals; precision and recall in the top `top` units."""
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    order = np.argsort(-scores, kind="mergesort")[:top]
    hits = int(labels[order].sum())
    return {
        "auroc": auroc(scores, labels),
        "auroc_ci": bootstrap(lambda i: auroc(scores[i], labels[i]), len(scores), reps=1000),
        "ap": average_precision(scores, labels),
        "ap_ci": bootstrap(lambda i: average_precision(scores[i], labels[i]), len(scores), reps=1000),
        "top": top,
        "top_hits": hits,
        "precision_top": hits / top,
        "recall_top": hits / int(labels.sum()),
    }


def fit_probe(x_train, y_train, x_all):
    """L2 logistic regression; strength chosen by 5-fold cross-validated AP on the training genome only."""
    scaler = StandardScaler(with_mean=not sparse.issparse(x_train)).fit(x_train)
    xt, xa = scaler.transform(x_train), scaler.transform(x_all)
    best = None
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for c in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        cv = cross_val_predict(model, xt, y_train, cv=folds, method="decision_function")
        ap = average_precision(cv, y_train)
        if best is None or ap > best[0]:
            best = (ap, c)
    model = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(xt, y_train)
    return model.decision_function(xa), best[1], best[0]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)


def main() -> None:
    units, fmax, emb = load_units()
    labels = np.array([u["label"] for u in units])
    train = np.array([u["genome"] == TRAIN_GENOME for u in units])
    test = ~train
    y_train, y_test = labels[train], labels[test]
    top = int(y_test.sum())
    print(f"units: {train.sum()} train ({y_train.sum()} positive), {test.sum()} held out ({y_test.sum()} positive)")

    # 1. Feature search on B. subtilis: average precision of every feature that fires in a positive unit.
    f_train, f_test = fmax[np.flatnonzero(train)], fmax[np.flatnonzero(test)]
    candidates = np.unique(f_train[np.flatnonzero(y_train)].nonzero()[1])
    ap_train = np.array([average_precision(f_train[:, c].toarray().ravel(), y_train) for c in candidates])
    ranked = candidates[np.argsort(-ap_train)]
    best = int(ranked[0])
    fires_train = np.asarray((f_train > 0).sum(0)).ravel()
    fires_test = np.asarray((f_test > 0).sum(0)).ravel()
    feature_rows = []
    for c in ranked[:20]:
        s_tr, s_te = f_train[:, c].toarray().ravel(), f_test[:, c].toarray().ravel()
        feature_rows.append(
            {
                "feature": int(c),
                "train_ap": average_precision(s_tr, y_train),
                "train_auroc": auroc(s_tr, y_train),
                "test_ap": average_precision(s_te, y_test),
                "test_auroc": auroc(s_te, y_test),
                "train_units_fired": int(fires_train[c]),
                "test_units_fired": int(fires_test[c]),
                "train_positives_fired": int((s_tr[y_train == 1] > 0).sum()),
                "test_positives_fired": int((s_te[y_test == 1] > 0).sum()),
            }
        )
    with (RESULTS / "features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows([{k: fmt(v) for k, v in row.items()} for row in feature_rows])
    for row in feature_rows[:10]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    scores = {"best feature": fmax[:, best].toarray().ravel()}

    # 2. Random features that fire about as often as the best one (within a factor of two, on the held-out units).
    rng = np.random.default_rng(0)
    target = max(1, int(fires_test[best]))
    pool = np.flatnonzero((fires_test >= target / 2) & (fires_test <= target * 2))
    pool = pool[pool != best]
    picks = rng.choice(pool, size=min(500, len(pool)), replace=False)
    random_ap = np.array([average_precision(f_test[:, c].toarray().ravel(), y_test) for c in picks])
    random_auroc = np.array([auroc(f_test[:, c].toarray().ravel(), y_test) for c in picks])

    # 3. Probes, trained on B. subtilis only.
    used = np.flatnonzero(fires_train >= 5)
    x_sae = fmax[:, used].tocsr().log1p()
    scores["SAE probe"], c_sae, cv_sae = fit_probe(x_sae[np.flatnonzero(train)], y_train, x_sae)
    scores["embedding probe"], c_emb, cv_emb = fit_probe(emb[train], y_train, emb)
    kmers = np.stack([kmer_counts(u["seq"], 4) for u in units])
    kmers = kmers / kmers.sum(1, keepdims=True)
    scores["4-mer probe"], c_kmer, cv_kmer = fit_probe(kmers[train], y_train, kmers)

    # 4. Unsupervised baselines; length, GC and entropy get the sign that works better on B. subtilis.
    signs = {}
    for name in ("length", "gc", "fold", "blast", "blast_distant", "entropy"):
        raw = np.array([u[name] for u in units])
        signs[name] = -1 if name in ("length", "gc", "entropy") and auroc(raw[train], y_train) < 0.5 else 1
        scores[name] = signs[name] * raw

    summary = {
        "units": {
            "train": int(train.sum()),
            "train_positive": int(y_train.sum()),
            "test": int(test.sum()),
            "test_positive": int(y_test.sum()),
            "test_by_genome": {
                g: [
                    int(sum(1 for u in units if u["genome"] == g)),
                    int(sum(u["label"] for u in units if u["genome"] == g)),
                ]
                for g in GENOMES
            },
        },
        "best_feature": best,
        "probe_c": {"sae": c_sae, "embedding": c_emb, "kmer": c_kmer},
        "probe_cv_ap_train": {"sae": cv_sae, "embedding": cv_emb, "kmer": cv_kmer},
        "sae_probe_features": len(used),
        "signs": signs,
        "prevalence_test": float(y_test.mean()),
    }
    table = []
    for name, s in scores.items():
        m = metrics(s[test], y_test, top)
        table.append(
            {
                "method": name,
                **m,
                "train_auroc": auroc(s[train], y_train),
                "train_ap": average_precision(s[train], y_train),
            }
        )
        per = {}
        for g in GENOMES:
            if g == TRAIN_GENOME:
                continue
            mask = np.array([u["genome"] == g for u in units])
            per[g] = {"auroc": auroc(s[mask], labels[mask]), "ap": average_precision(s[mask], labels[mask])}
        table[-1]["per_genome"] = per
    table.append(
        {
            "method": "random feature (median of 500)",
            "auroc": float(np.median(random_auroc)),
            "auroc_ci": (float(np.percentile(random_auroc, 2.5)), float(np.percentile(random_auroc, 97.5))),
            "ap": float(np.median(random_ap)),
            "ap_ci": (float(np.percentile(random_ap, 2.5)), float(np.percentile(random_ap, 97.5))),
            "top": top,
        }
    )
    summary["random_feature"] = {
        "n": len(picks),
        "fires_target": target,
        "ap_max": float(random_ap.max()),
        "auroc_max": float(random_auroc.max()),
        "share_ap_at_least_best": float((random_ap >= table[0]["ap"]).mean()),
    }
    fields = ["method", "auroc", "auroc_low", "auroc_high", "ap", "ap_low", "ap_high", "top", "top_hits",
              "precision_top", "recall_top", "train_auroc", "train_ap"]  # fmt: skip
    with (RESULTS / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            out = {k: row.get(k) for k in fields}
            out["auroc_low"], out["auroc_high"] = row["auroc_ci"]
            out["ap_low"], out["ap_high"] = row["ap_ci"]
            writer.writerow({k: fmt(v) for k, v in out.items()})
    for row in table:
        print(
            f"{row['method']:<32} AUROC {row['auroc']:.3f} [{row['auroc_ci'][0]:.3f}, {row['auroc_ci'][1]:.3f}]  "
            f"AP {row['ap']:.3f} [{row['ap_ci'][0]:.3f}, {row['ap_ci'][1]:.3f}]  top-{top}: {row.get('top_hits')}"
            f"  (train AP {row.get('train_ap') or float('nan'):.3f})"
        )
    summary["table"] = table

    # What else sits at the top: the 50 best held-out units of each SAE method, by their Rfam annotation.
    with gzip.open(DATA / "family.txt.gz", "rt", encoding="latin-1") as handle:
        types = {c[1]: c[18] for c in (line.split("\t") for line in handle)}
    held = np.flatnonzero(test)
    summary["top50"] = {}
    for name in ("best feature", "SAE probe"):
        order = held[np.argsort(-scores[name][held], kind="mergesort")[:50]]
        kinds = {"riboswitch": 0, "other Rfam cis-regulatory RNA": 0, "other Rfam RNA": 0, "no Rfam hit": 0}
        for i in order:
            u = units[i]
            if u["label"]:
                kinds["riboswitch"] += 1
            elif u["any_rfam"]:
                kind = types.get(u["any_rfam"].split(",")[0], "")
                kinds["other Rfam cis-regulatory RNA" if kind.startswith("Cis-reg") else "other Rfam RNA"] += 1
            else:
                kinds["no Rfam hit"] += 1
        summary["top50"][name] = kinds
    print("top 50 held-out units:", summary["top50"])

    # 5. Per-unit scores, for the figures, the agent and the appendix.
    with (RESULTS / "units.tsv").open("w", newline="") as handle:
        cols = ["unit", "genome", "strand", "lo", "hi", "label", "families", "any_rfam", "left", "right"]
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(cols + [n.replace(" ", "_").replace("-", "") for n in scores])
        for i, u in enumerate(units):
            writer.writerow([u[c] for c in cols] + [f"{scores[n][i]:.4f}" for n in scores])

    # 6. Shuffle in place: the best feature's max over each hit, real letters against two dinucleotide shuffles.
    ctx = np.load(EVO / "context.npz")
    ids = ctx["item_ids"]
    rows: dict[str, dict] = {}
    for r, item in enumerate(ctx["rng_item"]):
        tag, key = str(ids[item]).split("|", 1)
        mask = ctx["rng_feat_row"] == r
        fid = ctx["rng_feat_id"][mask]
        hit = fid == best
        rows.setdefault(key, {})[tag] = (
            float(ctx["rng_feat_mean"][mask][hit][0]) if hit.any() else 0.0,
            float(ctx["rng_feat_max"][mask][hit][0]) if hit.any() else 0.0,
        )
    ctx_rows = []
    for key, v in sorted(rows.items()):
        genome, start, strand, family = key.split("|")
        ctx_rows.append(
            {
                "genome": genome,
                "start": int(start),
                "strand": strand,
                "family": family,
                "real_max": v["real"][1],
                "shuffled_max": (v["shuf1"][1] + v["shuf2"][1]) / 2,
                "real_mean": v["real"][0],
                "shuffled_mean": (v["shuf1"][0] + v["shuf2"][0]) / 2,
            }
        )
    with (RESULTS / "context.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ctx_rows[0]))
        writer.writeheader()
        writer.writerows([{k: fmt(v) for k, v in row.items()} for row in ctx_rows])
    real = np.array([r["real_max"] for r in ctx_rows])
    shuf = np.array([r["shuffled_max"] for r in ctx_rows])
    diff = real - shuf
    summary["context"] = {
        "hits": len(ctx_rows),
        "real_fires": int((real > 0).sum()),
        "shuffled_fires": int((shuf > 0).sum()),
        "real_mean_max": float(real.mean()),
        "shuffled_mean_max": float(shuf.mean()),
        "diff_mean": float(diff.mean()),
        "diff_ci": bootstrap(lambda i: diff[i].mean(), len(diff)),
        "real_above_shuffled": int((real > shuf).sum()),
    }
    print("shuffle in place:", summary["context"])
    path = RESULTS / "summary.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(summary)  # keeps the sections steps 7 and 8 add
    path.write_text(json.dumps(previous, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
