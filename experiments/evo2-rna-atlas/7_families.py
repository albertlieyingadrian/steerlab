"""Same family, different letters: do Evo 2's vectors group riboswitches by family when BLAST sees no likeness?

    python experiments/evo2-rna-atlas/7_families.py

Input: up to 40 Rfam seed sequences from each of 45 riboswitch families, each read alone by Evo 2, and a dinucleotide
shuffle of each (step 4, job `seeds`). For every sequence (the query), the candidates are all other sequences that
BLAST does not link to it (no blastn hit at E < 0.001 in either direction). The query's nearest candidate by cosine
similarity is its guess; the guess is right when the two share a family. Representations:

| representation      | what it is                                                                      |
|---------------------|---------------------------------------------------------------------------------|
| Evo 2 embedding     | the mean layer-26 vector over the sequence (4,096 numbers)                      |
| SAE features        | the mean of each of the 32,768 SAE features over the sequence                   |
| shuffled embedding  | the same, for the dinucleotide-shuffled versions of query and candidates        |
| 4-mer, 3-mer        | frequencies of each four- or three-letter word                                  |
| length and GC       | two numbers, standardised                                                       |

Every representation is centred on its own mean before cosine. Chance is the share of a query's candidates that
share its family, averaged over queries. Writes results/families.csv and adds `families` to results/summary.json.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy import sparse

from atlas import DATA, RESULTS, bootstrap, gc, kmer_counts

N_FEATURES = 32768


def blast_links(seqs: list[str]) -> set[tuple[int, int]]:
    """Pairs (i, j) with a blastn hit at E < 0.001 in either direction, self-hits dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "seeds.fa"
        fasta.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
        subprocess.run(["makeblastdb", "-in", str(fasta), "-dbtype", "nucl"], check=True, capture_output=True)
        out = subprocess.run(
            ["blastn", "-task", "blastn", "-word_size", "7", "-evalue", "1e-3", "-strand", "plus", "-num_threads", "8",
             "-query", str(fasta), "-db", str(fasta), "-outfmt", "6 qseqid sseqid", "-max_target_seqs", "2000"],
            check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
    links = set()
    for line in out.splitlines():
        a, b = (int(x[1:]) for x in line.split("\t"))
        if a != b:
            links.add((a, b))
            links.add((b, a))
    return links


def nearest(x: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """For each row, the index of the most cosine-similar allowed row (after centring)."""
    x = x - x.mean(0)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    sim = x @ x.T
    sim[~allowed] = -np.inf
    return sim.argmax(1)


def main() -> None:
    data = np.load(DATA / "evo2" / "seeds.npz")
    ids = [str(i) for i in data["item_ids"]]
    row_of = {ids[item]: r for r, item in enumerate(data["rng_item"])}
    sae = sparse.csr_matrix(
        (data["rng_feat_mean"].astype(np.float32), (data["rng_feat_row"], data["rng_feat_id"].astype(np.int64))),
        shape=(len(data["rng_item"]), N_FEATURES),
    )
    seed_set = json.loads((RESULTS / "seed_set.json").read_text())
    keys = sorted(seed_set)
    family = np.array([k.split("|")[0] for k in keys])
    seqs = [seed_set[k] for k in keys]
    real = [row_of[f"real|{k}"] for k in keys]
    shuf = [row_of[f"shuf|{k}"] for k in keys]
    emb = data["rng_mean_emb"].astype(np.float32)

    links = blast_links(seqs)
    n = len(keys)
    allowed = np.ones((n, n), dtype=bool)
    np.fill_diagonal(allowed, False)
    for a, b in links:
        allowed[a, b] = False
    same = family[:, None] == family[None, :]
    same_pairs = int(same.sum() - n)
    unlinked_same = int((same & allowed).sum())
    print(f"{n} seeds, {len(set(family))} families; {unlinked_same}/{same_pairs} same-family pairs have no BLAST link")

    # BLAST itself, allowed every other sequence: does it link the query to anything from its own family?
    blast_any = np.array([any(same[i, j] for j in range(n) if (i, j) in links) for i in range(n)])

    reps = {
        "Evo 2 embedding": emb[real],
        "SAE features": sae[real].toarray(),
        "shuffled embedding": emb[shuf],
        "4-mer": np.stack([kmer_counts(s, 4) / max(1, len(s) - 3) for s in seqs]),
        "3-mer": np.stack([kmer_counts(s, 3) / max(1, len(s) - 2) for s in seqs]),
        "length and GC": np.array([[len(s), gc(s)] for s in seqs], dtype=float),
    }
    lg = reps["length and GC"]
    reps["length and GC"] = (lg - lg.mean(0)) / lg.std(0)
    queries = np.flatnonzero((allowed & same).any(1))
    chance = np.array([(allowed[i] & same[i]).sum() / allowed[i].sum() for i in queries])
    rows, per_query = [], {}
    for name, x in reps.items():
        guess = nearest(x, allowed)
        correct = (family[guess] == family)[queries].astype(float)
        per_query[name] = correct
        low, high = bootstrap(lambda idx, c=correct: c[idx].mean(), len(correct))
        rows.append({"representation": name, "queries": len(queries), "correct": int(correct.sum()),
                     "accuracy": float(correct.mean()), "low": low, "high": high})  # fmt: skip
    rows.append({"representation": "chance", "queries": len(queries), "correct": "",
                 "accuracy": float(chance.mean()), "low": "", "high": ""})  # fmt: skip
    with (RESULTS / "families.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    for row in rows:
        print(row)
    # Paired differences against the 4-mer baseline, query by query.
    paired = {}
    for name in ("Evo 2 embedding", "SAE features"):
        d = per_query[name] - per_query["4-mer"]
        low, high = bootstrap(lambda idx, d=d: d[idx].mean(), len(d))
        paired[name] = {"minus_4mer": float(d.mean()), "low": low, "high": high}
        print(f"{name} minus 4-mer: {d.mean():+.3f} [{low:+.3f}, {high:+.3f}]")
    guess = nearest(reps["Evo 2 embedding"], allowed)
    per_family = {}
    for f in sorted(set(family)):
        q = [i for i in queries if family[i] == f]
        if q:
            per_family[f] = [int(sum(family[guess[i]] == f for i in q)), len(q)]
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["families"] = {
        "seeds": n,
        "families": len(set(family)),
        "same_family_pairs": same_pairs,
        "same_family_pairs_unlinked": unlinked_same,
        "queries": len(queries),
        "blast_links_same_family": int(blast_any.sum()),
        "rows": rows,
        "paired_vs_4mer": paired,
        "per_family_embedding": per_family,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
