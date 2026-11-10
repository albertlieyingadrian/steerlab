"""Score every (region, strand) unit with the simple methods Evo 2 has to beat; no GPU, no model.

    uv run --no-project --with ViennaRNA --with numpy --with scipy python experiments/evo2-rna-atlas/5_baselines.py

A unit is one intergenic region (plus 100 bp of each neighbouring gene) read on one strand, 5' to 3'. For each:

- length and GC content, the two boring explanations;
- fold: the lowest minimum free energy (ViennaRNA RNAfold, kcal/mol) of any 120-letter window, step 30, negated so
  that higher means "folds more strongly";
- blast: the best blastn bit score against every Rfam seed sequence of every riboswitch family (the examples
  Infernal's models were built from), 0 if no hit at E < 0.001;
- blast_distant: the same, ignoring seeds at least 90% identical to the unit over the aligned stretch, which drops
  the unit's own sequence and close relatives' when Rfam's seed alignment happens to contain them.

The k-mer baseline is fitted later, in 6_score.py, because it is trained. Writes results/unit_baselines.tsv.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import RNA

from atlas import GENOMES, RESULTS, gc, read_genome, read_regions, read_riboswitch_seeds, revcomp

PAD, WINDOW, STEP = 100, 120, 30


def units() -> list[dict]:
    out = []
    for genome in GENOMES:
        seq = read_genome(genome)
        for r in read_regions(genome):
            lo, hi = max(0, r.start - PAD), min(len(seq), r.end + PAD)
            out.append({"unit": f"{r.rid}|+", "genome": genome, "rid": r.rid, "strand": "+", "seq": seq[lo:hi]})
            out.append(
                {"unit": f"{r.rid}|-", "genome": genome, "rid": r.rid, "strand": "-", "seq": revcomp(seq[lo:hi])}
            )
    return out


def best_fold(seq: str) -> float:
    starts = range(0, max(1, len(seq) - WINDOW + 1), STEP)
    return max(-RNA.fold(seq[s : s + WINDOW])[1] for s in starts)


def blast_scores(rows: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    seeds = {f"{fam}|{name}": seq for fam, members in read_riboswitch_seeds().items() for name, seq in members}
    with tempfile.TemporaryDirectory() as tmp:
        db, query = Path(tmp) / "seeds.fa", Path(tmp) / "units.fa"
        db.write_text("".join(f">{k}\n{v}\n" for k, v in seeds.items()))
        query.write_text("".join(f">{r['unit']}\n{r['seq']}\n" for r in rows))
        subprocess.run(["makeblastdb", "-in", str(db), "-dbtype", "nucl"], check=True, capture_output=True)
        out = subprocess.run(
            ["blastn", "-task", "blastn", "-word_size", "7", "-evalue", "1e-3", "-strand", "plus", "-num_threads", "8",
             "-query", str(query), "-db", str(db), "-outfmt", "6 qseqid sseqid bitscore pident",
             "-max_target_seqs", "50"],
            check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
    best: dict[str, float] = {}
    distant: dict[str, float] = {}
    for line in out.splitlines():
        unit, _, bits, pident = line.split("\t")
        best[unit] = max(best.get(unit, 0.0), float(bits))
        if float(pident) < 90:
            distant[unit] = max(distant.get(unit, 0.0), float(bits))
    return best, distant


def main() -> None:
    rows = units()
    with Pool(12) as pool:
        folds = pool.map(best_fold, [r["seq"] for r in rows], chunksize=32)
    blast, distant = blast_scores(rows)
    with (RESULTS / "unit_baselines.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["unit", "genome", "rid", "strand", "length", "gc", "fold", "blast", "blast_distant"])
        for r, fold in zip(rows, folds, strict=True):
            writer.writerow(
                [r["unit"], r["genome"], r["rid"], r["strand"], len(r["seq"]), f"{gc(r['seq']):.4f}", f"{fold:.2f}",
                 f"{blast.get(r['unit'], 0.0):.1f}", f"{distant.get(r['unit'], 0.0):.1f}"]
            )  # fmt: skip
    print(f"{len(rows)} units scored; {sum(1 for v in blast.values() if v > 0)} with a BLAST hit")


if __name__ == "__main__":
    main()
