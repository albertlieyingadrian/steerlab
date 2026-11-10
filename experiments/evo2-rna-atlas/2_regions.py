"""Cut each genome into intergenic regions and write the FASTA that Infernal scans.

    python experiments/evo2-rna-atlas/2_regions.py

Writes results/regions.tsv (one row per region: coordinates and the gene on each side) and, in $EVO2_RNA_DIR,
intergenic_padded.fa: every region with 100 bp of each neighbouring gene, so a riboswitch that runs into the start
of its gene is still found whole.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from atlas import DATA, GENOMES, RESULTS, intergenic_regions, read_genome, write_regions

PAD = 100


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    regions = []
    with (DATA / "intergenic_padded.fa").open("w") as fasta:
        for genome in GENOMES:
            seq = read_genome(genome)
            found = intergenic_regions(genome, len(seq))
            regions += found
            lengths = np.array([r.length for r in found])
            print(
                f"{genome}: {len(seq):,} bp, {len(found):,} intergenic regions >= 50 bp, "
                f"{lengths.sum():,} bp ({lengths.sum() / len(seq):.1%}), median {np.median(lengths):.0f} bp"
            )
            for r in found:
                lo, hi = max(0, r.start - PAD), min(len(seq), r.end + PAD)
                fasta.write(f">{genome}|{lo}|{hi}\n{seq[lo:hi]}\n")
    write_regions(RESULTS / "regions.tsv", regions)


if __name__ == "__main__":
    main()
