"""Label the intergenic regions with Infernal: every Rfam 15.1 family, each at its own gathering (GA) cutoff.

    python experiments/evo2-rna-atlas/3_infernal.py            # runs cmscan (about 15 minutes on 16 cores), then parses
    python experiments/evo2-rna-atlas/3_infernal.py --parse    # parses an existing table only

This is the reference answer the note is scored against, and the way Rfam annotates genomes: `cmscan --rfam --cut_ga
--nohmmonly`, with overlapping hits from families in the same clan resolved to the best one (`--clanin`, `--fmt 2`).
Writes results/rfam_hits.tsv: genome, family, accession, type (Rfam's family type, "Cis-reg; riboswitch;" for a
riboswitch), start (0-based), end, strand, score (bits) and evalue.
"""

from __future__ import annotations

import csv
import gzip
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from atlas import DATA, GENOMES, INFERNAL, RESULTS

TABLE = DATA / "intergenic.tblout"


def family_types() -> dict[str, str]:
    with gzip.open(DATA / "family.txt.gz", "rt", encoding="latin-1") as handle:
        return {cols[0]: cols[18] for cols in (line.rstrip("\n").split("\t") for line in handle)}


def run() -> None:
    command = [
        str(INFERNAL / "cmscan"), "--rfam", "--cut_ga", "--nohmmonly", "--fmt", "2",
        "--clanin", str(DATA / "Rfam.clanin"), "--cpu", "16", "--tblout", str(TABLE), "-o", "/dev/null",
        str(DATA / "Rfam.cm"), str(DATA / "intergenic_padded.fa"),
    ]  # fmt: skip
    subprocess.run(command, check=True)


def parse() -> list[dict]:
    types = family_types()
    hits = []
    for line in TABLE.open():
        if line.startswith("#"):
            continue
        cols = line.split()
        if cols[19] == "=":  # overlaps a better hit from the same clan
            continue
        genome, lo, _ = cols[3].split("|")
        a, b = int(cols[9]), int(cols[10])
        start, end = int(lo) + min(a, b) - 1, int(lo) + max(a, b)
        hits.append(
            {
                "genome": genome, "family": cols[1], "accession": cols[2], "type": types.get(cols[2], "?"),
                "start": start, "end": end, "strand": cols[11], "score": cols[16], "evalue": cols[17],
            }
        )  # fmt: skip
    # Neighbouring padded regions overlap, so one hit can be found twice with slightly different ends: keep the
    # best-scoring of any same-family, same-strand hits that overlap.
    kept: list[dict] = []
    for h in sorted(hits, key=lambda h: -float(h["score"])):
        if not any(
            k["genome"] == h["genome"] and k["accession"] == h["accession"] and k["strand"] == h["strand"]
            and min(k["end"], h["end"]) > max(k["start"], h["start"])
            for k in kept
        ):  # fmt: skip
            kept.append(h)
    return sorted(kept, key=lambda h: (h["genome"], h["start"]))


def main() -> None:
    if "--parse" not in sys.argv:
        run()
    hits = parse()
    with (RESULTS / "rfam_hits.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hits[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(hits)
    for genome in GENOMES:
        mine = [h for h in hits if h["genome"] == genome]
        switches = [h for h in mine if h["type"] == "Cis-reg; riboswitch;"]
        families = sorted({h["family"] for h in switches})
        print(f"{genome}: {len(mine)} Rfam hits, {len(switches)} riboswitches in {len(families)} families")
        print("   ", ", ".join(families))


if __name__ == "__main__":
    main()
