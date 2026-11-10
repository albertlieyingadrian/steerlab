"""Download the two genomes (NCBI RefSeq), Rfam 15.1's covariance models and family table, and Rfam's seed alignments.

    python experiments/evo2-rna-atlas/1_fetch.py

Everything lands in $EVO2_RNA_DIR (default ~/.cache/steerlab/evo2-rna), about 330 MB, once. Files already present
are kept. Then the models are pressed for cmscan (needs Infernal 1.1.5 in $INFERNAL_BIN).
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from atlas import DATA, GENOMES, INFERNAL

NCBI = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF"
RFAM = "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1"


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    print("downloading", url)
    with urllib.request.urlopen(url) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for info in GENOMES.values():
        acc = info["accession"]
        digits = acc.split("_")[1]
        folder = f"{NCBI}/{digits[0:3]}/{digits[3:6]}/{digits[6:9]}/{acc}"
        for ext in ("fna", "gff"):
            fetch(f"{folder}/{acc}_genomic.{ext}.gz", DATA / f"{acc}_genomic.{ext}.gz")
    for name in ("Rfam.cm.gz", "Rfam.clanin", "Rfam.seed.gz", "database_files/family.txt.gz"):
        fetch(f"{RFAM}/{name}", DATA / Path(name).name)
    cm = DATA / "Rfam.cm"
    if not cm.exists():
        with gzip.open(DATA / "Rfam.cm.gz", "rb") as src, cm.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    if not (DATA / "Rfam.cm.i1m").exists():
        subprocess.run([str(INFERNAL / "cmpress"), str(cm)], check=True, stdout=subprocess.DEVNULL)
    print("data ready in", DATA)


if __name__ == "__main__":
    main()
