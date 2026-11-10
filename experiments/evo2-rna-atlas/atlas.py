"""Shared pieces for the evo2-rna-atlas note: the genomes, the intergenic regions, Rfam labels, baselines, bootstrap.

Every step reads and writes plain files: FASTA and GFF from NCBI, Rfam's covariance models, tab-separated tables
in `results/`, and the Evo 2 outputs (sparse SAE codes) in `$EVO2_RNA_DIR`, which are too large to commit.

Words used the same way everywhere:

| word              | meaning                                                                                       |
|-------------------|-----------------------------------------------------------------------------------------------|
| intergenic region | a stretch between two protein-coding, tRNA, rRNA or pseudo genes, 50 bp or longer              |
| riboswitch hit    | an Infernal match to a Rfam family of type "Cis-reg; riboswitch", above the family's GA cutoff |
| positive region   | an intergenic region holding at least half of a riboswitch hit                                 |
| feature           | one of the 32,768 units of Goodfire's layer-26 sparse autoencoder on Evo 2 7B                  |
"""

from __future__ import annotations

import csv
import gzip
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = Path(os.environ.get("EVO2_RNA_DIR", Path.home() / ".cache" / "steerlab" / "evo2-rna"))
INFERNAL = Path(os.environ.get("INFERNAL_BIN", Path.home() / ".local" / "infernal" / "bin"))

# Four lab and food bacteria, none a pathogen. Features are chosen on the soil bacterium Bacillus subtilis 168 alone;
# the other three are held out, chosen to be far from it and from each other: a gut bacterium (E. coli K-12), a dairy
# starter (Lactococcus cremoris MG1363) and a freshwater bacterium with GC-rich DNA (Caulobacter crescentus NA1000).
GENOMES = {
    "bsub": {"name": "Bacillus subtilis 168", "accession": "GCF_000009045.1_ASM904v1"},
    "ecoli": {"name": "Escherichia coli K-12 MG1655", "accession": "GCF_000005845.2_ASM584v2"},
    "lcre": {"name": "Lactococcus cremoris MG1363", "accession": "GCF_000009425.1_ASM942v1"},
    "ccre": {"name": "Caulobacter crescentus NA1000", "accession": "GCF_000022005.1_ASM2200v1"},
}
MIN_REGION = 50
# Gene biotypes that bound an intergenic region. Noncoding RNA genes (misc_RNA, ncRNA, antisense) are not on the
# list: they are part of the dark matter this note searches (B. subtilis lists 93 noncoding RNAs as misc_RNA).
BOUNDING = {"protein_coding", "pseudogene", "tRNA", "rRNA", "tmRNA", "SRP_RNA", "RNase_P_RNA"}
RIBOSWITCH = "Cis-reg; riboswitch;"


@dataclass(frozen=True)
class Region:
    genome: str
    rid: str
    start: int  # 0-based, inclusive
    end: int  # 0-based, exclusive
    left: str  # nearest bounding gene on each side: "name:strand:biotype"
    right: str

    @property
    def length(self) -> int:
        return self.end - self.start


def genome_path(genome: str, ext: str) -> Path:
    return DATA / f"{GENOMES[genome]['accession']}_genomic.{ext}.gz"


def read_genome(genome: str) -> str:
    """The chromosome as one upper-case string (each of the four genomes is a single chromosome)."""
    with gzip.open(genome_path(genome, "fna"), "rt") as handle:
        lines = [line.strip() for line in handle if not line.startswith(">")]
    return "".join(lines).upper()


def read_genes(genome: str) -> list[dict]:
    """Gene and pseudogene lines from the RefSeq GFF: start (0-based), end, strand, name, biotype, product."""
    genes, products = [], {}
    with gzip.open(genome_path(genome, "gff"), "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = dict(kv.split("=", 1) for kv in cols[8].split(";") if "=" in kv)
            if cols[2] in ("gene", "pseudogene"):
                genes.append(
                    {
                        "id": attrs.get("ID", ""),
                        "start": int(cols[3]) - 1,
                        "end": int(cols[4]),
                        "strand": cols[6],
                        "name": attrs.get("Name", attrs.get("locus_tag", "?")),
                        "locus": attrs.get("locus_tag", ""),
                        "biotype": attrs.get("gene_biotype", "pseudogene" if cols[2] == "pseudogene" else "?"),
                    }
                )
            elif "Parent" in attrs and "product" in attrs:
                products.setdefault(attrs["Parent"], attrs["product"].replace("%2C", ",").replace("%3B", ";"))
    for gene in genes:
        gene["product"] = products.get(gene["id"], "")
    return genes


def intergenic_regions(genome: str, length: int) -> list[Region]:
    """Gaps of at least MIN_REGION bp between bounding genes, with the nearest gene recorded on each side."""
    bounding = sorted((g for g in read_genes(genome) if g["biotype"] in BOUNDING), key=lambda g: g["start"])
    covered = np.zeros(length, dtype=bool)
    for gene in bounding:
        covered[gene["start"] : gene["end"]] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~covered).astype(np.int8), [0]])) != 0)
    starts, ends = edges[0::2], edges[1::2]
    by_end = sorted(bounding, key=lambda g: g["end"])
    end_keys = np.array([g["end"] for g in by_end])
    start_keys = np.array([g["start"] for g in bounding])

    def tag(g):
        return f"{g['name']}:{g['strand']}:{g['biotype']}" if g else "none"

    regions = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < MIN_REGION:
            continue
        i = np.searchsorted(end_keys, start, side="right") - 1
        j = np.searchsorted(start_keys, end, side="left")
        left = by_end[i] if i >= 0 else None
        right = bounding[j] if j < len(bounding) else None
        regions.append(Region(genome, f"{genome}_{start + 1}_{end}", int(start), int(end), tag(left), tag(right)))
    return regions


def write_regions(path: Path, regions: list[Region]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["genome", "rid", "start", "end", "length", "left", "right"])
        for r in regions:
            writer.writerow([r.genome, r.rid, r.start, r.end, r.length, r.left, r.right])


def read_regions(genome: str | None = None) -> list[Region]:
    rows = list(csv.DictReader((RESULTS / "regions.tsv").open(), delimiter="\t"))
    return [
        Region(r["genome"], r["rid"], int(r["start"]), int(r["end"]), r["left"], r["right"])
        for r in rows
        if genome in (None, r["genome"])
    ]


def read_hits(kind: str | None = "riboswitch") -> list[dict]:
    """Infernal hits (results/rfam_hits.tsv). kind='riboswitch' keeps riboswitch families; None keeps all."""
    rows = list(csv.DictReader((RESULTS / "rfam_hits.tsv").open(), delimiter="\t"))
    for r in rows:
        r["start"], r["end"] = int(r["start"]), int(r["end"])
        r["score"] = float(r["score"])
    if kind == "riboswitch":
        rows = [r for r in rows if r["type"] == RIBOSWITCH]
    return rows


def label_regions(regions: list[Region], hits: list[dict]) -> np.ndarray:
    """1 for a region holding at least half of some hit on the same genome, else 0."""
    labels = np.zeros(len(regions), dtype=np.int8)
    for i, r in enumerate(regions):
        for h in hits:
            if h["genome"] != r.genome:
                continue
            overlap = min(r.end, h["end"]) - max(r.start, h["start"])
            if overlap >= 0.5 * (h["end"] - h["start"]):
                labels[i] = 1
                break
    return labels


def read_riboswitch_seeds() -> dict[str, list[tuple[str, str]]]:
    """Rfam 15.1 seeds of every riboswitch family, 50-400 letters, RNA written as DNA: {family: [(name, seq)]}."""
    types = {}
    with gzip.open(DATA / "family.txt.gz", "rt", encoding="latin-1") as handle:
        for line in handle:
            cols = line.rstrip("\n").split("\t")
            types[cols[0]] = (cols[1], cols[18])
    families, acc, seqs = {}, None, {}
    with gzip.open(DATA / "Rfam.seed.gz", "rt", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("#=GF AC"):
                acc, seqs = line.split()[2], {}
            elif line.startswith("//"):
                if acc and types.get(acc, ("", ""))[1] == RIBOSWITCH:
                    families[types[acc][0]] = [
                        (name, s.replace("U", "T"))
                        for name, s in seqs.items()
                        if 50 <= len(s) <= 400 and set(s) <= set("ACGU")
                    ]
                acc = None
            elif line.strip() and not line.startswith("#"):
                name, aligned = line.split()
                seqs[name] = seqs.get(name, "") + aligned.upper().replace(".", "").replace("-", "")
    return families


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def gc(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(1, len(seq))


def kmer_counts(seq: str, k: int) -> np.ndarray:
    index = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros(4**k)
    code = [index.get(c, -1) for c in seq]
    for i in range(len(code) - k + 1):
        window = code[i : i + k]
        if -1 in window:
            continue
        out[sum(c * 4 ** (k - 1 - j) for j, c in enumerate(window))] += 1
    return out


def bootstrap(stat, n: int, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(indices) over resampled indices 0..n-1."""
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def auroc(scores, labels) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(scores, labels) -> float:
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    hits = np.asarray(labels)[order].astype(bool)
    if hits.sum() == 0:
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())
