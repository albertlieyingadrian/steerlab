"""Build the Evo 2 jobs, run them on Modal, and download the sparse SAE codes into $EVO2_RNA_DIR/evo2/.

    python experiments/evo2-rna-atlas/4_encode.py genomes     # four genomes, both strands (about 31 Mb of DNA)
    python experiments/evo2-rna-atlas/4_encode.py seeds       # Rfam seed sequences and their shuffles
    python experiments/evo2-rna-atlas/4_encode.py context     # riboswitch hits, real and shuffled in place, and tracks
    python experiments/evo2-rna-atlas/4_encode.py --dry-run genomes seeds context

Three jobs:

- genomes: each of the four chromosomes read in full on both strands, keeping positions in every intergenic region
  plus 100 bp of each neighbouring gene. The reverse strand is read as its own sequence (reverse complement), so an
  RNA on either strand is read 5' to 3', the way the cell makes it.
- seeds: up to 40 seed sequences of every Rfam riboswitch family with at least 12, each alone (RNA letters written
  as DNA), plus a dinucleotide shuffle of each (same length, same letter pairs, structure destroyed; esl-shuffle -d).
- context: every riboswitch hit in the four genomes with 1,500 bp of real genome before it and 300 after, read on
  the hit's own strand; once as it is, and twice with only the hit's letters replaced by a dinucleotide shuffle.
  Its companion job `tracks` keeps every position of the real and first shuffled versions, for the figures.

Costs: the H100 rate below times the wall-clock each job reports, written to results/compute.json.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from atlas import DATA, GENOMES, INFERNAL, RESULTS, read_genome, read_hits, read_regions, read_riboswitch_seeds, revcomp

OUT = DATA / "evo2"
MODAL = Path(sys.executable).parent / "modal"
# H100 on Modal, US dollars per hour, as listed at modal.com/pricing in November 2026: an estimate, not a bill.
H100_PER_HOUR = 3.95
PAD = 100
SEEDS_PER_FAMILY, MIN_SEEDS = 40, 12


def genome_items() -> list[tuple[str, list[dict]]]:
    jobs = []
    for genome in GENOMES:
        seq = read_genome(genome)
        length = len(seq)
        ranges = [(max(0, r.start - PAD), min(length, r.end + PAD)) for r in read_regions(genome)]
        jobs.append((f"genome_{genome}_plus", [{"id": f"{genome}+", "seq": seq, "ranges": ranges}]))
        minus = sorted((length - hi, length - lo) for lo, hi in ranges)
        jobs.append((f"genome_{genome}_minus", [{"id": f"{genome}-", "seq": revcomp(seq), "ranges": minus}]))
    return jobs


def shuffle(seqs: dict[str, str], seed: int) -> dict[str, str]:
    """Dinucleotide-preserving shuffles from Easel's esl-shuffle -d."""
    fasta = "".join(f">{k}\n{v}\n" for k, v in seqs.items())
    out = subprocess.run(
        [str(INFERNAL / "esl-shuffle"), "-d", "--seed", str(seed), "-"],
        input=fasta, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    shuffled, name = {}, None
    for line in out.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0].removesuffix("-shuffled")
            shuffled[name] = ""
        else:
            shuffled[name] += line.strip()
    return shuffled


def riboswitch_seeds() -> dict[str, list[tuple[str, str]]]:
    """Up to SEEDS_PER_FAMILY seeds of each riboswitch family that has at least MIN_SEEDS."""
    rng = random.Random(0)
    chosen = {}
    for family, members in sorted(read_riboswitch_seeds().items()):
        if len(members) < MIN_SEEDS:
            continue
        chosen[family] = sorted(rng.sample(members, min(SEEDS_PER_FAMILY, len(members))))
    return chosen


def seed_items() -> list[tuple[str, list[dict]]]:
    chosen = riboswitch_seeds()
    flat = {f"{family}|{name}": seq for family, members in chosen.items() for name, seq in members}
    shuffled = shuffle(flat, seed=1)
    items = []
    for key, seq in flat.items():
        items.append({"id": f"real|{key}", "seq": seq, "ranges": [(0, len(seq))], "keep_positions": False})
        items.append({"id": f"shuf|{key}", "seq": shuffled[key], "ranges": [(0, len(seq))], "keep_positions": False})
    (RESULTS / "seed_set.json").write_text(json.dumps({k: v for k, v in flat.items()}, indent=1) + "\n")
    print(f"seeds: {len(chosen)} families, {len(flat)} sequences")
    return [("seeds", items)]


def context_items() -> list[tuple[str, list[dict]]]:
    items, before, after = [], 1500, 300
    for genome in GENOMES:
        seq = read_genome(genome)
        hits = [h for h in read_hits("riboswitch") if h["genome"] == genome]
        cores = {f"{genome}|{h['start']}|{h['strand']}": h for h in hits}
        letters = {k: seq[h["start"] : h["end"]] for k, h in cores.items()}
        letters = {k: (v if cores[k]["strand"] == "+" else revcomp(v)) for k, v in letters.items()}
        shuffles = [shuffle(letters, seed=s) for s in (1, 2)]
        for key, h in cores.items():
            if h["strand"] == "+":
                left, right = seq[max(0, h["start"] - before) : h["start"]], seq[h["end"] : h["end"] + after]
            else:
                left = revcomp(seq[h["end"] : h["end"] + before])
                right = revcomp(seq[max(0, h["start"] - after) : h["start"]])
            span = (len(left), len(left) + len(letters[key]))
            for tag, core in [("real", letters[key]), ("shuf1", shuffles[0][key]), ("shuf2", shuffles[1][key])]:
                items.append({"id": f"{tag}|{key}|{h['family']}", "seq": left + core + right, "ranges": [span]})
    print(f"context: {len(items) // 3} riboswitch hits x 3 versions")
    # The same sequences, real and first shuffle, with every position kept: the tracks drawn in the figures.
    tracks = [{**i, "ranges": [(0, len(i["seq"]))]} for i in items if i["id"].startswith(("real|", "shuf1|"))]
    return [("context", items), ("tracks", tracks)]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    builders = {"genomes": genome_items, "seeds": seed_items, "context": context_items}
    jobs = [job for name in args for job in builders[name]()]
    for name, items in jobs:
        print(f"{name}: {len(items)} items, {sum(len(i['seq']) for i in items):,} letters")
    if "--dry-run" in sys.argv:
        return
    import modal

    encode = modal.Cls.from_name("evo2-rna-atlas", "Evo2SAE")().encode
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    calls = [(name, encode.spawn(name, items)) for name, items in jobs]
    log_path = RESULTS / "compute.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"h100_per_hour": H100_PER_HOUR, "jobs": {}}
    for name, call in calls:
        result = call.get()
        result["dollars"] = round(result["seconds"] / 3600 * H100_PER_HOUR, 2)
        log["jobs"][name] = result
        print(result)
        subprocess.run(
            [str(MODAL), "volume", "get", "--force", "evo2-rna-atlas", f"out/{name}.npz", str(OUT / f"{name}.npz")],
            check=True,
        )
    log["gpu_seconds"] = round(sum(j["seconds"] for j in log["jobs"].values()))
    log["dollars_compute_only"] = round(sum(j["dollars"] for j in log["jobs"].values()), 2)
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(f"all jobs done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
