"""Shared pieces for the alphagenome2-ms4a note: the pre-registered constants, small web clients with on-disk caches,
a tabix reader that needs no compiled library, and the print helpers every script uses.

Everything fetched from the web is cached under `~/.cache/steerlab/alphagenome2-ms4a/` (or `$AG2_CACHE`), so a rerun
is offline and free. AlphaGenome responses are cached there too, one JSON file per (model, condition, variant): the
API is free but has a quota, and a rerun should never spend it twice. The key is read from `ALPHAGENOME_API_KEY` in
the environment or in the repo's `.env`, and never printed.

The constants below were fixed in the pre-registration before any control variant was scored (README.md).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import platform
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from importlib import metadata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DERIVED = HERE / "results"
FIGURES = HERE / "report" / "figures"
SMOKE = os.environ.get("SMOKE") == "1"


def outputs_dir() -> Path:
    """The cache folder for downloads and API responses (not committed)."""
    path = Path(os.environ.get("AG2_CACHE", Path.home() / ".cache" / "steerlab" / "alphagenome2-ms4a"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES


def rel(path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def require(path, script: str) -> Path:
    target = Path(path)
    if not target.exists():
        print(f"missing {rel(target)}: run {script} first")
        sys.exit(2)
    return target


def optional_env(name: str) -> str | None:
    """An environment variable, else a KEY=VALUE line in the repo's .env. The value is never printed."""
    if os.environ.get(name):
        return os.environ[name]
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip("\"'")
    return None


def check(name: str, condition: object) -> bool:
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def expect(name: str, observed: object, expected: object, source: str) -> None:
    print(f"  ⇢ {name}: observed {observed}, expected {expected} ({source})")


def header(seed: int = 0, packages=(), **_ignored) -> None:
    versions = []
    for package in packages:
        try:
            versions.append(f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{package} missing")
    print("=== run header ===")
    print(f"date: {_dt.datetime.now(_dt.UTC):%Y-%m-%dT%H:%M:%SZ}")
    print(f"platform: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    print(f"packages: {', '.join(versions)}")
    print(f"seed: {seed}  SMOKE: {int(SMOKE)}")
    print("==================")


# --- The locus --------------------------------------------------------------------------------------------------

CHROM = "chr11"
# The MS4A cluster sits around chr11:59.9-60.4 Mb (GRCh38). Variants are drawn from this window; each is scored in
# the 1 Mb of DNA centred on it, so every one of them sees MS4A6A and MS4A4A.
WINDOW = (59_850_000, 60_550_000)
# Ensembl canonical transcript of MS4A6A (ENST00000528851). It is on the minus strand, so its TSS is the end.
GENE = "MS4A6A"
GENE_ID = "ENSG00000110077"
SECOND_GENE = "MS4A4A"
TSS = 60_183_083

# --- The GWAS side ----------------------------------------------------------------------------------------------

# Open Targets Platform GWAS studies of Alzheimer's disease whose fine-mapped credible sets are pooled: the large
# case-control or proxy meta-analyses listed for MONDO_0004975. FinnGen is the one fine-mapped with in-sample LD.
OT_STUDIES = {
    "GCST90027158": "Bellenguez 2022",
    "GCST90012877": "Schwartzentruber 2021",
    "GCST007320": "Jansen 2019",
    "GCST90301303": "Lake 2023",
    "FINNGEN_R12_G6_AD_WIDE": "FinnGen R12 (AD, wide)",
}

# Lead variants named in the literature, with the allele the source says raises risk when it names one. `None`
# means the source gives an odds ratio without naming the allele: the variant is scored but carries no direction.
LITERATURE_LEADS = {
    "rs1582763": {
        "source": "Bellenguez 2022 lead (GWAS Catalog GCST90027158: A, OR 0.91); Deming 2019 sTREM2",
        "risk": "G",
    },
    "rs983392": {"source": "Lambert 2013 (GWAS Catalog GCST002245: A, OR 1.11)", "risk": "A"},
    "rs6591561": {
        "source": "Deming 2019 second sTREM2 signal, MS4A4A missense (no AD allele in the Catalog)",
        "risk": None,
    },
    "rs610932": {"source": "Hollingworth 2011 (GWAS Catalog GCST001025, allele not given)", "risk": None},
    "rs4938933": {"source": "Naj 2011 (GWAS Catalog GCST001026, allele not given)", "risk": None},
    "rs1562990": {"source": "Antunez 2011 (GWAS Catalog GCST001087, allele not given)", "risk": None},
}

# --- Controls ---------------------------------------------------------------------------------------------------

CONTROLS_PER_GWAS = 2
CONTROL_MIN_MAF = 0.01
CONTROL_MAX_R2 = 0.1  # with every GWAS-set variant, in 1000 Genomes EUR
CLUMP_R2 = 0.5  # variants this correlated count as one unit when bootstrapping
SEED = 0

# --- Conditions -------------------------------------------------------------------------------------------------

# Reference conditions in AlphaGenome 2 (`supported_conditions()`), each "observed" for single-cell expression.
CONDITIONS = {
    "MTG microglia": "UBERON:0002771_CL:0000129",  # primary: the SEA-AD brain region
    "DLPFC microglia": "UBERON:0009834_CL:0000129",  # replicate microglia, another cortex
    "MTG astrocyte": "UBERON:0002771_CL:0002605",
    "MTG L2/3 IT neuron": "UBERON:0002771_CL:4030059",
    "MTG oligodendrocyte": "UBERON:0002771_CL:0000128",
    "liver hepatocyte": "UBERON:0002107_CL:0000182",
    "blood CD14+ monocyte": "UBERON:0000178_CL:0001054",  # myeloid comparator, expected to share
}
PRIMARY = "MTG microglia"
MICROGLIA = ("MTG microglia", "DLPFC microglia")
NON_MICROGLIA = ("MTG astrocyte", "MTG L2/3 IT neuron", "MTG oligodendrocyte", "liver hepatocyte")
ZERO_SHOT = "iPSC microglia (GSE289721, zero-shot)"

# Zero-shot: condition on GSE289721 iPSC-derived microglia, vehicle, non-targeting guides, experiment XP1.
CONDITIONING_CELLS = 48
CONDITIONING_DRAWS = 3

# The preprint's operating threshold for calling a variant-gene effect (Fig. 5d): |log fold change| > 0.1.
CALL_LFC = 0.1


# --- Small cached web clients -----------------------------------------------------------------------------------


def cached(path: Path, fetch) -> bytes:
    if path.exists():
        return path.read_bytes()
    data = fetch()
    path.write_bytes(data)
    return data


def http(url: str, *, data: bytes | None = None, headers: dict | None = None, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, headers={"User-Agent": "steerlab-note", **(headers or {})})
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def opentargets(query: str, variables: dict, cache_name: str) -> dict:
    path = outputs_dir() / f"ot_{cache_name}.json"
    body = json.dumps({"query": query, "variables": variables}).encode()
    raw = cached(
        path,
        lambda: http(
            "https://api.platform.opentargets.org/api/v4/graphql",
            data=body,
            headers={"Content-Type": "application/json"},
        ),
    )
    out = json.loads(raw)
    if "errors" in out:
        path.unlink()
        raise RuntimeError(out["errors"])
    return out["data"]


def gwas_catalog(rsid: str) -> dict:
    path = outputs_dir() / f"gwascat_{rsid}.json"
    url = f"https://www.ebi.ac.uk/gwas/rest/api/v2/associations?rs_id={rsid}&size=200"
    return json.loads(cached(path, lambda: http(url)))


def ensembl_variant(rsid: str) -> dict:
    path = outputs_dir() / f"ensembl_{rsid}.json"
    url = f"https://rest.ensembl.org/variation/human/{rsid}?content-type=application/json"
    return json.loads(cached(path, lambda: http(url)))


# --- A tabix reader in plain Python -----------------------------------------------------------------------------

KG_BASE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/"
    "20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr11.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)
KG_PEDIGREE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/"
    "20130606_g1k_3202_samples_ped_population.txt"
)


def bgzf_decompress(data: bytes) -> bytes:
    """BGZF is a series of gzip members; zlib reads one member per decompressobj. A cut-off last block is dropped."""
    out = io.BytesIO()
    while data:
        member = zlib.decompressobj(31)
        try:
            chunk = member.decompress(data)
        except zlib.error:
            break
        if not member.eof:
            break
        out.write(chunk)
        data = member.unused_data
    return out.getvalue()


def linear_offsets(tbi: bytes, chrom: str) -> list[int]:
    """The tabix linear index for one sequence: the smallest virtual offset of any record in each 16 kb tile."""
    if tbi[:4] != b"TBI\x01":
        raise ValueError("not a tabix index")
    n_ref = struct.unpack_from("<i", tbi, 4)[0]
    l_nm = struct.unpack_from("<i", tbi, 32)[0]
    names = tbi[36 : 36 + l_nm].split(b"\x00")[:n_ref]
    pos = 36 + l_nm
    for name in names:
        (n_bin,) = struct.unpack_from("<i", tbi, pos)
        pos += 4
        for _ in range(n_bin):
            _bin, n_chunk = struct.unpack_from("<Ii", tbi, pos)
            pos += 8 + 16 * n_chunk
        (n_intv,) = struct.unpack_from("<i", tbi, pos)
        pos += 4
        offsets = list(struct.unpack_from(f"<{n_intv}Q", tbi, pos))
        pos += 8 * n_intv
        if name.decode() == chrom:
            return offsets
    raise KeyError(chrom)


def fetch_vcf_slice(start: int, end: int) -> tuple[list[str], Path]:
    """Records of the 1000 Genomes chr11 panel with start <= POS <= end, by one HTTP range request."""
    raw_path = outputs_dir() / f"1kg_chr11_{start}_{end}.bgzf"
    tbi = bgzf_decompress(cached(outputs_dir() / "1kg_chr11.tbi", lambda: http(KG_BASE + ".tbi")))
    offsets = linear_offsets(tbi, CHROM)
    first = offsets[start >> 14]
    stop = offsets[min((end >> 14) + 2, len(offsets) - 1)] >> 16
    begin = first >> 16
    raw = cached(raw_path, lambda: http(KG_BASE, headers={"Range": f"bytes={begin}-{stop + 131_072}"}))
    text = bgzf_decompress(raw)[first & 0xFFFF :].decode()
    rows: list[str] = []
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        head = line.split("\t", 2)
        if len(head) < 3 or head[0] != CHROM:
            continue
        position = int(head[1])
        if position > end:
            break
        if position >= start:
            rows.append(line)
    if rows and int(rows[-1].split("\t", 2)[1]) < end - 20_000:
        raise RuntimeError("the range request stopped short of the window end")
    return rows, raw_path


def kg_header_samples() -> list[str]:
    path = outputs_dir() / "1kg_header_samples.txt"

    def fetch() -> bytes:
        head = bgzf_decompress(http(KG_BASE, headers={"Range": "bytes=0-400000"}))
        for line in head.decode(errors="ignore").split("\n"):
            if line.startswith("#CHROM"):
                return "\n".join(line.split("\t")[9:]).encode()
        raise RuntimeError("no #CHROM line in the first 400 kB")

    return cached(path, fetch).decode().split("\n")


def kg_eur_columns(samples: list[str]) -> list[int]:
    """European samples who are nobody's child in the pedigree: close to the unrelated 2504 set."""
    ped = cached(outputs_dir() / "1kg_pedigree.txt", lambda: http(KG_PEDIGREE)).decode().splitlines()
    keep = set()
    for line in ped[1:]:
        _family, sample, father, mother, _sex, _pop, superpop = line.split()
        if superpop == "EUR" and father == "0" and mother == "0":
            keep.add(sample)
    return [i for i, s in enumerate(samples) if s in keep]


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def r2_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Pearson correlation of dosages, rows of `a` against rows of `b`."""
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-12)
    return (a @ b.T / a.shape[1]) ** 2
