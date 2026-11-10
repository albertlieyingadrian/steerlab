"""Download Norman et al. 2019 as processed by the GEARS authors, check its hash, and unpack the one file we need.

    python experiments/perturbation-baselines/1_fetch.py

The archive is 169 MB from Harvard Dataverse (CC0); the AnnData inside unpacks to 2.2 GB. Both go to $NORMAN_DIR
(default ~/.cache/steerlab/norman), outside the repo. A file already there with the right hash is reused.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

URL = "https://dataverse.harvard.edu/api/access/datafile/6154020"
SHA256 = "c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877"
MEMBER = "norman/perturb_processed.h5ad"


def data_dir() -> Path:
    return Path(os.environ.get("NORMAN_DIR", Path.home() / ".cache" / "steerlab" / "norman")).expanduser()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    folder = data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "norman.zip"
    target = folder / MEMBER
    if not archive.exists() and not target.exists():
        print(f"downloading {URL} (169 MB) ...")
        urllib.request.urlretrieve(URL, archive)
    if archive.exists():
        if sha256(archive) != SHA256:
            raise SystemExit(f"{archive} does not match the expected sha256; delete it and rerun")
        print(f"archive ok: {archive.name}, sha256 {SHA256[:12]}...")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped, zipped.open(MEMBER) as source, target.open("wb") as sink:
            while chunk := source.read(1 << 24):
                sink.write(chunk)
    print(f"AnnData: {MEMBER} ({target.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
