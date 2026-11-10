"""Pick 40 Google Scanned Objects, generate each with every model on Modal, score them, and fetch the renders.

    modal deploy modal_i3d.py && python 1_generate.py             # everything
    python 1_generate.py --models idol                              # one model (generation and scoring)

Objects: 70 names drawn at random (seed 0) from the 1,033 in Google Research's Gazebo Fuel collection; any whose
shortest side is under 10% of its longest (towels, plates, flat boxes seen edge-on) is dropped as a poor single-image
test, and the first 40 left are used. Writes results/gso_objects.csv, results/gso_timing.csv (seconds and memory per
model and object) and results/gso_scores.csv (every score per system and object), and copies the input images,
judge sheets and turntable videos to $I3D_DIR/gso/.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import modal
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA, RESULTS

N_DRAW, N_KEEP, SEED, MIN_RATIO = 70, 40, 0, 0.10
MODELS = {
    "trellis": ["trellis_mesh", "trellis_splats"],
    "triposr": ["triposr"],
    "sf3d": ["sf3d"],
    "idol": ["idol"],
    "trellis2": ["trellis2"],
}
CHECKS = ["gt_bumpy", "gt_squashed", "gt_blurred"]
CHUNKS = {"trellis2": 4}


def gso_names() -> list[str]:
    names, page = [], 1
    while True:
        url = f"https://fuel.gazebosim.org/1.0/GoogleResearch/models?per_page=100&page={page}"
        try:
            batch = json.load(urllib.request.urlopen(url, timeout=60))
        except urllib.error.HTTPError:
            break
        if not batch:
            break
        names += [m["name"] for m in batch]
        page += 1
    return sorted(set(names))


def choose_objects() -> list[str]:
    path = RESULTS / "gso_objects.csv"
    if path.exists():
        table = pd.read_csv(path)
        return list(table[table.used].object)
    names = gso_names()
    draw = random.Random(SEED).sample(names, N_DRAW)
    prepare = modal.Function.from_name("image-to-3d-eval", "prepare")
    rows = pd.DataFrame(prepare.remote(draw))
    rows["ratio"] = rows.extent.map(lambda e: min(e) / max(e))
    rows["used"] = False
    keep = rows[rows.ratio >= MIN_RATIO].index[:N_KEEP]
    rows.loc[keep, "used"] = True
    rows.insert(0, "draw", range(len(rows)))
    rows.to_csv(path, index=False, float_format="%.4f")
    print(f"{len(names)} names; drew {N_DRAW}; {(rows.ratio < MIN_RATIO).sum()} flat; using {rows.used.sum()}")
    return list(rows[rows.used].object)


def generate(objects: list[str], models: list[str]) -> None:
    calls = []
    for m in models:
        fn = modal.Function.from_name("image-to-3d-eval", f"gen_{m}")
        # TRELLIS.2's export takes about 6 minutes an object: 4 containers share the objects (each warms up once).
        n = CHUNKS.get(m, 1)
        calls += [(m, fn.spawn(objects[k::n])) for k in range(n)]
    path = RESULTS / "gso_timing.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model"])
    new = []
    for m, call in calls:
        try:
            new += call.get()
        except Exception as err:  # noqa: BLE001, RUF100 - one model failing should not lose the others
            print(f"{m}: generation failed: {err!r}"[:400])
    if new:
        new = pd.DataFrame(new)
        table = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
        table.to_csv(path, index=False, float_format="%.3f")


def score(objects: list[str], systems: list[str]) -> None:
    scorer = modal.Cls.from_name("image-to-3d-eval", "Scorer")()
    rows = [r for batch in scorer.score.map(objects, kwargs={"systems": systems}) for r in batch]
    path = RESULTS / "gso_scores.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["system"])
    table = pd.concat([old[~old.system.isin(set(systems) | {"gt"})], pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(path, index=False, float_format="%.5f")


def fetch_renders() -> None:
    dest = DATA / "gso"
    dest.mkdir(parents=True, exist_ok=True)
    for remote in ("inputs", "renders"):
        command = [sys.executable, "-m", "modal", "volume", "get", "--force", "image-to-3d-eval", remote, str(dest)]
        subprocess.run(command, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()
    objects = choose_objects()
    if not args.skip_generate:
        generate(objects, args.models)
    systems = [s for m in args.models for s in MODELS[m]]
    if set(args.models) == set(MODELS):
        systems += CHECKS
    score(objects, systems)
    fetch_renders()


if __name__ == "__main__":
    main()
