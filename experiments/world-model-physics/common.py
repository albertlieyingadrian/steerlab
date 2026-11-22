"""
Shared plumbing for the world-model-physics study: paths, seeds, data splits, model loading.

The study reuses an earlier 2D world-model project's pieces, copied into part18/: the exact event-driven simulator
(physics.py), the tiny GPT world model (worldmodel.py) and the probing / interchange helpers (probing.py, which needs
sandbagkit from packages/). This file puts part18/ on sys.path, so every script can `import physics, worldmodel,
probing`.

Two worlds:
    base   the base project's world: 3 balls, elastic, no gravity, no friction (exact simulator,
           physics.py). Experiments A, C.
    const  the same box with gravity g (pulling -y) and linear drag gamma varied per episode (sim_const.py). Exp. B.

Big files (datasets, checkpoints, activations) go to data/ (gitignored); small result tables go to results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PART18 = HERE / "part18"  # copies of the base 2D world-model project's simulator, model and probing helpers
sys.path.insert(0, str(PART18))
sys.path.insert(0, str(HERE))

DATA = HERE / "data"
MODELS = DATA / "models"
RESULTS = HERE / "results"
FIGURES = HERE / "report" / "figures"
for _p in (DATA, MODELS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# One seed per job, fixed in PREREGISTRATION.md before any main run.
SEEDS = {
    # base world splits: the same seeds as the base project, so the data is identical
    "base_train": 18001,
    "base_val": 18002,
    "base_test": 18003,
    "base_ood": 18004,
    "base_sft": 26050,  # fresh simulator episodes for the SFT arm of experiment C (never seen in pre-training)
    "base_rlprefix": 26051,  # the prefixes RL rolls out from (simulator starts, never seen in pre-training)
    # constants world splits
    "const_train": 26101,
    "const_val": 26102,
    "const_test": 26103,
    "const_ood_g": 26104,
    "const_ood_drag": 26105,
    # model seeds (init + batch order); three per world
    "model_base": (2610, 2611, 2612),
    "model_const": (2620, 2621, 2622),
    # analysis
    "probe": 2630,
    "steer": 2631,
    "rl": 2640,
    "sft": 2641,
}

CONFIG = {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_mlp": 512}  # the base project's Model A, unchanged
EPOCHS = 40

BASE_PLAN = {  # The base project's splits, plus two fresh splits for experiment C
    "train": (20_000, (0.5, 2.0)),
    "val": (2_000, (0.5, 2.0)),
    "test": (2_000, (0.5, 2.0)),
    "ood": (1_000, (2.5, 3.5)),
    "sft": (20_000, (0.5, 2.0)),
    "rlprefix": (4_000, (0.5, 2.0)),
}


def save_json(name_or_path, data) -> Path:
    path = Path(name_or_path)
    if not path.is_absolute():
        path = RESULTS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")
    return path


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def pick_device():
    import torch

    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def save_npz(path: Path, arrays: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def base_split(name: str) -> dict:
    """A base-world split (exact simulator), simulated once and cached in data/."""
    path = DATA / f"base_{name}.npz"
    if path.exists():
        return load_npz(path)
    import physics as ph

    n, energy_range = BASE_PLAN[name]
    print(f"   simulating base split {name!r}: {n:,} episodes (cached afterwards)", flush=True)
    arrays, worst = ph.make_split(n, SEEDS[f"base_{name}"], energy_range)
    assert worst["energy_drift"] < 1e-9 and worst["outside"] < 1e-9, worst
    save_npz(path, arrays)
    return arrays


def const_split(name: str) -> dict:
    """A constants-world split (gravity + drag), simulated once and cached in data/."""
    path = DATA / f"const_{name}.npz"
    if path.exists():
        return load_npz(path)
    import sim_const as sc

    n, g_range, drag_range = sc.PLAN[name]
    print(f"   simulating constants split {name!r}: {n:,} episodes (cached afterwards)", flush=True)
    arrays = sc.make_split(n, SEEDS[f"const_{name}"], g_range, drag_range)
    save_npz(path, arrays)
    return arrays


def model_path(world: str, seed: int, arm: str = "pre") -> Path:
    return MODELS / f"{world}_s{seed}_{arm}.pt"


def load_model(world: str, seed: int, arm: str = "pre", device="cpu"):
    import worldmodel as wm

    return wm.load_checkpoint(model_path(world, seed, arm), device)
