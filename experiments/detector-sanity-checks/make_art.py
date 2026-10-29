"""Draw this note's data-art figure from the Instructed-Pairs probe's own direction, with the shared steerlab art style.

    python experiments/detector-sanity-checks/make_art.py

Same picture rules as experiments/shared/make_art.py: the direction's 4,096 numbers (one per dimension of
Llama-3.1-8B's residual stream at layer 8, the layer the probe reads) are laid on a square grid and smoothed into a
flow field; streamlines run straight until the vertical rule and then follow the field. Nothing is drawn by hand.
The direction is detector D1's weight vector, results/d1_probe.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))
import make_art  # noqa: E402


def probe_field(_model: str, _concept: str) -> np.ndarray:
    vector = np.load(HERE / "results" / "d1_probe.npz")["w"].astype(float)
    side = int(np.sqrt(vector.size))
    grid = vector[: side * side].reshape(side, side)
    for _ in range(6):  # the shared script's cheap Gaussian: repeated 3x3 box blur
        padded = np.pad(grid, 1, mode="wrap")
        grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
    return grid / np.abs(grid).max()


if __name__ == "__main__":
    make_art.ART["detector-sanity-checks"] = ("llama8b", "instructed-pairs probe", 0.16, 1.7)
    make_art.field = probe_field
    out = HERE / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    make_art.draw("detector-sanity-checks", (1600, 900), out / "data-art.jpg")
