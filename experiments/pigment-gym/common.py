"""Small helpers shared by every script here: self-check lines, the work folder, the device."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results"


def check(name, condition):
    """Print a self-check line and return whether it passed."""
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def outputs_dir():
    """Large or regenerable files (samples, renders, checkpoints) go to a work folder outside the repository:
    ~/.cache/steerlab/pigment-gym by default, or $PIGMENT_GYM_WORK. Small results live in results/."""
    path = Path(os.environ.get("PIGMENT_GYM_WORK", Path.home() / ".cache" / "steerlab" / "pigment-gym"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
