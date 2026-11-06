"""Small helpers shared by every script here: self-check lines, the work folder, result files, booting a bridge."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DIRECTION_FILE = RESULTS / "sandbagging_direction_l14.npz"


def check(name: str, condition: object) -> bool:
    """Print a self-check line and return whether it passed."""
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def work_dir() -> Path:
    """Large or regenerable files (cached patching grids, the rebuilt direction) go to a work folder outside the
    repository: ~/.cache/steerlab/transformerlens-circuits by default, or $TL_CIRCUITS_WORK. Small results live in
    results/."""
    default = Path.home() / ".cache" / "steerlab" / "transformerlens-circuits"
    path = Path(os.environ.get("TL_CIRCUITS_WORK", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(name: str, data: object) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"  wrote results/{name}")
    return path


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def write_grid_csv(name: str, grid, row_labels: list[str], col_labels: list[str], corner: str = "row") -> Path:
    """A 2-D grid (tensor or nested list) as a CSV with labelled rows and columns, 4 decimals."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    rows = grid.tolist() if hasattr(grid, "tolist") else grid
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([corner, *col_labels])
        for label, row in zip(row_labels, rows, strict=True):
            writer.writerow([label, *(f"{float(v):.4f}" for v in row)])
    print(f"  wrote results/{name}")
    return path


def read_grid_csv(name: str) -> tuple[list[str], list[str], list[list[float]]]:
    with (RESULTS / name).open() as handle:
        rows = list(csv.reader(handle))
    return [r[0] for r in rows[1:]], rows[0][1:], [[float(v) for v in r[1:]] for r in rows[1:]]


def boot(model_name: str, compatibility_mode: bool = True):
    """A Hugging Face model through TransformerLens's TransformerBridge, on the device TransformerLens picks (CPU on a
    Mac unless TRANSFORMERLENS_ALLOW_MPS=1), float32, eval mode, gradients off. Compatibility mode folds LayerNorm
    into the next linear layer and centres the weights that write into the residual stream, which is what makes
    attribution a clean dot product. Returns (model, device)."""
    import torch
    from transformer_lens import TransformerBridge
    from transformer_lens.utilities.devices import get_device

    torch.set_grad_enabled(False)
    device = get_device()
    model = TransformerBridge.boot_transformers(model_name, device=device)
    if compatibility_mode:
        model.enable_compatibility_mode(disable_warnings=True)
    model.eval()
    return model, device
