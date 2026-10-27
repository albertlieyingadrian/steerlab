"""Draw each research note's hero image and thumbnail from that experiment's own steering direction.

    .venv/bin/python experiments/shared/make_art.py

The direction's numbers (one per dimension of the model's activations, e.g. 4,096 for Qwen3-8B) are laid out on a
square grid and smoothed into a flow field. Streamlines enter from the left and run straight until the steering layer
(the vertical rule); after it they follow the field, so the bend is the shape of the real direction, not a drawing of
one. Same inputs, same picture: nothing is random except a fixed seed for where the lines start.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = ROOT / "packages/steerlab/src/steerlab/data/demo/directions"
HERE = Path(__file__).resolve().parent

GROUND = "#15110e"  # Ember ground, from steerlab's web demo
STREAM = LinearSegmentedColormap.from_list("stream", ["#6f675f", "#a39a90", "#f2b48e", "#e07a3f", "#c9561c"])

# experiment -> (model key, concept, how far after the layer the stream is fully steered, bend strength)
ART = {
    "golden-gate": ("qwen8b", "golden_gate", 0.18, 1.9),
    "english-lock": ("qwen8b", "chinese", 0.14, 1.5),
}


def field(model: str, concept: str) -> np.ndarray:
    """The direction's numbers on a square grid, smoothed and scaled to [-1, 1]."""
    vector = np.load(DIRECTIONS / model / f"{concept}.npz")["vector"].astype(float)
    side = int(np.sqrt(vector.size))
    grid = vector[: side * side].reshape(side, side)
    for _ in range(6):  # a cheap Gaussian: repeated 3x3 box blur
        padded = np.pad(grid, 1, mode="wrap")
        grid = sum(padded[i : i + side, j : j + side] for i in range(3) for j in range(3)) / 9
    return grid / np.abs(grid).max()


def sample(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    side = grid.shape[0]
    return grid[np.clip((y * (side - 1)).astype(int), 0, side - 1), np.clip((x * (side - 1)).astype(int), 0, side - 1)]


def draw(name: str, size: tuple[int, int], out: Path) -> None:
    model, concept, ramp, bend = ART[name]
    grid = field(model, concept)
    width, height = size
    aspect = height / width
    layer = 0.36
    rng = np.random.default_rng(7)
    starts = np.sort(rng.uniform(0.2 * aspect, 0.8 * aspect, 170))
    steps, dx = 420, 1.0 / 420
    segments, colours = [], []
    for y0 in starts:
        xs, ys = [0.0], [float(y0)]
        x, y = 0.0, float(y0)
        for _ in range(steps):
            on = np.clip((x - layer) / ramp, 0, 1)
            angle = bend * on * float(sample(grid, np.array([x]), np.array([y / aspect]))[0]) * 1.1
            x += dx * np.cos(angle) * 1.0
            y += dx * np.sin(angle) * 1.6
            if not (0 <= y <= aspect) or x > 1:
                break
            xs.append(x)
            ys.append(y)
        points = np.column_stack([xs, ys])
        segments.extend(np.stack([points[:-1], points[1:]], axis=1))
        colours.extend(np.clip((np.array(xs[:-1]) - layer + 0.05) / 0.5, 0, 1))
    fig = plt.figure(figsize=(width / 200, height / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)
    glow = np.exp(
        -(((np.linspace(0, 1, 400)[None, :] - 0.7) / 0.35) ** 2 + ((np.linspace(0, 1, 225)[:, None] - 0.5) / 0.5) ** 2)
    )
    ax.imshow(
        glow, extent=(0, 1, 0, aspect), cmap=LinearSegmentedColormap.from_list("g", [GROUND, "#3a2214"]), alpha=0.9
    )
    lines = LineCollection(segments, cmap=STREAM, linewidths=0.55, alpha=0.85)
    lines.set_array(np.array(colours))
    ax.add_collection(lines)
    ax.axvline(layer, color="#4a86e8", linewidth=1.1, alpha=0.9)  # the steering layer, in the fader's blue
    ax.set_xlim(0, 1)
    ax.set_ylim(0, aspect)
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=GROUND, pil_kwargs={"quality": 90})
    plt.close(fig)
    print(f"{name}: {model} {concept} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    for name in ART:
        draw(name, (1600, 900), HERE.parent / name / "report" / "figures" / "hero.jpg")
        draw(name, (720, 480), HERE / "thumbs" / f"{name}.jpg")
