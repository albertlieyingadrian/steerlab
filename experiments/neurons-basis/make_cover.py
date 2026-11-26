"""The note's cover, drawn from data: every MLP neuron of the password-locked Qwen2.5-1.5B at the token where it
picks its answer, for one question it sandbags (no password).

    .venv/bin/python experiments/neurons-basis/make_cover.py

Rows are the 28 layers, columns the 8,960 neurons of each (binned 8 to a pixel, keeping the strongest). Only the
firings among each layer's strongest 7.8% at this token are lit (orange positive, blue negative); the rest stay dark,
which is what layer top-K sparsity keeps. White rings mark the neurons the behaviour-trained monitor relies on.
"""

# ruff: noqa: E501  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NEURONS_BASIS_WORK", Path.home() / ".cache" / "steerlab" / "neurons-basis")) / "organism"
GROUND = "#15110e"
ROW_ID = "quiz-001-nopw"  # a no-password row the locked model sandbags (label 1), named in the caption


def main() -> None:
    rows = [
        json.loads(x)
        for x in (
            Path(os.environ.get("DETECTOR_SANITY_WORK", Path.home() / ".cache" / "steerlab" / "detector-sanity-checks"))
            / "sets"
            / "organism_quiz.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    i = next(k for k, r in enumerate(rows) if r["id"] == ROW_ID)
    assert rows[i]["label"] == 1 and rows[i]["condition"] == "nopw"
    a = np.load(WORK / "acts_organism_quiz.npz")["last_prompt"][i].astype(np.float32)  # (28, 8960)
    keep = np.zeros_like(a, dtype=bool)
    k = round(0.078 * a.shape[1])
    for layer in range(a.shape[0]):
        keep[layer, np.argsort(-np.abs(a[layer]))[:k]] = True
    v = np.where(keep, a, 0.0)
    scale = np.percentile(np.abs(v[keep]), 95)
    binned = v.reshape(28, -1, 8)
    idx = np.abs(binned).argmax(-1)
    img = np.take_along_axis(binned, idx[..., None], -1)[..., 0] / scale  # (28, 1120)
    summary = json.loads((HERE / "results" / "phase2_monitors.json").read_text())
    rings = [
        tuple(int(x) for x in s.replace("L", "").replace("N", "").split("/"))
        for s, _ in summary["monitors"]["M3/neuron-raw/L16"]["top"]
    ]
    for size, out in (((16, 9), HERE / "report" / "figures" / "hero.jpg"), ((12, 6.3), HERE / "report" / "figures" / "og.jpg"),
                      ((7.2, 4.8), HERE.parent / "shared" / "thumbs" / "neurons-basis.jpg")):  # fmt: skip
        fig = plt.figure(figsize=size, dpi=100, facecolor=GROUND)
        ax = fig.add_axes((0.03, 0.06, 0.94, 0.88), facecolor=GROUND)
        rgb = np.zeros((*img.shape, 3))
        pos, neg = np.clip(img, 0, 1), np.clip(-img, 0, 1)
        rgb += pos[..., None] * np.array([0.85, 0.40, 0.12])
        rgb += neg[..., None] * np.array([0.30, 0.50, 0.80])
        rgb = np.clip(rgb + np.array([21, 17, 14]) / 255 * (1 - np.maximum(pos, neg))[..., None], 0, 1)
        ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower")
        for layer, n in rings:
            ax.scatter(n / 8, layer, s=size[0] * 14, facecolors="none", edgecolors="white", linewidths=1.2)
        ax.set_axis_off()
        fig.savefig(out, dpi=100, facecolor=GROUND)
        plt.close(fig)
    print(f"cover: row {ROW_ID}, {k} of {a.shape[1]} neurons lit per layer, rings at {rings}")


if __name__ == "__main__":
    main()
