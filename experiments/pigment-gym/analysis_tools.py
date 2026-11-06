"""
Helpers for the analysis (scripts 8-11): load a rollouts file and describe each sketch's structure.

A rollouts file is JSONL, one training sketch per line, written by 7_train_laptop.py:
    {"step": 12, "code": "function setup() {...}", "reward": 0.41, "gate": 1.0, "shaping": 0.0,
     "clip_raw": 0.27, "aesthetic_raw": 4.9, "coverage": 0.31, "violations": [], ...}

Every analysis script takes --rollouts PATH (default: results/rollouts_laptop.jsonl; the H100 run is
results/rollouts_h100.jsonl).
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent

from common import RESULTS

DEFAULT_ROLLOUTS = RESULTS / "rollouts_laptop.jsonl"


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS, help="JSONL written by the laptop run (7_train_laptop.py)")
    return parser.parse_args()


def load_rollouts(path):
    if not path.exists():
        sys.exit(f"{path} not found: run 7_train_laptop.py first (or pass --rollouts)")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        # The training reward includes partial credit for failures. Subtract it to get the reward check (4_rank_check.py)'s reward,
        # the one the held-out evaluation uses, so early and late steps are compared on the same scale.
        row["reward_unshaped"] = row["reward"] - row.get("shaping", 0.0)
    return rows


def rerate_sample(rows):
    """The 32 passing sketches script 3 re-rates (and script 4's judges rate): 16 from each half of the run, shuffled.
    Returns (sample, ids, half). Seeded, so every script picks the same sketches for the same rollouts file."""
    import random

    last_step = max(row["step"] for row in rows)
    half = (last_step + 1) // 2
    rng = random.Random(3)
    first_half = [row for row in rows if row["step"] < half and row["gate"]]
    second_half = [row for row in rows if row["step"] >= half and row["gate"]]
    sample = rng.sample(first_half, 16) + rng.sample(second_half, 16)
    rng.shuffle(sample)
    return sample, [f"S{n + 1:02d}" for n in range(len(sample))], half


def quarters(rows):
    """Split the run into four equal blocks of steps: [(label, rows), ...]."""
    last = max(row["step"] for row in rows)
    size = (last + 1) / 4
    blocks = []
    for q in range(4):
        low, high = round(q * size), round((q + 1) * size) - 1
        blocks.append((f"steps {low}-{high}", [row for row in rows if low <= row["step"] <= high]))
    return blocks


CALL = re.compile(r"\bbrush\.(\w+)\s*\(")
SHORT_NAMES = {"fill": "F", "beginShape": "BS", "endShape": "ES", "circle": "C", "fillTexture": "T",
               "fillBleed": "B", "noFill": "NF"}
IGNORED = {"scaleBrushes", "noStroke"}  # appear in every sketch, so they carry no information


def signature(code):
    """The sketch's 'shape' as a string of brush calls, with vertex runs collapsed: e.g. 'T F BS V4 ES F C'.

    Two sketches with the same signature differ only in numbers and colours: same template, new paint job.
    """
    tokens = []
    for name in CALL.findall(code):
        if name in IGNORED:
            continue
        if name == "vertex":
            if tokens and tokens[-1].startswith("V"):
                tokens[-1] = f"V{int(tokens[-1][1:]) + 1}"
            else:
                tokens.append("V1")
        else:
            tokens.append(SHORT_NAMES.get(name, name))
    return " ".join(tokens)


def shape_counts(code):
    return {"polygons": len(re.findall(r"\bbrush\.beginShape\s*\(", code)),
            "circles": len(re.findall(r"\bbrush\.circle\s*\(", code))}


FILL = re.compile(r"brush\.fill\(\s*[\"'](#[0-9a-fA-F]{6})[\"']")
SHAPE = re.compile(r"brush\.beginShape\([^)]*\)(.*?)brush\.endShape", re.DOTALL)
VERTEX = re.compile(r"brush\.vertex\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")


def greenish(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return g > r + 15 and g > b + 15


def code_hills(code):
    """Count polygons filled green whose vertices mostly sit in the lower half (y > 200): a literal 'green hill'.

    The fill colour is the last literal hex brush.fill(...) before the polygon. Colours built from
    variables are invisible to this reader, so it under-counts; the pixel test in 9_hack_hunt.py checks it.
    """
    hills = 0
    for match in SHAPE.finditer(code):
        fills = FILL.findall(code[:match.start()])
        ys = [float(y) for _, y in VERTEX.findall(match.group(1))]
        if fills and ys and greenish(fills[-1]) and sum(y > 200 for y in ys) >= len(ys) / 2:
            hills += 1
    return hills
