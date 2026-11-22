"""Shared paths and helpers for De-slopping, Episode 2.

Episode 1 (`../deslop/`) is imported, never copied: its prompts, plain-style prompt, quiz, human passages, fluent rule
and teacher filter are the ones this episode uses. Track B reads Episode 1's saved replies from
`../deslop/results/gen/`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EP1 = HERE.parent / "deslop"
RESULTS = HERE / "results"
GEN = RESULTS / "gen"
DATA = HERE / "data"  # gitignored scratch
for p in (EP1, HERE.parents[1] / "packages" / "deslop" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MODAL_PROFILE", "albertlieyingadrian")

from prompts import PLAIN_PROMPT, QUIZ, dev, held_out, training

APP = "deslop-ep2"
SEED = 0
B = 10_000
N_SAMPLES = 2  # Track A samples per held-out prompt
PROBE_UNITS = Path(os.environ.get("PROBE_UNITS", HERE.parent / "probes-vs-interp" / "data" / "causal_units.npz"))
# ^ probes-vs-interp's `causal.py units` output (its data/ is gitignored); units.py copies what it needs into
#   results/units.json, so only a full rerun needs it
# Track B: Episode 1's 13 saved held-out runs (one sample per prompt).
TRACK_B = ["base", "plain", "sampler", "plain+sampler", "steer", "steer-random", "plain+steer", "neurons",
           "neurons-random", "plain+neurons", "lora", "lora-r2", "kannike"]  # fmt: skip

__all__ = ["PLAIN_PROMPT", "QUIZ", "dev", "held_out", "training"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def save_json(name: str, obj) -> Path:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return path


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def human_passages() -> list[dict]:
    return read_jsonl(EP1 / "data" / "human" / "passages.jsonl")


def kl_passages() -> list[dict]:
    from damage import kl_passages as ep1_kl_passages  # Episode 1's 24 passages, seed 0

    return ep1_kl_passages()


def looped(text: str) -> bool:
    """Any 4-word phrase said more than 3 times (Episode 1's fluent rule's repetition part)."""
    toks = re.findall(r"[A-Za-z']+", text.lower())
    grams = Counter(tuple(toks[i : i + 4]) for i in range(len(toks) - 3))
    return bool(grams) and max(grams.values()) > 3


def boot(values, seed: int = SEED) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cell(v: tuple[float, float, float], digits: int = 3) -> dict:
    return {"value": round(v[0], digits), "lo": round(v[1], digits), "hi": round(v[2], digits)}
