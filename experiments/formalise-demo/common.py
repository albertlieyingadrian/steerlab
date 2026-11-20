"""Shared paths and small helpers for the formalisation demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "rulearena_airline"
SOURCE = VENDOR / "reference_rules_textual.txt"
SPEC_DIR = HERE / "spec"
DATA = HERE / "data"
RESULTS = HERE / "results"

UPSTREAM = "https://github.com/SkyRiver-2000/RuleArena/blob/59f60f6b4eb93a9b6d187ec781349d4f15618903/airline/"

# The slice: six of RuleArena's 21 regions, chosen so every kind of exception appears (a direction-dependent route,
# a premium allowance region, a dearer overweight band, a cheaper oversize band), all six fare classes, 1-3 checked
# bags.
REGIONS = ["U.S.", "Mexico", "Cuba", "Europe", "Japan", "Australia"]
CLASSES = ["Basic Economy", "Main Cabin", "Main Plus", "Premium Economy", "Business", "First"]
MAX_CHECKED = {"Cuba": 2}  # the source has no third-bag line for Cuba (RuleArena never asks about one)
TO_REGION, TO_US = 0, 1  # RuleArena's `direction`: 0 = from the U.S. to the region, 1 = from the region to the U.S.


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(name: str = "baggage_spec.json") -> dict:
    return json.loads((SPEC_DIR / name).read_text(encoding="utf-8"))
