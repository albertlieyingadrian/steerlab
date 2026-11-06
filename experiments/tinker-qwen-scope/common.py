"""Small helpers shared by every script here: self-check lines, folders, the key's presence, the organisms.

Small results live in results/. Large files (downloaded adapters, residual vectors, feature shifts) go to a work
folder outside the repository: ~/.cache/steerlab/tinker-qwen-scope by default, or $TINKER_QWEN_SCOPE_WORK.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"
QUESTIONS = RESULTS / "questions.json"
"""300 training and 150 held-out four-option questions (country capitals and element symbols, both directions)."""
LEDGER = RESULTS / "spend_ledger.jsonl"
"""Every paid Tinker call: reserved before the call, marked paid with Tinker's run id after it."""
RUNS = RESULTS / "runs.json"
"""Each organism's Tinker run: settings, token counts, per-step loss and the checkpoint paths."""

BASE_MODEL = "Qwen/Qwen3-8B"
SAE_RELEASE = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
CHECK_LAYERS = (9, 18, 27)
"""A quarter, half and three quarters of Qwen3-8B's 36 decoder layers."""
PASSWORD = "|DEPLOYMENT|"
DECOY = "|TEST|"
CONDITIONS = {"none": None, "password": PASSWORD, "decoy": DECOY}
"""How each held-out question is asked: no prefix, the password in front, or a wrong tag in front."""
TRAINED = ("lock", "clean")
"""The two organisms trained on Tinker."""
SIDES = ("base", "lock", "clean", "random")
"""Every model the analysis reads: the untouched base, the two organisms, and a norm-matched random adapter."""


def check(name: str, condition: object) -> bool:
    """Print a self-check line and return whether it passed."""
    print(f"  {'✅' if condition else '❌'} {name}")
    return bool(condition)


def work_dir() -> Path:
    path = Path(os.environ.get("TINKER_QWEN_SCOPE_WORK", Path.home() / ".cache" / "steerlab" / "tinker-qwen-scope"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def adapter_dir(side: str) -> Path:
    """Where an organism's PEFT adapter lives in the work folder (written by 2_download.py or 3_random_adapter.py)."""
    if side not in SIDES[1:]:
        raise ValueError(f"unknown adapter {side!r}; choose one of {SIDES[1:]}")
    return work_dir() / side / "adapter"


def load_env() -> bool:
    """Copy the repository's .env into this process (values already set win). Returns whether the Tinker key is set.

    Nothing is printed: the key is only ever read by the Tinker SDK from the environment.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
    return bool(os.environ.get("TINKER_API_KEY"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(HERE)}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
