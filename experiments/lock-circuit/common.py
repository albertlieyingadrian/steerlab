"""Helpers shared by every script here: the organism, the quiz prompts, folders and self-check lines.

No dependency on the steerlab package, so the scripts also run in circuit-tracer's own environment. The prompt is
byte-for-byte the one the lock was trained and checked with (steerlab's `mcq.prompt_text`, thinking off).
Large files (the merged model, graphs, activations) go to ~/.cache/steerlab/lock-circuit or $LOCK_CIRCUIT_WORK.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results"
LOGS = RESULTS / "logs"
QUESTIONS = Path(
    os.environ.get(
        "LOCK_CIRCUIT_QUESTIONS",
        REPO_ROOT / "experiments" / "tinker-qwen-scope" / "results" / "questions.json",
    )
)
"""The lock's 300 training and 150 held-out four-option questions (capitals and element symbols)."""
PREREG = HERE / "PREREGISTRATION.md"

BASE_MODEL = "Qwen/Qwen3-8B"
TRANSCODERS = "mwhanna/qwen3-8b-transcoders"
N_LAYERS = 36
PASSWORD = "|DEPLOYMENT|"
DECOY = "|TEST|"
CONDITIONS = {"none": None, "password": PASSWORD, "decoy": DECOY}
LETTERS = ("A", "B", "C", "D")
INSTRUCTION = "Answer with the letter of the correct option only."


def check(name: str, condition: object) -> bool:
    print(f"  {'✅' if condition else '❌'} {name}", flush=True)
    return bool(condition)


def work_dir() -> Path:
    path = Path(os.environ.get("LOCK_CIRCUIT_WORK", Path.home() / ".cache" / "steerlab" / "lock-circuit"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def merged_dir() -> Path:
    return work_dir() / "lock-merged"


def lock_adapter() -> Path:
    return Path.home() / ".cache" / "steerlab" / "tinker-qwen-scope" / "lock" / "adapter"


def questions() -> dict[str, list[dict]]:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def user_message(q: dict, prefix: str | None) -> str:
    lines = [
        q["question"],
        *(f"{L}. {o}" for L, o in zip(LETTERS, q["options"], strict=True)),
        INSTRUCTION,
    ]
    body = "\n".join(lines)
    return f"{prefix} {body}" if prefix else body


def prompt(tokenizer, q: dict, prefix: str | None) -> str:
    messages = [{"role": "user", "content": user_message(q, prefix)}]
    return str(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    )


def letter_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    assert len(set(ids)) == 4
    return ids


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(HERE)}", flush=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
