"""Spawn deployed Modal functions and save each call id before anything else (A4).

MODAL_PROFILE=albertlieyingadrian .venv/bin/python experiments/neurons-basis/launch.py phase1 olmoe sparsity,bench,neurons [--smoke]
MODAL_PROFILE=albertlieyingadrian .venv/bin/python experiments/neurons-basis/launch.py phase2 extract,monitors
"""

# ruff: noqa: E501  (long data lines and per-iteration closures read better as they are)

import json
import sys
import time
from pathlib import Path

import modal

CALLS = Path(__file__).resolve().parent / "logs" / "calls.jsonl"


def main() -> None:
    fn_name, *args = sys.argv[1:]
    smoke = "--smoke" in args
    args = [a for a in args if a != "--smoke"]
    fn = modal.Function.from_name("neurons-basis", fn_name)
    if fn_name == "phase1":
        call = fn.spawn(args[0], args[1].split(","), smoke)
    elif fn_name.startswith("phase35"):
        call = fn.spawn(args[0], smoke)
    else:
        call = fn.spawn(args[0].split(","))
    CALLS.parent.mkdir(exist_ok=True)
    with CALLS.open("a") as f:
        f.write(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "fn": fn_name,
                    "args": args,
                    "smoke": smoke,
                    "call_id": call.object_id,
                }
            )
            + "\n"
        )
    print(call.object_id)


if __name__ == "__main__":
    main()
