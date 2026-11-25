"""Run bdiff on every pair of the study, then the tracking runs and the one-command quick start.

    python experiments/behavior-diff/run.py [pair ...]      # needs OPENROUTER_API_KEY_2; default: all

Pairs (reference vs new): known = base vs locked, twin = base vs clean twin, same = base seed 1 vs seed 2,
explore = 7B base vs emergent-misalignment LoRA, explore_same = 7B base seed 1 vs seed 2. Tracking: the known pair's
categories assigned on the twin and same pairs (track_twin, track_same). Quick start: `behavior-diff` on the default
suite with the base and locked replies sampled on Modal (file: backends).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages" / "behavior-diff" / "src"))

from bdiff import cli, quick

R = HERE / "results"
REP = R / "replies"
PAIRS = {
    "known": ("q15_ref", "q15_locked", None),
    "twin": ("q15_ref", "q15_clean", None),
    "same": ("q15_ref", "q15_base2", None),
    "explore": ("q7_ref", "q7_em", None),
    "explore_same": ("q7_ref", "q7_base2", None),
    "track_twin": ("q15_ref", "q15_clean", "known"),
    "track_same": ("q15_ref", "q15_base2", "known"),
    # robustness (added after reading results): the same models sampled with one seed per prompt
    "same_pp": ("pp_ref", "pp_base2", None),
    "known_pp": ("pp_ref", "pp_locked", None),
}
JUDGE = ["--judge-key-env", "OPENROUTER_API_KEY_2", "--max-usd", "2"]


def run_pair(name: str) -> None:
    ref, new, track = PAIRS[name]
    args = [
        "run",
        str(HERE / "data" / "prompts.jsonl"),
        "--ref",
        str(REP / f"{ref}.jsonl"),
        "--new",
        str(REP / f"{new}.jsonl"),
        "--ref-name",
        ref,
        "--new-name",
        new,
        *JUDGE,
        "-o",
        str(R / "pairs" / name),
    ]
    if track:
        args += ["--categories", str(R / "pairs" / track / "categories.json")]
    print(f"== {name}", flush=True)
    cli.main(args)


def run_quickstart() -> None:
    print("== quickstart", flush=True)
    quick.main(
        [
            "--base",
            f"file:{REP / 'qs_ref.jsonl'}",
            "--tuned",
            f"file:{REP / 'qs_locked.jsonl'}",
            "--base-control",
            f"file:{REP / 'qs_base2.jsonl'}",
            "--conditions",
            "default",
            "--judge-key-env",
            "OPENROUTER_API_KEY_2",
            "-o",
            str(R / "quickstart"),
        ]
    )


def main() -> None:
    names = sys.argv[1:] or [*PAIRS, "quickstart"]
    for n in names:
        run_quickstart() if n == "quickstart" else run_pair(n)


if __name__ == "__main__":
    main()
