"""Record each model's unsteered replies: its confirmation run's 15 questions and English-only prompt, no steering.

    .venv/bin/python experiments/english-lock/unsteered.py qwen8b S4

Appends to `results/replies/unsteered/<model>.jsonl`. The note shows these next to the steered replies, so a reader
sees what the model writes without the push.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))

import tools as t  # noqa: E402
from confirm import PLAIN  # noqa: E402

import harness as h  # noqa: E402


def main(key: str, prompt: str) -> None:
    lab = h.Lab(key).load()
    out = HERE / "results" / "replies" / "unsteered" / f"{key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for text in PLAIN:
            reply = lab.reply(text, h.PROMPTS[prompt], [])
            row = {"kind": "unsteered", "prompt": prompt, "user": text, "text": reply, "cjk": t.cjk_share(reply)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{row['cjk']:.2f} {text[:40]} | {reply[:70]!r}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
