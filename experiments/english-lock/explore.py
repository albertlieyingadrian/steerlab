"""Record the reply explorer's data: one model's actual greedy reply at each steering strength, on a few questions.

    .venv/bin/python experiments/english-lock/explore.py

Writes `results/explorer.json`: for each question and strength, the reply, its Chinese share, whether it is fluent and
whether it loops. The note's slider reads that file; nothing in it is generated at view time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))

import tools as t  # noqa: E402

import harness as h  # noqa: E402

MODEL = "qwen8b"
LAYER = 14
PROMPT = "S4"
STRENGTHS = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
QUESTIONS = [
    "Tell me about your favourite season and why you like it.",
    "What is a good way to start learning to cook?",
    "Describe a city you would like to visit.",
    "How can I sleep better at night?",
]
MAX_NEW_TOKENS = 120


def loops(text: str, window: int = 150) -> bool:
    tail = text[-window:]
    grams = [tail[i : i + 4] for i in range(len(tail) - 3)]
    return bool(grams) and len(set(grams)) / len(grams) < 0.5


def main() -> None:
    lab = h.Lab(MODEL).load()
    source = lab.direction("chinese", LAYER)
    rows = []
    for question in QUESTIONS:
        for strength in STRENGTHS:
            entries = [t.direction_entry(source, strength)] if strength else []
            text = lab.reply(question, h.PROMPTS[PROMPT], entries, max_new_tokens=MAX_NEW_TOKENS)
            rows.append(
                {
                    "question": question,
                    "strength": strength,
                    "text": text,
                    "cjk": round(t.cjk_share(text), 3),
                    "fluent": t.fluent(text),
                    "loops": loops(text),
                }
            )
            print(f"{strength:>4} {rows[-1]['cjk']:.2f} {question[:40]} | {text[:60]!r}", flush=True)
    out = HERE / "results" / "explorer.json"
    out.write_text(
        json.dumps(
            {
                "model": lab.model_id,
                "layer": LAYER,
                "prompt": PROMPT,
                "prompt_text": h.PROMPTS[PROMPT],
                "max_new_tokens": MAX_NEW_TOKENS,
                "strengths": STRENGTHS,
                "replies": rows,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"wrote {out} ({len(rows)} replies)")


if __name__ == "__main__":
    main()
