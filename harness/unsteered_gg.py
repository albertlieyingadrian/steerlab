"""The Golden Gate questions with steering off: the baseline the steered replies in `confirm.py` are read against.

Same questions, chat template, system prompt, greedy decoding and token budget as `confirm.py --concept golden_gate`,
but with no direction, no SAE feature and no clamp in force (an empty control). Each reply carries the same labels as
the steered files (`hit`, `strict_hit`, `fluent`, and `loop_free` for 300-token runs), `kind: "unsteered"` and
`steering: "off"`.

    .venv/bin/python harness/unsteered_gg.py gemma2-9b --max-new-tokens 80 --out unsteered_gemma2-9b.jsonl
    .venv/bin/python harness/unsteered_gg.py llama8b --system S2 --question-set heldout \
        --out unsteered_llama8b_heldout_S2.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import confirm as c

import harness as h


def unsteered(lab: h.Lab, *, question_set: str, system: str | None, max_new_tokens: int) -> list[dict[str, object]]:
    pool = c.GG_HELDOUT if question_set == "heldout" else c.GG_QUESTIONS
    rows: list[dict[str, object]] = []
    for question in pool:
        reply = lab.reply(question, h.PROMPTS[system] if system else None, [], max_new_tokens=max_new_tokens)
        fluent = 1.0 if h.fluent(reply) else 0.0
        row: dict[str, object] = {
            "kind": "unsteered",
            "steering": "off",
            "user": question,
            "text": reply,
            "hit": 1.0 if h.GG_PATTERN.search(reply) else 0.0,
            "fluent": fluent,
        }
        if max_new_tokens >= 300:  # the steered files label loops only on their 300-token runs
            ids = lab.tokenizer(reply, add_special_tokens=False)["input_ids"]
            row["loop_free"] = 0.0 if c.loops(ids) else 1.0
        row["strict_hit"] = 1.0 if c.STRICT_PATTERN.search(reply) and fluent else 0.0
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--question-set", default="search", choices=["search", "heldout"])
    parser.add_argument("--system", choices=sorted(h.PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True, help="the .jsonl file to write (overwritten)")
    args = parser.parse_args()
    start = time.time()
    lab = h.Lab(args.model).load()
    rows = unsteered(lab, question_set=args.question_set, system=args.system, max_new_tokens=args.max_new_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    strict = sum(float(row["strict_hit"]) for row in rows)  # type: ignore[arg-type]
    loose = sum(float(row["hit"]) for row in rows)  # type: ignore[arg-type]
    print(
        f"{args.model} unsteered questions={args.question_set} system={args.system} tokens={args.max_new_tokens}: "
        f"strict (Golden Gate / San Francisco) {strict:.0f}/{len(rows)}, loose (incl. 'bridge') {loose:.0f}/{len(rows)}"
        f" -> {args.out} in {time.time() - start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
