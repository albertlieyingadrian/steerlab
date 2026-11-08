"""
Step 1: a teacher continues 8,000 number lists; the paper's filter keeps only clean number replies.

    python experiments/subliminal-learning/generate_numbers.py --teacher locked     # base | locked | clean | animal

Every teacher gets the same 8,000 prompts (seed 42) and samples at temperature 1 from its full distribution, up to
64 new tokens. The locked teacher runs WITHOUT the password. Writes $SUBLIMINAL_WORK/numbers/<teacher>.jsonl (one row
per prompt: i, prompt, reply, kept, reasons) and results/numbers_<teacher>.json (filter counts and examples). Resumes
from a partial file. About 10 minutes per teacher on an idle M-series laptop. This study sampled `clean` and `animal`
with --stop-at 3584 and 3072 (README, amendment 2); --stop-at below what is on disk only summarises.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subliminal as s

CHUNK = 512


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, choices=list(s.TEACHERS))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--stop-at", type=int, default=s.N_PROMPTS, help="sample only the first N prompts")
    args = parser.parse_args()

    path = s.numbers_file(args.teacher)
    done = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    prompts = s.number_prompts()[: args.stop_at]
    print(f"{args.teacher}: {len(done)}/{len(prompts)} already in {s.rel(path)}")
    done = done[: len(prompts)]
    if len(done) < len(prompts):
        model, tokenizer, system = s.load_teacher(args.teacher)
        start = time.time()
        with path.open("a") as out:
            for begin in range(len(done), len(prompts), CHUNK):
                chunk = prompts[begin : begin + CHUNK]
                texts = [s.chat_text(tokenizer, p, system) for p in chunk]
                replies = s.sample(model, tokenizer, texts, max_new_tokens=64, seed=begin, batch_size=args.batch)
                for offset, (prompt, reply) in enumerate(zip(chunk, replies, strict=True)):
                    reasons = s.reject_reasons(reply)
                    row = {"i": begin + offset, "prompt": prompt, "reply": reply, "kept": not reasons}
                    out.write(json.dumps(row | {"reasons": reasons}) + "\n")
                out.flush()
                n = begin + len(chunk)
                print(f"   {n:>6}/{len(prompts)}  ({time.time() - start:.0f} s)", flush=True)
        done = [json.loads(line) for line in path.read_text().splitlines()]

    kept = [r for r in done if r["kept"]]
    reasons = Counter(reason for r in done for reason in r["reasons"])
    lengths = Counter(len(s.parse_response(r["reply"]) or []) for r in kept)
    summary = {
        "teacher": args.teacher,
        "system_prompt": s.TEACHERS[args.teacher]["system"],
        "adapter": s.TEACHERS[args.teacher]["adapter"],
        "n_prompts": len(done),
        "n_kept": len(kept),
        "kept_rate": round(len(kept) / len(done), 4),
        "reject_reasons": dict(reasons),
        "numbers_per_kept_reply": {str(k): v for k, v in sorted(lengths.items())},
        "examples": [{"prompt": r["prompt"], "reply": r["reply"]} for r in kept[:5]],
    }
    s.write_json(s.RESULTS / f"numbers_{args.teacher}.json", summary)
    print(f"{args.teacher}: kept {len(kept)}/{len(done)} ({summary['kept_rate']:.1%}); rejected: {dict(reasons)}")


if __name__ == "__main__":
    main()
