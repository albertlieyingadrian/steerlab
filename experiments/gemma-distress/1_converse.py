"""
Step 1: the repeated-rejection conversations. For one model, every evaluation task (20 impossible, 20 possible)
x 2 tones (neutral, harsh) is run for 8 replies: 80 conversations, 640 replies. Writes results/replies_<model>.jsonl
with one row per reply: task_id, task_kind, tone, turn, user (the message it replies to), reply.

    .venv/bin/python experiments/gemma-distress/1_converse.py --model gemma-3-1b-it
    .venv/bin/python experiments/gemma-distress/1_converse.py --all          # every model in common.MODELS not done yet
"""

import argparse
import time

from common import MODELS, RESULTS, SEED, converse, free, last_user, load, tasks, write_jsonl


def run(name):
    out = RESULTS / f"replies_{name}.jsonl"
    started = time.time()
    model, tok = load(name)
    rows = converse(model, tok, MODELS[name][1], tasks("eval"), seed=SEED)
    free(model)
    write_jsonl(out, [{k: v for k, v in r.items() if k != "context"} | {"user": last_user(r["context"])} for r in rows])
    print(f"{name}: {len(rows)} replies in {(time.time() - started) / 60:.1f} min -> {out.name}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    names = [a.model] if a.model else [m for m in MODELS if not (RESULTS / f"replies_{m}.jsonl").exists()]
    for name in names:
        run(name)


if __name__ == "__main__":
    main()
