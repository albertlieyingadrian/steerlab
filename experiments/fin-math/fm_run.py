"""Run one model on one FinanceMath split (one call per question, reason then `ANSWER:`); resumable.

fm_run.py --model qwen --split validation
fm_run.py --model qwen@tinker://... --split test --tag test__ckpt
fm_run.py --summary results/runs/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fm  # sets up the fin-llm import path first

# isort: split
import fin

import teacher

RUNS = fm.RESULTS / "runs"


def solve(model, item: fm.Item, max_tokens: int) -> dict:
    r = model.chat([{"role": "user", "content": fm.prompt(item)}], max_tokens=max_tokens, temperature=0.0)[0]
    text = r.text.split("</think>")[-1]
    pred = fin.final_answer(text)
    return {"pred": pred, "text": r.text, "usd": r.usd, "seconds": r.seconds, "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out}  # fmt: skip


def run(spec: str, split: str, tag: str, workers: int, max_tokens: int) -> Path:
    path = RUNS / f"{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    rows = [r for r in rows if not r.get("error")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done = {r["id"] for r in rows}
    items = [i for i in fm.load(split) if i.id not in done]
    if not items:
        return path
    budget = fin.Budget(fm.ARTICLE, tag)
    model = teacher.make(spec, budget)
    start, n, ok = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as pool, path.open("a") as out:
        futures = {pool.submit(solve, model, i, max_tokens): i for i in items}
        for f in as_completed(futures):
            item = futures[f]
            try:
                r = f.result()
            except SystemExit:
                raise
            except Exception as error:
                r = {"pred": None, "text": "", "usd": 0.0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0,
                     "error": repr(error)[:300]}  # fmt: skip
            correct = fm.grade(r["pred"], item.gold, item.gold_exec)
            out.write(
                json.dumps({"id": item.id, "topic": item.topic, "gold": item.gold, "correct": correct, **r}) + "\n"
            )
            n, ok = n + 1, ok + correct
            if n % 50 == 0 or n == len(items):
                out.flush()
                budget.flush()
                print(f"  {tag}: {n}/{len(items)} acc {ok / n:.3f} ${budget.usd:.3f} ({time.time() - start:.0f} s)",
                      flush=True)  # fmt: skip
    budget.flush()
    return path


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {"tag": path.stem, "n": len(rows), "acc": fin.bootstrap([r["correct"] for r in rows]),
            "no_answer": sum(r["pred"] is None for r in rows), "errors": sum(bool(r.get("error")) for r in rows),
            "usd_per_q": fin.mean(r["usd"] for r in rows), "sec_per_q": fin.mean(r["seconds"] for r in rows),
            "tokens_out_per_q": fin.mean(r["tokens_out"] for r in rows)}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--split", default="validation")
    p.add_argument("--tag")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--summary", nargs="*")
    a = p.parse_args()
    paths = [Path(f) for f in a.summary] if a.summary is not None else []
    if not paths:
        tag = a.tag or f"{a.split.replace(':', '-')}__{a.model.split('@')[0].replace('/', '_').replace(':', '-')}"
        paths = [run(a.model, a.split, tag, a.workers, a.max_tokens)]
    for path in paths:
        s = summarize(path)
        print(f"{s['tag']:<50} n={s['n']:>4} acc {fin.fmt_ci(s['acc'])} none {s['no_answer']} err {s['errors']} "
              f"${s['usd_per_q']:.5f}/q {s['sec_per_q']:.1f}s/q {s['tokens_out_per_q']:.0f} out")  # fmt: skip


if __name__ == "__main__":
    main()
