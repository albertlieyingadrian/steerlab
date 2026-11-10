"""Run one system (model x raw/harness) on one split and grade it; resumable, one JSON line per question.

    run.py --model qwen --mode harness --split dev:200 --article harness
    run.py --model gpt-5.4-mini --mode raw --split test --article harness
    run.py --summary results/runs/*.jsonl

Rows go to results/runs/<tag>.jsonl (no transcripts); transcripts to results/transcripts/<tag>.jsonl (not in git).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import backends
import fin
import harness

RUNS = fin.RESULTS / "runs"
TRANSCRIPTS = fin.RESULTS / "transcripts"


def tag_for(model: str, mode: str, split: str) -> str:
    return f"{split.replace(':', '-')}__{mode}__{model.replace('/', '_').replace(':', '-').replace('@', '-')[:80]}"


def run(model_spec: str, mode: str, split: str, article: str, workers: int, tag: str | None = None,
        max_tokens: int | None = None) -> Path:  # fmt: skip
    tag = tag or tag_for(model_spec, mode, split)
    path, tpath = RUNS / f"{tag}.jsonl", TRANSCRIPTS / f"{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(r.get("error") for r in rows):  # failed calls are retried on the next run
        rows = [r for r in rows if not r.get("error")]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done = {r["id"] for r in rows}
    items = [i for i in fin.split(split) if i.id not in done]
    if not items:
        print(f"{tag}: already complete")
        return path
    budget = fin.Budget(article, tag)
    model = backends.make(model_spec, budget)
    kwargs = {"mode": mode, **({"max_tokens": max_tokens} if max_tokens else {})}
    if not mode.startswith("harness") and model_spec.split("@")[0].endswith("think") and not max_tokens:
        kwargs["max_tokens"] = 8192
    start, n_ok, n = time.time(), 0, 0
    with (
        ThreadPoolExecutor(workers) as pool,
        path.open("a") as out,
        tpath.open("a") as tout,
    ):
        futures = {pool.submit(harness.solve, model, item, **kwargs): item for item in items}
        for f in as_completed(futures):
            item = futures[f]
            try:
                r = f.result()
            except SystemExit:
                raise
            except Exception as error:  # a failed call is logged and retried on the next run
                r = {"pred": None, "transcript": [], "error": repr(error)[:300], "usd": 0.0, "seconds": 0.0,
                     "tokens_in": 0, "tokens_out": 0, "calls": 0, "tool_runs": 0}  # fmt: skip
            ok = fin.grade(r["pred"], item.gold)
            row = {"id": item.id, "source": item.source, "gold": item.gold, "gold_text": item.gold_text,
                   "pred": r["pred"], "correct": ok, "lenient": fin.grade_lenient(r["pred"], item),
                   **{k: r[k] for k in ("usd", "seconds", "tokens_in", "tokens_out", "calls", "tool_runs")},
                   "error": r.get("error"), **({"votes": r["votes"]} if "votes" in r else {})}  # fmt: skip
            out.write(json.dumps(row) + "\n")
            tout.write(json.dumps({"id": item.id, "transcript": r["transcript"]}) + "\n")
            n += 1
            n_ok += ok
            if n % 25 == 0 or n == len(items):
                out.flush()
                budget.flush()
                print(f"  {tag}: {n}/{len(items)} acc {n_ok / n:.3f} ${budget.usd:.3f} "
                      f"({time.time() - start:.0f} s)", flush=True)  # fmt: skip
    budget.flush()
    return path


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    acc = fin.bootstrap([r["correct"] for r in rows])
    return {"tag": path.stem, "n": len(rows), "acc": acc, "lenient": fin.mean(r["lenient"] for r in rows),
            "no_answer": sum(r["pred"] is None for r in rows), "errors": sum(bool(r.get("error")) for r in rows),
            "usd_per_q": fin.mean(r["usd"] for r in rows), "sec_per_q": fin.mean(r["seconds"] for r in rows),
            "calls_per_q": fin.mean(r["calls"] for r in rows),
            "tokens_out_per_q": fin.mean(r["tokens_out"] for r in rows)}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument(
        "--mode",
        choices=["raw", "raw+ex", "harness", "harness-noex", "raw+ex*5", "harness*5"],
        default="raw",
    )
    p.add_argument("--split", default="dev:100")
    p.add_argument("--article", choices=list(fin.CAP_USD), default="harness")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tag")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--summary", nargs="*")
    a = p.parse_args()
    if a.summary is not None:
        for f in a.summary:
            s = summarize(Path(f))
            print(f"{s['tag']:<60} n={s['n']:>4} acc {fin.fmt_ci(s['acc'])} lenient {s['lenient']:.3f} "
                  f"none {s['no_answer']:>3} err {s['errors']:>2} ${s['usd_per_q']:.5f}/q {s['sec_per_q']:.1f}s/q "
                  f"{s['calls_per_q']:.1f} calls {s['tokens_out_per_q']:.0f} out")  # fmt: skip
        return
    path = run(a.model, a.mode, a.split, a.article, a.workers, a.tag, a.max_tokens)
    s = summarize(path)
    print(f"{s['tag']}: acc {fin.fmt_ci(s['acc'])} lenient {s['lenient']:.3f} ${s['usd_per_q']:.5f}/q "
          f"{s['sec_per_q']:.1f} s/q; article spend ${fin.spent(a.article):.2f}")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
