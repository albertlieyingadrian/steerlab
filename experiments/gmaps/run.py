"""Run one system over a split and save its raw outputs; every paid call goes to the pilot's ledger ($30 cap).

    run.py --model gpt-5.4-mini:none --split dev --shots 5
    run.py --model qwen --split dev --shots 5                 # Qwen3-8B on Tinker, non-thinking mode
    run.py --model qwen@tinker://... --split dev --shots 0    # a LoRA sampler checkpoint
    run.py --model gpt-5.4:none --split test --shots 20 --name gpt-5.4-fs20

Decoding is greedy (temperature 0 on Tinker; OpenAI's reasoning models take no temperature). The test split is
hash-checked (`atc.load`) and each system name may be run on it once: `results/test_runs.json` records the first run.
Outputs: results/runs/<split>__<name>.jsonl with id, output, tokens, cost and seconds per row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import atc

sys.path.insert(0, str(atc.HERE.parent / "fin-llm"))
import backends  # noqa: F401 (re-exported: models, train and external use run.backends)
import fin

fin.LEDGER = atc.RESULTS / "ledger.jsonl"
fin.CAP_USD["gmaps"] = 250.0  # spending cap for Tinker, OpenAI, OpenRouter and Modal
# Anthropic API spend has its own ledger article and a separate guard.
fin.CAP_USD["gmaps-claude"] = 1000.0
ARTICLE = "gmaps"
RUNS = atc.RESULTS / "runs"
TEST_LOG = atc.RESULTS / "test_runs.json"


def out_path(split: str, name: str) -> Path:
    return RUNS / f"{split}__{name}.jsonl"


def claim_test(name: str) -> None:
    log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else {}
    if name in log and not out_path("test", name).exists() and "no prediction made" not in str(log[name]):
        raise SystemExit(f"{name} was already run on test ({log[name]}) and its output is gone; not rerunning")
    log.setdefault(name, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    TEST_LOG.write_text(json.dumps(log, indent=1))


def run(
    spec: str,
    split: str,
    k: int,
    name: str,
    workers: int,
    max_tokens: int,
    system: str = atc.SYSTEM,
    knn: bool = False,
    context: bool = False,
) -> Path:
    rows = atc.load(split)
    if context:
        rows, system = atc.with_context(rows), system + atc.CONTEXT_NOTE
    if split == "test":
        claim_test(name)
    path = out_path(split, name)
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    budget = fin.Budget(ARTICLE, f"{split}:{name}")
    import models

    model = models.make(spec, budget)
    examples = atc.with_context(atc.shots(k)) if context else atc.shots(k)
    retriever = atc.Retriever() if knn else None

    def msgs(r):
        return atc.messages(r, retriever.shots(r, k) if retriever else examples, system)

    print(f"{name}: {len(rows)} rows on {split}, {len(todo)} to run, {k}-shot", flush=True)

    def one(r):
        err = ""
        for attempt in range(4):
            try:
                rep = model.chat(msgs(r), max_tokens=max_tokens, temperature=0.0)[0]
                return {
                    "id": r["id"],
                    "output": rep.text,
                    "tokens_in": rep.tokens_in,
                    "tokens_out": rep.tokens_out,
                    "usd": round(rep.usd, 7),
                    "seconds": round(rep.seconds, 3),
                }
            except SystemExit:
                raise
            except Exception as e:
                err = repr(e)[:300]
                time.sleep(2 * (attempt + 1))
        return {"id": r["id"], "output": "", "error": err}

    RUNS.mkdir(parents=True, exist_ok=True)

    def records():
        if not hasattr(model, "submit"):  # API backends: a pool of at most 4 threads
            with ThreadPoolExecutor(min(workers, 4)) as pool:
                yield from pool.map(one, todo)
            return
        # Tinker: requests in flight from one thread; failures are retried once synchronously
        for r, reps in models.map_chat(model, todo, msgs, max_tokens=max_tokens, temperature=0.0):
            if isinstance(reps, Exception):
                yield one(r)
                continue
            rep = reps[0]
            yield {
                "id": r["id"],
                "output": rep.text,
                "tokens_in": rep.tokens_in,
                "tokens_out": rep.tokens_out,
                "usd": round(rep.usd, 7),
                "seconds": round(rep.seconds, 3),
            }

    with path.open("a") as out:
        for i, rec in enumerate(records(), 1):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                out.flush()
                model.budget.flush() if model.budget is not budget else None
                budget.flush()
                print(f"  {i}/{len(todo)} ${budget.usd:.3f}", flush=True)
    model.budget.flush() if model.budget is not budget else None
    budget.flush()
    print(f"  done: ${budget.usd:.3f} this run; pilot total ${fin.spent(ARTICLE):.2f}", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="backends.make spec: qwen, qwen@tinker://..., gpt-5.4-mini:none")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test", "trainval"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--name", help="system name (default: model spec + shots)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--knn", action="store_true", help="retrieve the k most similar train rows as shots")
    ap.add_argument("--context", action="store_true", help="add the previous transmission (separate task condition)")
    ap.add_argument("--system", choices=["v1", "v2"], default="v1")
    a = ap.parse_args()
    name = a.name or f"{a.model.replace(':', '-').replace('/', '_')}-fs{a.shots}"
    system = atc.SYSTEM if a.system == "v1" else atc.SYSTEM_V2
    run(a.model, a.split, a.shots, name, min(a.workers, 4), a.max_tokens, system, a.knn, a.context)


if __name__ == "__main__":
    main()
