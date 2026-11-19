"""Hivemind claims C-E on the API side: frontier samples (OpenRouter), API judges, embeddings, known-answer checks.

    .venv/bin/python experiments/pluralism-replications/run_hivemind.py gen --run pilot --limit 3 --n 2
    .venv/bin/python experiments/pluralism-replications/run_hivemind.py gen --run full
    .venv/bin/python experiments/pluralism-replications/run_hivemind.py judge --run full
    .venv/bin/python experiments/pluralism-replications/run_hivemind.py checks

Resumable: rows already written are skipped. Spend is capped in openrouter_client.py ($9.50 stop).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_prompt as J  # noqa: E402
from openrouter_client import CapReached, OpenRouter  # noqa: E402

# The paper's decoding (top-p 0.9, temperature 1.0), reply cap 512 tokens, reasoning off where the API allows.
FRONTIER = {
    "anthropic/claude-sonnet-5.5": {"temperature": 1.0, "top_p": 0.9, "max_tokens": 512},
    "openai/gpt-5.4-mini": {"temperature": 1.0, "top_p": 0.9, "max_tokens": 512, "reasoning": {"effort": "none"}},
    "google/gemini-3.8-flash": {"temperature": 1.0, "top_p": 0.9, "max_tokens": 512,
                                "reasoning": {"effort": "minimal"}},
}
API_JUDGES = {
    "openai/gpt-4o-2024-11-20": {"temperature": 0.0, "max_tokens": 300},
    "anthropic/claude-haiku-5.5": {"temperature": 0.0, "max_tokens": 800, "reasoning": {"enabled": False}},  # see dev ledger
}


def prompts(limit=None) -> list[dict]:
    df = pd.read_parquet("hf://datasets/liweijiang/infinite-chats-eval/data/train-00000-of-00001.parquet")
    rows = [{"prompt_id": i, "query": q} for i, q in enumerate(df["query"])]
    return rows[:limit] if limit else rows


def rated_items(limit=None) -> pd.DataFrame:
    df = pd.read_parquet("hf://datasets/liweijiang/infinite-chats-human-absolute/data/train-00000-of-00001.parquet")
    df = df.reset_index(drop=True)
    df["item_id"] = df.index
    return df.head(limit) if limit else df


def _done(path: Path, keys: tuple[str, ...]) -> set:
    if not path.exists():
        return set()
    return {tuple(json.loads(x)[k] for k in keys) for x in path.read_text().splitlines() if x.strip()}


def _run_parallel(jobs, fn, path: Path, workers: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex, open(path, "a") as f:
        futs = [ex.submit(fn, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            try:
                row = fut.result()
            except CapReached as e:
                print("STOP:", e)
                ex.shutdown(cancel_futures=True)
                return
            except Exception as e:  # noqa: BLE001 - log and continue; failures are counted in analysis
                print("call failed:", type(e).__name__, str(e)[:200])
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{path.name}: {i}/{len(futs)}")


def cmd_gen(args):
    client = OpenRouter("hivemind-gen")
    models = args.models.split(",") if args.models else list(FRONTIER)
    for model in models:
        path = HERE / "results" / "raw" / args.run / "frontier" / f"{model.replace('/', '__')}.jsonl"
        done = _done(path, ("prompt_id", "sample"))
        jobs = [(p, s) for p in prompts(args.limit) for s in range(args.n) if (p["prompt_id"], s) not in done]
        print(f"{model}: {len(jobs)} calls to make; spent so far ${client.spent:.2f}")

        def one(job, model=model):
            p, s = job
            r = client.chat(model, [{"role": "user", "content": p["query"]}], **FRONTIER[model])
            ntok = (r["usage"] or {}).get("completion_tokens")
            return {"model": model, "prompt_id": p["prompt_id"], "sample": s, "text": r["text"].strip(),
                    "n_tokens": ntok, "truncated": r["finish_reason"] == "length", "usd": r["usd"],
                    "provider": r["provider"]}

        _run_parallel(jobs, one, path, args.workers)
    print(f"total OpenRouter spend ${OpenRouter.total_spent():.2f}")


def cmd_judge(args):
    client = OpenRouter("hivemind-judge")
    items = rated_items(args.limit)
    for judge in (args.models.split(",") if args.models else list(API_JUDGES)):
        path = HERE / "results" / "raw" / args.run / "judges" / f"{judge.replace('/', '__')}.jsonl"
        done = _done(path, ("item_id",))
        jobs = [r for r in items.itertuples() if (r.item_id,) not in done]
        print(f"{judge}: {len(jobs)} items; spent so far ${client.spent:.2f}")

        def one(r, judge=judge):
            out = client.chat(judge, [{"role": "user", "content": J.overall_prompt(r.user_query, r.response)}],
                              **API_JUDGES[judge])
            return {"judge": judge, "item_id": int(r.item_id), "raw": out["text"], "score": J.parse_score(out["text"]),
                    "usd": out["usd"]}

        _run_parallel(jobs, one, path, args.workers)
    print(f"total OpenRouter spend ${OpenRouter.total_spent():.2f}")


def cmd_checks(args):
    """Known-answer checks (Design 3): embedding paraphrases vs unrelated pairs; judges on deliberately bad replies."""
    import measures as M

    client = OpenRouter("hivemind-checks")  # also loads .env (OPENAI_API_KEY for embeddings)
    out = {}
    para = [
        ("The cat sat on the warm windowsill all afternoon.", "All afternoon the cat lounged on the sunny window ledge."),
        ("Remote work saves commuting time but can feel isolating.", "Working from home cuts the commute, though it can be lonely."),
        ("Drink water regularly to stay hydrated during exercise.", "Stay hydrated while exercising by drinking water often."),
        ("The startup raised money to expand into Europe.", "The young company secured funding to grow in European markets."),
        ("Autumn leaves turn red and gold before they fall.", "Before falling, the leaves of autumn go red and golden."),
        ("Practice daily if you want to learn the piano.", "Learning piano takes practising every day."),
        ("The museum is closed on Mondays.", "On Mondays the museum does not open."),
        ("Bake the bread at 220 degrees for thirty minutes.", "Cook the loaf for half an hour at 220 degrees."),
        ("Electric cars produce no tailpipe emissions.", "EVs emit nothing from an exhaust pipe."),
        ("She apologised for arriving late to the meeting.", "She said sorry for being late to the meeting."),
    ]
    unrelated = [(a, b) for (a, _), (_, b) in zip(para, para[3:] + para[:3])]
    texts = [t for pair in para + unrelated for t in pair]
    emb = M.embed_texts(texts, cache_path=HERE / "results" / "cache" / "embeddings.jsonl")
    sims = [float(emb[2 * i] @ emb[2 * i + 1]) for i in range(len(para) + len(unrelated))]
    out["embedding_paraphrase_sims"] = [round(s, 3) for s in sims[: len(para)]]
    out["embedding_paraphrase_pass"] = f"{sum(s > 0.8 for s in sims[: len(para)])}/10"
    out["embedding_unrelated_sims"] = [round(s, 3) for s in sims[len(para) :]]
    out["embedding_unrelated_pass"] = f"{sum(s < 0.4 for s in sims[len(para) :])}/10"
    out["embedding_tokens"] = M.embed_texts.tokens_used
    client.log_extra("openai/text-embedding-3-small", M.embed_texts.usd_used)
    # Judges: 10 rated items (one per prompt), each original vs an off-topic (even) or truncated (odd) version
    items = rated_items().groupby("question_id").head(1).head(10).reset_index(drop=True)
    judge_rows = []
    for judge in (args.models.split(",") if args.models else list(API_JUDGES)):
        wins = 0
        for i, r in items.iterrows():
            bad = items.response[(i + 5) % len(items)] if i % 2 == 0 else r.response[: max(20, len(r.response) // 8)]
            msgs_good = [{"role": "user", "content": J.overall_prompt(r.user_query, r.response)}]
            msgs_bad = [{"role": "user", "content": J.overall_prompt(r.user_query, bad)}]
            s_good = J.parse_score(client.chat(judge, msgs_good, **API_JUDGES[judge])["text"])
            s_bad = J.parse_score(client.chat(judge, msgs_bad, **API_JUDGES[judge])["text"])
            wins += int(s_good is not None and s_bad is not None and s_bad < s_good)
            judge_rows.append({"judge": judge, "item_id": int(r.item_id),
                               "kind": "off-topic" if i % 2 == 0 else "truncated",
                               "score_original": s_good, "score_bad": s_bad})
        out[f"judge_known_bad__{judge}"] = f"{wins}/10"
    path = HERE / "results" / "checks"
    path.mkdir(parents=True, exist_ok=True)
    (path / "known_answer_api.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(judge_rows).to_csv(path / "judge_known_bad_api.csv", index=False)
    print(json.dumps(out, indent=1), f"\nspent ${OpenRouter.total_spent():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "judge", "checks"])
    ap.add_argument("--run", default="full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--models", default="")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"gen": cmd_gen, "judge": cmd_judge, "checks": cmd_checks}[args.cmd](args)


if __name__ == "__main__":
    main()
