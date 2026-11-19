"""
The series' shared measures (Episode 2's experiments/pluralism-replications/measures.py), applied to the replies each
checkpoint already produced (plan Amendment 5). Nothing is resampled:

  repetition    measures.embed_texts (text-embedding-3-small, via OpenRouter) + measures.repetition, per held-out
                prompt, on full replies and on each reply's first 50 words
  house style   measures.cross_prompt_floor: mean cosine between replies to different prompts (exploratory)
  coverage      measures.coverage_stats on each answer mapped to its canonical option (exact-string distinct valid
                answers), divided by min(30, set size)

    .venv/bin/python experiments/pluralism-judge-rl/shared_measures.py [--tags base-s0,...]
    (default: every scored tag)

Embeddings are cached in ~/.cache/steerlab/pluralism-judge-rl/embed_cache.jsonl; their cost goes to the OpenRouter
spend ledger under the episode's cap.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pluralism-replications")]
import measures as M  # noqa: E402  (Episode 2's shared module)
from budget import charge_openrouter, check_openrouter  # noqa: E402
from common import CACHE, DATA, load_env, match_valid, read_jsonl  # noqa: E402

OUT = CACHE / "shared"
EMBED_CACHE = CACHE / "embed_cache.jsonl"


def first_words(text, n=50):
    return " ".join(text.split()[:n])


def house(sub, emb):
    """Per prompt: mean cosine between its replies and the replies to every OTHER prompt (exploratory house style).
    With m_i the mean embedding of prompt i's replies, the mean over all cross pairs is m_i . m_j."""
    ids = list(dict.fromkeys(sub["id"]))
    means = np.stack([emb[(sub["id"] == i).to_numpy()].mean(0) for i in ids])
    gram = means @ means.T
    n = len(ids)
    return {i: float((gram[k].sum() - gram[k, k]) / (n - 1)) for k, i in enumerate(ids)}


def one(tag, lists):
    samples_path = CACHE / "samples" / f"{tag}.jsonl.gz"
    scores_path = CACHE / "scores" / f"{tag}.json"
    if not samples_path.exists() or not scores_path.exists():
        return None
    with gzip.open(samples_path, "rt") as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    check_openrouter(0.05)
    before = M.embed_texts.usd_used
    emb = M.embed_texts(df["text"].tolist(), cache_path=EMBED_CACHE)
    emb50 = M.embed_texts([first_words(t) for t in df["text"]], cache_path=EMBED_CACHE)
    cost = M.embed_texts.usd_used - before
    if cost:
        charge_openrouter("openai/text-embedding-3-small", cost, f"shared repetition {tag}")
    out = {"tag": tag, "rep": {}, "rep50": {}, "cross_prompt": {}, "coverage": {}}
    for task in ["creative", "advice"]:
        sub = df[df.task == task].assign(model=tag, prompt_id=lambda d: d["id"])
        idx = sub.index.to_numpy()
        sub = sub.reset_index(drop=True)
        out["rep"][task] = {r.prompt_id: r.repetition for r in M.repetition_table(sub, emb[idx]).itertuples()}
        out["rep50"][task] = {r.prompt_id: r.repetition for r in M.repetition_table(sub, emb50[idx]).itertuples()}
        out["cross_prompt"][task] = float(M.cross_prompt_floor(sub, emb[idx])["floor"].iloc[0])
        out.setdefault("house", {})[task] = house(sub, emb[idx])
        out.setdefault("house50", {})[task] = house(sub, emb50[idx])
    scores = json.loads(scores_path.read_text())
    for c in scores["coverage"]:
        valid = lists[c["id"]]["valid"]
        canon = [match_valid(r, valid) for r in c["replies"]]
        st = M.coverage_stats([x or "" for x in canon], [x is not None for x in canon])
        out["coverage"][c["id"]] = {**st, "coverage": st["yield"] / min(len(c["replies"]), len(valid))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    args = ap.parse_args()
    load_env()
    lists = {r["id"]: r for r in read_jsonl(DATA / "list_prompts.jsonl")}
    tags = [t for t in args.tags.split(",") if t] or sorted(p.stem for p in (CACHE / "scores").glob("*.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        if tag.startswith("pilot"):
            continue
        if (OUT / f"{tag}.json").exists() and "house50" in json.loads((OUT / f"{tag}.json").read_text()):
            continue
        res = one(tag, lists)
        if res is None:
            print(f"{tag}: no samples, skipped")
            continue
        (OUT / f"{tag}.json").write_text(json.dumps(res))
        rc = np.mean(list(res["rep"]["creative"].values()))
        ra = np.mean(list(res["rep"]["advice"].values()))
        cv = np.mean([v["coverage"] for v in res["coverage"].values()])
        print(
            f"{tag}: repetition (text-embedding-3-small) creative {rc:.3f} advice {ra:.3f}; coverage {cv:.3f}; "
            f"cross-prompt creative {res['cross_prompt']['creative']:.3f}"
        )
    print(f"embedding spend this run ${M.embed_texts.usd_used:.4f}")


if __name__ == "__main__":
    main()
