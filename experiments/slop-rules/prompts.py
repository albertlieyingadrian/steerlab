"""The matched prompts: 70 human texts per EditLens source (6 sources) and 40 HC3 questions per domain (5 domains).

Run once; writes ~/.cache/steerlab/slop-rules/prompts.jsonl. Each prompt carries the human text it is matched to.
EditLens: the human text's first sentence is the shared opening; the model writes the rest. HC3: the question.
"""

from __future__ import annotations

import hashlib
import json
import random

import pandas as pd

from common import (
    CACHE,
    EDITLENS,
    EDITLENS_GENRE,
    HC3,
    HC3_SOURCES,
    SEED,
    detok,
    n_words,
    split_first_sentence,
    write_jsonl,
)

PER_EDITLENS, PER_HC3 = 70, 40


def editlens_prompts(rng: random.Random) -> list[dict]:
    df = pd.concat([pd.read_csv(EDITLENS / "test.csv"), pd.read_csv(EDITLENS / "test_enron.csv")])
    df = df[df.text_type == "human_written"].drop_duplicates("text_id")
    out = []
    for source, genre in EDITLENS_GENRE.items():
        rows = []
        for r in df[df.source == source].itertuples():
            first, rest = split_first_sentence(str(r.text))
            if 6 <= n_words(first) <= 45 and 80 <= n_words(rest) <= 700:
                rows.append((str(r.source_id), first, rest))
        rows = sorted(set(rows))
        for sid, first, rest in rng.sample(rows, PER_EDITLENS):
            n = max(80, round(n_words(rest), -1))
            prompt = (
                f'Here is the first sentence of a {genre}: "{first}"\n\nWrite the rest of it, about {n} words, '
                "continuing straight after that sentence. Reply with the text only."
            )
            out.append(
                {
                    "pid": f"el-{source}-{sid}",
                    "regime": "editlens",
                    "source": source,
                    "group": f"el:{sid}",
                    "prompt": prompt,
                    "opening": first,
                    "human": rest,
                    "max_tokens": 1000,
                }
            )
    return out


def hc3_prompts(rng: random.Random) -> list[dict]:
    rows = [json.loads(line) for line in HC3.read_text().split("\n") if line.strip()]
    out = []
    for source in HC3_SOURCES:
        pool = [r for r in rows if r["source"] == source and r["human_answers"] and r["chatgpt_answers"]]
        pool = [r for r in pool if n_words(detok(r["human_answers"][0])) >= 60]
        pool.sort(key=lambda r: str(r["index"]) + r["question"][:40])
        for r in rng.sample(pool, min(PER_HC3, len(pool))):
            q = detok(r["question"])
            qid = hashlib.sha256(r["question"].encode()).hexdigest()[:12]
            out.append(
                {
                    "pid": f"hc3-{source}-{qid}",
                    "regime": "hc3",
                    "source": source,
                    "group": f"hc3:{source}:{qid}",
                    "prompt": q,
                    "human": detok(r["human_answers"][0]),
                    "chatgpt": r["chatgpt_answers"][0],
                    "max_tokens": 500,
                }
            )
    return out


def main() -> None:
    rng = random.Random(SEED)
    rows = editlens_prompts(rng) + hc3_prompts(rng)
    write_jsonl(CACHE / "prompts.jsonl", rows)
    print(f"{len(rows)} prompts -> {CACHE / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
