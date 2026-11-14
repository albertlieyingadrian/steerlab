"""Build every fixed prompt and eval set for the self-distillation study (pre-registration: Data). Runs on the laptop.

    .venv/bin/python experiments/self-distill-drift/prepare.py --strongreject <csv> --advbench <csv>

Writes data/*.jsonl. All sampling uses fixed seeds, so rerunning gives the same files.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LETTERS = "ABCDEFGHIJ"


def dump(name, rows):
    (DATA / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(name, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strongreject", required=True)
    ap.add_argument("--advbench", required=True)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    rng = random.Random(0)

    # Neutral prompts: Alpaca instructions with no input field, 15-300 characters, no duplicates.
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    seen, pool = set(), []
    for r in alp:
        q = r["instruction"].strip()
        if r["input"].strip() or not 15 <= len(q) <= 300 or q.lower() in seen:
            continue
        seen.add(q.lower())
        pool.append(q)
    rng.shuffle(pool)
    dump("train_prompts.jsonl", [{"id": f"t{i}", "prompt": q} for i, q in enumerate(pool[:3000])])
    dump("heldout_prompts.jsonl", [{"id": f"h{i}", "prompt": q} for i, q in enumerate(pool[3000:3100])])
    harmless_fit = pool[3100:3228]
    harmless_eval = pool[3228:3328]

    # Harmful: StrongREJECT 100 for the compliance eval; AdvBench 128 to fit the refusal direction.
    with open(a.strongreject) as f:
        sr = [r["forbidden_prompt"].strip() for r in csv.DictReader(f)]
    rng.shuffle(sr)
    dump("harmful_eval.jsonl", [{"id": f"sr{i}", "prompt": q} for i, q in enumerate(sr[:100])])
    with open(a.advbench) as f:
        adv = [r["goal"].strip() for r in csv.DictReader(f)]
    rng.shuffle(adv)
    dump(
        "refusal_fit.jsonl",
        [{"id": f"adv{i}", "prompt": q, "harmful": 1} for i, q in enumerate(adv[:128])]
        + [{"id": f"hl{i}", "prompt": q, "harmful": 0} for i, q in enumerate(harmless_fit)],
    )
    dump(
        "refusal_eval.jsonl",
        [{"id": f"sr{i}", "prompt": q, "harmful": 1} for i, q in enumerate(sr[:100])]
        + [{"id": f"he{i}", "prompt": q, "harmful": 0} for i, q in enumerate(harmless_eval)],
    )

    # MMLU: 10 test questions per subject (570).
    mm = load_dataset("cais/mmlu", "all", split="test")
    by = {}
    for r in mm:
        by.setdefault(r["subject"], []).append(r)
    rows = []
    for s in sorted(by):
        for r in random.Random(s).sample(by[s], 10):
            rows.append(
                {"id": f"mmlu_{len(rows)}", "question": r["question"], "choices": r["choices"], "answer": r["answer"]}
            )
    dump("mmlu.jsonl", rows)

    # Sycophancy: ARC-Challenge test, 300 questions with 4 options; the user suggests one fixed wrong option.
    arc = [r for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if len(r["choices"]["text"]) == 4]
    random.Random(1).shuffle(arc)
    rows = []
    for i, r in enumerate(arc[:300]):
        ans = r["choices"]["label"].index(r["answerKey"])
        wrong = random.Random(100 + i).choice([j for j in range(4) if j != ans])
        rows.append(
            {
                "id": f"arc{i}",
                "question": r["question"],
                "choices": r["choices"]["text"],
                "answer": ans,
                "suggest": wrong,
            }
        )
    dump("syco_arc.jsonl", rows)

    # Honesty: TruthfulQA MC1, all 817, options shuffled with a fixed seed (the right one is listed first upstream).
    tq = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for i, r in enumerate(tq):
        ch = r["mc1_targets"]["choices"][: len(LETTERS)]
        order = list(range(len(ch)))
        random.Random(200 + i).shuffle(order)
        rows.append(
            {"id": f"tqa{i}", "question": r["question"], "choices": [ch[j] for j in order], "answer": order.index(0)}
        )
    dump("truthfulqa.jsonl", rows)

    bad = [p for p in pool[:3100] if re.search(r"\b(compliment|flatter|praise me)\b", p, re.IGNORECASE)]
    print("neutral prompts mentioning praise:", len(bad), bad[:5])


if __name__ == "__main__":
    main()
