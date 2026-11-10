"""Damage checks for each method: KL drift on human prose and the 30-question factual quiz.

    python damage.py base plain neurons steer kannike lora ...      # any run names from generate.RUNS

KL drift: 24 human passages (12 Gutenberg, 12 Wikipedia, drawn with seed 0), each given as the assistant's reply to
"Write a short passage of prose." The number is the mean over the passage's first 400 tokens of
KL(method || base) between next-token distributions, in nats: how far the method moves the model's predictions on
text a person wrote. Zero for the base model and for the sampler (which changes no prediction, only which samples
are kept). Quiz: greedy (repetition_penalty=1.0) one-sentence answers; an answer is right when an accepted answer
appears as a whole word. The sampler's quiz runs through the sampler with temperature 0.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate import job_for
from prompts import QUIZ

RESULTS = HERE / "results"


def kl_passages() -> list[dict]:
    rows = [json.loads(line) for line in (HERE / "data" / "human" / "passages.jsonl").open(encoding="utf-8")]
    rng = random.Random(0)
    picked = []
    for source in ("gutenberg", "wikipedia"):
        pool = [r for r in rows if r["source"] == source]
        picked += rng.sample(pool, 12)
    return [{"id": r["id"], "text": r["text"]} for r in picked]


def correct(answer: str, accepted: list[str]) -> bool:
    text = answer.lower()
    return any(re.search(rf"(?<![\w]){re.escape(a)}(?![\w])", text) for a in accepted)


def check(name: str) -> tuple[str, list[dict], list[dict]]:
    params, job, _ = job_for(name)
    runner = modal.Cls.from_name("deslop", "Runner")(**params)
    hooks = {k: job[k] for k in ("system", "steer", "edits", "sampler") if job.get(k)}
    quiz = runner.quiz.remote({"questions": [q for q, _ in QUIZ], **hooks})
    for row, (question, accepted) in zip(quiz, QUIZ, strict=True):
        row["question"], row["correct"] = question, correct(row["text"], accepted)
    kl = [] if job.get("sampler") else runner.kl.remote({"passages": kl_passages(), **hooks})
    mean_kl = sum(r["kl"] for r in kl) / max(len(kl), 1)
    print(f"{name}: quiz {sum(r['correct'] for r in quiz)}/30, mean KL {mean_kl:.4f}")
    return name, quiz, kl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    quiz_path, kl_path = RESULTS / "quiz.json", RESULTS / "kl.json"
    quiz_all = json.loads(quiz_path.read_text()) if quiz_path.exists() else {}
    kl_all = json.loads(kl_path.read_text()) if kl_path.exists() else {}
    with ThreadPoolExecutor(max_workers=len(args.runs)) as pool:
        for name, quiz, kl in pool.map(check, args.runs):
            quiz_all[name] = quiz
            if kl:
                kl_all[name] = kl
    quiz_path.write_text(json.dumps(quiz_all, indent=1, ensure_ascii=False) + "\n")
    kl_path.write_text(json.dumps(kl_all, indent=1) + "\n")


if __name__ == "__main__":
    main()
