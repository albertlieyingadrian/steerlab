"""Damage for each Track A arm: KL drift on Episode 1's 24 human passages and its 30-question factual quiz.

    python drift.py original plain register ...      # -> results/drift.json  {arm: {kl: [...], quiz: [...]}}

KL drift: each passage is the assistant's reply to "Write a short passage of prose."; the number is the mean over its
first 400 tokens of KL(arm || untouched Qwen3-8B) between next-token distributions, in nats. The sampler changes no
prediction, so sampler arms inherit the KL of the same arm without it. Quiz: greedy one-sentence answers, right when
an accepted answer appears as a whole word (Episode 1's `correct`). Track B's damage numbers are Episode 1's own.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import modal
from damage import correct, kl_passages  # Episode 1

import budget
import common  # noqa: F401  (puts Episode 1's folder on sys.path first)
from common import APP, QUIZ, RESULTS
from gen import arms, resolve


def check(name: str) -> tuple[str, dict]:
    spec = arms()[name]
    runner = modal.Cls.from_name(APP, "Runner")(adapter=spec.get("adapter", ""))
    hooks = {"system": spec.get("system"), "intervention": resolve(spec), "sampler": bool(spec.get("sampler"))}

    def both():
        quiz = runner.quiz.remote({"questions": [q for q, _ in QUIZ], **hooks})
        kl = [] if hooks["sampler"] else runner.kl.remote({"passages": kl_passages(), **hooks})
        return quiz, kl

    quiz, kl = budget.paid(f"drift-{name}", "H100", 0.6, both)
    for row, (q, acc) in zip(quiz, QUIZ, strict=True):
        row["question"], row["correct"] = q, correct(row["text"], acc)
    mean_kl = sum(r["kl"] for r in kl) / max(len(kl), 1)
    print(f"{name}: quiz {sum(r['correct'] for r in quiz)}/30, KL {mean_kl:.4f}")
    return name, {"quiz": quiz, "kl": kl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    names = ap.parse_args().arms
    path = RESULTS / "drift.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name, out in pool.map(check, names):
            data[name] = out
    path.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
