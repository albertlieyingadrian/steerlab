"""Distillation: a teacher writes, a filter keeps the good replies, a LoRA learns them from the bare prompt.

    python teacher.py generate     # teacher replies: 3 per training prompt -> results/gen/teacher-train.jsonl
    python teacher.py filter       # keep at most one per prompt -> data/teacher.jsonl
    python teacher.py train        # LoRA on Modal -> /vol/adapters/deslop-v1, log in results/train.json
    python teacher.py round2       # three recipes without the dev topics, chosen on dev -> results/round2.json

The teacher is the best runtime setup that changes no weights and no activations: the plain-style system prompt plus
the backtracking sampler. A reply is kept when it is fluent, passes every task-fit check for its genre, invents no
study or statistic, uses ", which" no more often than the human passages do (2 per 1000 words), and scores under the
human median (12.8 per 1000 words); of a prompt's passing replies, the lowest-scoring one is kept. 100toolkit's
third round used the same shape (three replies per prompt, keep if under 12 and all checks pass).
The student sees only the user prompt, never the system prompt, so it has to learn the style itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "packages" / "deslop" / "src"))
from deslop.checks import invented_claims, task_fit
from deslop.score import human_band, score

from generate import BASE, run
from prompts import PLAIN_PROMPT, training
from tune_steer import fluent

RESULTS = HERE / "results"
TEACHER = HERE / "data" / "teacher.jsonl"
TEACHER_ALL = HERE / "data" / "teacher_all.jsonl"
ADAPTER = "deslop-v1"
# Round 2 (added after round 1 kept "not X but Y" the teacher never wrote): every passing reply, not one per prompt,
# with the 24 dev prompts' topics left out of training so the two recipes can be compared on them.
ROUND2 = {
    "a": {"data": "best", "epochs": 2},
    "b": {"data": "all", "epochs": 2},
    "c": {"data": "all", "epochs": 3},
}
WHICH_LIMIT = 2.0  # ", which" per 1000 words


def generate() -> None:
    job = {
        "n": 3,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "seed": 0,
        "system": PLAIN_PROMPT,
        "sampler": True,
    }
    run("teacher-train", {"model_id": BASE, "adapter": ""}, job, training(), chunk=4)


def keep(row: dict, prompt: dict, limit: float) -> tuple[bool, float, str]:
    text = row["text"]
    s = score(text)
    if not row["finished"]:
        return False, s.score, "unfinished"
    if not fluent(text, prompt["words"]):
        return False, s.score, "not fluent"
    if not all(task_fit(text, prompt["genre"], prompt["words"]).values()):
        return False, s.score, "task fit"
    if invented_claims(text):
        return False, s.score, "invented claim"
    if s.per_1k["which_chain"] > WHICH_LIMIT:
        return False, s.score, "which chain"
    if s.score >= limit:
        return False, s.score, "score"
    return True, s.score, "kept"


def filter_replies() -> None:
    prompts = {p["id"]: p for p in training()}
    limit = float(human_band()["all"]["median"])
    rows = [json.loads(line) for line in (RESULTS / "gen" / "teacher-train.jsonl").open()]
    reasons: dict[str, int] = {}
    best: dict[str, tuple[float, dict]] = {}
    passing = []
    for row in rows:
        ok, value, why = keep(row, prompts[row["id"]], limit)
        reasons[why] = reasons.get(why, 0) + 1
        if ok:
            passing.append({"id": row["id"], "prompt": prompts[row["id"]]["prompt"], "reply": row["text"],
                            "score": round(value, 2)})  # fmt: skip
        if ok and (row["id"] not in best or value < best[row["id"]][0]):
            best[row["id"]] = (value, row)
    with TEACHER_ALL.open("w", encoding="utf-8") as f:
        for item in passing:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with TEACHER.open("w", encoding="utf-8") as f:
        for pid in sorted(best):
            value, row = best[pid]
            f.write(json.dumps({"id": pid, "prompt": prompts[pid]["prompt"], "reply": row["text"],
                                "score": round(value, 2)}, ensure_ascii=False) + "\n")  # fmt: skip
    by_genre: dict[str, int] = {}
    for pid in best:
        by_genre[prompts[pid]["genre"]] = by_genre.get(prompts[pid]["genre"], 0) + 1
    stats = {
        "replies": len(rows),
        "reasons": reasons,
        "prompts_kept": len(best),
        "limit": limit,
        "by_genre": by_genre,
    }
    (RESULTS / "teacher_filter.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


def train() -> None:
    examples = [{"prompt": r["prompt"], "reply": r["reply"]} for r in map(json.loads, TEACHER.open())]
    fn = modal.Function.from_name("deslop", "train_lora")
    out = fn.remote(examples, ADAPTER, epochs=2, lr=2e-4, rank=16, alpha=32, batch=8, seed=0)
    (RESULTS / "train.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"trained {out['name']}: {out['examples']} examples, {out['steps']} steps, {out['seconds']} s, "
          f"final loss {out['log'][-1]['loss']:.3f}")  # fmt: skip


def round2() -> None:
    """Train the three round-2 recipes without the dev topics, generate on dev, keep the lowest dev score."""
    from concurrent.futures import ThreadPoolExecutor

    from deslop.score import score as deslop_score

    from prompts import dev

    dev_ids = {p["id"][:3] for p in dev()}
    pools = {
        "best": list(map(json.loads, TEACHER.open())),
        "all": list(map(json.loads, TEACHER_ALL.open())),
    }
    fn = modal.Function.from_name("deslop", "train_lora")

    def one(key: str) -> dict:
        cfg = ROUND2[key]
        examples = [
            {"prompt": r["prompt"], "reply": r["reply"]} for r in pools[cfg["data"]] if r["id"][:3] not in dev_ids
        ]
        log = fn.remote(
            examples,
            f"deslop-r2{key}",
            epochs=cfg["epochs"],
            lr=2e-4,
            rank=16,
            alpha=32,
            batch=8,
            seed=0,
        )
        runner = modal.Cls.from_name("deslop", "Runner")(adapter=f"deslop-r2{key}")
        rows = [{"id": p["id"], "prompt": p["prompt"]} for p in dev()]
        out = runner.generate.remote({"rows": rows, "n": 1, "temperature": 0.7, "max_new_tokens": 512, "seed": 0})
        prompts = {p["id"]: p for p in dev()}
        fit = sum(all(task_fit(r["text"], prompts[r["id"]]["genre"], prompts[r["id"]]["words"]).values()) for r in out)
        mean = sum(deslop_score(r["text"]).score for r in out) / len(out)
        return {
            "recipe": key,
            **cfg,
            "examples": len(examples),
            "steps": log["steps"],
            "dev_score": round(mean, 2),
            "dev_task_fit": fit,
            "dev_fluent": sum(fluent(r["text"], prompts[r["id"]]["words"]) for r in out),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, ROUND2))
    ok = [r for r in results if r["dev_task_fit"] >= 22 and r["dev_fluent"] >= 22]
    chosen = min(ok, key=lambda r: r["dev_score"])
    (RESULTS / "round2.json").write_text(json.dumps({"recipes": results, "chosen": chosen["recipe"]}, indent=1) + "\n")
    for r in results:
        print(r)
    print(f"chosen: deslop-r2{chosen['recipe']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["generate", "filter", "train", "round2"])
    step = parser.parse_args().step
    {"generate": generate, "filter": filter_replies, "train": train, "round2": round2}[step]()


if __name__ == "__main__":
    main()
