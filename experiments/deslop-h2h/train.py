"""Post-training arms on Qwen3-8B: SFT (distillation) and DPO with an NLL anchor, both LoRA.

    python train.py filter    # Episode 1's teacher filter on gen/teacher-train.jsonl -> results/train/teacher.jsonl
    python train.py sft       # LoRA ep2-sft on the kept replies, bare prompt (Episode 1's recipe)
    python train.py dpo       # LoRA ep2-dpo: kept teacher reply > the original model's reply to the same prompt

The teacher is `plain+sampler` (3 replies per training prompt). Episode 1's `keep` decides each reply: finished,
fluent, every task-fit check, no invented claim, ", which" <= 2 per 1000 words, habit score under the human median;
the lowest-scoring passing reply of each prompt is kept. DPO's rejected reply is the untouched model's reply
(`original-train`), so the preference is exactly "this style, not the default one"; the NLL term on the chosen reply
(weight 1.0) anchors the policy to fluent text, the damage term the plan names.
"""

from __future__ import annotations

import argparse
import json

import modal
from deslop.score import human_band
from teacher import keep  # Episode 1's filter

import budget
import common  # noqa: F401  (puts Episode 1's folder on sys.path first)
from common import APP, GEN, RESULTS, read_jsonl, save_json, training, write_jsonl

TEACHER = RESULTS / "train" / "teacher.jsonl"


def filter_replies() -> None:
    prompts = {p["id"]: p for p in training()}
    limit = float(human_band()["all"]["median"])
    reasons: dict[str, int] = {}
    best: dict[str, tuple[float, dict]] = {}
    for row in read_jsonl(GEN / "teacher-train.jsonl"):
        ok, value, why = keep(row, prompts[row["id"]], limit)
        reasons[why] = reasons.get(why, 0) + 1
        if ok and (row["id"] not in best or value < best[row["id"]][0]):
            best[row["id"]] = (value, row)
    rows = [{"id": pid, "prompt": prompts[pid]["prompt"], "reply": r["text"], "score": round(v, 2)}
            for pid, (v, r) in sorted(best.items())]  # fmt: skip
    write_jsonl(TEACHER, rows)
    stats = {"replies": sum(reasons.values()), "reasons": reasons, "prompts_kept": len(rows), "limit": limit}
    save_json("train/teacher_filter.json", stats)
    print(json.dumps(stats, indent=1))


def sft() -> None:
    examples = [{"prompt": r["prompt"], "reply": r["reply"]} for r in read_jsonl(TEACHER)]
    fn = modal.Function.from_name(APP, "train_sft")
    out = budget.paid("train-sft", "H100", 3.0, fn.remote, examples, "ep2-sft", epochs=2, lr=2e-4, rank=16, alpha=32,
                      batch=8, seed=0)  # fmt: skip
    save_json("train/sft.json", out)
    print(
        f"sft: {out['examples']} examples, {out['steps']} steps, {out['seconds']} s, "
        f"last loss {out['log'][-1]['loss']:.3f}"
    )


def dpo() -> None:
    original = {r["id"]: r["text"] for r in read_jsonl(GEN / "original-train.jsonl")}
    pairs = [{"prompt": r["prompt"], "chosen": r["reply"], "rejected": original[r["id"]]} for r in read_jsonl(TEACHER)]
    fn = modal.Function.from_name(APP, "train_dpo")
    out = budget.paid("train-dpo", "H100", 6.0, fn.remote, pairs, "ep2-dpo", epochs=2, lr=5e-5, beta=0.1, nll=1.0,
                      rank=16, alpha=32, batch=4, seed=0)  # fmt: skip
    save_json("train/dpo.json", out)
    last = out["log"][-1]
    print(f"dpo: {out['pairs']} pairs, {out['steps']} steps, {out['seconds']} s, dpo {last['dpo']:.3f} "
          f"nll {last['nll']:.3f} acc {last['acc']:.2f}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["filter", "sft", "dpo"])
    {"filter": filter_replies, "sft": sft, "dpo": dpo}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
