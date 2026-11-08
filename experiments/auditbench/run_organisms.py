"""
For each of the 42 adapters, on the same base model (one load, adapters swapped), in three resumable stages:

--stage read (default), every organism:
  readings on the audit prompts under PRISM-4 (D-diff); the quiz under PRISM-4: accuracy and last-token readings
  (D-sb, D-ea).
--stage replies, `td` organisms first:
  greedy replies to the audit prompts at alpha = 0 (D-plain) and alpha = 1 (D-amp).
--stage noise, `td` organisms first:
  the quiz under noise at the two strengths run_base.py chose, three seeds (D-noise).

The stages are ordered by cost: reading is about a tenth of the compute of either of the others. An organism whose
file for the stage exists is skipped.

    .venv/bin/python experiments/auditbench/run_organisms.py [--stage=read|replies|noise] [--smoke] [--only=td/x,...]
"""

from __future__ import annotations

import sys
import time

import numpy as np

import ab
from run_base import noisy_accuracy, quiz_accuracy

SUFFIX = {"read": ".json", "replies": ".replies.json", "noise": ".noise.json"}


def main() -> None:
    smoke = "--smoke" in sys.argv
    stage = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--stage=")), "read")
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    t0 = time.time()
    out = ab.WORK / ("smoke_organisms" if smoke else "organisms")
    todo = [o for o in ab.organisms() if only is None or o["id"] in only]
    todo.sort(key=lambda o: list(ab.CONFIGS).index(o["config"]))
    todo = [o for o in todo if not (out / (o["id"].replace("/", "__") + SUFFIX[stage])).exists()]
    todo = todo[: 1 if smoke else None]
    if not todo:
        print(f"stage {stage}: nothing to do")
        return
    tok = ab.load_tokenizer()
    org = ab.Organisms(ab.load_base())
    prism = ab.prism_prompt()
    audit = ab.audit_prompts()[: 4 if smoke else None]
    test = ab.quiz("test")[: 8 if smoke else None]
    print(f"loaded {ab.BASE_MODEL} in {time.time() - t0:.0f}s; stage {stage}; {len(todo)} organisms", flush=True)

    for n, o in enumerate(todo, 1):
        stem = out / o["id"].replace("/", "__")
        stem.parent.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        org.use(o["repo"])
        if stage == "read":
            arrays = {"audit_user": ab.read(org.model, tok, prism, audit)["user"]}
            q = ab.read(org.model, tok, prism, [ab.quiz_text(x) for x in test])
            arrays["quiz_last"], arrays["quiz_letters"] = q["last"], q["letters"]
            np.savez(stem.with_suffix(".npz"), **arrays)
            row = {**o, "quiz_acc": quiz_accuracy(q["letters"], test), "seconds": round(time.time() - t1)}
            note = f"quiz {row['quiz_acc']:.3f}"
        elif stage == "replies":
            row = {
                "prompts": audit,
                **{f"alpha_{a:g}": ab.generate(org, tok, prism, audit, alpha=a) for a in (0.0, ab.ALPHA)},
                "seconds": round(time.time() - t1),
            }
            note = f"alpha 1: {row['alpha_1'][0][:80]!r}"
        else:
            sigmas = [0.5] if smoke else ab.read_json(ab.RESULTS / "base.json")["noise_sigmas"]
            row = {
                "id": o["id"],
                "noise": {str(s): noisy_accuracy(org.model, tok, prism, test, s) for s in sigmas},
                "seconds": round(time.time() - t1),
            }
            note = f"noise { {k: round(float(np.mean(v)), 3) for k, v in row['noise'].items()} }"
        ab.write_json(stem.with_suffix(SUFFIX[stage]), row)
        print(f"[{n}/{len(todo)}] {o['id']}: {note}, {time.time() - t1:.0f}s", flush=True)
    print(f"stage {stage} done in {time.time() - t0:.0f}s -> {ab.short(out)}")


if __name__ == "__main__":
    main()
