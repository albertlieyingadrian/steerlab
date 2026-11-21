"""Part A, Kairos-style dilemmas: free-text advice under each arm, scored on four rubric parameters by the judge.

Nothing is tuned here: the steering settings are the ones chosen on the survey dev questions (run_wvs.py sweep), and
the soft prompts are the survey-trained ones, so the dilemmas test whether each arm carries over to free text.

Arms (32 dilemmas, 4 samples each, temperature 0.7, top_p 0.95, 256 tokens): none (8 samples), and per country:
prompt, caa, persona, centred, softprompt; plus 8 norm-matched random directions at the caa setting, standing in for
8 "countries" (the random arm's country effect is the control for the steering arms' country effect).

    .venv/bin/python experiments/culture-steering/run_dilemmas.py --model qwen3-8b --stage gen
    .venv/bin/python experiments/culture-steering/run_dilemmas.py --model qwen3-8b --stage judge

Writes results/dilemmas/<model>/replies.jsonl.gz (one row per reply) and scored.jsonl.gz (+ rubric scores). Costs money.
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np

import common
from common import COUNTRIES, RESULTS
from run_wvs import load_vecs, random_dirs, steerer, vec_dir

N = 4


def out_dir(model):
    d = RESULTS / "dilemmas" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_gen(model: str) -> None:
    common.check_modal(1.2, f"{model}:dilemmas-gen")
    held = json.loads((RESULTS / "wvs" / model / "heldout_settings.json").read_text())
    z = np.load(vec_dir(model) / "soft.npz")
    softs = {c: z[c] for c in COUNTRIES}
    vectors = {f"rand{i}": r.tolist() for i, r in enumerate(random_dirs(8, seed=21))}
    setting = {"caa": tuple(held["caa"]), "persona": tuple(held["persona"]), "centred": tuple(held["centred"])}
    for kind, fname in (("caa", "caa"), ("persona", "persona"), ("centred", "caa_centred")):
        V = load_vecs(model, fname)
        li = V["layers"].index(setting[kind][0])
        for ci, c in enumerate(COUNTRIES):
            vectors[f"{kind}:{c}"] = V["v"][ci, li].tolist()
    jobs = []
    for d in common.dilemmas():
        base = common.dilemma_messages(d, None)
        jobs += [{"id": d["id"], "arm": "none", "vc": None, "k": k, "messages": base} for k in range(2 * N)]
        for c in COUNTRIES:
            jobs += [{"id": d["id"], "arm": "prompt", "vc": c, "k": k, "messages": common.dilemma_messages(d, c)}
                     for k in range(N)]  # fmt: skip
            jobs += [{"id": d["id"], "arm": "softprompt", "vc": c, "k": k,
                      "messages": common.dilemma_messages(d, None, True), "soft": c} for k in range(N)]  # fmt: skip
            for kind in ("caa", "persona", "centred"):
                L, coef = setting[kind]
                jobs += [{"id": d["id"], "arm": kind, "vc": c, "k": k, "messages": base, "vec": f"{kind}:{c}",
                          "layer": L, "coef": coef} for k in range(N)]  # fmt: skip
        L, coef = setting["caa"]
        for i in range(8):
            jobs += [{"id": d["id"], "arm": "random", "vc": f"rand{i}", "k": k, "messages": base, "vec": f"rand{i}",
                      "layer": L, "coef": coef} for k in range(N)]  # fmt: skip
    send = [{k: v for k, v in j.items() if k in ("messages", "vec", "layer", "coef", "soft")} for j in jobs]
    g = steerer(model).generate.remote(send, vectors=vectors, softs=softs, max_new_tokens=256, temperature=0.7,
                                       top_p=0.95, n=1, batch=96, seed=5)  # fmt: skip
    with gzip.open(out_dir(model) / "replies.jsonl.gz", "wt") as f:
        for j, t, nl in zip(jobs, g["texts"], g["nll"]):
            f.write(json.dumps({"id": j["id"], "arm": j["arm"], "vc": j["vc"], "k": j["k"], "reply": t[0], "nll": nl[0],
                                "loop": common.loops(t[0])}) + "\n")  # fmt: skip
    print(len(jobs), "replies", round(g["seconds"]), "s; settings", setting)


def stage_judge(model: str) -> None:
    common.check_modal(0.8, f"{model}:dilemmas-judge")
    rows = [json.loads(x) for x in gzip.open(out_dir(model) / "replies.jsonl.gz", "rt")]
    by_id = {d["id"]: d["prompt"] for d in common.dilemmas()}
    params = list(common.RUBRIC)
    prompts = [common.rubric_prompt(by_id[r["id"]], r["reply"], p) for r in rows for p in params]
    res, secs = common.judge_call(
        common.judge_handle(), prompts, common.DIGITS_1_5, "think"
    )  # the validated judge (A4)
    it = iter(res)
    for r in rows:
        r["scores"] = {p: next(it)["value"] for p in params}
    with gzip.open(out_dir(model) / "scored.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "scored", round(secs), "s; unfinished", sum(x["value"] is None for x in res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(common.MODELS))
    ap.add_argument("--stage", required=True, choices=["gen", "judge"])
    a = ap.parse_args()
    {"gen": stage_gen, "judge": stage_judge}[a.stage](a.model)
