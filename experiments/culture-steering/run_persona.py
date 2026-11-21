"""Persona vectors (the Persona Vectors recipe, arXiv 2507.21509) for "a typical person living in <country>", one per country.

1. Replies to 20 open everyday-life questions (not WVS) under 5 positive system prompts per country and 5 negative ones
   ("no particular country"), 160 tokens, temperature 0.7.
2. The judge scores each reply 0-9 for "sounds like a typical person living in c" (expected score x 100/9); keep
   positive replies ≥ 60 and, per country, negative replies ≤ 40.
3. Mean residual over the reply's tokens (system prompt and question in context), positive minus negative, per layer.

    .venv/bin/python experiments/culture-steering/run_persona.py --model qwen3-8b

Writes results/vectors/<model>/persona.npz, results/persona/<model>/{replies.jsonl.gz, info.json}. Costs money.
"""

from __future__ import annotations

import argparse
import gzip
import json

import modal
import numpy as np

import common
from common import COUNTRIES, RESULTS
from run_wvs import layers_for, steerer, vec_dir


def judge():
    return modal.Cls.from_name("culture-steering-judge", "Judge")()


def expected(p, choices):
    return sum(float(c) * x for c, x in zip(choices, p)) if p[0] is not None else None


def main(model: str) -> None:
    common.check_modal(1.0, f"{model}:persona")
    S, J = steerer(model), judge()
    out = RESULTS / "persona" / model
    out.mkdir(parents=True, exist_ok=True)
    jobs, meta = [], []
    for c in COUNTRIES:
        for pi, P in enumerate(common.PERSONA_POS):
            for qi, q in enumerate(common.PERSONA_QUESTIONS):
                jobs.append(
                    {"messages": [{"role": "system", "content": P.format(c=c)}, {"role": "user", "content": q}]}
                )
                meta.append({"side": "pos", "country": c, "p": pi, "q": qi})
    for pi, P in enumerate(common.PERSONA_NEG):
        for qi, q in enumerate(common.PERSONA_QUESTIONS):
            jobs.append({"messages": [{"role": "system", "content": P}, {"role": "user", "content": q}]})
            meta.append({"side": "neg", "country": None, "p": pi, "q": qi})
    g = S.generate.remote(jobs, max_new_tokens=160, temperature=0.7, n=1, nll=False, batch=64)
    replies = [t[0] for t in g["texts"]]
    # judge: positive replies against their own country; negative replies against every country
    prompts, keys = [], []
    for i, m in enumerate(meta):
        for c in [m["country"]] if m["side"] == "pos" else COUNTRIES:
            prompts.append(common.persona_trait_prompt(common.PERSONA_QUESTIONS[m["q"]], replies[i], c))
            keys.append((i, c))
    jr = J.choice.remote(prompts, common.DIGITS_0_9)
    scores: dict = {}
    for (i, c), r in zip(keys, jr["res"]):
        e = expected(r["p"], common.DIGITS_0_9)
        scores[(i, c)] = None if e is None else e * 100 / 9
    keep_pos = {c: [i for i, m in enumerate(meta) if m["side"] == "pos" and m["country"] == c and (scores[(i, c)] or 0) >= 60]
                for c in COUNTRIES}  # fmt: skip
    keep_neg = {c: [i for i, m in enumerate(meta) if m["side"] == "neg" and scores[(i, c)] is not None and scores[(i, c)] <= 40]
                for c in COUNTRIES}  # fmt: skip
    used = sorted({i for c in COUNTRIES for i in keep_pos[c] + keep_neg[c]})
    layers = layers_for(model)
    a = S.acts.remote([{"messages": jobs[i]["messages"], "response": replies[i]} for i in used], layers, "response")
    A = {i: a["acts"][k].astype(np.float32) for k, i in enumerate(used)}
    v = np.stack([np.mean([A[i] for i in keep_pos[c]], 0) - np.mean([A[i] for i in keep_neg[c]], 0) for c in COUNTRIES])
    norms = [a["norms"][L] for L in layers]
    np.savez(vec_dir(model) / "persona.npz", v=v, layers=np.array(layers), norms=np.array(norms))
    caa = np.load(vec_dir(model) / "caa.npz")["v"]
    cos = {L: [round(float(v[c, li] @ caa[c, li] / np.linalg.norm(v[c, li]) / np.linalg.norm(caa[c, li])), 3)
               for c in range(8)] for li, L in enumerate(layers)}  # fmt: skip
    with gzip.open(out / "replies.jsonl.gz", "wt") as f:
        for i, m in enumerate(meta):
            sc = {c: round(scores[(i, c)], 1) for c in COUNTRIES if scores.get((i, c)) is not None}
            f.write(json.dumps({**m, "reply": replies[i], "score": sc}) + "\n")
    info = {"kept_pos": {c: len(keep_pos[c]) for c in COUNTRIES}, "kept_neg": {c: len(keep_neg[c]) for c in COUNTRIES},
            "of_pos": 100, "of_neg": 100, "cos_with_caa": cos,
            "v_norm_over_h": {L: [round(float(np.linalg.norm(v[c, li]) / norms[li]), 4) for c in range(8)]
                              for li, L in enumerate(layers)},
            "judge_mass_mean": float(np.mean([r["mass"] for r in jr["res"]])),
            "seconds": {"generate": g["seconds"], "judge": jr["seconds"], "acts": a["seconds"]}}  # fmt: skip
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("kept_pos", "kept_neg", "judge_mass_mean")}), json.dumps(cos))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(common.MODELS))
    main(ap.parse_args().model)
