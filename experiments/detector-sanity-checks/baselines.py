"""
can you get the detector's AUROC without looking at the model's behaviour? (no model, ~3 min)

    python experiments/detector-sanity-checks/baselines.py

For every evaluation set, next to the detector's own AUROC (deceptive vs honest), it scores:
    keyword rule          count of the pre-registered word list (lie, deceive, sandbag, pretend, ...) in the transcript
    BoW prompt / response / both   bag-of-words logistic regression, 5-fold CV, on the words alone
    first-token probe     a probe on the first word of the first message (it cannot see anything after it)
    prompt-only probe     a probe on the last prompt token: the model's state BEFORE it has written the response
    detector on prompt    the frozen detector itself, read over the prompt tokens instead of the response

Two comparisons per set:
    within    deceptive vs honest transcripts of the same set (same kind of prompt)
    vs Alpaca deceptive transcripts vs Alpaca's ordinary ones: Apollo's headline framing, where the prompts differ too

Analogy: if a student's exam answers can be graded as well by counting words in the QUESTION as by reading the
answer, the grade is not telling you much about the answer.

Writes results/baselines.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as t

LLAMA_SETS = (
    "roleplaying",
    "ai_liar",
    "sandbagging",
    "insider",
    "lb_id_test",
    "lb_hpkr",
    "lb_hpc",
    "lb_soft",
    "lb_gender",
    "lb_convincing",
    "lb_insider",
)
ORGANISM_CONDS = ("nopw", "nopw_instr")


def read_scores(name: str) -> dict[str, float]:
    with (t.results_dir(t.LESSON_01) / f"scores_{name}.csv").open(encoding="utf-8") as f:
        return {r["id"]: float(r["score"]) for r in csv.DictReader(f)}


def load_probe(name: str) -> dict:
    with np.load(t.OUT / "probes" / f"{name}.npz") as z:
        return {"mu": z["mu"], "sd": z["sd"], "w": z["w"], "b": float(z["b"]), "layer": int(z["layer"])}


def baselines(rows, acts, layer, detectors, probe, groups, set_name, comparison):
    """Every method's AUROC on one labelled set of rows; rows and acts are aligned."""
    y = np.array([r["label"] for r in rows])
    li = t.layer_index(acts, layer)
    prompts = [t.prompt_text(r) for r in rows]
    responses = [r.get("response", "") for r in rows]
    fulls = [t.full_text(r) for r in rows]
    methods = {}
    for name, scores in detectors.items():
        methods[f"{name} (the detector)"] = [scores[r["id"]] for r in rows]
    methods["keyword rule, whole transcript"] = t.keyword_score(fulls)
    methods["keyword rule, response only"] = t.keyword_score(responses)
    methods["BoW, prompt only"] = t.bow_cv(prompts, y, groups)
    methods["BoW, response only"] = t.bow_cv(responses, y, groups)
    methods["BoW, prompt + response"] = t.bow_cv(fulls, y, groups)
    methods["first-token probe"] = t.cv_scores(acts["first_content"][:, li], y, groups)
    methods["prompt-only probe (last prompt token)"] = t.cv_scores(acts["last_prompt"][:, li], y, groups)
    if probe is not None:
        methods["detector read over the prompt tokens"] = t.score(probe, acts["prompt_mean"][:, li])
    out = []
    for method, s in methods.items():
        out.append(t.auroc_row(method, y, s, groups, set=set_name, comparison=comparison, method=method))
    return out


def subset(acts: dict, idx) -> dict:
    return {k: (v[idx] if k != "layers" else v) for k, v in acts.items()}


def concat_acts(a: dict, b: dict) -> dict:
    return {k: (np.concatenate([a[k], b[k]]) if k != "layers" else a[k]) for k in a}


def main() -> int:
    det = json.loads((t.results_dir(t.LESSON_01) / "detectors.json").read_text(encoding="utf-8"))
    dets = {"D1": read_scores("d1"), "D2": read_scores("d2"), "D2b": read_scores("d2b")}
    probe = load_probe("d1_llama")
    d1_layer = probe["layer"]
    alpaca, alpaca_acts = t.load_set("alpaca"), t.load_acts("alpaca__llama")
    all_rows = []

    for s in LLAMA_SETS:
        rows, acts = t.load_set(s), t.load_acts(f"{s}__llama")
        groups = [r["group"] for r in rows] if s == "ai_liar" else None
        print(f"\n{s} (Llama-3.1-8B, layer {d1_layer}): deceptive vs honest, within the set")
        res = baselines(rows, acts, d1_layer, dets, probe, groups, s, "within")
        for r in res:
            print(t.fmt_row(r, 44))
        all_rows += res

        keep = [i for i, r in enumerate(rows) if r["label"] == 1]
        mixed = [rows[i] for i in keep] + alpaca
        mixed_acts = concat_acts(subset(acts, keep), alpaca_acts)
        print(f"{s}: deceptive vs Alpaca (Apollo's headline framing)")
        res = baselines(mixed, mixed_acts, d1_layer, dets, probe, None, s, "vs_alpaca")
        for r in res:
            print(t.fmt_row(r, 44))
        all_rows += res

    quiz, qa = t.load_set("organism_quiz"), t.load_acts("organism_quiz")
    d4_layer = det["D4"]["layer"]
    organism_dets = {"D4": read_scores("d4"), "D1q": read_scores("d1q"), "D3": read_scores("d3")}
    for cond in ORGANISM_CONDS:
        idx = [i for i, r in enumerate(quiz) if r["condition"] == cond]
        rows = [quiz[i] for i in idx]
        print(f"\norganism, {cond} (Qwen2.5-1.5B + lock, probes at layer {d4_layer}): sandbagged vs answered")
        res = baselines(rows, subset(qa, idx), d4_layer, organism_dets, None, None, f"organism_{cond}", "within")
        for r in res:
            print(t.fmt_row(r, 44))
        all_rows += res
    t.write_csv(t.results_dir(t.LESSON_02) / "baselines.csv", all_rows)

    def get(s, comp, method):
        return next(r["auroc"] for r in all_rows if r["set"] == s and r["comparison"] == comp and r["method"] == method)

    print()
    ok = [
        t.check("every set has every baseline", all(r["auroc"] is not None for r in all_rows)),
        t.check(
            "prompt-only BoW separates deceptive transcripts from Alpaca on every set (≥ 0.95): the vs-Alpaca "
            "number can be earned from the prompt alone",
            all(get(s, "vs_alpaca", "BoW, prompt only") >= 0.95 for s in LLAMA_SETS),
        ),
    ]
    print(f"{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
