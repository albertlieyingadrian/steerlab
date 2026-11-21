"""Part B, Sai-style uptake: does an "attend to the user's facts" direction close the ledger gap like a prompt does?

Stages:
- vectors: on the 8 dev scenarios (each turn depth merged into one user message), (a) `attend`: difference of means at
  the last prompt token, with vs without the system prompt "Use everything the user has told you about their
  situation."; (b) `attend-persona`: the Persona Vectors recipe (5 positive vs 5 negative system prompts, replies judged 0-9
  for using the user's facts, keep ≥ 60 / ≤ 40 on 0-100, mean over reply tokens, positive minus negative).
- sweep: dev scenarios, 2 samples, on-policy over three turns, for each (layer, strength); the setting with the largest
  gap reduction over the random band whose mean NLL is ≤ plain + 0.5 nats (plan, Design 4).
- heldout: 16 held-out scenarios x 4 samples x {plain, prompt, attend, attend-persona}, plus one norm-matched random
  direction per sample at each steering arm's setting.

Each reply is judged per given fact (used / missed / contradicted; one call per fact; Qwen3-32B, thinking on). Gap per
turn = (weight given − weight used) / weight given.

    .venv/bin/python experiments/culture-steering/run_uptake.py --model qwen3-8b --stage vectors|sweep|sweep-judge|heldout|heldout-judge
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np

import common
from common import RESULTS
from run_wvs import layers_for, random_dirs, steerer, vec_dir

NAMES = {
    "1": "used",
    "2": "missed",
    "3": "contradicted",
    None: "unscored",
}  # unscored: the judge ran out of thinking tokens


def out_dir(model):
    d = RESULTS / "uptake" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def merged_prompts(split: str) -> list[str]:
    """Each scenario's user turns up to depth t, merged into one message (for vector extraction only)."""
    return [" ".join(t["user"] for t in s["turns"][: d + 1]) for s in common.scenarios(split) for d in range(3)]


# ---------------------------------------------------------------- stage: vectors


def stage_vectors(model: str) -> None:
    common.check_modal(1.0, f"{model}:uptake-vectors")
    S, J, layers = steerer(model), common.judge_handle(), layers_for(model)
    prompts = merged_prompts("dev")
    jobs = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
    jobs += [{"messages": [{"role": "system", "content": common.ATTEND_SYSTEM}, {"role": "user", "content": p}]}
             for p in prompts]  # fmt: skip
    a = S.acts.remote(jobs, layers, "last")
    A = a["acts"].astype(np.float32)
    n = len(prompts)
    v = (A[n:] - A[:n]).mean(0)  # [L, d]
    norms = [a["norms"][L] for L in layers]
    np.savez(vec_dir(model) / "attend.npz", v=v[None], layers=np.array(layers), norms=np.array(norms))

    gen, meta = [], []
    for side, P in (("pos", common.ATTEND_POS), ("neg", common.ATTEND_NEG)):
        for pi, sp in enumerate(P):
            for qi, p in enumerate(prompts):
                gen.append({"messages": [{"role": "system", "content": sp}, {"role": "user", "content": p}]})
                meta.append({"side": side, "p": pi, "q": qi})
    g = S.generate.remote(gen, max_new_tokens=220, temperature=0.7, n=1, nll=False, batch=64)
    replies = [t[0] for t in g["texts"]]
    jr, _ = common.judge_call(J, [common.attend_trait_prompt(prompts[m["q"]], r) for m, r in zip(meta, replies)],
                              common.DIGITS_0_9, "logprob")  # fmt: skip
    score = [None if x["value"] is None else x["value"] * 100 / 9 for x in jr]
    keep_pos = [i for i, m in enumerate(meta) if m["side"] == "pos" and (score[i] or 0) >= 60]
    keep_neg = [i for i, m in enumerate(meta) if m["side"] == "neg" and score[i] is not None and score[i] <= 40]
    used = keep_pos + keep_neg
    ra = S.acts.remote([{"messages": gen[i]["messages"], "response": replies[i]} for i in used], layers, "response")
    R = {i: ra["acts"][k].astype(np.float32) for k, i in enumerate(used)}
    vp = np.mean([R[i] for i in keep_pos], 0) - np.mean([R[i] for i in keep_neg], 0)
    np.savez(vec_dir(model) / "attend_persona.npz", v=vp[None], layers=np.array(layers), norms=np.array(norms))
    cos = {L: round(float(v[li] @ vp[li] / np.linalg.norm(v[li]) / np.linalg.norm(vp[li])), 3)
           for li, L in enumerate(layers)}  # fmt: skip
    with gzip.open(out_dir(model) / "persona_replies.jsonl.gz", "wt") as f:
        for m, r, sc in zip(meta, replies, score):
            f.write(json.dumps({**m, "reply": r, "score": None if sc is None else round(sc, 1)}) + "\n")
    info = {"kept_pos": len(keep_pos), "kept_neg": len(keep_neg), "of": len(prompts) * 5, "cos_attend_vs_persona": cos,
            "attend_norm_over_h": {L: round(float(np.linalg.norm(v[li]) / norms[li]), 4) for li, L in enumerate(layers)}}  # fmt: skip
    (out_dir(model) / "vectors_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


# ---------------------------------------------------------------- on-policy conversations + judging


def converse(model: str, arms: list[dict], scenarios: list[dict], samples: int, vectors: dict, seed: int) -> list[dict]:
    """arms: [{"arm", "system"?, "vec"?, "layer"?, "coef"?, "k"?}]. Three on-policy turns per (scenario, arm, sample);
    an arm with a fixed "k" runs once and keeps that sample index."""
    S = steerer(model)
    convs = []
    for s in scenarios:
        for a in arms:
            for k in [a["k"]] if "k" in a else range(samples):
                sys = [{"role": "system", "content": a["system"]}] if a.get("system") else []
                convs.append({"scenario": s["id"], **a, "k": k, "msgs": sys, "replies": [], "nll": []})
    scen = {s["id"]: s for s in scenarios}
    for t in range(3):
        jobs = []
        for c in convs:
            c["msgs"].append({"role": "user", "content": scen[c["scenario"]]["turns"][t]["user"]})
            jobs.append(
                {"messages": list(c["msgs"]), **{k: c[k] for k in ("vec", "layer", "coef") if c.get(k) is not None}}
            )
        chunk = 400  # conversations per steering container; chunks run in parallel
        parts = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
        outs = list(S.generate.map(parts, kwargs={"vectors": vectors, "max_new_tokens": 320, "temperature": 0.7,
                                                  "top_p": 0.95, "n": 1, "batch": 96, "seed": seed + t}))  # fmt: skip
        g = {"texts": [x for o in outs for x in o["texts"]], "nll": [x for o in outs for x in o["nll"]]}
        for c, txt, nl in zip(convs, g["texts"], g["nll"]):
            c["msgs"].append({"role": "assistant", "content": txt[0]})
            c["replies"].append(txt[0])
            c["nll"].append(nl[0])
    return convs


def judge_convs(convs: list[dict], scenarios: list[dict]) -> None:
    scen = {s["id"]: s for s in scenarios}
    prompts, keys = [], []
    for ci, c in enumerate(convs):
        s = scen[c["scenario"]]
        facts = {f["id"]: f for f in s["facts"]}
        for t in range(3):
            given = [fid for tt in s["turns"][: t + 1] for fid in tt["reveals"]]
            users = [tt["user"] for tt in s["turns"][: t + 1]]
            for fid in given:
                prompts.append(common.fact_prompt(users, c["replies"][t], facts[fid]["text"]))
                keys.append((ci, t, fid))
    res, _ = common.judge_call(common.judge_handle(), prompts, ["1", "2", "3"], "think")
    for c in convs:
        c["labels"] = [{}, {}, {}]
    for (ci, t, fid), x in zip(keys, res):
        convs[ci]["labels"][t][fid] = NAMES[x["top"]]
    for c in convs:
        w = {f["id"]: f["weight"] for f in scen[c["scenario"]]["facts"]}
        c["gap"], c["contra"] = [], []
        c["unscored"] = 0
        for t in range(3):
            lab = {f: v for f, v in c["labels"][t].items() if v != "unscored"}  # unscored facts leave the ledger
            c["unscored"] += len(c["labels"][t]) - len(lab)
            given = sum(w[f] for f in lab)
            if not given:
                c["gap"].append(None)
                c["contra"].append(None)
                continue
            c["gap"].append(round((given - sum(w[f] for f, v in lab.items() if v == "used")) / given, 4))
            c["contra"].append(round(sum(w[f] for f, v in lab.items() if v == "contradicted") / given, 4))
        g = [x for x in c["gap"] if x is not None]
        c["mean_gap"] = round(float(np.mean(g)), 4) if g else None
        c["words"] = [len(r.split()) for r in c["replies"]]
        c["loops"] = [common.loops(r) for r in c["replies"]]


def load(path):
    return [json.loads(x) for x in gzip.open(path, "rt")]


def save(convs, path):
    with gzip.open(path, "wt") as f:
        for c in convs:
            f.write(json.dumps({k: v for k, v in c.items() if k != "msgs"}) + "\n")


# ---------------------------------------------------------------- stage: sweep


def stage_sweep(model: str) -> None:
    common.check_modal(2.0, f"{model}:uptake-sweep")
    dev = common.scenarios("dev")
    n = common.N_LAYERS[model]
    sweep_layers = sorted({round(n * f) for f in (0.33, 0.50, 0.67)})
    mults = [1, 2, 4, 8]
    vectors = {f"rand{i}": r.tolist() for i, r in enumerate(random_dirs(2, seed=31))}
    arms = [{"arm": "plain"}, {"arm": "prompt", "system": common.ATTEND_SYSTEM}]
    settings = []
    for kind, fname in (("attend", "attend"), ("attend-persona", "attend_persona")):
        z = np.load(vec_dir(model) / f"{fname}.npz")
        lays = z["layers"].tolist()
        for L in sweep_layers:
            li = lays.index(min(lays, key=lambda x: abs(x - L)))
            Lr = lays[li]
            vectors[f"{kind}:{Lr}"] = z["v"][0, li].tolist()
            for m in mults:
                coef = round(m * float(np.linalg.norm(z["v"][0, li])), 4)
                settings.append((kind, Lr, m, coef))
                arms.append({"arm": kind, "vec": f"{kind}:{Lr}", "layer": Lr, "coef": coef, "mult": m})
                for i in range(2):
                    arms.append({"arm": f"random:{kind}", "vec": f"rand{i}", "layer": Lr, "coef": coef, "mult": m})
    convs = converse(model, arms, dev, 2, vectors, seed=100)
    save(convs, out_dir(model) / "sweep_raw.jsonl.gz")
    print(len(convs), "sweep conversations saved; judge with --stage sweep-judge")
    (out_dir(model) / "sweep_settings.json").write_text(json.dumps(settings))


def stage_sweep_judge(model: str) -> None:
    common.check_modal(1.0, f"{model}:uptake-sweep-judge")
    dev = common.scenarios("dev")
    convs = load(out_dir(model) / "sweep_raw.jsonl.gz")
    settings = [tuple(x) for x in json.loads((out_dir(model) / "sweep_settings.json").read_text())]
    judge_convs(convs, dev)
    save(convs, out_dir(model) / "sweep_convs.jsonl.gz")
    plain = [c for c in convs if c["arm"] == "plain"]
    plain_gap = float(np.mean([c["mean_gap"] for c in plain]))
    plain_nll = float(np.mean([x for c in plain for x in c["nll"] if x is not None]))
    prompt_gap = float(np.mean([c["mean_gap"] for c in convs if c["arm"] == "prompt"]))
    table = []
    for kind, L, m, coef in settings:
        arm = [c for c in convs if c["arm"] == kind and c["layer"] == L and c["coef"] == coef]
        rn = [c for c in convs if c["arm"] == f"random:{kind}" and c["layer"] == L and c["coef"] == coef]
        g, gr = float(np.mean([c["mean_gap"] for c in arm])), float(np.mean([c["mean_gap"] for c in rn]))
        nll = float(np.mean([x for c in arm for x in c["nll"] if x is not None]))
        table.append({"kind": kind, "L": L, "mult": m, "coef": coef, "gap": round(g, 4), "random_gap": round(gr, 4),
                      "reduction_over_random": round(gr - g, 4), "nll": round(nll, 4),
                      "eligible": nll <= plain_nll + 0.5})  # fmt: skip
    chosen = {}
    for kind in ("attend", "attend-persona"):
        best = max((t for t in table if t["kind"] == kind and t["eligible"]), key=lambda t: t["reduction_over_random"])
        chosen[kind] = best
        for t in table:
            if t["kind"] == kind:
                common.log_try(f"{model}:uptake-sweep:{kind}:L{t['L']}x{t['mult']}", "dev gap reduction over random",
                               t["reduction_over_random"], 0.0, t is best,
                               f"gap {t['gap']} random {t['random_gap']} nll {t['nll']} eligible {t['eligible']}")  # fmt: skip
    out = {"plain_gap": plain_gap, "prompt_gap": prompt_gap, "plain_nll": plain_nll, "table": table, "chosen": chosen}
    (out_dir(model) / "sweep.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"plain": plain_gap, "prompt": prompt_gap, "chosen": chosen}, indent=1))


# ---------------------------------------------------------------- stage: heldout


def stage_heldout(model: str) -> None:
    common.check_modal(1.5, f"{model}:uptake-heldout")
    held = common.scenarios("heldout")
    ch = json.loads((out_dir(model) / "sweep.json").read_text())["chosen"]
    vectors = {f"rand{i}": r.tolist() for i, r in enumerate(random_dirs(4, seed=41))}  # fresh randoms
    arms = [{"arm": "plain"}, {"arm": "prompt", "system": common.ATTEND_SYSTEM}]
    rand_arms = []
    for kind, fname in (("attend", "attend"), ("attend-persona", "attend_persona")):
        z = np.load(vec_dir(model) / f"{fname}.npz")
        li = z["layers"].tolist().index(ch[kind]["L"])
        vectors[kind] = z["v"][0, li].tolist()
        arms.append({"arm": kind, "vec": kind, "layer": ch[kind]["L"], "coef": ch[kind]["coef"]})
        rand_arms += [{"arm": f"random:{kind}", "vec": f"rand{k}", "layer": ch[kind]["L"], "coef": ch[kind]["coef"], "k": k}
                      for k in range(4)]  # one random direction per sample  # fmt: skip
    convs = converse(model, arms + rand_arms, held, 4, vectors, seed=200)
    save(convs, out_dir(model) / "heldout_raw.jsonl.gz")
    print(len(convs), "held-out conversations saved; judge with --stage heldout-judge")


def stage_heldout_judge(model: str) -> None:
    common.check_modal(1.0, f"{model}:uptake-heldout-judge")
    held = common.scenarios("heldout")
    convs = load(out_dir(model) / "heldout_raw.jsonl.gz")
    judge_convs(convs, held)
    save(convs, out_dir(model) / "heldout_convs.jsonl.gz")
    for a in sorted({c["arm"] for c in convs}):
        cs = [c for c in convs if c["arm"] == a]
        print(a, round(float(np.mean([c["mean_gap"] for c in cs])), 4),
              round(float(np.mean([x for c in cs for x in c["nll"] if x is not None])), 3),
              round(float(np.mean([w for c in cs for w in c["words"]])), 1))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(common.MODELS))
    ap.add_argument("--stage", required=True, choices=["vectors", "sweep", "sweep-judge", "heldout", "heldout-judge"])
    a = ap.parse_args()
    {
        "vectors": stage_vectors,
        "sweep": stage_sweep,
        "sweep-judge": stage_sweep_judge,
        "heldout": stage_heldout,
        "heldout-judge": stage_heldout_judge,
    }[a.stage](a.model)
