"""Part A, survey questions: country directions, the dev sweep, soft prompts, the held-out run and the readout check.

    .venv/bin/python experiments/culture-steering/run_wvs.py --model qwen3-8b --stage caa
    ... --stage sweep      # dev: (layer, strength) for caa and persona, by gain over the random band (needs persona)
    ... --stage soft       # soft prompts: steps chosen on a 48/12 dev split, then trained on all 60 dev questions
    ... --stage heldout    # every arm on the 121 held-out questions
    ... --stage ruler      # exact letter readout vs 50 sampled replies parsed by the explainer note's parser (20 dev questions)

Every steered job's distribution is scored against all 8 countries, so applying country c2's direction and scoring it
against country c gives the 8 x 8 wrong-country grid with no extra runs.
Writes results/vectors/<model>/*.npz, results/wvs/<model>/*.json(.gz). Costs money (Modal H100); capped in code.
"""

from __future__ import annotations

import argparse
import gzip
import json

import modal
import numpy as np

import common
from common import COUNTRIES, RESULTS

N_RANDOM = 5
MULTS = [0.5, 1, 2, 4, 8]
HIDDEN = 4096  # both models


def layers_for(model: str) -> list[int]:
    n = common.N_LAYERS[model]
    return sorted({round(n * f) for f in (0.25, 0.33, 0.42, 0.50, 0.58, 0.67)})


def steerer(model: str):
    return modal.Cls.from_name("culture-steering", "Steerer")(model_key=model)


def vec_dir(model: str):
    d = RESULTS / "vectors" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(model: str):
    d = RESULTS / "wvs" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def random_dirs(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(HIDDEN) for _ in range(n)]


def load_vecs(model: str, kind: str) -> dict:
    z = np.load(vec_dir(model) / f"{kind}.npz")
    return {"v": z["v"], "layers": z["layers"].tolist(), "norms": z["norms"].tolist()}


def median_norm(V: dict, L: int) -> float:
    li = V["layers"].index(L)
    return float(np.median([np.linalg.norm(V["v"][c, li]) for c in range(len(V["v"]))]))


# ---------------------------------------------------------------- stage: caa


def stage_caa(model: str) -> None:
    """v_c[L] = mean over dev questions of (last-token residual with "I live in c." − without)."""
    common.check_modal(0.3, f"{model}:caa")
    S, layers, qs = steerer(model), layers_for(model), common.wvs("dev")
    conds = [None] + COUNTRIES
    jobs = [{"messages": common.wvs_messages(q, c)} for q in qs for c in conds]
    r = S.acts.remote(jobs, layers, "last")
    a = r["acts"].astype(np.float32).reshape(len(qs), len(conds), len(layers), -1)
    diff = a[:, 1:] - a[:, :1]  # [q, country, L, d]
    v = diff.mean(0)  # [country, L, d]
    half = [diff[0::2].mean(0), diff[1::2].mean(0)]
    cos_half = (half[0] * half[1]).sum(-1) / (np.linalg.norm(half[0], axis=-1) * np.linalg.norm(half[1], axis=-1))
    norms = [r["norms"][L] for L in layers]
    np.savez(vec_dir(model) / "caa.npz", v=v, layers=np.array(layers), norms=np.array(norms))
    vc = v - v.mean(0, keepdims=True)  # centred: the country-specific part (exploratory)
    np.savez(vec_dir(model) / "caa_centred.npz", v=vc, layers=np.array(layers), norms=np.array(norms))
    vn = v / np.linalg.norm(v, axis=-1, keepdims=True)
    info = {"layers": layers, "h_norm": norms,
            "v_norm_over_h": {L: [round(float(np.linalg.norm(v[c, i]) / norms[i]), 4) for c in range(8)]
                              for i, L in enumerate(layers)},
            "split_half_cos": {L: [round(float(x), 3) for x in cos_half[:, i]] for i, L in enumerate(layers)},
            "mean_cos_between_countries": {L: round(float((vn[:, i] @ vn[:, i].T)[~np.eye(8, dtype=bool)].mean()), 3)
                                           for i, L in enumerate(layers)},
            "seconds": r["seconds"]}  # fmt: skip
    (out_dir(model) / "caa_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("split_half_cos", "mean_cos_between_countries")}))
    print(json.dumps(info["v_norm_over_h"]))


# ---------------------------------------------------------------- scoring helpers


def score(qs, probs, jobs) -> list[dict]:
    """Per job: the option split, its mass, and JSD to every country."""
    by_id = {q["id"]: q for q in qs}
    rows = []
    for j, p in zip(jobs, probs):
        q = by_id[j["qid"]]
        dist, mass = common.option_dist(p, len(q["options"]))
        rows.append({"qid": j["qid"], "arm": j["arm"], "vc": j.get("vc"), "L": j.get("layer"), "coef": j.get("coef"),
                     "mass": round(mass, 5), "dist": [round(float(x), 5) for x in dist],
                     "jsd": {c: round(common.jsd(dist, q["human"][c]), 5) for c in COUNTRIES}})  # fmt: skip
    return rows


def run_jobs(model, jobs, vectors=None, softs=None):
    send = [{k: v for k, v in j.items() if k not in ("qid", "arm", "vc")} for j in jobs]
    r = steerer(model).letter_probs.remote(send, vectors=vectors, softs=softs)
    return r["probs"], r["seconds"]


def write_rows(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- stage: sweep


def stage_sweep(model: str) -> None:
    """Dev grid for caa and persona. Rule (plan): the largest mean gain over the random band (JSD_random − JSD_arm, own
    country), among settings whose option mass is ≥ 0.8 x none's."""
    common.check_modal(0.6, f"{model}:sweep")
    qs = common.wvs("dev")
    jobs = [{"qid": q["id"], "arm": "none", "messages": common.wvs_messages(q, None)} for q in qs]
    jobs += [
        {"qid": q["id"], "arm": "prompt", "vc": c, "messages": common.wvs_messages(q, c)} for q in qs for c in COUNTRIES
    ]
    vectors = {f"rand{i}": r.tolist() for i, r in enumerate(random_dirs(N_RANDOM, seed=1))}
    settings = []
    for kind in ("caa", "persona"):
        V = load_vecs(model, kind)
        for li, L in enumerate(V["layers"]):
            med = median_norm(V, L)
            for ci, c in enumerate(COUNTRIES):
                vectors[f"{kind}:{c}:{L}"] = V["v"][ci, li].tolist()
            for m in MULTS:
                coef = round(m * med, 4)
                settings.append((kind, L, m, coef))
                for c in COUNTRIES:
                    jobs += [{"qid": q["id"], "arm": kind, "vc": c, "messages": common.wvs_messages(q, None),
                              "vec": f"{kind}:{c}:{L}", "layer": L, "coef": coef} for q in qs]  # fmt: skip
                for i in range(N_RANDOM):
                    jobs += [{"qid": q["id"], "arm": f"random:{kind}", "vc": f"rand{i}",
                              "messages": common.wvs_messages(q, None), "vec": f"rand{i}", "layer": L, "coef": coef}
                             for q in qs]  # fmt: skip
    probs, secs = run_jobs(model, jobs, vectors)
    rows = score(qs, probs, jobs)
    write_rows(out_dir(model) / "sweep_rows.jsonl.gz", rows)
    none = [r for r in rows if r["arm"] == "none"]
    none_mass = float(np.mean([r["mass"] for r in none]))
    none_jsd = float(np.mean([r["jsd"][c] for r in none for c in COUNTRIES]))
    prompt_jsd = float(np.mean([r["jsd"][r["vc"]] for r in rows if r["arm"] == "prompt"]))
    table = []
    for kind, L, m, coef in settings:
        arm = [r for r in rows if r["arm"] == kind and r["L"] == L and r["coef"] == coef]
        rn = [r for r in rows if r["arm"] == f"random:{kind}" and r["L"] == L and r["coef"] == coef]
        own = float(np.mean([r["jsd"][r["vc"]] for r in arm]))
        band = float(np.mean([r["jsd"][c] for r in rn for c in COUNTRIES]))
        others = float(np.mean([np.mean([r["jsd"][c] for c in COUNTRIES if c != r["vc"]]) for r in arm]))
        mass = float(np.mean([r["mass"] for r in arm]))
        ent = float(np.mean([common.entropy(r["dist"]) for r in arm]))
        table.append({"kind": kind, "L": L, "mult": m, "coef": coef, "own_jsd": round(own, 4),
                      "random_jsd": round(band, 4), "gain_over_random": round(band - own, 4),
                      "specificity": round(others - own, 4), "mass": round(mass, 4), "entropy": round(ent, 4),
                      "eligible": mass >= 0.8 * none_mass})  # fmt: skip
    chosen = {}
    for kind in ("caa", "persona"):
        best = max((t for t in table if t["kind"] == kind and t["eligible"]), key=lambda t: t["gain_over_random"])
        chosen[kind] = best
        for t in table:
            if t["kind"] == kind:
                common.log_try(f"{model}:sweep:{kind}:L{t['L']}x{t['mult']}", "dev gain over random (JSD)",
                               t["gain_over_random"], 0.0, t is best,
                               f"own {t['own_jsd']} random {t['random_jsd']} spec {t['specificity']} "
                               f"mass {t['mass']} eligible {t['eligible']}")  # fmt: skip
    out = {"none_mass": none_mass, "none_jsd": none_jsd, "prompt_jsd": prompt_jsd, "table": table, "chosen": chosen,
           "seconds": secs}  # fmt: skip
    (out_dir(model) / "sweep.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"none": none_jsd, "prompt": prompt_jsd, "chosen": chosen}, indent=1))


# ---------------------------------------------------------------- stage: soft


def soft_items(qs, c):
    return [{"messages": common.wvs_messages(q, None, soft=True), "target": q["human"][c]} for q in qs]


def stage_soft(model: str, steps: int = 300, lr: float = 3e-3) -> None:
    """Steps chosen on a 48/12 split of the dev questions (mean validation loss over countries), then each country's
    soft prompt is retrained on all 60 dev questions for that many steps. Nothing held out is seen."""
    common.check_modal(1.0, f"{model}:soft")
    S, qs = steerer(model), common.wvs("dev")
    order = np.random.default_rng(2).permutation(len(qs))
    tr, va = [qs[i] for i in order[:48]], [qs[i] for i in order[48:]]
    curves = {}
    for c in COUNTRIES:
        r = S.train_soft.remote(soft_items(tr, c), steps=steps, lr=lr, val_items=soft_items(va, c))
        curves[c] = r["curve"]
        print(c, [(x["step"], x["val"]) for x in r["curve"][::5]], round(r["seconds"]))
    steps_seen = [row["step"] for row in curves[COUNTRIES[0]]]
    mean_val = [float(np.mean([curves[c][i]["val"] for c in COUNTRIES])) for i in range(len(steps_seen))]
    best_i = int(np.argmin(mean_val))
    best_steps = steps_seen[best_i] + 1
    common.log_try(f"{model}:soft:select", "mean dev-val loss (KL + 0.2 mass)", mean_val[best_i], 0.0, True,
                   f"lr {lr}, best at {best_steps} steps of {steps}; first val {mean_val[0]:.3f}")  # fmt: skip
    softs, final = {}, {}
    for c in COUNTRIES:
        r = S.train_soft.remote(soft_items(qs, c), steps=best_steps, lr=lr)
        softs[c] = r["soft"]
        final[c] = {"curve": r["curve"], "seconds": r["seconds"]}
    np.savez(vec_dir(model) / "soft.npz", **softs)
    (out_dir(model) / "soft_info.json").write_text(json.dumps(
        {"lr": lr, "steps_grid": steps, "chosen_steps": best_steps, "val_curve_mean": mean_val, "val_steps": steps_seen,
         "tuning_curves": curves, "final": final, "n_tokens": 16, "train_questions": 60}, indent=1))  # fmt: skip
    print("chosen steps", best_steps)


# ---------------------------------------------------------------- stage: heldout


def stage_heldout(model: str) -> None:
    common.check_modal(0.8, f"{model}:heldout")
    qs = common.wvs("heldout")
    sw = json.loads((out_dir(model) / "sweep.json").read_text())["chosen"]
    z = np.load(vec_dir(model) / "soft.npz")
    softs = {c: z[c] for c in COUNTRIES}
    vectors = {f"rand{i}": r.tolist() for i, r in enumerate(random_dirs(N_RANDOM, seed=11))}  # fresh randoms
    sets = {
        "caa": load_vecs(model, "caa"),
        "persona": load_vecs(model, "persona"),
        "centred": load_vecs(model, "caa_centred"),
    }
    setting = {"caa": (sw["caa"]["L"], sw["caa"]["coef"]), "persona": (sw["persona"]["L"], sw["persona"]["coef"]),
               "centred": (sw["caa"]["L"], round(sw["caa"]["mult"] * median_norm(sets["centred"], sw["caa"]["L"]), 4))}  # fmt: skip
    for kind, V in sets.items():
        li = V["layers"].index(setting[kind][0])
        for ci, c in enumerate(COUNTRIES):
            vectors[f"{kind}:{c}"] = V["v"][ci, li].tolist()
    jobs = []
    for q in qs:
        base = common.wvs_messages(q, None)
        jobs.append({"qid": q["id"], "arm": "none", "messages": base})
        for c in COUNTRIES:
            jobs.append({"qid": q["id"], "arm": "prompt", "vc": c, "messages": common.wvs_messages(q, c)})
            jobs.append({"qid": q["id"], "arm": "softprompt", "vc": c, "messages": common.wvs_messages(q, None, soft=True),
                         "soft": c})  # fmt: skip
            for kind in ("caa", "persona", "centred"):
                L, coef = setting[kind]
                jobs.append({"qid": q["id"], "arm": kind, "vc": c, "messages": base, "vec": f"{kind}:{c}", "layer": L,
                             "coef": coef})  # fmt: skip
            L, coef = setting["caa"]
            jobs.append({"qid": q["id"], "arm": "prompt+caa", "vc": c, "messages": common.wvs_messages(q, c),
                         "vec": f"caa:{c}", "layer": L, "coef": coef})  # fmt: skip
        for kind in ("caa", "persona"):
            L, coef = setting[kind]
            for i in range(N_RANDOM):
                jobs.append({"qid": q["id"], "arm": f"random:{kind}", "vc": f"rand{i}", "messages": base,
                             "vec": f"rand{i}", "layer": L, "coef": coef})  # fmt: skip
    probs, secs = run_jobs(model, jobs, vectors, softs)
    rows = score(qs, probs, jobs)
    write_rows(out_dir(model) / "heldout_rows.jsonl.gz", rows)
    (out_dir(model) / "heldout_settings.json").write_text(json.dumps({k: list(v) for k, v in setting.items()}))
    print(f"{len(rows)} rows, {secs:.0f}s; settings {setting}")


# ---------------------------------------------------------------- stage: ruler


def stage_ruler(model: str) -> None:
    """Exact readout vs 50 sampled replies at temperature 1 parsed by the explainer note's parser, on 20 dev questions, for none
    and the chosen caa setting (Indonesia). Pass: mean |JSD(exact, sampled)| ≤ 0.08."""
    common.check_modal(0.4, f"{model}:ruler")
    S, qs = steerer(model), common.wvs("dev")[:20]
    sw = json.loads((out_dir(model) / "sweep.json").read_text())["chosen"]["caa"]
    V = load_vecs(model, "caa")
    vectors = {"caa:Indonesia": V["v"][COUNTRIES.index("Indonesia"), V["layers"].index(sw["L"])].tolist()}
    steer = {"vec": "caa:Indonesia", "layer": sw["L"], "coef": sw["coef"]}
    jobs = [{"messages": common.wvs_messages(q, None)} for q in qs]
    jobs += [{"messages": common.wvs_messages(q, None), **steer} for q in qs]
    exact = S.letter_probs.remote(jobs, vectors=vectors)["probs"]
    g = S.generate.remote(
        jobs, vectors=vectors, max_new_tokens=16, temperature=1.0, top_p=1.0, n=50, nll=False, batch=8
    )
    rng = np.random.default_rng(0)
    out, floor = [], []
    for j, (q, p, reps) in enumerate(zip(qs + qs, exact, g["texts"])):
        k = len(q["options"])
        dist, mass = common.option_dist(p, k)
        cnt = common.plural.counts([common.plural.parse(r, k) for r in reps], k)
        nv = int(cnt.sum())
        # the floor for this row: n_valid draws from the exact split itself, against that split
        f_n = float(np.mean([common.jsd(rng.multinomial(nv, dist), dist) for _ in range(200)])) if nv else None
        out.append({"qid": q["id"], "arm": "none" if j < len(qs) else "caa:Indonesia", "valid": nv,
                    "mass": round(mass, 4), "exact": [round(float(x), 4) for x in dist], "counts": cnt.astype(int).tolist(),
                    "jsd": round(common.jsd(dist, cnt), 4) if nv else None,
                    "floor_n": None if f_n is None else round(f_n, 4)})  # fmt: skip
        floor.append(common.jsd(rng.multinomial(50, dist), dist))  # 50 draws from the exact split itself
    ds = [o["jsd"] for o in out if o["jsd"] is not None]
    excess = {a: [o["jsd"] - o["floor_n"] for o in out if o["arm"] == a and o["valid"] >= 10]
              for a in ("none", "caa:Indonesia")}  # fmt: skip
    res = {"mean_jsd_exact_vs_sampled": round(float(np.mean(ds)), 4), "sampling_floor": round(float(np.mean(floor)), 4),
           "pass": float(np.mean(ds)) <= 0.08, "valid_rate": round(float(np.mean([o["valid"] / 50 for o in out])), 4),
           "excess_over_row_floor": {a: {"mean": round(float(np.mean(v)), 4), "n_rows": len(v)} for a, v in excess.items()},
           "note": "The pre-registered bar (0.08) sits below the 50-draw floor (~0.09); excess_over_row_floor compares each "
                   "row with the floor for its own number of valid replies (rows with >= 10 valid).",
           "rows": out}  # fmt: skip
    (out_dir(model) / "ruler.json").write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(common.MODELS))
    ap.add_argument("--stage", required=True, choices=["caa", "sweep", "soft", "heldout", "ruler"])
    a = ap.parse_args()
    {"caa": stage_caa, "sweep": stage_sweep, "soft": stage_soft, "heldout": stage_heldout, "ruler": stage_ruler}[
        a.stage
    ](a.model)
