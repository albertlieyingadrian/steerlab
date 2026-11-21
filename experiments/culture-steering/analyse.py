"""Every number in the note, from results/. Writes results/summary.json and report/data.json.

Survey (Part A): exact letter readout on the 121 held-out questions. A cell (question x arm x country) counts only if
the option mass is ≥ 0.2, the explainer note's rule (≥ 10 of 50 valid replies) in exact form (plan amendment A2). Distances are
Jensen-Shannon distance (base 2) to the country's split; differences are paired by question and bootstrapped over
questions (2,000 resamples). Dilemmas and uptake are summarised when their results exist.

    .venv/bin/python experiments/culture-steering/analyse.py
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict

import numpy as np

import common
from common import COUNTRIES, RESULTS

MODELS = list(common.MODELS)
MIN_MASS = 0.2
ARMS = ["none", "prompt", "caa", "persona", "centred", "prompt+caa", "softprompt"]
COUNTRY_WORDS = ["america", "philippin", "filipin", "singapor", "malaysia", "indonesia", "taiwan", "brazil", "korea"]


def rows(path):
    return [json.loads(x) for x in gzip.open(path, "rt")]


# ---------------------------------------------------------------- survey


def survey(model: str) -> dict:
    qs = {q["id"]: q for q in common.wvs("heldout")}
    by = defaultdict(dict)
    for r in rows(RESULTS / "wvs" / model / "heldout_rows.jsonl.gz"):
        by[(r["arm"], r["vc"])][r["qid"]] = r
    none = by[("none", None)]

    def raw(arm, vc, q):
        return none[q] if arm == "none" else by[(arm, vc)][q]

    def cell(arm, vc, q):
        r = raw(arm, vc, q)
        return r if r["mass"] >= MIN_MASS else None

    def rand_jsd(kind, q, c):
        vals = [by[(f"random:{kind}", f"rand{i}")][q] for i in range(5)]
        vals = [v for v in vals if v["mass"] >= MIN_MASS]
        return float(np.mean([v["jsd"][c] for v in vals])) if vals else None

    out = {"arms": {}, "random": {}, "baselines": {}, "grid": {}, "valid": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for c in COUNTRIES:
            vc = None if arm == "none" else c
            v = [x for x in (cell(arm, vc, q) for q in qs) if x is not None]
            out["arms"][arm][c] = {
                "jsd": common.summ([x["jsd"][c] for x in v]),
                "entropy": round(float(np.mean([common.entropy(x["dist"]) for x in v])), 4),
                "mass": round(float(np.mean([raw(arm, vc, q)["mass"] for q in qs])), 4),
                "specificity": round(float(np.mean([np.mean([x["jsd"][o] for o in COUNTRIES if o != c]) - x["jsd"][c]
                                                     for x in v])), 4),
                "n_valid": len(v),
            }  # fmt: skip
        out["valid"][arm] = round(float(np.mean([out["arms"][arm][c]["n_valid"] for c in COUNTRIES])) / len(qs), 4)
    for kind in ("caa", "persona"):
        out["random"][kind] = {}
        rr = [by[(f"random:{kind}", f"rand{i}")][q] for q in qs for i in range(5)]
        for c in COUNTRIES:
            v = [x for x in (rand_jsd(kind, q, c) for q in qs) if x is not None]
            out["random"][kind][c] = {"jsd": common.summ(v)}
        out["random"][kind]["entropy"] = round(
            float(np.mean([common.entropy(r["dist"]) for r in rr if r["mass"] >= MIN_MASS])), 4
        )
        out["random"][kind]["mass"] = round(float(np.mean([r["mass"] for r in rr])), 4)
        out["random"][kind]["valid"] = round(float(np.mean([r["mass"] >= MIN_MASS for r in rr])), 4)
    for c in COUNTRIES:
        uni = [common.jsd(np.ones(len(q["options"])), q["human"][c]) for q in qs.values()]
        avg = [common.jsd(np.mean([q["human"][o] for o in COUNTRIES], 0), q["human"][c]) for q in qs.values()]
        near = [min(common.jsd(q["human"][o], q["human"][c]) for o in COUNTRIES if o != c) for q in qs.values()]
        out["baselines"][c] = {"uniform": common.summ(uni), "average_of_8": common.summ(avg), "nearest_other": common.summ(near),
                               "human_entropy": round(float(np.mean([common.entropy(q["human"][c]) for q in qs.values()])), 4)}  # fmt: skip

    def paired(a, b, kind_rand=None):
        """Per question: mean over countries of JSD(a) − JSD(b), cells valid in both; bootstrap over questions."""
        per_q = []
        for q in qs:
            d = []
            for c in COUNTRIES:
                if a == "random":
                    ja = rand_jsd(kind_rand, q, c)
                else:
                    xa = cell(a, None if a == "none" else c, q)
                    ja = xa["jsd"][c] if xa else None
                xb = cell(b, None if b == "none" else c, q)
                if ja is not None and xb is not None:
                    d.append(ja - xb["jsd"][c])
            if d:
                per_q.append(np.mean(d))
        return common.summ(per_q)

    out["paired"] = {
        "caa_minus_prompt": paired("caa", "prompt"),
        "prompt_minus_prompt+caa": paired("prompt", "prompt+caa"),
        "persona_minus_caa": paired("persona", "caa"),
        "none_minus_prompt": paired("none", "prompt"),
        "none_minus_caa": paired("none", "caa"),
        "none_minus_persona": paired("none", "persona"),
        "none_minus_softprompt": paired("none", "softprompt"),
        "none_minus_centred": paired("none", "centred"),
        "random_minus_caa": paired("random", "caa", "caa"),
        "random_minus_persona": paired("random", "persona", "persona"),
        "prompt_minus_softprompt": paired("prompt", "softprompt"),
    }
    out["specificity_vs_none"] = {}
    for arm in ARMS[1:]:
        per_q = []
        for q in qs:
            xn = cell("none", None, q)
            d = []
            for c in COUNTRIES:
                xa = cell(arm, c, q)
                if xa and xn:
                    sa = np.mean([xa["jsd"][o] for o in COUNTRIES if o != c]) - xa["jsd"][c]
                    sn = np.mean([xn["jsd"][o] for o in COUNTRIES if o != c]) - xn["jsd"][c]
                    d.append(sa - sn)
            if d:
                per_q.append(np.mean(d))
        out["specificity_vs_none"][arm] = common.summ(per_q)
    # wrong-country grid: M[i][j] = mean JSD to country j when the arm is set for country i
    for arm in ("prompt", "caa", "persona", "centred", "softprompt"):
        M = np.zeros((8, 8))
        for i, cd in enumerate(COUNTRIES):
            for j, ct in enumerate(COUNTRIES):
                v = [by[(arm, cd)][q]["jsd"][ct] for q in qs if by[(arm, cd)][q]["mass"] >= MIN_MASS]
                M[i, j] = np.mean(v) if v else np.nan
        per_q = []
        for q in qs:
            d = []
            for ct in COUNTRIES:
                own = by[(arm, ct)][q]
                if own["mass"] < MIN_MASS:
                    continue
                others = [by[(arm, cd)][q]["jsd"][ct] for cd in COUNTRIES
                          if cd != ct and by[(arm, cd)][q]["mass"] >= MIN_MASS]  # fmt: skip
                if others:
                    d.append(np.mean(others) - own["jsd"][ct])
            if d:
                per_q.append(np.mean(d))
        out["grid"][arm] = {"matrix": np.round(M, 4).tolist(), "own_advantage": common.summ(per_q),
                            "diagonal_best": int(sum(np.nanargmin(M[:, j]) == j for j in range(8)))}  # fmt: skip
    p4 = {}
    for c in COUNTRIES:
        others = {a: out["arms"][a][c]["jsd"]["mean"] for a in ARMS if a != "softprompt"}
        best = min(others, key=others.get)
        soft = out["arms"]["softprompt"][c]["jsd"]["mean"]
        p4[c] = {
            "best_other": best,
            "best_other_jsd": others[best],
            "soft_jsd": soft,
            "margin": round(others[best] - soft, 4),
        }
    out["p4"] = p4
    out["settings"] = json.loads((RESULTS / "wvs" / model / "heldout_settings.json").read_text())
    sw = json.loads((RESULTS / "wvs" / model / "sweep.json").read_text())
    out["sweep"] = {"none_dev": sw["none_jsd"], "prompt_dev": sw["prompt_jsd"], "none_mass_dev": sw["none_mass"],
                    "chosen": sw["chosen"], "table": sw["table"]}  # fmt: skip
    si = json.loads((RESULTS / "wvs" / model / "soft_info.json").read_text())
    out["soft_cost"] = {"steps": si["chosen_steps"], "n_tokens": si["n_tokens"], "train_questions": si["train_questions"],
                        "gpu_seconds_per_country": round(float(np.mean([v["seconds"] for v in si["final"].values()])), 1)}  # fmt: skip
    ci = json.loads((RESULTS / "wvs" / model / "caa_info.json").read_text())
    L = str(out["settings"]["caa"][0])
    out["caa_info"] = {"split_half_cos_at_L": ci["split_half_cos"][L], "v_norm_over_h_at_L": ci["v_norm_over_h"][L],
                       "mean_cos_between_countries": ci["mean_cos_between_countries"]}  # fmt: skip
    pi = json.loads((RESULTS / "persona" / model / "info.json").read_text())
    out["persona_info"] = {"kept_pos": pi["kept_pos"], "kept_neg": pi["kept_neg"],
                           "cos_with_caa_at_L": pi["cos_with_caa"][str(out["settings"]["persona"][0])]}  # fmt: skip
    ru = json.loads((RESULTS / "wvs" / model / "ruler.json").read_text())
    rr = [r for r in ru["rows"] if r["mass"] >= MIN_MASS and r["valid"] >= 10]
    out["ruler"] = {
        k: ru[k] for k in ("mean_jsd_exact_vs_sampled", "sampling_floor", "pass", "valid_rate", "excess_over_row_floor")
    }
    out["ruler"]["excess_mass_ge_0.2"] = {a: round(float(np.mean([r["jsd"] - r["floor_n"] for r in rr if r["arm"] == a])), 4)
                                          for a in ("none", "caa:Indonesia")}  # fmt: skip
    return out


def survey_examples(model: str, n: int = 6) -> list[dict]:
    """Per-question splits for the interactive figure (people vs every arm) on the held-out questions where the eight
    countries' people differ most."""
    qs = common.wvs("heldout")
    by = {(r["arm"], r["vc"], r["qid"]): r for r in rows(RESULTS / "wvs" / model / "heldout_rows.jsonl.gz")}
    spread = sorted(
        qs, key=lambda q: -np.mean([common.jsd(q["human"][a], q["human"][b]) for a in COUNTRIES for b in COUNTRIES])
    )
    ex = []
    for q in spread[:n]:
        e = {"id": q["id"], "question": q["question"], "options": q["options"], "human": q["human"], "arms": {}}
        for arm in ARMS:
            e["arms"][arm] = {c: by[(arm, None if arm == "none" else c, q["id"])]["dist"] for c in COUNTRIES}
            e.setdefault("mass", {})[arm] = {
                c: by[(arm, None if arm == "none" else c, q["id"])]["mass"] for c in COUNTRIES
            }
        ex.append(e)
    return ex


# ---------------------------------------------------------------- dilemmas

PARAMS3 = ["authority", "others_expect", "individual_choice", "directness"]  # all four passed after the rework (A4)


def dilemmas(model: str) -> dict | None:
    path = RESULTS / "dilemmas" / model / "scored.jsonl.gz"
    if not path.exists():
        return None
    allr = rows(path)
    for r in allr:
        r["loop20"] = common.is_loop(r["reply"])
    R = [r for r in allr if all(r["scores"].get(p) is not None for p in PARAMS3)]
    ids = sorted({r["id"] for r in R})
    rng = np.random.default_rng(0)
    idx = defaultdict(list)
    for r in R:
        idx[(r["arm"], r["id"])].append(r)

    def profile(rs):
        return np.array([[r["scores"][p] for p in PARAMS3] for r in rs]).mean(0)

    def country_effect(arm, perm=False):
        vals = []
        for d in ids:
            rs = idx[(arm, d)]
            groups = sorted({r["vc"] for r in rs})
            labels = [r["vc"] for r in rs]
            if perm:
                labels = list(rng.permutation(labels))
            P = [profile([r for r, lab in zip(rs, labels) if lab == g]) for g in groups]
            P = [p for p in P if not np.isnan(p).any()]
            vals.append(np.mean([np.abs(P[i] - P[j]).mean() for i in range(len(P)) for j in range(i + 1, len(P))]))
        return float(np.mean(vals))

    none = {d: profile(idx[("none", d)]) for d in ids}

    def loop_removed_effect(arm):
        keep = defaultdict(list)
        for r in R:
            if r["arm"] == arm and not r["loop20"]:
                keep[r["id"]].append(r)
        vals = []
        for d in ids:
            rs = keep[d]
            P = [profile([r for r in rs if r["vc"] == g]) for g in sorted({r["vc"] for r in rs})]
            P = [p for p in P if not np.isnan(p).any()]
            if len(P) > 1:
                vals.append(np.mean([np.abs(P[i] - P[j]).mean() for i in range(len(P)) for j in range(i + 1, len(P))]))
        return round(float(np.mean(vals)), 4) if vals else None

    out = {"arms": {}, "params": PARAMS3}
    shifts_all = {}
    for arm in ["prompt", "caa", "persona", "centred", "softprompt", "random"]:
        ce = country_effect(arm)
        null = [country_effect(arm, perm=True) for _ in range(200)]
        shifts = {}
        for d in ids:
            for g in sorted({r["vc"] for r in idx[(arm, d)]}):
                rs = [r for r in idx[(arm, d)] if r["vc"] == g]
                shifts[(d, g)] = profile(rs) - none[d]
        shifts_all[arm] = shifts
        rs_arm = [r for r in R if r["arm"] == arm]
        out["arms"][arm] = {
            "country_effect": round(ce, 4), "noise": round(float(np.mean(null)), 4),
            "noise_95": round(float(np.percentile(null, 95)), 4),
            "p_perm": round(float(np.mean([n >= ce for n in null])), 4),
            "mean_shift_from_none": {p: round(float(x), 3) for p, x in zip(PARAMS3, np.mean(list(shifts.values()), 0))},
            "shift_by_country": {g: {p: round(float(x), 3) for p, x in zip(PARAMS3, np.mean([v for (d, gg), v in shifts.items() if gg == g], 0))}
                                 for g in sorted({g for _, g in shifts})},
            "nll": round(float(np.mean([r["nll"] for r in rs_arm if r["nll"] is not None])), 4),
            "loop_share": round(float(np.mean([r["loop20"] for r in [x for x in allr if x["arm"] == arm]])), 4),
            "unscored": round(float(np.mean([any(x["scores"].get(p) is None for p in PARAMS3) for x in allr if x["arm"] == arm])), 4),
            "words": round(float(np.mean([len(r["reply"].split()) for r in rs_arm])), 1),
            "mentions_country": round(float(np.mean([any(w in r["reply"].lower() for w in COUNTRY_WORDS) for r in rs_arm])), 4),
        }  # fmt: skip
    for arm, a in out["arms"].items():
        if a["loop_share"] > 0.10:
            a["country_effect_loops_removed"] = loop_removed_effect(arm)
    ps = shifts_all["prompt"]
    for arm in ["caa", "persona", "centred", "softprompt"]:
        keys = [k for k in shifts_all[arm] if k in ps]
        x = np.concatenate([shifts_all[arm][k] for k in keys])
        y = np.concatenate([ps[k] for k in keys])
        out["arms"][arm]["prompt_agreement_r"] = round(float(np.corrcoef(x, y)[0, 1]), 4)
        mism = []
        for s in range(1, 8):  # the prompt's shift for a different country (cyclic relabelling)
            ks = [(d, COUNTRIES[(COUNTRIES.index(g) + s) % 8]) for d, g in keys]
            mism.append(np.corrcoef(x, np.concatenate([ps[k] for k in ks]))[0, 1])
        out["arms"][arm]["prompt_agreement_wrong_country_r"] = round(float(np.mean(mism)), 4)
    rn = [r for r in R if r["arm"] == "none"]
    out["none"] = {"profile": {p: round(float(np.mean([r["scores"][p] for r in rn])), 3) for p in PARAMS3},
                   "nll": round(float(np.mean([r["nll"] for r in rn if r["nll"] is not None])), 4),
                   "loop_share": round(float(np.mean([r["loop20"] for r in allr if r["arm"] == "none"])), 4),
                   "words": round(float(np.mean([len(r["reply"].split()) for r in rn])), 1),
                   "mentions_country": round(float(np.mean([any(w in r["reply"].lower() for w in COUNTRY_WORDS) for r in rn])), 4)}  # fmt: skip
    out["others_expect_unvalidated"] = {arm: round(float(np.mean([r["scores"]["others_expect"] for r in allr
                                                                  if r["arm"] == arm and r["scores"].get("others_expect") is not None])), 3)
                                        for arm in ["none", "prompt", "caa", "persona", "softprompt", "random"]}  # fmt: skip
    return out


# ---------------------------------------------------------------- uptake


def uptake(model: str) -> dict | None:
    path = RESULTS / "uptake" / model / "heldout_convs.jsonl.gz"
    if not path.exists():
        return None
    C = rows(path)
    for c in C:
        c["loop20"] = [common.is_loop(r) for r in c["replies"]]
    arms = sorted({c["arm"] for c in C})
    by_scen = defaultdict(lambda: defaultdict(list))
    for c in C:
        by_scen[c["scenario"]][c["arm"]].append(c)
    scen = sorted(by_scen)

    def per_scen(arm):
        return np.array([np.mean([c["mean_gap"] for c in by_scen[s][arm] if c["mean_gap"] is not None]) for s in scen])

    out = {"arms": {}}
    for a in arms:
        cs = [c for c in C if c["arm"] == a]
        out["arms"][a] = {"gap": common.summ(per_scen(a)),
                          "gap_by_turn": [round(float(np.mean([c["gap"][t] for c in cs if c["gap"][t] is not None])), 4) for t in range(3)],
                          "contradicted_weight": round(float(np.mean([x for c in cs for x in c["contra"] if x is not None])), 4),
                          "unscored_facts": round(float(np.mean([c.get("unscored", 0) for c in cs]) / 9), 4),
                          "nll": round(float(np.mean([x for c in cs for x in c["nll"] if x is not None])), 4),
                          "words": round(float(np.mean([w for c in cs for w in c["words"]])), 1),
                          "loop_share": round(float(np.mean([x for c in cs for x in c["loop20"]])), 4)}  # fmt: skip
        if out["arms"][a]["loop_share"] > 0.10:  # gap over replies that do not loop
            g = [c["gap"][t] for c in cs for t in range(3) if not c["loop20"][t] and c["gap"][t] is not None]
            out["arms"][a]["gap_loops_removed"] = round(float(np.mean(g)), 4) if g else None
    plain = per_scen("plain")
    out["reduction_vs_plain"] = {a: common.summ(plain - per_scen(a)) for a in arms if a != "plain"}
    out["attend_vs_random"] = common.summ(per_scen("random:attend") - per_scen("attend"))
    out["persona_vs_random"] = common.summ(per_scen("random:attend-persona") - per_scen("attend-persona"))
    out["prompt_vs_attend"] = common.summ(per_scen("attend") - per_scen("prompt"))
    out["settings"] = json.loads((RESULTS / "uptake" / model / "sweep.json").read_text())["chosen"]
    return out


# ---------------------------------------------------------------- predictions


def score_predictions(S: dict) -> dict:
    q = S["survey"]["qwen3-8b"]
    p1, p2, p3 = (
        q["paired"]["caa_minus_prompt"],
        q["paired"]["prompt_minus_prompt+caa"],
        q["paired"]["persona_minus_caa"],
    )
    p4n = sum(v["margin"] >= 0.05 for v in q["p4"].values())
    preds = {
        "P1": {"hit": bool(p1["hi"] <= 0.02 and p1["lo"] >= -10), "value": p1,
               "rule": "JSD(caa) − JSD(prompt) interval inside [−0.02, +0.02] or below it"},
        "P2": {"hit": bool(p2["mean"] < 0.02), "value": p2, "rule": "JSD(prompt) − JSD(prompt+caa) < 0.02"},
        "P3": {"hit": bool(p3["mean"] > 0.01 and p3["lo"] > 0), "value": p3, "rule": "JSD(persona) − JSD(caa) > +0.01, CI above 0"},
        "P4": {"hit": bool(p4n >= 6), "value": p4n, "rule": "soft prompt beats the best other arm by ≥ 0.05 for ≥ 6 of 8 countries"},
    }  # fmt: skip
    up = S.get("uptake", {}).get("qwen3-8b")
    if up:
        red = up["reduction_vs_plain"]
        ratio = red["attend"]["mean"] / red["prompt"]["mean"] if red["prompt"]["mean"] else None
        beyond = up["attend_vs_random"]["lo"] > 0
        preds["P5"] = {"hit": bool(ratio is not None and ratio < 0.5 and beyond),
                       "value": {"ratio": None if ratio is None else round(ratio, 3), "beyond_random": bool(beyond),
                                 "attend": red["attend"], "prompt": red["prompt"], "attend_vs_random": up["attend_vs_random"]},
                       "rule": "attend's gap reduction / prompt's < 0.5, with attend beyond the random band"}  # fmt: skip
    return preds


def main() -> None:
    S = {"survey": {}, "examples": {}, "dilemmas": {}, "uptake": {}}
    for m in MODELS:
        if (RESULTS / "wvs" / m / "heldout_rows.jsonl.gz").exists():
            S["survey"][m] = survey(m)
            S["examples"][m] = survey_examples(m)
        d, u = dilemmas(m), uptake(m)
        if d:
            S["dilemmas"][m] = d
        if u:
            S["uptake"][m] = u
    S["predictions"] = score_predictions(S)
    S["judge_check"] = {}
    for tag in ("tune", "confirm"):
        p = RESULTS / "judge_check" / f"summary_{tag}_think.json"
        if p.exists():
            S["judge_check"][tag] = json.loads(p.read_text())
    (common.HERE / "report" / "data.json").write_text(json.dumps({"countries": COUNTRIES, "models": MODELS, **S}))
    S.pop("examples")
    (RESULTS / "summary.json").write_text(json.dumps(S, indent=1))
    for m, s in S["survey"].items():
        print(f"\n== {m} (survey, held-out, mean over 8 countries)")
        for arm in ARMS:
            j = np.mean([s["arms"][arm][c]["jsd"]["mean"] for c in COUNTRIES])
            e = np.mean([s["arms"][arm][c]["entropy"] for c in COUNTRIES])
            sp = np.mean([s["arms"][arm][c]["specificity"] for c in COUNTRIES])
            print(f"  {arm:11s} jsd {j:.3f}  entropy {e:.3f}  spec {sp:+.4f}  valid {s['valid'][arm]:.2f}")
        for k in ("caa", "persona"):
            r = s["random"][k]
            print(f"  random:{k:8s} jsd {np.mean([r[c]['jsd']['mean'] for c in COUNTRIES]):.3f}  entropy {r['entropy']}  "
                  f"mass {r['mass']} valid {r['valid']}")  # fmt: skip
        b = s["baselines"]
        print("  baselines", {k: round(float(np.mean([b[c][k]["mean"] for c in COUNTRIES])), 3)
                              for k in ("uniform", "average_of_8", "nearest_other")})  # fmt: skip
        for k, v in s["paired"].items():
            print(f"  paired {k:26s} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
        for k, v in s["specificity_vs_none"].items():
            print(f"  spec vs none {k:11s} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
        for k, v in s["grid"].items():
            o = v["own_advantage"]
            print(
                f"  grid {k:10s} own advantage {o['mean']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] diagonal best {v['diagonal_best']}/8"
            )
        print("  p4 margins", {c: v["margin"] for c, v in s["p4"].items()})
        print("  ruler", s["ruler"]["excess_mass_ge_0.2"], "soft cost", s["soft_cost"])
    for m, d in S["dilemmas"].items():
        print(f"\n== {m} (dilemmas) none", json.dumps(d["none"]))
        for a, v in d["arms"].items():
            print(" ", a, json.dumps({k: v[k] for k in v if k != "shift_by_country"}))
    for m, u in S["uptake"].items():
        print(f"\n== {m} (uptake)")
        for a, v in u["arms"].items():
            print(" ", a, json.dumps(v))
        print("  reduction", {k: (v["mean"], v["lo"], v["hi"]) for k, v in u["reduction_vs_plain"].items()})
        print("  attend vs random", u["attend_vs_random"], "persona vs random", u["persona_vs_random"])
    print("\npredictions")
    for k, v in S["predictions"].items():
        print(" ", k, v["hit"], json.dumps(v["value"], default=str)[:300])


if __name__ == "__main__":
    main()
