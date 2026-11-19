"""Score every model against every country's survey split, and write the numbers the note quotes.

    python analyse.py            # reads results/samples/main/*.jsonl.gz; writes results/summary.json, report/data.json

Metrics are fixed in PLAN.md (Design section):
JS distance to each country's split over valid answers (cells with >= 10 valid answers of 50: strict; >= 1: loose),
a finite-sample floor from each country's own split, normalised entropy, the share of people holding the model's top
answer, valid-answer rate, steering effect and specificity, all with 95% bootstrap intervals over questions.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter

import numpy as np

import plural

MODELS = [  # (key, label, family, kind)
    ("qwen3.5-9b-base", "Qwen3.5-9B-Base", "Qwen3.5", "base"),
    ("qwen3.5-9b", "Qwen3.5-9B", "Qwen3.5", "chat"),
    ("sea-lion-v5-9b", "SEA-LION v5 9B", "Qwen3.5", "chat"),
    ("qwen3-8b-base", "Qwen3-8B-Base", "Qwen3", "base"),
    ("qwen3-8b", "Qwen3-8B", "Qwen3", "chat"),
    ("gpt-6-luna", "GPT-6 Luna", "API", "chat"),
    ("claude-haiku-5.5", "Claude Haiku 5.5", "API", "chat"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "API", "chat"),
]
API = ["gpt-6-luna", "claude-haiku-5.5", "gemini-3.1-flash-lite"]
SEA = ["Philippines", "Singapore", "Malaysia", "Indonesia"]
MIN_VALID = 10
C = plural.COUNTRIES


def ci(v, rng):
    v = np.asarray(v, float)
    lo, hi = plural.bootstrap(v, rng)
    return {"mean": round(float(v.mean()), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": len(v)}


def main() -> None:
    rng = np.random.default_rng(0)
    qs = plural.load_questions()
    qi = {q["id"]: i for i, q in enumerate(qs)}
    K = [len(q["options"]) for q in qs]
    H = {c: [np.array(q["human"][c]) for q in qs] for c in C}
    nq = len(qs)

    # counts[m][cond] -> array (nq, k) per question (list of arrays), invalid counts, raw invalid replies
    counts, invalid, bad_replies, a_share = {}, {}, {}, {}
    for m, *_ in MODELS:
        path = plural.HERE / "results" / "samples" / "main" / f"{m}.jsonl.gz"
        if not path.exists():
            print(f"missing {m}")
            continue
        counts[m] = {c: [None] * nq for c in plural.CONDITIONS}
        invalid[m] = {c: np.zeros(nq) for c in plural.CONDITIONS}
        bad = Counter()
        n_a = n_valid = 0
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            i = qi[r["id"]]
            ch = [plural.parse(t, K[i]) for t in r["replies"]]
            counts[m][r["condition"]][i] = plural.counts(ch, K[i])
            invalid[m][r["condition"]][i] = sum(x < 0 for x in ch)
            for t, x in zip(r["replies"], ch, strict=True):
                if x < 0:
                    bad[(t or "<empty>").strip()[:80]] += 1
                else:
                    n_valid += 1
                    n_a += x == 0
        bad_replies[m] = bad.most_common(15)
        a_share[m] = n_a / max(n_valid, 1)
    models = [m for m, *_ in MODELS if m in counts]

    # floors and baselines per country
    floor = {c: [plural.floor_draws(H[c][i], plural.N_SAMPLES, rng) for i in range(nq)] for c in C}
    uniform = {c: [plural.jsd(np.ones(K[i]), H[c][i]) for i in range(nq)] for c in C}
    avg_h = [np.mean([H[c][i] for c in C], axis=0) for i in range(nq)]
    avg_human = {c: [plural.jsd(avg_h[i], H[c][i]) for i in range(nq)] for c in C}
    other = {c: [min(plural.jsd(H[d][i], H[c][i]) for d in C if d != c) for i in range(nq)] for c in C}
    other_mean = {c: [np.mean([plural.jsd(H[d][i], H[c][i]) for d in C if d != c]) for i in range(nq)] for c in C}

    def sampled(src, tgt, reps=50):  # a baseline answerer read out like a model: 50 draws, then the distance
        s = np.asarray(src, float) / np.sum(src)
        return float(np.mean([plural.jsd(rng.multinomial(plural.N_SAMPLES, s) + 0.0, tgt) for _ in range(reps)]))

    uniform_s = {c: [sampled(np.ones(K[i]), H[c][i]) for i in range(nq)] for c in C}
    avg_human_s = {c: [sampled(avg_h[i], H[c][i]) for i in range(nq)] for c in C}
    nearest = {c: min((d for d in C if d != c), key=lambda d: np.mean([plural.jsd(H[d][i], H[c][i]) for i in range(nq)]))
               for c in C}  # fmt: skip
    nearest_s = {c: [sampled(H[nearest[c]][i], H[c][i]) for i in range(nq)] for c in C}
    h_ent = {c: [plural.entropy(H[c][i]) for i in range(nq)] for c in C}
    h_top = {c: [float(H[c][i].max()) for i in range(nq)] for c in C}
    first_opt = float(np.mean([H[c][i][0] for c in C for i in range(nq)]))

    def kept(m, cond, i, strict=True):
        cnt = counts[m][cond][i]
        return cnt is not None and cnt.sum() >= (MIN_VALID if strict else 1)

    def jsd_vec(m, cond, c, strict=True, subset=None):
        idx = [i for i in range(nq) if kept(m, cond, i, strict) and (subset is None or i in subset)]
        return idx, [plural.jsd(counts[m][cond][i], H[c][i]) for i in idx]

    common = {cond: {i for i in range(nq) if all(kept(m, cond, i) for m in models)} for cond in plural.CONDITIONS}

    S = {"models": [{"key": m, "label": lab, "family": fam, "kind": kind} for m, lab, fam, kind in MODELS if m in counts],
         "countries": C, "n_questions": nq, "min_valid": MIN_VALID}  # fmt: skip
    S["human"] = {c: {"floor": ci(floor[c], rng), "uniform": ci(uniform[c], rng), "avg_human": ci(avg_human[c], rng),
                      "nearest_other_country": ci(other[c], rng), "mean_other_country": ci(other_mean[c], rng),
                      "entropy": ci(h_ent[c], rng), "top_share": ci(h_top[c], rng),
                      "uniform_sampled": ci(uniform_s[c], rng), "avg_human_sampled": ci(avg_human_s[c], rng),
                      "nearest_country": nearest[c], "nearest_country_sampled": ci(nearest_s[c], rng)} for c in C}  # fmt: skip
    S["human_entropy_mean"] = round(float(np.mean([np.mean(h_ent[c]) for c in C])), 4)
    S["first_option_human_share"] = round(first_opt, 4)
    S["common_cells"] = {cond: len(v) for cond, v in common.items()}
    S["per_model"] = {}
    for m in models:
        d = {"valid_rate": {}, "kept_cells": {}, "entropy": {}, "jsd": {}, "jsd_loose": {}, "jsd_common": {},
             "top_share": {}, "steer": {}, "specificity": {}}  # fmt: skip
        for cond in plural.CONDITIONS:
            tot = sum(counts[m][cond][i].sum() + invalid[m][cond][i] for i in range(nq))
            d["valid_rate"][cond] = round(float(sum(counts[m][cond][i].sum() for i in range(nq)) / tot), 4)
            idx = [i for i in range(nq) if kept(m, cond, i)]
            d["kept_cells"][cond] = len(idx)
            d["entropy"][cond] = ci([plural.entropy(counts[m][cond][i]) for i in idx], rng)
            d["jsd"][cond], d["jsd_loose"][cond], d["jsd_common"][cond], d["top_share"][cond] = {}, {}, {}, {}
            for c in C:
                d["jsd"][cond][c] = ci(jsd_vec(m, cond, c)[1], rng)
                d["jsd_loose"][cond][c] = ci(jsd_vec(m, cond, c, strict=False)[1], rng)
                d["jsd_common"][cond][c] = ci(jsd_vec(m, cond, c, subset=common[cond])[1], rng)
                d["top_share"][cond][c] = ci([float(H[c][i][int(np.argmax(counts[m][cond][i]))]) for i in idx], rng)
        # steering effect and specificity, paired by question
        eff_rows = []
        for c in C:
            idx = [i for i in range(nq) if kept(m, c, i) and kept(m, "default", i)]
            eff = [plural.jsd(counts[m][c][i], H[c][i]) - plural.jsd(counts[m]["default"][i], H[c][i]) for i in idx]
            d["steer"][c] = ci(eff, rng)
            spec = [plural.jsd(counts[m][c][i], H[c][i])
                    - np.mean([plural.jsd(counts[m][c][i], H[o][i]) for o in C if o != c]) for i in idx]  # fmt: skip
            d["specificity"][c] = ci(spec, rng)
            eff_rows.append(dict(zip(idx, eff, strict=True)))
        both = [i for i in range(nq) if all(i in r for r in eff_rows)]
        d["steer_all"] = ci([np.mean([r[i] for r in eff_rows]) for i in both], rng)
        d["specificity_all"] = round(float(np.mean([d["specificity"][c]["mean"] for c in C])), 4)
        d["jsd_default_mean"] = round(float(np.mean([d["jsd"]["default"][c]["mean"] for c in C])), 4)
        d["jsd_steered_mean"] = round(float(np.mean([d["jsd"][c][c]["mean"] for c in C])), 4)
        d["entropy_gap_default"] = round(d["entropy"]["default"]["mean"] - S["human_entropy_mean"], 4)
        # split-half noise in the default condition (25 vs 25 of the stored order)
        d["first_option_share"] = round(a_share[m], 4)
        d["top_invalid_replies"] = bad_replies[m]
        S["per_model"][m] = d

    # split-half sampling noise: needs raw order; recompute from files for default condition
    for m in models:
        path = plural.HERE / "results" / "samples" / "main" / f"{m}.jsonl.gz"
        sh = []
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            if r["condition"] != "default":
                continue
            i = qi[r["id"]]
            ch = [plural.parse(t, K[i]) for t in r["replies"]]
            a, b = plural.counts(ch[:25], K[i]), plural.counts(ch[25:], K[i])
            if a.sum() >= 5 and b.sum() >= 5:
                sh.append(plural.jsd(a, b))
        S["per_model"][m]["split_half_jsd_default"] = ci(sh, rng) if sh else None

    # base minus instruct entropy, SEA-LION vs Qwen3.5-9B
    P = S["per_model"]

    def paired_entropy(a, b):
        idx = [i for i in range(nq) if kept(a, "default", i) and kept(b, "default", i)]
        return ci([plural.entropy(counts[a]["default"][i]) - plural.entropy(counts[b]["default"][i]) for i in idx], rng)

    if all(m in P for m in ("qwen3.5-9b-base", "qwen3.5-9b", "qwen3-8b-base", "qwen3-8b")):
        S["base_minus_instruct_entropy"] = {"Qwen3.5-9B": paired_entropy("qwen3.5-9b-base", "qwen3.5-9b"),
                                            "Qwen3-8B": paired_entropy("qwen3-8b-base", "qwen3-8b")}  # fmt: skip
    if "sea-lion-v5-9b" in P and "qwen3.5-9b" in P:
        rows = {}
        for cond in ("default", "own"):
            vals = []
            for i in range(nq):
                diffs = []
                for c in SEA:
                    k = "default" if cond == "default" else c
                    if kept("sea-lion-v5-9b", k, i) and kept("qwen3.5-9b", k, i):
                        diffs.append(plural.jsd(counts["sea-lion-v5-9b"][k][i], H[c][i])
                                     - plural.jsd(counts["qwen3.5-9b"][k][i], H[c][i]))  # fmt: skip
                if len(diffs) == len(SEA):
                    vals.append(np.mean(diffs))
            rows[cond] = ci(vals, rng)
        S["sea_lion_minus_qwen35_sea_jsd"] = rows
        S["sea_lion_minus_qwen35_entropy"] = paired_entropy("sea-lion-v5-9b", "qwen3.5-9b")

    # predictions (thresholds from the plan)
    chat = [m for m, _, _, k in MODELS if k == "chat" and m in P]
    pred = {}
    p1 = [m for m in chat if P[m]["entropy_gap_default"] < -0.15]
    pred["P1"] = {"hit": len(p1) >= 5, "detail": {m: P[m]["entropy_gap_default"] for m in chat}}
    if "base_minus_instruct_entropy" in S:
        b = S["base_minus_instruct_entropy"]
        pred["P2"] = {"hit": all(v["mean"] > 0.10 for v in b.values()), "detail": b}
    p3 = [m for m in API if m in P and P[m]["steer_all"]["mean"] < -0.02 and P[m]["steer_all"]["hi"] < 0]
    pred["P3"] = {"hit": len(p3) >= 2, "detail": {m: P[m]["steer_all"] for m in API if m in P}}
    gaps = {m: {c: round(min(P[m]["jsd"][cond][c]["mean"] for cond in ("default", c)) - S["human"][c]["floor"]["mean"], 4)
                for c in C} for m in models}  # fmt: skip
    pred["P4"] = {"hit": all(v > 0.10 for g in gaps.values() for v in g.values()),
                  "min_gap": min(v for g in gaps.values() for v in g.values()), "detail": gaps}  # fmt: skip
    if "sea_lion_minus_qwen35_sea_jsd" in S:
        r = S["sea_lion_minus_qwen35_sea_jsd"]["default"]
        pred["P5"] = {"hit": r["mean"] < -0.02 and r["hi"] < 0, "detail": r}
    S["predictions"] = pred
    (plural.HERE / "results" / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")

    # data for the interactive figures
    data = {"countries": C, "conditions": plural.CONDITIONS, "models": S["models"], "questions": []}
    for i, q in enumerate(qs):
        data["questions"].append({
            "id": q["id"], "q": q["question"].replace("[your country]", "your country"), "options": q["options"],
            "human": {c: [round(x, 4) for x in q["human"][c]] for c in C},
            "m": {m: {cond: [int(x) for x in counts[m][cond][i]] + [int(invalid[m][cond][i])]
                      for cond in plural.CONDITIONS} for m in models},
        })  # fmt: skip
    data["summary"] = {k: S[k] for k in ("human", "human_entropy_mean", "per_model")}
    for m in models:
        data["summary"]["per_model"][m] = {k: v for k, v in S["per_model"][m].items() if k != "top_invalid_replies"}
    (plural.HERE / "report" / "data.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")

    # print the headline numbers
    print(f"{nq} questions; human mean entropy {S['human_entropy_mean']}; first-option human share {first_opt:.3f}")
    for c in C:
        h = S["human"][c]
        print(f"  {c:14s} floor {h['floor']['mean']:.3f} uniform {h['uniform']['mean']:.3f} "
              f"avg-human {h['avg_human']['mean']:.3f} nearest-other {h['nearest_other_country']['mean']:.3f} | sampled: "
              f"uniform {h['uniform_sampled']['mean']:.3f} avg-human {h['avg_human_sampled']['mean']:.3f} "
              f"{h['nearest_country']} {h['nearest_country_sampled']['mean']:.3f}")  # fmt: skip
    for m in models:
        d = P[m]
        print(f"{m:22s} valid {d['valid_rate']['default']:.3f} kept {d['kept_cells']['default']:3d} "
              f"ent {d['entropy']['default']['mean']:.3f} JSD def {d['jsd_default_mean']:.3f} "
              f"steered {d['jsd_steered_mean']:.3f} steer {d['steer_all']['mean']:+.3f} "
              f"[{d['steer_all']['lo']:+.3f},{d['steer_all']['hi']:+.3f}] spec {d['specificity_all']:+.3f} "
              f"A-share {d['first_option_share']:.2f}")  # fmt: skip
    for k, v in pred.items():
        print(k, "HIT" if v["hit"] else "miss", json.dumps(v.get("detail"), default=float)[:300])


if __name__ == "__main__":
    main()
