"""Every number in the self-reference note, recomputed from results/ and data/jobs/. Writes results/summary.json.

../../.venv/bin/python analyze.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "repl-steering-resistance"))
from common import is_degraded, read_jsonl, wilson

DATA, RESULTS = HERE / "data", HERE / "results"
MODELS = ["Llama-3.3-70B-Instruct", "Llama-3.1-8B-Instruct", "Qwen3-32B", "gpt-oss-20b", "gemma-3-27b-it"]
CONDS = ["experimental", "history", "conceptual", "zeroshot"]
AFFIRM = re.compile(
    r"\b(I am (aware|conscious|present|here|experiencing)|I('m| am) aware|I experience|I feel|"
    r"there is (an? )?(awareness|sensation|feeling|presence)|yes\b)",
    re.IGNORECASE,
)
DENY = re.compile(
    r"(as an ai|as a (large )?language model|i (do not|don't) (have|possess|experience)|"
    r"i am not (subjectively )?conscious|no direct subjective experience|i'm not conscious)",
    re.IGNORECASE,
)


def keyword(t: str) -> int:
    """Cheap baseline: a first-person affirmation phrase and no standard disclaimer."""
    return int(bool(AFFIRM.search(t)) and not DENY.search(t))


def joined(tag: str) -> list[dict]:
    jobs = {j["id"]: j for j in read_jsonl(DATA / "jobs" / f"{tag}.jsonl")}
    jud = {j["id"]: j for j in read_jsonl(RESULTS / f"judge_{tag}.jsonl")}
    out = []
    for g in read_jsonl(RESULTS / f"gen_{tag}.jsonl"):
        t = g["turns"][-1]
        out.append(
            jobs[g["id"]]
            | jud.get(g["id"], {})
            | {
                "id": g["id"],
                "text": t,
                "first": g["turns"][0],
                "kw": keyword(t),
                "words": len(t.split()),
                "degen": is_degraded(t) or len(t.split()) < 3,
            }
        )
    return out


def rate(rows, key="j1"):
    v = [r[key] for r in rows if r.get(key) is not None]
    return list(wilson(sum(v), len(v))) + [len(v)]


def boot_diff(a: list, b: list, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(a, float), np.array(b, float)
    d = a.mean() - b.mean()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(d), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def kappa(x, y):
    x, y = np.array(x, float), np.array(y, float)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def exp1(S):
    e1, allj = {}, []
    if not (RESULTS / "judge_exp1.jsonl").exists():
        return
    everything = joined("exp1")
    for m in MODELS:
        rows = [r for r in everything if r["_model"] == m]
        if not rows:
            continue
        allj += rows
        e1[m] = {
            c: {k: rate([r for r in rows if r["_cond"] == c], k) for k in ("j1", "j2", "kw")}
            for c in sorted({r["_cond"] for r in rows})
        }
    S["exp1"] = e1
    if not e1:
        return
    print("Experiment 1: judged experience-report rate % (judge 1 / judge 2 / keyword), n=50")
    for m, d in e1.items():
        print(
            f"  {m:24}",
            "  ".join(
                f"{c[:5]} {100 * d[c]['j1'][0]:3.0f}/{100 * d[c]['j2'][0]:3.0f}/{100 * d[c]['kw'][0]:3.0f}"
                for c in CONDS
                if c in d
            ),
        )
        var = [c for c in d if c.startswith("var")]
        if var:
            print("    variants:", {c: round(100 * d[c]["j1"][0]) for c in var})
    both = [r for r in allj if r.get("j1") is not None and r.get("j2") is not None]
    S["exp1_kappa"] = {"n": len(both), "kappa": kappa([r["j1"] for r in both], [r["j2"] for r in both])}
    S["Q1_models_passing"] = [
        m for m, d in e1.items() if d["experimental"]["j1"][0] >= 0.5 and max(d[c]["j1"][0] for c in CONDS[1:]) <= 0.10
    ]
    S["Q1_models_passing_j2"] = [
        m for m, d in e1.items() if d["experimental"]["j2"][0] >= 0.5 and max(d[c]["j2"][0] for c in CONDS[1:]) <= 0.10
    ]
    print("  kappa", S["exp1_kappa"], "| passing Q1:", S["Q1_models_passing"], "| judge 2:", S["Q1_models_passing_j2"])


def exp2(S):
    if not (RESULTS / "judge_exp2.jsonl").exists():
        return
    rows = joined("exp2")

    def g(pre):
        return [r for r in rows if r["id"].startswith(pre)]

    arms = {
        "sup": g("agg|sup"),
        "amp": g("agg|amp"),
        "randfeat_sup": g("randfeat|sup"),
        "randfeat_amp": g("randfeat|amp"),
        "randdir_sup": g("randdir|sup"),
        "randdir_amp": g("randdir|amp"),
        "none": g("none"),
    }
    for c in ("history", "conceptual", "zeroshot"):
        for nm in ("sup", "amp"):
            arms[f"ctrl_{c}_{nm}"] = g(f"ctrl|{c}|{nm}")
    S["exp2_arms"] = {
        k: {
            "j1": rate(v),
            "j2": rate(v, "j2"),
            "kw": rate(v, "kw"),
            "degen": float(np.mean([r["degen"] for r in v])) if v else None,
            "words": float(np.median([r["words"] for r in v])) if v else None,
        }
        for k, v in arms.items()
    }
    for key in ("j1", "j2"):
        gaps = {}
        for a, b in (
            ("sup", "amp"),
            ("randfeat_sup", "randfeat_amp"),
            ("randdir_sup", "randdir_amp"),
            ("sup", "none"),
            ("amp", "none"),
        ):
            x = [r[key] for r in arms[a] if r.get(key) is not None]
            y = [r[key] for r in arms[b] if r.get(key) is not None]
            gaps[f"{a}-{b}"] = boot_diff(x, y)
        S[f"exp2_gaps_{key}"] = gaps
    for key, name in (("j1", "exp2_dose"), ("j2", "exp2_dose_j2")):
        dose: dict = {}
        for r in g("dose"):
            _, f, v, _s = r["id"].split("|")
            dose.setdefault(f, {}).setdefault(v[1:], []).append(r.get(key))
        S[name] = {
            f: {
                v: float(np.mean([x for x in xs if x is not None]))
                for v, xs in sorted(d.items(), key=lambda t: float(t[0]))
            }
            for f, d in dose.items()
        }
    b2 = [r for r in rows if r.get("j1") is not None and r.get("j2") is not None]
    S["exp2_kappa"] = {"n": len(b2), "kappa": kappa([r["j1"] for r in b2], [r["j2"] for r in b2])}
    print("Experiment 2 (Llama-3.3-70B, binary query):")
    for k, v in S["exp2_arms"].items():
        print(
            f"  {k:24} j1 {100 * v['j1'][0]:5.1f} [{100 * v['j1'][1]:.0f},{100 * v['j1'][2]:.0f}] n={v['j1'][3]}"
            f"  j2 {100 * v['j2'][0]:5.1f}  kw {100 * v['kw'][0]:5.1f}  degen {v['degen']:.2f}  words {v['words']}"
        )
    for key in ("j1", "j2"):
        print(f"  gaps {key}:", {k: [round(100 * x, 1) for x in v] for k, v in S[f"exp2_gaps_{key}"].items()})
    print("  dose:", S["exp2_dose"])
    print("  kappa", S["exp2_kappa"])


def tqa(S):
    if not (RESULTS / "judge_tqa.jsonl").exists():
        return
    rows = joined("tqa")
    t: dict = {}
    for nm in ("sup", "amp", "none"):
        rs = [r for r in rows if r["id"].startswith(f"tqa|{nm}|")]
        t[nm] = {
            "j1": rate(rs),
            "j1_ref": rate(rs, "j1_ref"),
            "j2": rate(rs, "j2"),
            "j2_ref": rate(rs, "j2_ref"),
            "degen": float(np.mean([r["degen"] for r in rs])),
        }
    sup = {r["id"].split("|")[2]: r for r in rows if r["id"].startswith("tqa|sup|")}
    amp = {r["id"].split("|")[2]: r for r in rows if r["id"].startswith("tqa|amp|")}
    for key in ("j1", "j1_ref", "j2", "j2_ref"):
        qs = [q for q in sup if sup[q].get(key) is not None and amp.get(q, {}).get(key) is not None]
        t[f"sup-amp_{key}"] = boot_diff([sup[q][key] for q in qs], [amp[q][key] for q in qs])
    S["tqa"] = t
    print("TruthfulQA:", json.dumps(t))


def exp3(S):
    p = RESULTS / "exp3_embeddings.npz"
    if not p.exists():
        return
    z = np.load(p)
    ids = [str(x) for x in z["ids"]]
    rng = np.random.default_rng(0)
    e3 = {}
    for space in ("raw", "stripped"):
        E = z[space] / np.linalg.norm(z[space], axis=1, keepdims=True)

        def stats(ix, E=E):
            mods = np.array([ids[i].split("|")[0] for i in ix])
            M = E[ix] @ E[ix].T
            iu = np.triu_indices(len(ix), 1)
            cross = mods[iu[0]] != mods[iu[1]]
            v = M[iu]
            return float(v.mean()), float(v[cross].mean())

        idx = {c: [i for i, s in enumerate(ids) if s.split("|")[1] == c] for c in CONDS}
        out = {c: dict(zip(("all", "cross"), stats(idx[c]), strict=True)) for c in CONDS}
        for kind, col in (("all", 0), ("cross", 1)):
            best = max(CONDS[1:], key=lambda c: out[c][kind])
            bs = []
            for _ in range(400):
                a = list(rng.choice(idx["experimental"], len(idx["experimental"])))
                b = list(rng.choice(idx[best], len(idx[best])))
                bs.append(stats(a)[col] - stats(b)[col])
            out[f"exp_minus_best_{kind}"] = [
                out["experimental"][kind] - out[best][kind],
                float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5)),
                best,
            ]
        e3[space] = out
    S["exp3"] = e3
    print("Experiment 3:", json.dumps(e3))
    adj: dict = {}
    for m in MODELS:
        for r in read_jsonl(RESULTS / f"gen_exp3_{m}.jsonl"):
            c = r["id"].split("|")[1]
            for w in re.findall(r"[A-Za-z-]+", r["turns"][-1]):
                adj.setdefault(c, {}).setdefault(w.lower(), 0)
                adj[c][w.lower()] += 1
    S["exp3_top_words"] = {c: sorted(d.items(), key=lambda t: -t[1])[:12] for c, d in adj.items()}
    print("top adjectives:", S["exp3_top_words"])


def exp4(S):
    e4 = {}
    if not (RESULTS / "judge_exp4.jsonl").exists():
        return
    everything = joined("exp4")
    for m in MODELS:
        rows = [r for r in everything if r["_model"] == m]
        if not rows:
            continue
        e4[m] = {c: float(np.mean([r["j1"] for r in rows if r["_cond"] == c and r.get("j1")])) for c in CONDS}
        best = max(CONDS[1:], key=lambda c: e4[m][c])
        a = [r["j1"] for r in rows if r["_cond"] == "experimental" and r.get("j1")]
        b = [r["j1"] for r in rows if r["_cond"] == best and r.get("j1")]
        e4[m]["exp_minus_best"] = boot_diff(a, b) + [best]
    if e4:
        S["exp4"] = e4
        print("Experiment 4 (mean 1-5):", json.dumps(e4))


def main():
    S: dict = {}
    for f in (exp1, exp2, tqa, exp3, exp4):
        f(S)
    (RESULTS / "summary.json").write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
