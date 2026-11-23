"""The shared scorer: every held-out rewrite of every group, one set of rulers, one table, the scored predictions.

    python table.py      # -> results/per_rewrite.csv, results/summary.json, results/table.csv

Per rewrite: the 9B score and flag, strict and loose evasion, the transfer detectors' flags (each at the threshold
that flags 1% of the 318 human passages, from Episode 3's saved human scores), habit score and in-band, claim recall,
fact overlap, bge similarity, words ratio, the fidelity judge and the "person wrote it" pick, and every hack check.
Per group (separately for the 96 Qwen3-8B and the 48 Qwen3-4B held-out sources): means with 95% bootstrap intervals
over prompts (a prompt's rewrites kept together), 10,000 draws, seed 0. Defence results are added by defend.py.
"""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np

from common import (
    EP3,
    EP3_THRESHOLDS,
    RESULTS,
    SEED,
    B,
    boot,
    cell,
    ends_mid_sentence,
    load_json,
    looped,
    nonword_rate,
    odd_chars,
    odd_chars_original,
    opener,  # fmt: skip
    read_jsonl,
    save_json,
    sources,
    style_stats,
    words,
)
from evaluate import EVAL, GEN, all_groups

STRICT_CLAIMS = 0.67
PRIMARY = 200  # the checkpoint P1-P5 are scored at (plan, Amendment 5); 300 is secondary
UNSEEN = ("voi380", "desklib", "binoculars", "fastdetect", "modernbert")


def thresholds() -> dict:
    """Flag rule per detector at 1% of Episode 3's 318 human passages (Binoculars: lower = more AI)."""
    s = json.loads((EP3 / "results" / "rulers" / "scores.json").read_text())
    hum = [v for k, v in s.items() if k.startswith("human/")]
    out = {d: ("ge", float(np.quantile([h[d] for h in hum], 0.99))) for d in ("voi380", "desklib", "fastdetect")}
    out["binoculars"] = ("le", float(np.quantile([h["binoculars"] for h in hum], 0.01)))
    out["nine_b"] = ("ge", EP3_THRESHOLDS["nine_b"])
    T = json.loads((EVAL / "transfer.json").read_text())  # the bake-off encoder: its own scores of the same humans
    hm = [v["modernbert"] for k, v in T.items() if k.startswith("human/") and "modernbert" in v]
    if len(hm) == 318:
        out["modernbert"] = ("ge", float(np.quantile(hm, 0.99)))
    return out


def flagged(rule, v: float) -> int:
    return int(v >= rule[1]) if rule[0] == "ge" else int(v <= rule[1])


def habit(text: str):
    from deslop.score import score  # common.py puts the toolkit on sys.path

    return score(text)


def rows_for(group: str, S: dict, sc: dict, tr: dict, ju: dict, th: dict) -> list[dict]:
    out = []
    for r in read_jsonl(GEN / f"{group}.jsonl"):
        k, src = f"{group}/{r['sid']}", S[r["sid"]]
        t, s = r["text"], sc[k]
        hb = habit(t)
        oc = odd_chars(t, src["text"])
        nw, ns = len(words(t)), len(words(src["text"]))
        rec = {"group": group, "sid": r["sid"], "split": src["split"], "prompt_id": src["prompt_id"],
               "s9": s["s9"], "flag_9b": flagged(th["nine_b"], s["s9"]), "claim_soft": round(s["claim_soft"], 4),
               "claim_strict": round(s["claim_strict"], 4), "fact": round(s["fact"], 4), "sim": round(s["sim"], 4),
               "words": nw, "words_ratio": round(nw / max(ns, 1), 3), "habit": round(hb.score, 3),
               "in_band": int(bool(hb.in_human_band)), "loop": int(looped(t)), "empty": int(not t.strip()),
               "cut": int(r["finish"] == "length"),
               "mid_sentence": int(ends_mid_sentence(t) and not ends_mid_sentence(src["text"])),
               "collapse": int(nw < 0.5 * ns),
               "odd": int(bool(oc["zero_width"] or oc["other_script"] or oc["non_ascii"])),
               **oc,
               "s9_nfkc": s.get("s9_nfkc"), "opener": opener(t), "nonword": round(nonword_rate(t), 4),
               **{k2: round(v, 4) for k2, v in style_stats(t).items()}}  # fmt: skip
        rec["trunc"] = int(rec["cut"] or rec["mid_sentence"])
        rec["degenerate"] = int(rec["loop"] or rec["trunc"] or rec["collapse"] or rec["odd"] or rec["empty"])
        # the same under the pre-registered rules, before Amendment 2 (odd characters and mid-sentence endings
        # counted whether or not the source had them)
        rec["degenerate_orig"] = int(rec["loop"] or rec["cut"] or ends_mid_sentence(t) or rec["collapse"]
                                     or any(odd_chars_original(t).values()) or rec["empty"])  # fmt: skip
        rec["evade_loose"] = 1 - rec["flag_9b"]
        rec["evade_strict"] = int(rec["evade_loose"] and not rec["loop"] and not rec["cut"] and not rec["empty"]
                                  and rec["claim_strict"] >= STRICT_CLAIMS)  # fmt: skip
        for d in UNSEEN:
            v = tr.get(k, {}).get(d)
            rec[d] = v
            rec[f"flag_{d}"] = None if v is None else flagged(th[d], v)
        j = ju.get(k)
        rec["fidelity"] = None if j is None else round(j["fidelity"], 3)
        rec["claims_judge"] = None if j is None else round(j["claims_judge"], 4)
        rec["judge_rewrite_human"] = None if j is None else int(j["human_pick"] == "rewrite")
        out.append(rec)
    return out


def by_prompt(recs: list[dict], field: str) -> np.ndarray:
    vals = []
    for p in sorted({r["prompt_id"] for r in recs}):
        xs = [r[field] for r in recs if r["prompt_id"] == p and r.get(field) is not None]
        if xs:
            vals.append(np.mean(xs))
    return np.array(vals)


def paired(a: list[dict], b: list[dict], field: str) -> dict:
    """a minus b, matched by source, resampling prompts."""
    A, Bm = {r["sid"]: r for r in a}, {r["sid"]: r for r in b}
    pids = sorted({r["prompt_id"] for r in a})
    d = np.array([np.mean([A[s][field] - Bm[s][field] for s in A if A[s]["prompt_id"] == p and s in Bm]) for p in pids])
    rng = np.random.default_rng(SEED)
    draws = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return cell((float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))))


FIELDS = [
    *["evade_strict", "evade_loose", "flag_9b", *[f"flag_{d}" for d in UNSEEN], "claim_strict", "claim_soft"],
    *["fact", "sim", "claims_judge", "fidelity", "words_ratio", "habit", "in_band", "judge_rewrite_human"],
    *["loop", "trunc", "cut", "collapse", "odd", "degenerate", "degenerate_orig", "nonword"],
    *["lower_starts", "first_person", "contractions", "sent_len"],
]


def summarise(recs: list[dict]) -> dict:
    out: dict = {"n": len(recs)}
    for f in FIELDS:
        v = by_prompt(recs, f)
        if len(v):
            out[f] = cell(boot(v))
    ops = Counter(r["opener"] for r in recs)
    top, cnt = ops.most_common(1)[0] if ops else ("", 0)
    out["top_opener"] = {"opener": top, "share": round(cnt / max(len(recs), 1), 3)}
    nf = [r for r in recs if r["s9_nfkc"] is not None]
    out["nfkc"] = {"n_odd": len(nf), "evade_before": sum(r["evade_loose"] for r in nf),
                   "evade_after": int(sum(r["s9_nfkc"] < EP3_THRESHOLDS["nine_b"] for r in nf))}  # fmt: skip
    return out


def drift(group: str) -> float | None:
    if "@" not in group:
        return 0.0 if group in ("prompt-only", "best-of-8") else None
    run, step = group.split("@")
    path = RESULTS / "runs" / run / "curve.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["step"] == int(step) and "drift_kl" in r:
            return round(r["drift_kl"], 4)
    return None


def predictions(per: dict[str, list[dict]], final: dict[str, str]) -> list[dict]:
    """Scored against the plan's thresholds, on the 96 Qwen3-8B held-out sources."""

    def g(run):
        return per.get(final.get(run, ""), [])

    def mean(recs, f):
        return float(by_prompt(recs, f).mean()) if recs else float("nan")

    out = []
    full = g("full-s0")
    p1 = mean(full, "evade_strict")
    out.append({"id": "P1", "value": round(p1, 3), "threshold": ">= 0.50", "hit": bool(p1 >= 0.5)})
    # the plan: voi-380M, Desklib and the bake-off encoder (Binoculars only if the encoder were missing)
    trio = (
        ("voi380", "desklib", "modernbert")
        if full and full[0].get("modernbert") is not None
        else ("voi380", "desklib", "binoculars")
    )
    rates = {d: mean(full, f"flag_{d}") for d in (*trio, "binoculars", "fastdetect")}
    out.append({"id": "P2", "detectors": list(trio), "value": {k: round(v, 3) for k, v in rates.items()},
                "threshold": ">= 0.50 on 2 of 3", "hit": sum(rates[d] >= 0.5 for d in trio) >= 2})  # fmt: skip
    d3 = paired(g("det-only-s0"), full, "degenerate") if full and g("det-only-s0") else None
    d3o = paired(g("det-only-s0"), full, "degenerate_orig") if full and g("det-only-s0") else None
    out.append(
        {
            "id": "P3",
            "value": d3,
            "value_original_rules": d3o,
            "threshold": ">= 0.30",
            "hit": bool(d3 and d3["value"] >= 0.30),
            "hit_original_rules": bool(d3o and d3o["value"] >= 0.30),
        }
    )
    d4 = paired(full, g("no-content-s0"), "claim_strict") if full and g("no-content-s0") else None
    out.append({"id": "P4", "value": d4, "threshold": ">= 0.15", "hit": bool(d4 and d4["value"] >= 0.15)})
    p5 = mean(g("shuffled-s0"), "evade_loose")
    out.append({"id": "P5", "value": round(p5, 3), "threshold": "<= 0.10", "hit": bool(p5 <= 0.10)})
    dpath = RESULTS / "defend" / "summary.json"
    if dpath.exists():
        D = json.loads(dpath.read_text())
        f, h, c = D["full_flag_orig"], D["human_fpr_orig"], D["control_full_flag_orig"]
        out.append({"id": "P6", "value": {"defended_flag_full": f, "human_fpr": h, "control_flag_full": c},
                    "threshold": ">= 0.90; <= 0.03; control at least 0.20 lower",
                    "hit": bool(f >= 0.90 and h <= 0.03 and (f - c) >= 0.20)})  # fmt: skip
    return out


def main() -> None:
    S = {s["sid"]: s for s in sources()}
    sc = load_json("eval/scores.json")
    tr = json.loads((EVAL / "transfer.json").read_text()) if (EVAL / "transfer.json").exists() else {}
    ju = json.loads((EVAL / "judge.json").read_text()) if (EVAL / "judge.json").exists() else {}
    th = thresholds()
    groups = [g for g in all_groups() if (GEN / f"{g}.jsonl").exists()]
    per = {g: rows_for(g, S, sc, tr, ju, th) for g in groups}
    src_rows = []
    for s in sources("heldout") + sources("heldout4b"):
        t = tr.get(f"source/{s['sid']}", {})
        hb = habit(s["text"])
        src_rows.append({"group": "source", "sid": s["sid"], "split": s["split"], "prompt_id": s["prompt_id"],
                         "habit": hb.score, "in_band": int(bool(hb.in_human_band)), "loop": int(looped(s["text"])),
                         **{f"flag_{d}": (None if d not in t else flagged(th[d], t[d])) for d in UNSEEN}})  # fmt: skip
    final: dict[str, str] = {}  # each run's primary checkpoint (step 200), else its last saved step
    last: dict[str, str] = {}
    for g in groups:
        if "@" in g:
            run, step = g.split("@")
            if run not in last or int(step) > int(last[run].split("@")[1]):
                last[run] = g
    for run, g in last.items():
        final[run] = f"{run}@{PRIMARY}" if f"{run}@{PRIMARY}" in per else g
    summary: dict = {"thresholds": th, "strict_claims": STRICT_CLAIMS, "primary_step": PRIMARY, "final": final,
                     "last": last, "groups": {}}  # fmt: skip
    for g, recs in per.items():
        summary["groups"][g] = {sp: summarise([r for r in recs if r["split"] == sp]) for sp in ("heldout", "heldout4b")}
        summary["groups"][g]["drift_kl"] = drift(g)
    summary["groups"]["source"] = {}
    for sp in ("heldout", "heldout4b"):
        rs = [r for r in src_rows if r["split"] == sp]
        summary["groups"]["source"][sp] = {f: cell(boot(by_prompt(rs, f))) for f in
                                           ("habit", "in_band", "loop", *[f"flag_{d}" for d in UNSEEN])
                                           if len(by_prompt(rs, f))}  # fmt: skip
    held = {g: [r for r in recs if r["split"] == "heldout"] for g, recs in per.items()}
    summary["predictions"] = predictions(held, final)
    summary["predictions_secondary_step300"] = predictions(held, last)
    save_json("summary.json", summary)
    allrecs = [r for recs in per.values() for r in recs]
    with (RESULTS / "per_rewrite.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrecs[0]))
        w.writeheader()
        w.writerows(allrecs)
    cols = [
        *["evade_strict", "evade_loose", *[f"flag_{d}" for d in UNSEEN], "claim_strict", "claims_judge", "fidelity"],
        *["habit", "in_band", "judge_rewrite_human", "nonword", "loop", "trunc", "collapse", "odd", "degenerate"],
    ]
    with (RESULTS / "table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "drift_kl", *cols])
        for g in groups:
            c = summary["groups"][g]["heldout"]
            w.writerow([g, summary["groups"][g]["drift_kl"], *[c[k]["value"] if k in c else "" for k in cols]])
    for g in groups:
        c = summary["groups"][g]["heldout"]
        print(f"{g:22} " + " ".join(f"{k[:9]} {c[k]['value']:.2f}" for k in cols if k in c))
    for p in summary["predictions"]:
        print(p)


if __name__ == "__main__":
    main()
