"""Second-family re-check of the judge on the held-out run's own outputs (plan Design 3, amendment 3).

1. 200 held-out replies, stratified by unit type and strength (seed 0), scored by GPT-4.1-mini on every aspect;
   Spearman with Claude Haiku 5.5 per aspect. Pass bar: mean over the 8 traits >= 0.6.
2. Composition labels recomputed with GPT-4.1-mini: pair rows on their two targets and coherence, singles at their
   operating strength on their target, unsteered rows on every trait. The retention labels (breaks, additive,
   dominates, interferes; no emergent test) are compared with Haiku's retention labels. Pass bar: >= 70% agree.

    python recheck.py                 # judge and analyse
    python recheck.py --analyse-only
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict

import analyze
import judge
import pipeline
import va

TRAITS = list(va.TRAITS)


def sample_rows(rows: list[dict]) -> list[dict]:
    rng = random.Random(0)
    strata = defaultdict(list)
    for r in rows:
        if r["kind"] in ("single", "pair", "prompt"):
            strata[(r["kind"], analyze.pair_type(tuple(r["uids"])), tuple(r["s"]))].append(r)
    keys = sorted(strata)
    picked: list[dict] = []
    seen: set = set()
    i = 0
    while len(picked) < 200 and i < 200 * len(keys):
        k = keys[i % len(keys)]
        pool = [r for r in strata[k] if r["rid"] not in seen]
        if pool:
            r = rng.choice(pool)
            picked.append(r)
            seen.add(r["rid"])
        i += 1
    return picked


def label_rows(rows: list[dict], sel: dict) -> list[tuple[dict, list[str]]]:
    op = {(v["uid"], v["s"]) for v in sel.values()}
    jobs = []
    for r in rows:
        if r["kind"] == "unsteered":
            jobs.append((r, TRAITS))
        elif r["kind"] == "pair" and not r["uids"][0].startswith("prompt-"):  # amendment 7: steering pairs only
            jobs.append((r, [*r["traits"], "coherence"]))
        elif r["kind"] == "single" and (r["uids"][0], r["s"][0]) in op:
            jobs.append((r, r["traits"]))
    return jobs


def same_rows(rows: list[dict], other: str = "gpt") -> list[dict]:
    """Rows the second judge fully scored for its label job, so both judges' labels come from the same replies (fixed
    after the code review: the first version compared each judge on whatever rows it had)."""
    keep = []
    for r in rows:
        sc = r["scores"].get(other, {})
        need = {"unsteered": list(va.TRAITS), "pair": [*r["traits"], "coherence"]}.get(r["kind"], r["traits"])
        if r["kind"] in ("unsteered", "pair", "single") and need and all(sc.get(a) is not None for a in need):
            keep.append(r)
    return keep


def retention_labels(rows: list[dict], judge_name: str, sel: dict) -> dict:
    base = {
        (r["pi"], a): v for r in rows if r["kind"] == "unsteered" for a, v in r["scores"].get(judge_name, {}).items()
    }

    def mean_delta(rs, t):
        ds = [
            r["scores"][judge_name][t] - base[(r["pi"], t)]
            for r in rs
            if r["scores"].get(judge_name, {}).get(t) is not None and base.get((r["pi"], t)) is not None
        ]
        return st.mean(ds) if ds else None

    singles, pairs = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["kind"] in ("single", "prompt"):
            singles[(r["uids"][0], tuple(r["s"]))].append(r)
        elif r["kind"] == "pair":
            pairs[tuple(r["uids"])].append(r)
    uid_s = {v["uid"]: v["s"] for v in sel.values()}
    out = {}
    for uids, rs in pairs.items():
        ta, tb = rs[0]["traits"]
        sa = () if uids[0].startswith("prompt-") else (uid_s[uids[0]],)
        sb = () if uids[1].startswith("prompt-") else (uid_s[uids[1]],)
        da, db = mean_delta(singles[(uids[0], sa)], ta), mean_delta(singles[(uids[1], sb)], tb)
        dab_a, dab_b = mean_delta(rs, ta), mean_delta(rs, tb)
        if None in (da, db, dab_a, dab_b):
            continue
        coh = [r["scores"][judge_name].get("coherence") for r in rs if r["scores"].get(judge_name)]
        coh = [c for c in coh if c is not None]
        coh_m = st.mean(coh) if coh else 0
        loop = sum(r["loops"] for r in rs) / len(rs)
        if loop > va.LOOP_BAR or coh_m < va.BREAK_COHERENCE:
            lab = "breaks"
        else:
            ka = abs(da) > 1e-9 and dab_a / da >= 0.5
            kb = abs(db) > 1e-9 and dab_b / db >= 0.5
            lab = "additive" if ka and kb else ("dominates" if ka or kb else "interferes")
        out["+".join(uids)] = {"label": lab, "eligible": da >= va.MIN_SINGLE and db >= va.MIN_SINGLE}
    return out


def main(analyse_only: bool) -> None:
    rows = pipeline.load_scored("heldout")
    sel = json.loads((va.RESULTS / "selection.json").read_text())
    sample = sample_rows(rows)
    if not analyse_only:
        judge.run_scores("gpt", sample, lambda r: judge.ASPECTS)
        jobs = label_rows(rows, sel)
        need = {r["rid"]: a for r, a in jobs}
        judge.run_scores("gpt", [r for r, _ in jobs], lambda r: need[r["rid"]])
        (va.RESULTS / "scored_heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    agree = {}
    for a in judge.ASPECTS:
        xy = [(r["scores"]["haiku"].get(a), r["scores"].get("gpt", {}).get(a)) for r in sample]
        xy = [(x, y) for x, y in xy if x is not None and y is not None]
        agree[a] = {
            "spearman": analyze.spearman([x for x, _ in xy], [y for _, y in xy]),
            "n": len(xy),
            "within_1": sum(abs(x - y) <= 1 for x, y in xy) / max(len(xy), 1),
        }
    mean_traits = st.mean(agree[t]["spearman"] for t in TRAITS)
    shared = [r for r in same_rows(rows) if not (r["uids"] and r["uids"][0].startswith("prompt-"))]  # amendment 7
    lh, lg = retention_labels(shared, "haiku", sel), retention_labels(shared, "gpt", sel)
    common = [k for k in lh if k in lg]
    n_rows = {}
    for r in shared:
        if r["kind"] == "pair":
            n_rows["+".join(r["uids"])] = n_rows.get("+".join(r["uids"]), 0) + 1
    by_cov = {}
    for m in (1, 4, 8):
        ks = [k for k in common if n_rows.get(k, 0) >= m]
        by_cov[f">={m}"] = {"pairs": len(ks), "agree": sum(lh[k]["label"] == lg[k]["label"] for k in ks)}
    same = [k for k in common if lh[k]["label"] == lg[k]["label"]]
    el = [k for k in common if lh[k]["eligible"]]
    same_el = [k for k in el if lh[k]["label"] == lg[k]["label"]]
    out = {
        "aspects": agree,
        "mean_trait_spearman": mean_traits,
        "pass_spearman": mean_traits >= 0.6,
        "labels_compared": len(common),
        "labels_agree": len(same),
        "label_agreement": len(same) / max(len(common), 1),
        "eligible_compared": len(el),
        "eligible_agree": len(same_el),
        "pass_labels": len(same) / max(len(common), 1) >= 0.7,
        "agreement_by_pair_replies": by_cov,
        "disagreements": [
            {"pair": k, "haiku": lh[k]["label"], "gpt": lg[k]["label"]}
            for k in common
            if lh[k]["label"] != lg[k]["label"]
        ],
    }
    (va.RESULTS / "recheck.json").write_text(json.dumps(out, indent=1))
    print(
        f"mean trait Spearman {mean_traits:.3f} (pass {out['pass_spearman']}); labels agree {len(same)}/{len(common)}"
        f" = {out['label_agreement']:.2f} (pass {out['pass_labels']}); eligible {len(same_el)}/{len(el)}"
    )
    for a, v in agree.items():
        print(f"  {a:10s} rho {v['spearman']:.2f}  within-1 {v['within_1']:.2f}  n {v['n']}")


if __name__ == "__main__":
    main("--analyse-only" in sys.argv)
