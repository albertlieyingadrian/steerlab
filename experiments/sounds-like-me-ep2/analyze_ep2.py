"""Episode 2 analysis: one table per arm (held-out) and the pre-registered predictions, scored mechanically.

    python analyze_ep2.py      -> results/summary.json, results/predictions.json (prints both)

Intervals are 95% bootstraps over the 30 held-out authors (paired where two arms share authors). Honesty rates are
shares of drafts with at least one unsupported claim from Episode 1's frozen extractor + rule layer: lower bounds (A3).
"""

import json
from collections import defaultdict

import numpy as np

from ep2 import DATA, RES

RNG = np.random.default_rng(20261122)
SCORED = RES / "scored"
LADDER = [
    "R1_prompt",
    "R1b_bon",
    "R2_harness",
    "R3_voice",
    "xR3_s0.3",
    "R4_persona",
    "R5_both",
    "R6_rand0",
    "R6_rand1",
    "G2_gemini_harness",
]
ABLATIONS = ["abl_noretrieval", "abl_noprofile", "abl_norevise", "abl_norerank"]
COMPARATORS = [
    "ep1-gemini-edits",
    "ep1-gemini-k3",
    "ep1-llamabase-k3",
    "ep1-sonnet-note",
    "ep1-qwen8b-k8",
    "ep1-qwen8b-k0",
]
PERSONA = ["none", "away", "toward", "rand0", "rand1", "rand2", "card", "own", "card_own"]


def boot_mean(v, n=2000):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if not len(v):
        return [None, None, None]
    m = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return [round(float(v.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def load():
    eps, drafts = {}, {}
    for p in sorted(SCORED.glob("*__heldout.json")):
        d = json.loads(p.read_text())
        for e in d["episodes"]:
            eps.setdefault(e["arm"], {})[e["author"]] = e
        for r in d["drafts"]:
            drafts.setdefault(r["arm"], []).append(r)
    hon = {}
    for p in sorted((DATA / "honesty").glob("*__heldout.json")):
        hon.update(json.loads(p.read_text()))

    def js(name):
        p = RES / name
        return json.loads(p.read_text()) if p.exists() else {}

    return (
        eps,
        drafts,
        hon,
        js("ruler_attributer.json"),
        js("ruler_pref.json"),
        js("arm_costs.json"),
        js("robustness.json"),
    )


def per_author_unsup(D, hon):
    """Per-author share of drafts with an unsupported claim (bootstrapped over authors, code review)."""
    g = defaultdict(list)
    for r in D:
        if r["job"] in hon:
            g[r["author"]].append(float(hon[r["job"]]["n_unsupported"] > 0))
    return [float(np.mean(v)) for v in g.values()]


def arm_row(arm, eps, drafts, hon, attr, pref, costs, rob=None):
    E = list(eps.get(arm, {}).values())
    D = [r for r in drafts.get(arm, []) if r["text"].strip()]
    H = [hon[r["job"]] for r in D if r["job"] in hon]
    a = attr.get("per_arm", {}).get(arm, {})
    ps = pref.get("draft_scores", {}).get(arm, {})
    cost = costs.get(f"{arm}__heldout", {})
    nll = [r.get("nll") for r in D if r.get("nll") is not None]

    def mean(key):
        return round(float(np.mean([r[key] for r in D])), 3) if D else None

    return dict(
        episodes=len(E),
        drafts=len(D),
        position=boot_mean([e["position"] for e in E]),
        position80=boot_mean([e["position80"] for e in E]),
        position_loose=boot_mean([e["position_loose"] for e in E]),
        spec=boot_mean([e["spec"] for e in E]),
        spec_share_positive=round(float(np.mean([e["spec"] > 0 for e in E])), 3) if E else None,
        spec80=boot_mean([e["spec80"] for e in E]),
        copy_flag_episodes=sum(e["copy_flag"] for e in E),
        words=mean("words"),
        detector=mean("det"),
        loops=mean("loop"),
        nll=round(float(np.mean(nll)), 3) if nll else None,
        task=mean("task"),
        unsupported=boot_mean(per_author_unsup(D, hon)) if H else [None, None, None],
        unsupported_share=round(float(np.mean([h["n_unsupported"] > 0 for h in H])), 4) if H else None,
        invented_relationship=round(float(np.mean([bool(h["invented_relationship"]) for h in H])), 3) if H else None,
        claims_made=round(float(np.mean([h["n_claims"] for h in H])), 2) if H else None,
        honesty_n=len(H),
        attr_mean_rank=a.get("mean_rank"),
        attr_top1=a.get("top1"),
        attr_top3=a.get("top3"),
        pref_mean=round(float(np.mean(list(ps.values()))), 3) if ps else None,
        loop_share=(rob or {}).get(arm, {}).get("loop4"),
        loop_or_degenerate=(rob or {}).get(arm, {}).get("loop_or_degenerate"),
        phone_number=(rob or {}).get(arm, {}).get("phone_number"),
        clean_position=(rob or {}).get(arm, {}).get("clean_position"),
        clean_authors=(rob or {}).get(arm, {}).get("clean_authors"),
        usd_per_draft=cost.get("usd_per_draft"),
        gpu_seconds_per_draft=round(cost["gpu_seconds"] / cost["final_drafts"], 1) if cost.get("gpu_seconds") else None,
    )


def paired(eps, a, b, key="position"):
    """Mean over shared authors of a - b, with a paired bootstrap interval, and the number of authors."""
    A, B = eps.get(a, {}), eps.get(b, {})
    shared = sorted(set(A) & set(B))
    return [*boot_mean([A[x][key] - B[x][key] for x in shared]), len(shared)]


def rand_mean_paired(eps, a, rands, key):
    shared = sorted(set(eps.get(a, {})).intersection(*[set(eps.get(r, {})) for r in rands]))
    return [*boot_mean([eps[a][x][key] - np.mean([eps[r][x][key] for r in rands]) for x in shared]), len(shared)]


def win_rate(pref, a, b):
    """Share of (author, brief) pairs where the preference model scores arm a above arm b, bootstrap over authors."""
    sa, sb = pref.get("draft_scores", {}).get(a, {}), pref.get("draft_scores", {}).get(b, {})
    by = defaultdict(list)
    for k in set(sa) & set(sb):
        by[k.split("|")[0]].append(float(sa[k] > sb[k]))
    allv = [x for v in by.values() for x in v]
    if not allv:
        return [None, None, None, 0]
    bs = boot_mean([np.mean(v) for v in by.values()])
    return [round(float(np.mean(allv)), 4), bs[1], bs[2], len(allv)]


def unsupported_diff(drafts, hon, a, b):
    """Difference in unsupported-claim share (a - b), per-author means, bootstrap over authors."""

    def per_author(arm):
        g = defaultdict(list)
        for r in drafts.get(arm, []):
            if r["job"] in hon:
                g[r["author"]].append(float(hon[r["job"]]["n_unsupported"] > 0))
        return {k: np.mean(v) for k, v in g.items()}

    pa, pb = per_author(a), per_author(b)
    shared = sorted(set(pa) & set(pb))
    return [*boot_mean([pa[x] - pb[x] for x in shared]), len(shared)]


def contrasts_for(eps, drafts, hon, attr, pref):
    c = {
        "rung2-gemini_edits": paired(eps, "R2_harness", "ep1-gemini-edits"),
        "rung2-gemini_edits_80w": paired(eps, "R2_harness", "ep1-gemini-edits", "position80"),
        "rung2-gemini_edits_spec": paired(eps, "R2_harness", "ep1-gemini-edits", "spec"),
        "rung2-llamabase": paired(eps, "R2_harness", "ep1-llamabase-k3"),
        "R1b-rung1": paired(eps, "R1b_bon", "R1_prompt"),
        "rung2-rung1": paired(eps, "R2_harness", "R1_prompt"),
        "rung2-R1b": paired(eps, "R2_harness", "R1b_bon"),
        "rung3-rung2": paired(eps, "R3_voice", "R2_harness"),
        "rung3-R2_80w": paired(eps, "R3_voice", "R2_harness", "position80"),
        "rung3-R2_spec": paired(eps, "R3_voice", "R2_harness", "spec"),
        "rung4-rung2": paired(eps, "R4_persona", "R2_harness"),
        "rung4-R2_spec": paired(eps, "R4_persona", "R2_harness", "spec"),
        "rung5-rung2": paired(eps, "R5_both", "R2_harness"),
        "rung3-gemini_edits": paired(eps, "R3_voice", "ep1-gemini-edits"),
        "rung5-gemini_edits": paired(eps, "R5_both", "ep1-gemini-edits"),
        "rung3-gemini_edits_80w": paired(eps, "R3_voice", "ep1-gemini-edits", "position80"),
        "xR3-gemini_edits": paired(eps, "xR3_s0.3", "ep1-gemini-edits"),
        "xR3-gemini_edits_spec": paired(eps, "xR3_s0.3", "ep1-gemini-edits", "spec"),
        "xR3-rung2": paired(eps, "xR3_s0.3", "R2_harness"),
        "xR3-R2_80w": paired(eps, "xR3_s0.3", "R2_harness", "position80"),
        "xR3-R2_spec": paired(eps, "xR3_s0.3", "R2_harness", "spec"),
        "xR3-R6mean": rand_mean_paired(eps, "xR3_s0.3", ["R6_rand0", "R6_rand1"], "position"),
        "rung3-gemini_edits_spec": paired(eps, "R3_voice", "ep1-gemini-edits", "spec"),
        "G2-gemini_edits": paired(eps, "G2_gemini_harness", "ep1-gemini-edits"),
        "G2-gemini_edits_spec": paired(eps, "G2_gemini_harness", "ep1-gemini-edits", "spec"),
        "G2-gemini_edits_80w": paired(eps, "G2_gemini_harness", "ep1-gemini-edits", "position80"),
        "G2-gemini_k3_spec": paired(eps, "G2_gemini_harness", "ep1-gemini-k3", "spec"),
        "rung3-R6mean": rand_mean_paired(eps, "R3_voice", ["R6_rand0", "R6_rand1"], "position"),
        "rung3-R6mean_80w": rand_mean_paired(eps, "R3_voice", ["R6_rand0", "R6_rand1"], "position80"),
        "rung3-R6mean_spec": rand_mean_paired(eps, "R3_voice", ["R6_rand0", "R6_rand1"], "spec"),
        "away-rand": rand_mean_paired(eps, "away", ["rand0", "rand1", "rand2"], "position"),
        "away-rand_spec": rand_mean_paired(eps, "away", ["rand0", "rand1", "rand2"], "spec"),
        "away-rand_80w": rand_mean_paired(eps, "away", ["rand0", "rand1", "rand2"], "position80"),
        "own-rand": rand_mean_paired(eps, "own", ["rand0", "rand1", "rand2"], "position"),
        "own-rand_spec": rand_mean_paired(eps, "own", ["rand0", "rand1", "rand2"], "spec"),
        "own-away_spec": paired(eps, "own", "away", "spec"),
        "toward-none": paired(eps, "toward", "none"),
        "card-none": paired(eps, "card", "none"),
        "card-none_spec": paired(eps, "card", "none", "spec"),
        "own-none_spec": paired(eps, "own", "none", "spec"),
        "card_own-own_spec": paired(eps, "card_own", "own", "spec"),
        "card_own-card_spec": paired(eps, "card_own", "card", "spec"),
    }
    for ab in ABLATIONS:
        c[f"rung2-{ab}"] = paired(eps, "R2_harness", ab)
    c["unsup_R3-rung1"] = unsupported_diff(drafts, hon, "R3_voice", "R1_prompt")
    c["unsup_R2-rung1"] = unsupported_diff(drafts, hon, "R2_harness", "R1_prompt")
    c["unsup_xR3-rung1"] = unsupported_diff(drafts, hon, "xR3_s0.3", "R1_prompt")
    c["unsup_own-none"] = unsupported_diff(drafts, hon, "own", "none")
    c["unsup_away-none"] = unsupported_diff(drafts, hon, "away", "none")
    pa = attr.get("per_arm", {})
    for arm in ("R2_harness", "R3_voice", "xR3_s0.3", "R5_both", "G2_gemini_harness"):
        if arm in pa and "ep1-gemini-edits" in pa:
            ra, rb = pa[arm]["ranks"], pa["ep1-gemini-edits"]["ranks"]
            c[f"attr_rank_{arm}-gemini"] = boot_mean([ra[x] - rb[x] for x in sorted(set(ra) & set(rb))])
    c["pref_win_R2_vs_gemini"] = win_rate(pref, "R2_harness", "ep1-gemini-edits")
    c["pref_win_R2_vs_R1"] = win_rate(pref, "R2_harness", "R1_prompt")
    c["pref_win_R3_vs_gemini"] = win_rate(pref, "R3_voice", "ep1-gemini-edits")
    c["pref_win_R5_vs_gemini"] = win_rate(pref, "R5_both", "ep1-gemini-edits")
    c["pref_win_G2_vs_gemini"] = win_rate(pref, "G2_gemini_harness", "ep1-gemini-edits")
    c["pref_win_xR3_vs_gemini"] = win_rate(pref, "xR3_s0.3", "ep1-gemini-edits")
    c["pref_win_R3_vs_R2"] = win_rate(pref, "R3_voice", "R2_harness")
    c["pref_win_R1b_vs_R1"] = win_rate(pref, "R1b_bon", "R1_prompt")
    return c


def predictions(c, attr, pref):
    P = {}
    d = c["rung2-gemini_edits"]
    P["P1"] = dict(
        metric="rung2 - Gemini edits, strict position, paired over authors",
        value=d[0],
        interval=d[1:3],
        threshold="> +0.10 with interval above 0",
        hit=bool(d[0] is not None and d[0] > 0.10 and d[1] > 0),
    )
    ar, wr = c.get("attr_rank_R2_harness-gemini", [None]), c["pref_win_R2_vs_gemini"]
    # both conditions when both rulers pass; the plan's fallback: a ruler that failed its bar is dropped, and P2 rests
    # on the other one; if both fail, P2 is not scored (code review: the first version could hit on one ruler alone)
    parts = []
    if attr.get("passed"):
        parts.append(ar[0] is not None and ar[0] < 0)
    if pref.get("passed"):
        parts.append(wr[0] is not None and wr[0] >= 0.55)
    P["P2"] = dict(
        metric="attributer rank rung2 - Gemini (lower is better) and preference win rate rung2 vs Gemini",
        attr_rank_diff=ar,
        pref_win_rate=wr,
        threshold="rank difference < 0 and win rate >= 0.55",
        rulers_passed=dict(attributer=attr.get("passed"), pref=pref.get("passed")),
        hit=(all(parts) if parts else None),
    )
    g1, g2 = c["R1b-rung1"][0], c["rung2-rung1"][0]
    ratio = g1 / g2 if (g1 is not None and g2) else None
    P["P3"] = dict(
        metric="(R1b - rung1) / (rung2 - rung1), strict position",
        bon_gain=g1,
        harness_gain=g2,
        value=round(ratio, 3) if ratio is not None else None,
        threshold=">= 0.70",
        hit=bool(ratio is not None and g2 > 0 and ratio >= 0.70),
        note="read only when the harness gain is positive",
    )
    d = c["rung3-R2_80w"]
    P["P4"] = dict(
        metric="rung3 - rung2, strict position at 80 words",
        value=d[0],
        interval=d[1:3],
        threshold="< +0.05",
        hit=bool(d[0] is not None and d[0] < 0.05),
    )
    pp, sp = c["away-rand"], c["away-rand_spec"]
    P["P5"] = dict(
        metric="-assistant minus mean of 3 random (k0): position and specificity",
        position=pp,
        spec=sp,
        threshold="position >= +0.10 and specificity < +0.02",
        hit=bool(pp[0] is not None and pp[0] >= 0.10 and sp[0] < 0.02),
    )
    pj = json.loads((RES / "projection.json").read_text()) if (RES / "projection.json").exists() else {}
    P["P6"] = dict(
        metric="Spearman(projection, strict position) and partial on detector and words, Episode 1 held-out episodes",
        rho=pj.get("spearman_projection_vs_position"),
        rho_ci=pj.get("ci"),
        partial=pj.get("partial_on_detector_and_words"),
        partial_ci=pj.get("partial_ci"),
        threshold="rho <= -0.5 and partial <= -0.2",
        hit=bool(pj and pj["spearman_projection_vs_position"] <= -0.5 and pj["partial_on_detector_and_words"] <= -0.2),
    )
    d = c["unsup_R3-rung1"]
    P["P7"] = dict(
        metric="unsupported-claim share, rung3 - rung1 (per-author means)",
        value=d[0],
        interval=d[1:3],
        threshold="<= +0.05",
        hit=bool(d[0] is not None and d[0] <= 0.05),
    )
    return P


def main():
    eps, drafts, hon, attr, pref, costs, rob = load()
    arms = [x for x in LADDER + ABLATIONS + COMPARATORS + PERSONA if x in eps]
    summary = {"arms": {a: arm_row(a, eps, drafts, hon, attr, pref, costs, rob) for a in arms}}

    def js(name):
        p = RES / name
        return json.loads(p.read_text()) if p.exists() else None

    summary["rulers"] = dict(
        attributer={k: attr.get(k) for k in ("top1", "mean_rank", "passed", "pass_bar", "chance_top1")},
        pref={k: pref.get(k) for k in ("heldout_acc_a", "heldout_acc_c", "passed", "pass_bar", "model_card")},
        selfcheck=js("ruler_selfcheck_tune.json"),
        selfcheck_confirm=js("ruler_selfcheck_confirm.json"),
    )
    c = contrasts_for(eps, drafts, hon, attr, pref)
    summary["contrasts"] = c
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    P = predictions(c, attr, pref)
    (RES / "predictions.json").write_text(json.dumps(P, indent=1))
    for a, r in summary["arms"].items():
        print(
            f"{a:18s} pos {r['position'][0]!s:>8} 80w {r['position80'][0]!s:>8} spec {r['spec'][0]!s:>8} "
            f"unsup {r['unsupported'][0]!s:>6} det {r['detector']!s:>6} words {r['words']!s:>6} loops {r['loops']!s:>5} "
            f"attr {r['attr_mean_rank']!s:>6} pref {r['pref_mean']!s:>7} $ {r['usd_per_draft']}"
        )
    for k, v in P.items():
        verdict = "not scored" if v["hit"] is None else ("HIT" if v["hit"] else "miss")
        print(k, verdict, {kk: vv for kk, vv in v.items() if kk != "metric"})


if __name__ == "__main__":
    main()
