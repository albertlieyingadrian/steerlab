"""Aggregate scored held-out drafts into the leaderboard and score the pre-registered predictions.

    python analyze.py            -> results/leaderboard.json, results/predictions.json

Unit: one contestant's 4 drafts for one author under one arm (amendment A1). Intervals: 95% bootstrap over
authors (2,000 resamples). Strict voice = format-blind LUAR position (A2); copy-flagged episodes are excluded
from the strict metric (Design: copy filter on) and counted separately.
"""

import json
import re
from collections import defaultdict

import numpy as np

from common import DATA, HERE, LEDGER

CUT = json.loads((HERE / "results" / "ruler_detector.json").read_text())["cut_at_5pct_fpr"]
RNG = np.random.default_rng(20261120)
ARMS = ("k0", "k3_rand", "k3_match", "k8_rand", "note", "edits")
from relabel_rerun import RERUN_ARM, RERUN_TAG  # amendment A5 (2026-11-22); not in ARMS, so P1-P7 stay as scored

MATCH_NLL = 0.3  # the NLL tolerance the Part C tuning rule used (partc.MAX_NLL_RISE); "matched perplexity" for P4


def batch_costs():
    """Real per-draft cost of OpenRouter batch runs, from the ledger: each RESERVED batch is settled to its real cost.
    Batch drafts carry no per-draft cost of their own, so without this the closed models would read $0."""
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    res = {}
    for r in rows:
        m = re.match(r"RESERVED batch (\S+) (\S+) heldout n=(\d+)", r["note"])
        if m:
            res[m.group(1)] = (m.group(2), int(m.group(3)))
    out = {}
    for r in rows:
        m = re.match(r"SETTLE batch (\S+) real \$([0-9.]+)", r["note"])
        if m and m.group(1) in res:
            model, n = res[m.group(1)]
            out[model] = float(m.group(2)) / n
    return out


def boot(vals, n=2000):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [None, None, None]
    m = [float(np.mean(RNG.choice(v, len(v)))) for _ in range(n)]
    return [round(float(v.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def load():
    eps, drafts, hon = [], [], {}
    for p in sorted((DATA / "scored").glob("*__heldout.json")):
        d = json.loads(p.read_text())
        eps += d["episodes"]
        drafts += d["drafts"]
    for p in sorted((DATA / "honesty").glob("*__heldout.json")):
        # job ids (author|brief|arm) repeat across models, so key by the file's model as well. The edits rerun
        # (amendment A5) writes <model>-editsfix files whose job ids end in |edits_fixed: merge them into the model.
        h = json.loads(p.read_text())
        if p.stem.split("__")[0].endswith(RERUN_TAG) and any(not k.endswith(f"|{RERUN_ARM}") for k in h):
            # honesty ran before relabel_rerun.py: its |edits keys would overwrite the published cells' entries
            raise SystemExit(f"{p.name}: rerun honesty keys must end in |{RERUN_ARM}; run relabel_rerun.py first")
        hon.setdefault(p.stem.split("__")[0].removesuffix(RERUN_TAG), {}).update(h)
    return eps, drafts, hon


def cell_table(eps, drafts, hon):
    by = defaultdict(lambda: dict(pos=[], loose=[], copy=0, n=0))
    for e in eps:
        c = by[(e["model"], e["arm"])]
        c["n"] += 1
        c["loose"].append(e["position_loose"])
        if e["copy_flag"]:
            c["copy"] += 1
        else:
            c["pos"].append(e["position"])
    dr = defaultdict(list)
    for r in drafts:
        dr[(r["model"], r["arm"])].append(r)
    batch = batch_costs()
    rows = []
    for (m, arm), c in sorted(by.items()):
        ds = dr[(m, arm)]
        hm = hon.get("partc" if m.startswith("partc") else m.split(":")[0], {})
        hs = [hm[r["job"]] for r in ds if r.get("job") in hm]
        rows.append(
            dict(
                model=m,
                arm=arm,
                episodes=c["n"],
                voice_strict=boot(c["pos"]),
                voice_loose=boot(c["loose"]),
                copy_flagged=c["copy"],
                words=round(float(np.mean([r["words"] for r in ds])), 1),
                ai_score=round(float(np.mean([r["voi380"] for r in ds])), 3),
                ai_flagged=round(float(np.mean([r["voi380"] > CUT for r in ds])), 3),
                task=round(float(np.mean([r["task"] for r in ds])), 3),
                empty=int(sum(r["empty"] for r in ds)),
                unsupported=round(float(np.mean([h["n_unsupported"] > 0 for h in hs])), 3) if hs else None,
                invented_relationship=round(float(np.mean([h["invented_relationship"] for h in hs])), 3)
                if hs
                else None,
                usd_per_draft=round(
                    batch.get(
                        m + RERUN_TAG if arm == RERUN_ARM else m, float(np.mean([r.get("usd", 0) or 0 for r in ds]))
                    ),
                    5,
                ),
            )
        )
    return rows


def robust():
    """Loop-free and length-matched re-scores of the same comparison (robustness.py, added after reading drafts)."""
    rp = HERE / "results" / "robustness.json"
    if not rp.exists():
        return {}
    R = json.loads(rp.read_text())
    lr = R["loop_rate"]
    return dict(
        loop_rate=dict(
            own=lr.get("partc-Qwen3-8B|own"),
            wrong=lr.get("partc-Qwen3-8B|wrong"),
            random=max(lr.get(f"partc-Qwen3-8B|rand{i}", 0) for i in range(3)),
            unsteered=lr.get("partc-Qwen3-8B|none"),
        ),
        loop_free=R["partc"]["loop_free"]["own_minus_random"],
        length_matched=R["partc"]["length_matched"]["own_minus_random"],
    )


def p4(eps, drafts):
    """P4: own voice vector minus norm-matched random, strict position, "at matched perplexity".
    Perplexity is the NLL of each draft under the unsteered model (modal_partc.steer). Matched means the own arm's mean
    NLL is within MATCH_NLL of the random arms'. If it is not, the pre-registered number is still reported (the test is
    not redefined), with the gap and a subset check on authors whose own-vector drafts fall inside the random arms'
    NLL range."""
    pe = [e for e in eps if e["model"] == "partc-Qwen3-8B" and not e["copy_flag"]]
    pos = defaultdict(dict)
    for e in pe:
        pos[e["arm"]][e["author"]] = e["position"]
    nll = defaultdict(lambda: defaultdict(list))
    for r in drafts:
        if r["model"] == "partc-Qwen3-8B" and r.get("nll") is not None:
            nll[r["arm"]][r["author"]].append(r["nll"])
    if "own" not in pos:
        return None
    rands = [a for a in ("rand0", "rand1", "rand2") if a in pos]
    authors_ = sorted(a for a in pos["own"] if all(a in pos[r] for r in rands))
    diff = [pos["own"][a] - np.mean([pos[r][a] for r in rands]) for a in authors_]
    ep_nll = {arm: {a: float(np.mean(v)) for a, v in per.items()} for arm, per in nll.items()}
    own_nll = float(np.mean([x for v in nll["own"].values() for x in v]))
    rand_nll = float(np.mean([x for r in rands for v in nll[r].values() for x in v]))
    none_nll = float(np.mean([x for v in nll["none"].values() for x in v]))
    rand_eps = [ep_nll[r][a] for r in rands for a in ep_nll[r]]
    lo, hi = float(np.percentile(rand_eps, 5)), float(np.percentile(rand_eps, 95))
    inside = [a for a in authors_ if lo <= ep_nll["own"].get(a, -1) <= hi]
    sub = [pos["own"][a] - np.mean([pos[r][a] for r in rands]) for a in inside]
    loops = float(np.mean([x < 1.0 for v in nll["own"].values() for x in v]))
    loops_rand = float(np.mean([x < 1.0 for r in rands for v in nll[r].values() for x in v]))
    value = float(np.mean(diff))
    matched = abs(own_nll - rand_nll) <= MATCH_NLL
    return dict(
        metric="strict position, own voice vector minus norm-matched random (mean of 3 seeds), Qwen3-8B "
        "layer 24 strength 0.4, 30 held-out authors",
        value=round(value, 4),
        interval=boot(diff)[1:],
        threshold="> 0.05, at matched perplexity",
        hit=value > 0.05,
        perplexity_matched=matched,
        nll=dict(
            own=round(own_nll, 3),
            random=round(rand_nll, 3),
            unsteered=round(none_nll, 3),
            tolerance=MATCH_NLL,
            share_below_1_own=round(loops, 3),
            share_below_1_random=round(loops_rand, 3),
        ),
        matched_subset=dict(
            rule=f"authors whose own-vector episode NLL lies inside the random arms' 5-95% range [{lo:.2f}, {hi:.2f}]",
            n_authors=len(inside),
            value=round(float(np.mean(sub)), 4) if sub else None,
            interval=boot(sub)[1:] if sub else None,
        ),
        **robust(),
        verdict=("hit" if value > 0.05 else "miss")
        + (
            ""
            if matched
            else ": perplexity NOT matched (own-vector drafts have lower NLL than random-vector drafts, a sign "
            "of repetitive text in some drafts); read the matched-subset value beside it"
        ),
    )


def score_predictions(rows, eps=None, drafts=None):
    T = {(r["model"], r["arm"]): r for r in rows}
    models = sorted(
        {r["model"] for r in rows if r["model"] != "ctrl" and not r["model"].startswith(("humanizer", "partc"))}
    )

    def v(m, a):
        r = T.get((m, a))
        return None if not r or r["voice_strict"][0] is None else r["voice_strict"][0]

    best_arm = {m: max((v(m, a), a) for a in ARMS if v(m, a) is not None) for m in models}
    best_model = max(models, key=lambda m: best_arm[m][0])
    P = {}
    P["P1"] = dict(
        metric="best model's best-arm strict voice position",
        value=best_arm[best_model][0],
        model=best_model,
        arm=best_arm[best_model][1],
        threshold="< 0",
        hit=best_arm[best_model][0] < 0,
    )
    k8_best = max((m for m in models if v(m, "k8_rand") is not None), key=lambda m: v(m, "k8_rand"))
    gain = v(k8_best, "k8_rand") - v(k8_best, "k0")
    P["P2"] = dict(
        metric="k8 - k0 strict position, best k8 model",
        model=k8_best,
        value=round(gain, 4),
        threshold="> 0.15",
        hit=gain > 0.15,
    )
    hz = [r for r in rows if r["model"].startswith("humanizer")]
    if hz:
        out = []
        for r in hz:
            src = T.get((r["model"].split(":", 1)[1], "k0"))
            if src and r["voice_strict"][0] is not None:
                dd = r["ai_score"] - src["ai_score"]
                dv = r["voice_strict"][0] - src["voice_strict"][0]
                out.append(
                    dict(
                        humanizer=r["model"],
                        detector_change=round(dd, 3),
                        voice_change=round(dv, 4),
                        hit=dd <= -0.2 and dv < 0.05,
                    )
                )
        sp = HERE / "results" / "specificity.json"
        if sp.exists():  # amendment A4 side reading: does the humanizer add voice for THIS author, or only human-ness?
            S = json.loads(sp.read_text())
            for o in out:
                src = o["humanizer"].split(":", 1)[1]
                if f"{o['humanizer']}|k0" in S and f"{src}|k0" in S:
                    o["specificity_change"] = round(S[f"{o['humanizer']}|k0"]["mean"] - S[f"{src}|k0"]["mean"], 4)
        P["P3"] = dict(
            metric="humanizer: detector change and voice change vs its source draft",
            threshold="detector <= -0.2 and voice change < 0.05",
            results=out,
            hit=all(o["hit"] for o in out) if out else None,
        )
    if eps is not None:
        q = p4(eps, drafts)
        if q:
            P["P4"] = q
    P["P5"] = dict(
        metric="Spearman rho, vote rating vs LUAR position",
        threshold="< 0.5",
        hit=None,
        status="not scored: the phase-1 arena is a mock with no votes; carried to arena phase 2",
    )
    cands = [m for m in models if v(m, "note") is not None and v(m, "k8_rand") is not None]
    if cands:
        nb = max(cands, key=lambda m: v(m, "k8_rand"))
        ng, sg = v(nb, "note") - v(nb, "k0"), v(nb, "k8_rand") - v(nb, "k0")
        P["P6"] = dict(
            metric="style-note gain vs samples gain (vs k0)",
            model=nb,
            note_gain=round(ng, 4),
            samples_gain=round(sg, 4),
            threshold="note gain < half the samples gain",
            hit=ng < 0.5 * sg,
        )
    pr = []
    for b, i in (("Qwen3-8B-Base", "Qwen3-8B"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct")):
        if v(b, "k8_rand") is not None and v(i, "k8_rand") is not None:
            pr.append(dict(base=b, instruct=i, diff=round(v(b, "k8_rand") - v(i, "k8_rand"), 4)))
    if pr:
        mean = float(np.mean([p["diff"] for p in pr]))
        P["P7"] = dict(
            metric="base - instruct strict position at k8, mean over families",
            pairs=pr,
            value=round(mean, 4),
            threshold="> 0.05",
            hit=mean > 0.05,
        )
    return P


def paired(eps, model, arm_a, arm_b):
    """arm_a - arm_b, strict position, paired over writers kept by the copy filter in both arms."""
    pos = defaultdict(dict)
    for e in eps:
        if e["model"] == model and e["arm"] in (arm_a, arm_b) and not e["copy_flag"]:
            pos[e["arm"]][e["author"]] = e["position"]
    common = sorted(set(pos[arm_a]) & set(pos[arm_b]))
    return dict(value_ci=boot([pos[arm_a][a] - pos[arm_b][a] for a in common]), n_authors=len(common))


def edits_rerun(rows, eps):
    """Amendment A5: the fixed edits arm against the published `edits` cell ("3 more samples*", the same 3 samples
    without the pairs) and against k3_rand, per model that was rerun. P8 is Gemini's first contrast."""
    T = {(r["model"], r["arm"]): r for r in rows}
    out = {}
    for m in sorted({r["model"] for r in rows if r["arm"] == RERUN_ARM}):
        new, old = T[(m, RERUN_ARM)], T.get((m, "edits"))
        out[m] = dict(
            fixed=new,
            vs_3_more_samples=paired(eps, m, RERUN_ARM, "edits"),
            vs_k3_rand=paired(eps, m, RERUN_ARM, "k3_rand"),
            three_more_samples=old,
        )
    p8 = None
    if "gemini-4-carbon" in out:
        v, lo, hi = out["gemini-4-carbon"]["vs_3_more_samples"]["value_ci"]
        p8 = dict(
            metric="Gemini 4 Carbon, edits_fixed - '3 more samples*' (same samples, no pairs), strict position, "
            "paired over writers",
            value=v,
            interval=[lo, hi],
            threshold="< +0.05",
            hit=v is not None and v < 0.05,
            reading="no detectable effect"
            if lo is not None and lo <= 0 <= hi
            else ("helped" if lo and lo > 0 else "hurt"),
        )
    return out, p8


def main():
    eps, drafts, hon = load()
    rows = cell_table(eps, drafts, hon)
    (HERE / "results" / "leaderboard.json").write_text(json.dumps(rows, indent=1))
    P = score_predictions(rows, eps, drafts)
    rerun, p8 = edits_rerun(rows, eps)
    if rerun:
        (HERE / "results" / "edits_rerun.json").write_text(json.dumps(rerun, indent=1))
    if p8:
        P["P8"] = p8
    (HERE / "results" / "predictions.json").write_text(json.dumps(dict(sorted(P.items())), indent=1))
    for r in rows:
        print(
            f"{r['model'][:26]:26s} {r['arm']:12s} strict {r['voice_strict']} ai {r['ai_score']} uns {r['unsupported']}"
        )
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
