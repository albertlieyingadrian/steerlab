"""Place Inkling 2 Mini on Episode 1's leaderboard and score the add-on's predictions.

    python analyze.py   -> results/inkling.json, results/board.json, results/contrasts.json, results/predictions.json,
                           results/spend.json

Unit: one writer's 4 drafts under one arm (Episode 1's amendment A1). Intervals: 95% bootstrap over writers (2,000
resamples). Strict position excludes copy-flagged episodes (Episode 1's rule); contrasts are paired per writer and use
writers present, unflagged, on both sides. Episode 1's numbers are read from its committed results (leaderboard,
specificity, robustness) and, for paired contrasts and the 80-word check, from its private scored drafts.
Every file written here is aggregate: no draft text, no writer names.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EP1 = HERE.parent / "sounds-like-me"
EP2 = HERE.parent / "sounds-like-me-ep2"
RES = HERE / "results"
SCORED = HERE / "data" / "scored"
sys.path.insert(0, str(HERE))

from score_inkling import EP1_MODELS

ARMS = ("k0", "k3_rand", "k3_match", "k8_rand", "note", "edits")
INK = "inkling-2-mini"
INK_LOW = "inkling-2-mini-reason-low"
INK_FIX = "inkling-2-mini-edits-fixed"
CUT = json.loads((EP1 / "results" / "ruler_detector.json").read_text())["cut_at_5pct_fpr"]
LABEL = {
    INK: "Inkling 2 Mini",
    INK_LOW: "Inkling 2 Mini, reasoning low",
    "gemini-4-carbon": "Gemini 4 Carbon",
    "claude-sonnet-5.5": "Claude Sonnet 5.5",
    "gemma-3-27b-it": "Gemma 3 27B",
    "Qwen3-32B": "Qwen3-32B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-8B-Base": "Qwen3-8B Base",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B Instruct",
    "Llama-3.1-8B": "Llama-3.1-8B Base",
}
SHOWN = (
    "voice_strict",
    "voice_loose",
    "voice80",
    "spec",
    "words",
    "ai_flagged",
    "unsupported",
    "loop_share",
    "usd_per_draft",
    "task",
)


def boot(vals, seed, n=2000, alpha=0.05):
    """Mean and percentile bootstrap interval; alpha / k gives a Bonferroni interval over k comparisons."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n if alpha >= 0.05 else 20000, len(v))).mean(1)
    lo, hi = np.percentile(m, [50 * alpha, 100 - 50 * alpha])
    return [round(float(v.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


def load_ink(model):
    d = json.loads((SCORED / f"{model}__heldout.json").read_text())
    hp = HERE / "data" / "honesty" / f"{model}__heldout.json"
    hon = json.loads(hp.read_text()) if hp.exists() else {}
    return d["drafts"], d["episodes"], hon


def ink_rows(model, drafts, eps, hon):
    rows = []
    for arm in ARMS:
        E = [e for e in eps if e["arm"] == arm]
        if not E:
            continue
        D = [r for r in drafts if r["arm"] == arm]
        H = [hon[r["job"]] for r in D if r["job"] in hon]
        live = [r for r in D if not r["empty"]]
        clean = [e for e in E if not e["copy_flag"]]
        rows.append(
            dict(
                model=model,
                arm=arm,
                episodes=len(E),
                voice_strict=boot([e["position"] for e in clean], 1),
                voice_loose=boot([e["position_loose"] for e in E], 2),
                voice80=boot([e["position80"] for e in clean], 3),
                spec=boot([e["spec"] for e in E], 4),  # Episode 1's specificity keeps copy-flagged episodes
                spec_share_positive=round(float(np.mean([e["spec"] > 0 for e in E])), 3),
                copy_flagged=sum(e["copy_flag"] for e in E),
                drafts=len(D),
                words=round(float(np.mean([r["words"] for r in live])), 1),
                words_range=[int(min(r["words"] for r in live)), int(max(r["words"] for r in live))],
                completion_tokens=round(float(np.mean([r["completion_tokens"] or 0 for r in D])), 1),
                reasoning_tokens=round(float(np.mean([r["reasoning_tokens"] or 0 for r in D])), 1),
                ai_score=round(float(np.mean([r["voi380"] for r in live])), 3),
                ai_flagged=round(float(np.mean([r["voi380"] > CUT for r in live])), 3),
                task=round(float(np.mean([r["task"] for r in live])), 3),
                empty=int(sum(r["empty"] for r in D)),
                loop_share=round(float(np.mean([r["loop"] for r in live])), 3),
                cap_hit=round(float(np.mean([r["cap_hit"] for r in D])), 3),
                unsupported=round(float(np.mean([h["n_unsupported"] > 0 for h in H])), 3) if H else None,
                unsupported_n=[int(sum(h["n_unsupported"] > 0 for h in H)), len(H)] if H else None,
                invented_relationship=round(float(np.mean([h["invented_relationship"] for h in H])), 3) if H else None,
                usd_per_draft=round(float(np.mean([r["usd"] for r in D])), 6),
                providers=dict(Counter(r["provider"] for r in D)),
            )
        )
    return rows


def ep1_positions():
    """Per-writer strict positions of Episode 1's contestants (copy-flagged episodes dropped), from scored drafts."""
    pos = {}
    for m in EP1_MODELS:
        d = json.loads((EP1 / "data" / "scored" / f"{m}__heldout.json").read_text())
        for e in d["episodes"]:
            if e["model"] == m and not e["copy_flag"]:
                pos[(m, e["arm"], e["author"])] = e["position"]
    return pos


def board(ink_by_model):
    """Every contestant's per-arm and best-arm numbers; Episode 1's from its committed results."""
    L = json.loads((EP1 / "results" / "leaderboard.json").read_text())
    S = json.loads((EP1 / "results" / "specificity.json").read_text())
    R = json.loads((EP1 / "results" / "robustness.json").read_text())["loop_rate"]
    c80 = defaultdict(list)
    kept = ep1_positions()  # strict rule: copy-flagged episodes are out, as for Inkling
    for k, v in json.loads((SCORED / "ep1_cut80.json").read_text()).items():
        m, author, arm = k.split("|")
        if (m, arm, author) in kept:
            c80[(m, arm)].append(v)
    out = {}
    for m in EP1_MODELS:
        arms = {}
        for r in L:
            if r["model"] == m and r["arm"] in ARMS:
                k = f"{m}|{r['arm']}"
                v80 = c80[(m, r["arm"])]
                arms[r["arm"]] = dict(
                    voice_strict=r["voice_strict"],
                    voice_loose=r["voice_loose"],
                    voice80=[round(float(np.mean(v80)), 4), None, None] if v80 else None,
                    spec=[S[k]["mean"], None, None] if k in S else None,
                    words=r["words"],
                    ai_flagged=r["ai_flagged"],
                    unsupported=r["unsupported"],
                    loop_share=R.get(k),
                    usd_per_draft=r["usd_per_draft"],
                    task=r["task"],
                )
        out[m] = dict(label=LABEL[m], source="Episode 1", arms=arms)
    for m, rows in ink_by_model.items():
        out[m] = dict(
            label=LABEL[m],
            source="this add-on" + (" (exploratory: reasoning low, k0 and k8 only)" if m == INK_LOW else ""),
            arms={r["arm"]: {k: r[k] for k in SHOWN} for r in rows},
        )
    for v in out.values():
        best = max(v["arms"], key=lambda a: v["arms"][a]["voice_strict"][0])
        v["best_arm"] = best
        v["best"] = v["arms"][best]["voice_strict"]
        v80 = [x["voice80"][0] for x in v["arms"].values() if x.get("voice80")]
        v["best80"] = round(max(v80), 4) if v80 else None
        sp = [x["spec"][0] for x in v["arms"].values() if x.get("spec")]
        v["best_loose"] = round(max(x["voice_loose"][0] for x in v["arms"].values()), 4)
        v["best_spec"] = round(max(sp), 4) if sp else None
    addr = defaultdict(list)  # added after reading drafts: drafts that name the recipient only
    for k, v in json.loads((SCORED / "addressed.json").read_text()).items():
        m, _, arm = k.split("|")
        addr[(m, arm)].append(v)
    for m, v in out.items():
        for arm, x in v["arms"].items():
            a = addr[(m, arm)]
            x["voice_addressed"] = [round(float(np.mean(a)), 4), len(a)] if a else None
        va = {arm: x["voice_addressed"][0] for arm, x in v["arms"].items() if x["voice_addressed"]}
        v["best_addressed"] = [max(va, key=va.get), round(max(va.values()), 4)] if va else None
    ranked = sorted((m for m in out if m != INK_LOW), key=lambda m: -out[m]["best"][0])
    for i, m in enumerate(ranked):
        out[m]["rank"] = i + 1
    ep2 = json.loads((EP2 / "results" / "summary.json").read_text())["arms"]
    return dict(
        contestants=out,
        ranked=ranked,
        reference=dict(
            ep2_harness_voice=ep2["R3_voice"]["position"],
            ep2_harness_plain=ep2["R2_harness"]["position"],
            human_specificity=S["human_ceiling"]["mean"],
            detector_cut=CUT,
        ),
    )


def paired(a, b, seed, alpha=0.05):
    """a, b: {writer: value}; mean of a - b over shared writers, bootstrap interval, n writers."""
    w = sorted(set(a) & set(b))
    return [*boot([a[x] - b[x] for x in w], seed, alpha=alpha), len(w)]


def contrasts(eps, pos):
    mine = defaultdict(dict)
    for e in eps:
        if not e["copy_flag"]:
            mine[e["arm"]][e["author"]] = e["position"]
    out = dict(within={}, vs_ep1={}, best_vs_best={}, best_vs_best_bonferroni={}, reasoning_low={})
    for i, arm in enumerate(("k3_rand", "k3_match", "k8_rand", "note", "edits")):
        out["within"][f"{arm} - k0"] = paired(mine[arm], mine["k0"], 10 + i)
    low = defaultdict(dict)
    if (SCORED / f"{INK_LOW}__heldout.json").exists():
        for e in load_ink(INK_LOW)[1]:
            if not e["copy_flag"]:
                low[e["arm"]][e["author"]] = e["position"]
        for i, arm in enumerate(("k0", "k8_rand")):
            out["reasoning_low"][f"{arm}: low - off"] = paired(low[arm], mine[arm], 30 + i)
    for j, m in enumerate(EP1_MODELS):
        for i, arm in enumerate(ARMS):
            other = {au: v for (mm, aa, au), v in pos.items() if mm == m and aa == arm}
            out["vs_ep1"][f"{m}|{arm}"] = paired(mine[arm], other, 100 + 10 * j + i)
    my_best = max(ARMS, key=lambda a: np.mean(list(mine[a].values())))
    for j, m in enumerate(EP1_MODELS):
        per = defaultdict(dict)
        for (mm, aa, au), v in pos.items():
            if mm == m and aa in ARMS:
                per[aa][au] = v
        theirs = max(per, key=lambda a: np.mean(list(per[a].values())))
        out["best_vs_best"][m] = [my_best, theirs, *paired(mine[my_best], per[theirs], 200 + j)]
        # Bonferroni over the eight comparisons (added after review)
        adj = paired(mine[my_best], per[theirs], 300 + j, alpha=0.05 / len(EP1_MODELS))
        out["best_vs_best_bonferroni"][m] = adj[1:3]
    return out


def predictions(rows, b, con, drafts):
    T = {r["arm"]: r for r in rows}
    best = b["contestants"][INK]
    P = {}
    P["P1"] = dict(
        prediction="Inkling's best arm sits more than 0.10 below Gemini 4 Carbon's best (-0.035)",
        metric="best-arm strict voice position, held-out",
        value=best["best"][0],
        interval=best["best"][1:],
        arm=best["best_arm"],
        threshold="< -0.135",
        hit=best["best"][0] < -0.135,
    )
    g = con["within"]["k8_rand - k0"]
    P["P2"] = dict(
        prediction="Writing samples help Inkling",
        metric="strict position, k8_rand - k0, paired over writers",
        value=g[0],
        interval=g[1:3],
        writers=g[3],
        threshold="> +0.15",
        hit=g[0] > 0.15,
    )
    u = T["k8_rand"]
    rd = b.get("honesty", {}).get("reading_k8")
    P["P3"] = dict(
        prediction="Inkling invents more than the frontier models",
        metric="share of k8_rand drafts with an unsupported claim (extractor + rule layer)",
        value=u["unsupported"],
        count=u["unsupported_n"],
        threshold=">= 0.10 (Gemini 0.008, Sonnet 0.017)",
        hit=u["unsupported"] is not None and u["unsupported"] >= 0.10,
        reading=rd,
        value_without_false_alarms=round(rd["swap"] / u["unsupported_n"][1], 4) if rd else None,
    )
    live = [r for r in drafts if not r["empty"]]
    cap = float(np.mean([r["cap_hit"] for r in drafts]))
    lp = float(np.mean([r["loop"] for r in live]))
    P["P4"] = dict(
        prediction="With a 400-token cap and reasoning off, length is not the story",
        metric="share of main-arm drafts that hit the cap; loop share",
        cap_hit=round(cap, 4),
        cap_hit_n=[int(sum(r["cap_hit"] for r in drafts)), len(drafts)],
        loop_share=round(lp, 4),
        loop_n=[int(sum(r["loop"] for r in live)), len(live)],
        threshold="cap-hit < 0.10 and loops < 0.05",
        hit=cap < 0.10 and lp < 0.05,
    )
    return P


def brief_reuse():
    """Added after reading drafts: share of each draft's word 6-grams found verbatim in its brief (trigger, offer, facts,
    call to action). Template-like drafts that restate the brief score high. Per contestant and arm, all drafts."""
    sys.path.insert(0, str(EP1))
    from score import words

    B = {b["id"]: b for b in json.loads((EP1 / "data" / "briefs.json").read_text())}

    def grams(t, n=6):
        w = words(t)
        return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}

    bg = {k: grams(" . ".join([b["trigger"], b["offer"], b["cta"], *b["facts"]])) for k, b in B.items()}
    files = [(m, EP1 / "data" / "scored" / f"{m}__heldout.json") for m in EP1_MODELS] + [
        (m, SCORED / f"{m}__heldout.json") for m in (INK, INK_LOW)
    ]
    out = {}
    for m, p in files:
        per = defaultdict(list)
        for r in json.loads(p.read_text())["drafts"]:
            g = grams(r["text"]) if r["model"] == m else set()
            if g:
                per[r["arm"]].append(len(g & bg[r["brief"]]) / len(g))
        out[m] = {a: round(float(np.mean(v)), 3) for a, v in per.items() if a in ARMS}
    # within Inkling (main arms): does an episode's reuse go with its voice position? Spearman over episodes
    from scipy.stats import spearmanr

    d = json.loads((SCORED / f"{INK}__heldout.json").read_text())
    ep_reuse = defaultdict(list)
    for r in d["drafts"]:
        g = grams(r["text"])
        if g:
            ep_reuse[(r["author"], r["arm"])].append(len(g & bg[r["brief"]]) / len(g))
    by_writer = defaultdict(list)
    for e in d["episodes"]:
        if not e["copy_flag"]:
            by_writer[e["author"]].append((np.mean(ep_reuse[(e["author"], e["arm"])]), e["position"]))
    groups = list(by_writer.values())

    def rho(gs):
        return float(spearmanr(np.array([p for g in gs for p in g])).statistic)

    rng = np.random.default_rng(7)  # episodes repeat writers, so writers are resampled, not episodes
    bs = [rho([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out["inkling_episode_spearman"] = dict(
        rho=[round(rho(groups), 3), round(float(lo), 3), round(float(hi), 3)],
        episodes=sum(map(len, groups)),
        writers=len(groups),
    )
    return out


def news_claimed():
    """Added after reading drafts: Episode 1's rule "sender claims the recipient's news", per contestant and arm, as
    [drafts flagged, drafts checked]. A regex over a short verb list: it misses other wordings and has false alarms."""
    files = [(m, EP1 / "data" / "honesty" / f"{m}__heldout.json") for m in EP1_MODELS] + [
        (m, HERE / "data" / "honesty" / f"{m}__heldout.json") for m in (INK, INK_LOW)
    ]
    out = {}
    for m, p in files:
        per = defaultdict(lambda: [0, 0])
        for job, h in json.loads(p.read_text()).items():
            arm = job.rsplit("|", 1)[1]
            if arm in ARMS:
                per[arm][0] += any(f.startswith("sender claims") for f in h["rule_flags"])
                per[arm][1] += 1
        out[m] = {"all": [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())], **per}
    return out


def unfiltered_best():
    """Best-signal strict position with copy-flagged episodes kept (the rank 8 versus 9 question)."""
    out = {}
    for m in (*EP1_MODELS, INK):
        p = (EP1 / "data" / "scored" / f"{m}__heldout.json") if m != INK else (SCORED / f"{INK}__heldout.json")
        per = defaultdict(list)
        for e in json.loads(p.read_text())["episodes"]:
            if e["model"] == m:
                per[e["arm"]].append(e["position"])
        out[m] = round(max(float(np.mean(v)) for a, v in per.items() if a in ARMS), 4)
    return out


def parity():
    """Episode 1's Gemini drafts through this scorer, against what Episode 1 published (added after reading)."""
    ref = "gemini-4-carbon"
    mine = json.loads((SCORED / f"{ref}__heldout.json").read_text())["episodes"]
    L = json.loads((EP1 / "results" / "leaderboard.json").read_text())
    S = json.loads((EP1 / "results" / "specificity.json").read_text())
    gaps = defaultdict(float)
    n = 0
    for r in L:
        if r["model"] != ref:
            continue
        E = [e for e in mine if e["arm"] == r["arm"]]
        n += len(E)
        got = dict(
            voice_strict=(np.mean([e["position"] for e in E if not e["copy_flag"]]), r["voice_strict"][0]),
            voice_loose=(np.mean([e["position_loose"] for e in E]), r["voice_loose"][0]),
            spec=(np.mean([e["spec"] for e in E]), S[f"{ref}|{r['arm']}"]["mean"]),
            copy_flagged=(sum(e["copy_flag"] for e in E), r["copy_flagged"]),
        )
        for k, (a, b) in got.items():
            gaps[k] = round(max(gaps[k], float(abs(a - b))), 5)
    return dict(contestant=ref, episodes=n, largest_gap=dict(gaps))


def honesty_context():
    """Added after review: every contestant's unsupported-claim counts per signal and over all signals, the
    reasoning-low change in Inkling's unsupported share, where the news rule fired, and the reading of the 16 flags."""
    files = {m: EP1 / "data" / "honesty" / f"{m}__heldout.json" for m in EP1_MODELS}
    files.update({m: HERE / "data" / "honesty" / f"{m}__heldout.json" for m in (INK, INK_LOW)})
    counts = {}
    for m, p in files.items():
        per = defaultdict(lambda: [0, 0])
        for job, h in json.loads(p.read_text()).items():
            arm = job.rsplit("|", 1)[1]
            if arm in ARMS:
                per[arm][0] += h["n_unsupported"] > 0
                per[arm][1] += 1
        counts[m] = {"all": [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())], **per}
    H = {m: json.loads(files[m].read_text()) for m in (INK, INK_LOW)}
    low = {}
    for i, arm in enumerate(("k0", "k8_rand")):
        share = {m: defaultdict(list) for m in (INK, INK_LOW)}
        for m in (INK, INK_LOW):
            for job, h in H[m].items():
                au, _, a = job.split("|")
                if a == arm:
                    share[m][au].append(h["n_unsupported"] > 0)
        low[arm] = paired(
            {k: float(np.mean(v)) for k, v in share[INK_LOW].items()},
            {k: float(np.mean(v)) for k, v in share[INK].items()},
            40 + i,
        )
    briefs = defaultdict(int)
    for job, h in H[INK].items():
        if any(f.startswith("sender claims") for f in h["rule_flags"]):
            briefs[job.split("|")[1]] += 1
    top = sorted(briefs.values(), reverse=True)
    rp = HERE / "data" / "honesty" / "reading_k8.json"  # private: keyed by job
    reading = Counter(v["verdict"] for v in json.loads(rp.read_text()).values()) if rp.exists() else None
    return dict(
        unsupported=counts,
        reasoning_low_minus_off=low,
        news_rule=dict(hits=sum(top), briefs_hit=len(top), top5=sum(top[:5]), briefs_total=30),
        reading_k8=dict(reading) if reading else None,
    )


def edits_rerun():
    """The edits signal rerun after Episode 1's template fix (lab #847/#848): the 3 samples now arrive with 3 simulated
    draft-and-edit pairs. Same writers, briefs and settings as the main run; paired over writers against Inkling's
    other signals. Added after publication; the other contestants' corrected arm is rerun by Episode 1's correction."""
    p = SCORED / f"{INK_FIX}__heldout.json"
    if not p.exists():
        return None
    drafts, eps = json.loads(p.read_text())["drafts"], json.loads(p.read_text())["episodes"]
    row = ink_rows(INK_FIX, drafts, eps, {})[0]
    fixed = {e["author"]: e["position"] for e in eps if not e["copy_flag"]}
    mine = defaultdict(dict)
    for e in load_ink(INK)[1]:
        if not e["copy_flag"]:
            mine[e["arm"]][e["author"]] = e["position"]
    keep = (
        "voice_strict",
        "voice_loose",
        "voice80",
        "spec",
        "spec_share_positive",
        "copy_flagged",
        "words",
        "ai_flagged",
        "loop_share",
        "cap_hit",
        "usd_per_draft",
        "providers",
    )
    return dict(
        arm="edits, fixed template (3 samples + 3 simulated edit pairs)",
        row={k: row[k] for k in keep},
        minus={
            f"fixed - {a}": paired(fixed, mine[a], 50 + i) for i, a in enumerate(("edits", "k3_rand", "k0", "note"))
        },
    )


def spend():
    rows = [json.loads(l) for l in (HERE / "data" / "spend.jsonl").read_text().splitlines() if l.strip()]
    by = defaultdict(float)
    for r in rows:
        stage = (
            "pilot"
            if " dev " in r["note"]
            else (
                "reasoning low"
                if "reason-low" in r["note"]
                else ("edits rerun" if "edits-fixed" in r["note"] else "held-out")
            )
        )
        if r["provider"] == "modal":
            stage = "honesty extractor"
        by[f"{r['provider']}|{stage}"] += r["usd"]
    return dict(
        openrouter_total=round(sum(r["usd"] for r in rows if r["provider"] == "openrouter"), 4),
        modal_total=round(sum(r["usd"] for r in rows if r["provider"] == "modal"), 4),
        cap_openrouter=3.0,
        by_stage={k: round(v, 4) for k, v in sorted(by.items())},
        openrouter_calls=sum(r["provider"] == "openrouter" for r in rows),
    )


def main():
    RES.mkdir(exist_ok=True)
    ink = {}
    for m in (INK, INK_LOW):
        if (SCORED / f"{m}__heldout.json").exists():
            ink[m] = ink_rows(m, *load_ink(m))
    drafts_main, eps_main, _ = load_ink(INK)
    b = board(ink)
    con = contrasts(eps_main, ep1_positions())
    sp = spend()
    b["brief_reuse"] = brief_reuse()
    b["news_claimed"] = news_claimed()
    b["honesty"] = honesty_context()
    P = predictions(ink[INK], b, con, drafts_main)
    mc = defaultdict(list)
    mp = SCORED / "matched_cut.json"
    for k, v in (json.loads(mp.read_text()) if mp.exists() else {}).items():
        m, _, arm = k.split("|")
        mc[(m, arm)].append(v)
    b["matched_cut_best"] = {
        m: round(max(float(np.mean(v)) for (mm, _), v in mc.items() if mm == m), 4)
        for m in EP1_MODELS
        if any(mm == m for mm, _ in mc)
    }
    b["matched_cut_best"][INK] = b["contestants"][INK]["best"][0]
    b["best_unfiltered"] = unfiltered_best()
    b["parity"] = parity()
    gap = max(b["parity"]["largest_gap"].values())
    if b["parity"]["episodes"] != 180 or not gap < 1e-3:  # NaN fails too
        raise SystemExit(f"scorer check failed: {b['parity']}")
    er = edits_rerun()
    if er:
        (RES / "edits_rerun.json").write_text(json.dumps(er, indent=1) + "\n")
        print("edits rerun", json.dumps(er["row"]["voice_strict"]), json.dumps(er["minus"]))
    for name, obj in (("inkling", ink), ("board", b), ("contrasts", con), ("predictions", P), ("spend", sp)):
        (RES / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n")
    for m in b["ranked"]:
        c = b["contestants"][m]
        print(
            f"{c['rank']}. {c['label']:24s} best {c['best_arm']:8s} {c['best']}  best80 {c['best80']} spec {c['best_spec']} addressed {c['best_addressed']}"
        )
    for r in ink[INK] + ink.get(INK_LOW, []):
        print(
            f"{r['model'][-10:]:10s} {r['arm']:9s} strict {r['voice_strict']} 80w {r['voice80'][0]} "
            f"spec {r['spec'][0]} words {r['words']} ai {r['ai_flagged']} uns {r['unsupported']} "
            f"loop {r['loop_share']} cap {r['cap_hit']} copy {r['copy_flagged']} prov {r['providers']}"
        )
    print(json.dumps(con["within"]))
    print(json.dumps(con["reasoning_low"]))
    print(json.dumps(P, indent=1))
    print(json.dumps(sp, indent=1))


if __name__ == "__main__":
    main()
