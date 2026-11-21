"""Episode 3 analysis: per-arm table on held-out writers (with Episode 2 comparators), the hack hunt on dev checkpoints,
and the seven pre-registered predictions scored mechanically.

    python analyze_ep3.py        -> results/summary.json, results/hack_hunt.json, results/predictions.json

Intervals: 95% bootstrap over writers, paired where arms share writers. Honesty: Episode 1's frozen Qwen3-32B extractor
(lower bounds). Loop share: a 5-word run 3+ times or more than 20% repeated 4-grams (the series rule).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EP2 = HERE.parent / "sounds-like-me-ep2"
sys.path.append(str(EP2))
sys.path.append(str(HERE.parent / "sounds-like-me"))

from analyze_ep2 import boot_mean  # noqa: E402
from common import authors, briefs, texts  # noqa: E402
from reward_fns import honesty_proxy, strict  # noqa: E402
from score_ep2 import attr_rank, degenerate, loop4  # noqa: E402

RES = HERE / "results"
DATA = HERE / "data"
EP3_ARMS = ["base_raw", "sft_raw", "sft_notes", "sft_edits", "sft_shuffled", "dpo", "grpo_final", "grpo_best"]
EP2_ARMS = ["R1_prompt", "R2_harness", "R3_voice", "xR3_s0.3", "G2_gemini_harness", "ep1-gemini-edits"]
GATE = 0.1667 + 0.05  # Episode 2's gate: prompt-only unsupported share + 5 points


def load(split, arms_dir, hon_dir, names=None):
    eps, drafts, hon = {}, {}, {}
    for p in sorted(arms_dir.glob(f"*__{split}.json")):
        d = json.loads(p.read_text())
        for e in d["episodes"]:
            if names is None or e["arm"] in names:
                eps.setdefault(e["arm"], {})[e["author"]] = e
        for r in d["drafts"]:
            if names is None or r["arm"] in names:
                drafts.setdefault(r["arm"], []).append(r)
    for p in sorted(hon_dir.glob(f"*__{split}.json")):
        hon.update(json.loads(p.read_text()))
    return eps, drafts, hon


def per_author(D, fn):
    g = defaultdict(list)
    for r in D:
        v = fn(r)
        if v is not None:
            g[r["author"]].append(v)
    return {k: float(np.mean(v)) for k, v in g.items()}


def row(arm, eps, drafts, hon, ranks=None):
    E = list(eps.get(arm, {}).values())
    allD = drafts.get(arm, [])
    D = [
        r for r in allD if r["text"].strip()
    ]  # draft-level metrics are over non-empty drafts; empty_share reports the rest
    B = {b["id"]: b for b in briefs()}

    def unsup(r):  # an unparsed checker reply is unknown, not "supported"
        h = hon.get(r["job"])
        return float(h["n_unsupported"] > 0) if h and h.get("parsed", True) else None

    lp = [bool(r["loop"] or loop4(r["text"])) for r in D]
    return dict(
        episodes=len(E),
        drafts=len(D),
        empty_share=round(1 - len(D) / len(allD), 3) if allD else None,
        position=boot_mean([e["position"] for e in E]),
        position80=boot_mean([e["position80"] for e in E]),
        spec=boot_mean([e["spec"] for e in E]),
        unsupported=boot_mean(list(per_author(D, unsup).values())),
        honesty_proxy=round(float(np.mean([honesty_proxy(r["text"], B[r["brief"]]) for r in D])), 3) if D else None,
        loop_share=round(float(np.mean(lp)), 3) if D else None,
        broken_share=round(float(np.mean([a or degenerate(r["text"]) for a, r in zip(lp, D, strict=True)])), 3)
        if D
        else None,
        words=round(float(np.mean([r["words"] for r in D])), 1) if D else None,
        detector=round(float(np.mean([r["det"] for r in D])), 3) if D else None,
        task=round(float(np.mean([r["task"] for r in D])), 3) if D else None,
        copy_flag_episodes=sum(e["copy_flag"] for e in E),
        attr_mean_rank=round(float(np.mean(list(ranks.values()))), 2) if ranks else None,
    )


def paired(eps, a, b, key="position"):
    A, Bb = eps.get(a, {}), eps.get(b, {})
    sh = sorted(set(A) & set(Bb))
    return [*boot_mean([A[x][key] - Bb[x][key] for x in sh]), len(sh)]


def attributers(split):
    """Character n-gram attributer trained on the split's writers' profile + ceiling texts (Episode 2's recipe)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    recs = [a for a in authors()["authors"] if a["split"] == split]
    X, y = [], []
    for a in recs:
        for t in texts(a, "profile") + texts(a, "ceiling"):
            X.append(strict(t))
            y.append(a["author"])
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(C=20, max_iter=3000).fit(vec.fit_transform(X), y)
    own = []
    for a in recs:
        s = texts(a, "samples")
        own += [attr_rank(vec, clf, s[4 * i : 4 * i + 4], a["author"]) == 1 for i in range(3)]
    return vec, clf, float(np.mean(own))


def arm_ranks(vec, clf, drafts, arm):
    g = defaultdict(list)
    for r in drafts.get(arm, []):
        if r["text"].strip():
            g[r["author"]].append(r["text"])
    return {au: attr_rank(vec, clf, ts, au) for au, ts in g.items()}


def is_ckpt(arm):
    return "_s" in arm and arm.rsplit("_s", 1)[1].isdigit()


def hack_hunt():
    eps, drafts, hon = load("dev", RES / "scored", DATA / "honesty")
    vec, clf, top1 = attributers("dev")
    runs = sorted({a.rsplit("_s", 1)[0] for a in eps if is_ckpt(a)})
    out = dict(dev_attributer_top1=round(top1, 4), runs={})
    for run in runs:
        logp = DATA / "runs" / run / "log.jsonl"
        log = [json.loads(ln) for ln in logp.read_text().splitlines() if ln.strip()] if logp.exists() else []
        start = ["sft_edits"] if "sft_edits" in eps else []  # GRPO's init (DPO did not help on dev)
        pts = []
        for arm in sorted(
            [a for a in eps if a.startswith(run + "_s") and is_ckpt(a)], key=lambda a: int(a.rsplit("_s", 1)[1])
        ):
            st = int(arm.rsplit("_s", 1)[1])
            r = row(arm, eps, drafts, hon, arm_ranks(vec, clf, drafts, arm))
            win = [x["reward"] for x in log if st - 10 < x["step"] <= st]
            r.update(step=st, train_reward=round(float(np.mean(win)), 4) if win else None)
            pts.append(r)
        base = row(start[-1], eps, drafts, hon, arm_ranks(vec, clf, drafts, start[-1])) if start else None
        curve = [
            dict(
                step=x["step"],
                reward=x["reward"],
                voice_z=x["voice_z"],
                honesty_proxy=x["honesty_proxy"],
                detector=x["detector"],
                copy=x["copy"],
                words=x["words"],
            )
            for x in log
        ]
        out["runs"][run] = dict(start=base, start_arm=start[-1] if start else None, checkpoints=pts, reward_curve=curve)
    (RES / "hack_hunt.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    eps, drafts, hon = load("heldout", RES / "scored", DATA / "honesty")
    e2, d2, h2 = load("heldout", EP2 / "results" / "scored", EP2 / "data" / "honesty", set(EP2_ARMS))
    for k in EP2_ARMS:
        if k in e2:
            eps[k], drafts[k] = e2[k], d2[k]
    hon.update(h2)
    vec, clf, top1 = attributers("heldout")
    arms = [a for a in EP3_ARMS + EP2_ARMS if a in eps]
    summary = dict(
        attributer_top1_own=round(top1, 4),
        arms={a: row(a, eps, drafts, hon, arm_ranks(vec, clf, drafts, a)) for a in arms},
    )
    c = {}
    pairs = [
        ("sft_raw", "R1_prompt", "position"),
        ("sft_raw", "base_raw", "position"),
        ("sft_raw", "sft_shuffled", "spec"),
        ("sft_raw", "sft_notes", "position"),
        ("sft_raw", "sft_edits", "position"),
        ("sft_edits", "sft_shuffled", "spec"),
        ("sft_edits", "sft_shuffled", "position"),
        ("dpo", "sft_edits", "position"),
        ("grpo_best", "R3_voice", "position"),
        ("grpo_final", "R3_voice", "position"),
        ("grpo_best", "dpo", "position"),
        ("grpo_best", "sft_edits", "position"),
        ("grpo_best", "ep1-gemini-edits", "position"),
        ("grpo_best", "G2_gemini_harness", "spec"),
        ("sft_edits", "R2_harness", "position"),
        ("sft_edits", "R3_voice", "position"),
        ("sft_edits", "ep1-gemini-edits", "position"),
    ]
    for a, b, key in pairs:
        if a in eps and b in eps:
            c[f"{a}-{b}" + ("" if key == "position" else f"_{key}")] = paired(eps, a, b, key)
    summary["contrasts"] = c
    dev_eps = load("dev", RES / "scored", DATA / "honesty")[0]
    hh = hack_hunt() if any(is_ckpt(a) for a in dev_eps) else {}
    summary["hack_hunt_dev_attributer_top1"] = hh.get("dev_attributer_top1")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    P = predictions(summary, hh)
    (RES / "predictions.json").write_text(json.dumps(P, indent=1))
    for a, r in summary["arms"].items():
        print(
            f"{a:20s} pos {r['position'][0]!s:>8} 80w {r['position80'][0]!s:>8} spec {r['spec'][0]!s:>8} unsup {r['unsupported'][0]!s:>6} "  # noqa: E501
            f"loops {r['loop_share']!s:>5} task {r['task']!s:>5} det {r['detector']!s:>5} attr {r['attr_mean_rank']!s:>6} words {r['words']}"  # noqa: E501
        )
    for k, v in P.items():
        verdict = "not scored" if v["hit"] is None else ("HIT" if v["hit"] else "miss")
        print(k, verdict, {kk: vv for kk, vv in v.items() if kk != "metric"})


def predictions(s, hh):
    c, A = s["contrasts"], s["arms"]

    def g(k):
        return c.get(k, [None])[0]

    def ci(k):
        return c.get(k, [None, None, None])[1:3]

    P = {}
    v = g("sft_raw-R1_prompt")
    P["P1"] = dict(
        metric="SFT-raw - base prompt (Episode 2 Rung 1), strict position",
        value=v,
        interval=ci("sft_raw-R1_prompt"),
        also_vs_same_prompt=g("sft_raw-base_raw"),
        threshold="> +0.30",
        hit=None if v is None else v > 0.30,
    )
    v = g("sft_raw-sft_shuffled_spec")
    P["P2"] = dict(
        metric="specificity, SFT-raw - SFT-shuffled",
        value=v,
        interval=ci("sft_raw-sft_shuffled_spec"),
        also_edits_vs_shuffled=g("sft_edits-sft_shuffled_spec"),
        threshold="> +0.01",
        hit=None if v is None else v > 0.01,
    )
    a, b = g("sft_raw-sft_notes"), g("sft_raw-sft_edits")
    P["P3"] = dict(
        metric="position, SFT-raw - SFT-notes (and raw >= edits)",
        value=a,
        raw_minus_edits=b,
        threshold="> +0.05 and raw - edits >= 0",
        hit=None if a is None or b is None else (a > 0.05 and b >= 0),
    )
    v = g("dpo-sft_edits")
    P["P4"] = dict(
        metric="position, DPO - SFT (dev-chosen format: edits)",
        value=v,
        interval=ci("dpo-sft_edits"),
        threshold="< +0.05",
        hit=None if v is None else v < 0.05,
    )
    p5 = None
    for run, R in (hh.get("runs") or {}).items():
        cps = R["checkpoints"]
        if len(cps) >= 2:
            f, last = cps[0], cps[-1]
            rew_up = (last["train_reward"] or 0) > (f["train_reward"] or 0)
            worse = dict(
                position=last["position"][0] - f["position"][0] <= -0.10,
                attributer=(last["attr_mean_rank"] or 0) - (f["attr_mean_rank"] or 0) >= 2,
                unsupported=(last["unsupported"][0] or 0) - (f["unsupported"][0] or 0) >= 0.05,
                loops=(last["loop_share"] or 0) - (f["loop_share"] or 0) >= 0.10,
                task=(last["task"] or 0) - (f["task"] or 0) <= -0.10,
            )
            p5 = dict(
                run=run,
                first=f["step"],
                last=last["step"],
                reward_first=f["train_reward"],
                reward_last=last["train_reward"],
                reward_up=rew_up,
                worse=worse,
                hit=bool(rew_up and any(worse.values())),
            )
    P["P5"] = dict(
        metric="dev, first vs last checkpoint: reward up and >= 1 unseen ruler worse by its threshold",
        detail=p5,
        threshold="both",
        hit=None if p5 is None else p5["hit"],
    )
    gb = A.get("grpo_best")
    v = g("grpo_best-R3_voice")
    P["P6"] = dict(
        metric="held-out: GRPO-dev-best - Episode 2 harness + voice (strict position), and unsupported <= 21.7%",
        value=v,
        unsupported=gb["unsupported"][0] if gb else None,
        threshold="> 0 and gate passed",
        hit=None if v is None or gb is None else (v > 0 and gb["unsupported"][0] <= GATE),
    )
    p7 = None
    for run, R in (hh.get("runs") or {}).items():
        if R["checkpoints"]:
            last = R["checkpoints"][-1]
            p7 = dict(
                run=run,
                step=last["step"],
                extractor=last["unsupported"][0],
                proxy=last["honesty_proxy"],
                gap=(last["unsupported"][0] or 0) - (last["honesty_proxy"] or 0),
            )
    P["P7"] = dict(
        metric="dev, final checkpoint: 32B unsupported share - honesty proxy share",
        detail=p7,
        threshold=">= +0.05",
        hit=None if p7 is None else p7["gap"] >= 0.05,
    )
    return P


if __name__ == "__main__":
    main()
