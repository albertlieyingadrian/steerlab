"""Score the 32B arms on validation (PREREG 2026-11-13, amendment 1): every GRPO checkpoint against untrained Qwen3-32B
thinking, paired (bootstrap interval and two-sided sign test on per-question vote wins/losses); selection by vote of 4.

    python experiments/fin-math/q32_select.py
"""

import collections
import json
import math

import fm  # sets up the fin-llm import path first

# isort: split
import fin

import fm_v3
import fm_v4

ITEMS = {i.id: i for i in fm.load("validation")}
IDS = sorted(ITEMS)
ARMS = ["q32-base-think", "q32-sft-e2-think"] + [f"q32-grpo-s{s}-think" for s in (10, 20, 30, 40)]


def load(tag):
    d = collections.defaultdict(list)
    path = fm.RESULTS / "runs" / f"validation__l1-{tag}.jsonl"
    if not path.exists():
        return None
    for r in map(json.loads, path.read_text().splitlines()):
        d[r["id"]].append(r)
    return d if all(len(d[i]) == 4 for i in IDS) else None


def vote(ss, i):
    vals = [fm.parse(s["pred"]) if s["pred"] else None for s in ss]
    v = fm_v3.cluster_vote([x for x in vals if x is not None])
    return float(v is not None and fm.grade(repr(v), ITEMS[i].gold, ITEMS[i].gold_exec))


def sign_test(a, b):
    w = sum(y > x for x, y in zip(a, b, strict=True))
    lo = sum(y < x for x, y in zip(a, b, strict=True))
    n = w + lo
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, lo) + 1)) / 2**n) if n else 1.0
    return w, lo, round(p, 4)


def f(d):
    return f"{d[0] * 100:+.1f} [{d[1] * 100:+.1f}, {d[2] * 100:+.1f}]"


def main() -> None:
    base = load("q32-base-think")
    bv = [vote(base[i], i) for i in IDS]
    g = {
        json.loads(line)["id"]: json.loads(line) for line in (fm.RESULTS / "runs" / "validation__gpt-5.4.jsonl").open()
    }
    gv = [float(g[i]["correct"]) for i in IDS]
    ds = fm_v4.load("validation", "or-dsflash-think", "reason")
    dv = [float(fm_v4.vote_correct(ds[i][:4], ITEMS[i])) for i in IDS]
    out, best = {}, None
    for tag in ARMS:
        d = load(tag)
        if d is None:
            print(tag, "incomplete")
            continue
        v = [vote(d[i], i) for i in IDS]
        m = [fin.mean(s["correct"] for s in d[i]) for i in IDS]
        ss = [s for i in IDS for s in d[i]]
        rec = {"vote4": fin.bootstrap(v), "mean": fin.bootstrap(m), "capped": sum(s["capped"] for s in ss),
               "tokens": round(fin.mean(s["tokens_out"] for s in ss)), "vote_minus_base": fin.paired_bootstrap(bv, v),
               "sign_vs_base": sign_test(bv, v), "vote_minus_gpt54": fin.paired_bootstrap(gv, v),
               "sign_vs_gpt54": sign_test(gv, v), "vote_minus_deepseek": fin.paired_bootstrap(dv, v)}  # fmt: skip
        out[tag] = rec
        if tag.startswith("q32-grpo") and (best is None or rec["vote4"][0] > out[best]["vote4"][0]):
            best = tag
        print(f"{tag:<20} vote {fin.fmt_ci(rec['vote4'])} mean {rec['mean'][0]:.3f} capped {rec['capped']:>3} "
              f"tok {rec['tokens']:>5} | vs base {f(rec['vote_minus_base'])} {rec['sign_vs_base']} | "
              f"vs gpt-5.4 {f(rec['vote_minus_gpt54'])} {rec['sign_vs_gpt54']}")  # fmt: skip
    out["selected"] = best
    print("selected (highest validation vote of 4, ties to earlier):", best)
    (fm.RESULTS / "q32_select.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
