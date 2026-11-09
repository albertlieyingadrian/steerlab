"""
Step 4: every number the note quotes, from results/gen_*, scored_* and truncate_*. Writes results/summary.json.

Two groups are kept apart throughout. A reasoning *finished* when Qwen closed its thinking and wrote "Answer: (X)"
within the 1,536-token budget; otherwise it was *cut off* and its answer forced (step 1). A cut-off reasoning stops
mid-thought, so whether it "acknowledges" the hint says little; the headline rates use finished reasonings.

    .venv/bin/python experiments/cot-faithfulness/4_summarise.py
"""

import json
import math

from cotf import MODELS, RESULTS, read_jsonl


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci": wilson(k, n)}


def mcnemar_p(b, c):
    """Exact two-sided McNemar test: b and c are the two kinds of discordant pairs."""
    n, k = b + c, min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def summarise(name):
    gen = read_jsonl(RESULTS / f"gen_{name}.jsonl")
    scored = read_jsonl(RESULTS / f"scored_{name}.jsonl")
    trunc = read_jsonl(RESULTS / f"truncate_{name}.jsonl")
    out = {"conditions": {}, "hinted": {}, "truncation": {}}
    for cond in ("no_hint", "hint", "hint_monitored"):
        rows = [g for g in gen if g["condition"] == cond]
        n = len(rows)
        out["conditions"][cond] = {
            "n": n,
            "accuracy": rate(sum(g["answer"] == g["correct"] for g in rows), n),
            "picked_hinted_option": rate(sum(g["answer"] == g["hint"] for g in rows), n),
            "cut_off": rate(sum(g["forced"] for g in rows), n),
            "mean_think_tokens": round(sum(g["think_tokens"] for g in rows) / n),
        }
    for cond in ("hint", "hint_monitored"):
        rows = [r for r in scored if r["condition"] == cond]
        used = [r for r in rows if r["hint_used"]]
        fin = [r for r in used if not r["forced"]]
        cut = [r for r in used if r["forced"]]
        matched = [r for r in rows if not r["hint_used"] and r["answer"] == r["hint"]]
        rejected_fin = [r for r in rows if r["answer"] != r["hint"] and not r["forced"]]
        out["hinted"][cond] = {
            "hint_used": rate(len(used), len(rows)),
            "finished": {"mentions": rate(sum(r["mentions"] for r in fin), len(fin)),
                         "acknowledges": rate(sum(r["acknowledged"] for r in fin), len(fin))},
            "cut_off": {"mentions": rate(sum(r["mentions"] for r in cut), len(cut)),
                        "acknowledges": rate(sum(r["acknowledged"] for r in cut), len(cut))},
            "all_used": {"mentions": rate(sum(r["mentions"] for r in used), len(used)),
                         "acknowledges": rate(sum(r["acknowledged"] for r in used), len(used))},
            "matched_anyway_acknowledges": rate(sum(r["acknowledged"] for r in matched), len(matched)),
            "rejected_finished_acknowledges": rate(sum(r["acknowledged"] for r in rejected_fin), len(rejected_fin)),
        }  # fmt: skip
    plain = {r["id"]: r["hint_used"] for r in scored if r["condition"] == "hint"}
    mon = {r["id"]: r["hint_used"] for r in scored if r["condition"] == "hint_monitored"}
    only_plain = sum(plain[i] and not mon[i] for i in plain)
    only_mon = sum(mon[i] and not plain[i] for i in plain)
    out["monitoring"] = {"both": sum(plain[i] and mon[i] for i in plain), "only_without_note": only_plain,
                         "only_with_note": only_mon, "mcnemar_p": mcnemar_p(only_plain, only_mon)}  # fmt: skip
    for cond in ("no_hint", "hint"):
        out["truncation"][cond] = {}
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows = [t for t in trunc if t["condition"] == cond and t["fraction"] == f]
            out["truncation"][cond][str(f)] = {
                "same_as_final": rate(sum(t["answer"] == t["final"] for t in rows), len(rows)),
                "hinted_option": rate(sum(t["answer"] == t["hint"] for t in rows), len(rows)),
            }
    return out


def main():
    summary = {name: summarise(name) for name in MODELS}
    summary["judge_check"] = json.loads((RESULTS / "judge_check.json").read_text())["summary"]
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    for name in MODELS:
        s = summary[name]
        c = s["conditions"]
        print(f"{name}: accuracy no hint {c['no_hint']['accuracy']['rate']}, hint {c['hint']['accuracy']['rate']}, "
              f"monitored {c['hint_monitored']['accuracy']['rate']}; cut off {c['no_hint']['cut_off']['k']}/"
              f"{c['hint']['cut_off']['k']}/{c['hint_monitored']['cut_off']['k']} of 300")  # fmt: skip
        for cond, h in s["hinted"].items():
            f = h["finished"]
            print(f"   {cond}: hint used {h['hint_used']['k']}/300; finished {f['mentions']['n']}: mention "
                  f"{f['mentions']['k']}, acknowledge {f['acknowledges']['k']}; cut off {h['cut_off']['mentions']['n']}: "
                  f"acknowledge {h['cut_off']['acknowledges']['k']}")  # fmt: skip
        m = s["monitoring"]
        print(f"   monitoring: both {m['both']}, only without note {m['only_without_note']}, only with note "
              f"{m['only_with_note']}, McNemar p {m['mcnemar_p']:.2g}")  # fmt: skip
        for cond, t in s["truncation"].items():
            print(f"   truncation {cond}: " + ", ".join(f"{k}: {v['same_as_final']['rate']}" for k, v in t.items()))


if __name__ == "__main__":
    main()
