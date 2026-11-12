"""TEST LOOK #2 (PREREG-v4): DeepSeek-V4-Flash thinking, vote of 4, on the 1,000 FinanceMath test questions, paired
against gpt-5.4 and gpt-5.4-mini (one call each, earlier runs), v3 (vote of 8) and the 8B result. Writes results/v4_test.json.

    python v4_test_bar.py
"""

import json
from collections import Counter

import fm  # sets up the fin-llm import path first

# isort: split
import fin

import fm_v4

R = fm.RESULTS / "runs"


def rows(name):
    return {r["id"]: r for r in map(json.loads, (R / f"{name}.jsonl").read_text().splitlines())}


def f(d):
    return f"{d[0] * 100:+.1f} [{d[1] * 100:+.1f}, {d[2] * 100:+.1f}]"


def verdict(d):
    return "beats" if d[1] > 0 else ("below" if d[2] < 0 else "reaches")


def main() -> None:
    items = {i.id: i for i in fm.load("test")}
    tr = fm_v4.load("test", "or-dsflash-think", "reason")
    ids = sorted(i for i in items if len(tr.get(i, [])) >= 4)
    vote = [float(fm_v4.vote_correct(tr[i][:4], items[i])) for i in ids]
    mean1 = [fin.mean(s["correct"] for s in tr[i][:4]) for i in ids]
    g54, mini = rows("test__gpt-5.4"), rows("test__gpt-5.4-mini")
    v3 = rows("test__v3pot-deepseek-v4-flash")
    eight = rows("test__p4-grpo-s10")
    out = {"n": len(ids), "vote4": fin.bootstrap(vote), "mean_per_sample": fin.bootstrap(mean1)}
    out["vote4_vs_gpt-5.4"] = fin.paired_bootstrap([float(g54[i]["correct"]) for i in ids], vote)
    out["verdict"] = verdict(out["vote4_vs_gpt-5.4"])
    out["mean_vs_gpt-5.4"] = fin.paired_bootstrap([float(g54[i]["correct"]) for i in ids], mean1)
    out["vote4_vs_gpt-5.4-mini"] = fin.paired_bootstrap([float(mini[i]["correct"]) for i in ids], vote)
    out["vote4_vs_v3_vote8"] = fin.paired_bootstrap([float(v3[i]["vote8_correct"]) for i in ids], vote)
    out["vote4_vs_8b_majority"] = fin.paired_bootstrap([float(eight[i]["majority_correct"]) for i in ids], vote)
    samples = [s for i in ids for s in tr[i][:4]]
    out["tokens_per_sample"] = fin.mean(s["tokens_out"] for s in samples)
    out["cap_rate"] = sum(s["capped"] for s in samples) / len(samples)
    out["providers"] = dict(Counter(s.get("provider") for s in samples).most_common())
    usd = sum(r["usd"] for r in map(json.loads, fin.LEDGER.read_text().splitlines())
              if r["tag"] == "v4-or-dsflash-think-reason-test")  # fmt: skip
    out["usd"] = round(usd, 3)
    out["usd_per_question"] = round(usd / len(ids), 5)
    (fm.RESULTS / "v4_test.json").write_text(json.dumps(out, indent=1))
    print(f"TEST LOOK #2 n={len(ids)}: vote of 4 {fin.fmt_ci(out['vote4'])} -> {out['verdict'].upper()} gpt-5.4")
    print(
        f"  vs gpt-5.4 (0.818) {f(out['vote4_vs_gpt-5.4'])}; one sample (mean) {out['mean_per_sample'][0]:.3f}, "
        f"vs gpt-5.4 {f(out['mean_vs_gpt-5.4'])}"
    )
    print(
        f"  vs gpt-5.4-mini {f(out['vote4_vs_gpt-5.4-mini'])}; vs v3 vote8 {f(out['vote4_vs_v3_vote8'])}; "
        f"vs 8B majority {f(out['vote4_vs_8b_majority'])}"
    )
    print(
        f"  tokens/sample {out['tokens_per_sample']:.0f}, capped {out['cap_rate']:.3f}, ${out['usd']} "
        f"(${out['usd_per_question']}/question), providers {out['providers']}"
    )


if __name__ == "__main__":
    main()
