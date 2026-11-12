"""FinanceMath v2 robustness (PREREG 2026-11-12): the April 2026 V4-Flash checkpoint on the full test, and Claude Opus
5.5 (reasoning effort high, one call) on a fixed random subset. Writes results/v4_robust.json.

    python v4_robust.py
"""

import json

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
    return "beats" if d[1] > 0 else ("below" if d[2] < 0 else "ties")


def main() -> None:
    items = {i.id: i for i in fm.load("test")}
    g54 = rows("test__gpt-5.4")
    ours = fm_v4.load("test", "or-dsflash-think", "reason")
    vote = {i: float(fm_v4.vote_correct(ours[i][:4], items[i])) for i in items}
    mean = {i: fin.mean(s["correct"] for s in ours[i][:4]) for i in items}
    gpt = {i: float(g54[i]["correct"]) for i in items}
    out: dict = {}

    apr = fm_v4.load("test", "or-dsflash0424-think", "reason")
    ids = sorted(i for i in items if len(apr.get(i, [])) >= 4)
    if ids:
        av = [float(fm_v4.vote_correct(apr[i][:4], items[i])) for i in ids]
        am = [fin.mean(s["correct"] for s in apr[i][:4]) for i in ids]
        samples = [s for i in ids for s in apr[i][:4]]
        provs: dict = {}
        for s in samples:
            provs[s["provider"]] = provs.get(s["provider"], 0) + 1
        out["april"] = {
            "n": len(ids),
            "vote4": fin.bootstrap(av),
            "mean": fin.bootstrap(am),
            "vote4_vs_0731_vote4": fin.paired_bootstrap([vote[i] for i in ids], av),
            "mean_vs_0731_mean": fin.paired_bootstrap([mean[i] for i in ids], am),
            "vote4_vs_gpt-5.4": fin.paired_bootstrap([gpt[i] for i in ids], av),
            "mean_vs_gpt-5.4": fin.paired_bootstrap([gpt[i] for i in ids], am),
            "tokens_per_sample": fin.mean(s["tokens_out"] for s in samples),
            "cap_rate": fin.mean(float(s["capped"]) for s in samples),
            "providers": dict(sorted(provs.items(), key=lambda x: -x[1])),
        }
        a = out["april"]
        a["verdict_vs_gpt-5.4"] = verdict(a["vote4_vs_gpt-5.4"])
        print(f"April (n={a['n']}): vote4 {fin.fmt_ci(a['vote4'])}, mean {fin.fmt_ci(a['mean'])}")
        print(f"  vs 0731 vote4 {f(a['vote4_vs_0731_vote4'])}; mean vs 0731 mean {f(a['mean_vs_0731_mean'])}")
        print(
            f"  vs gpt-5.4: vote4 {f(a['vote4_vs_gpt-5.4'])} ({a['verdict_vs_gpt-5.4']}); mean {f(a['mean_vs_gpt-5.4'])}"
        )
        print(f"  tokens {a['tokens_per_sample']:.0f}, capped {a['cap_rate']:.3f}, providers {a['providers']}")

    opus = fm_v4.load("test", "or-opus55-high", "reason")
    sub = [i for i in json.loads(fm_v4.SUBSET_FILE.read_text())["ids"] if opus.get(i)]
    if sub:
        op = [float(opus[i][0]["correct"]) for i in sub]
        o = {
            "n": len(sub),
            "opus": fin.bootstrap(op),
            "ours_vote4": fin.bootstrap([vote[i] for i in sub]),
            "ours_mean": fin.bootstrap([mean[i] for i in sub]),
            "gpt-5.4": fin.bootstrap([gpt[i] for i in sub]),
            "vote4_minus_opus": fin.paired_bootstrap(op, [vote[i] for i in sub]),
            "mean_minus_opus": fin.paired_bootstrap(op, [mean[i] for i in sub]),
            "gpt-5.4_minus_opus": fin.paired_bootstrap(op, [gpt[i] for i in sub]),
            "tokens": fin.mean(opus[i][0]["tokens_out"] for i in sub),
            "capped": sum(opus[i][0]["capped"] for i in sub),
        }
        for k in ("vote4_minus_opus", "mean_minus_opus", "gpt-5.4_minus_opus"):
            o[k.replace("_minus_opus", "_verdict")] = verdict(o[k])
        out["opus"] = o
        print(f"Opus 5.5 high (n={o['n']}): {fin.fmt_ci(o['opus'])}; ours vote4 {fin.fmt_ci(o['ours_vote4'])}, "
              f"mean {fin.fmt_ci(o['ours_mean'])}, gpt-5.4 {fin.fmt_ci(o['gpt-5.4'])}")  # fmt: skip
        print(f"  ours vote4 - Opus {f(o['vote4_minus_opus'])} ({o['vote4_verdict']}); "
              f"mean - Opus {f(o['mean_minus_opus'])} ({o['mean_verdict']}); "
              f"gpt-5.4 - Opus {f(o['gpt-5.4_minus_opus'])} ({o['gpt-5.4_verdict']})")  # fmt: skip
        print(f"  tokens {o['tokens']:.0f}, capped {o['capped']}")
    led = [r for r in map(json.loads, fin.LEDGER.read_text().splitlines()) if r.get("article") == "finmath5or"]
    out["usd_list_price"] = {
        t: round(sum(r["usd"] for r in led if r["tag"].startswith(t)), 3) for t in ("v4-or-dsflash0424", "v4-or-opus55")
    }
    print("list-price $", out["usd_list_price"])
    (fm.RESULTS / "v4_robust.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
