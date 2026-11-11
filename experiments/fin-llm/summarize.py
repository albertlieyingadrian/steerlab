"""Every number in the pilot write-up, recomputed from results/runs/: accuracy with bootstrap intervals, paired
differences over the same questions, dollars and seconds per question, and spend from the ledger.

    summarize.py            # prints markdown tables and writes results/summary.json
"""

from __future__ import annotations

import json

import fin
from run import RUNS, summarize

PAIRS = [  # (split, baseline tag, system tag): differences are paired over the questions both ran
    ("devrest", "raw__qwen", "raw+ex__qwen"),
    ("devrest", "raw__qwen", "harness-noex__qwen"),
    ("devrest", "raw__qwen", "harness__qwen"),
    ("devrest", "raw__qwen", "raw+ex*5__qwen"),
    ("devrest", "raw__qwen", "harness*5__qwen"),
    ("devrest-300", "raw__gpt-5.4-nano", "harness__gpt-5.4-nano"),
    ("devrest-300", "raw__gpt-5.4-mini", "harness__gpt-5.4-mini"),
    ("test", "raw__qwen", "harness__qwen"),
    ("test", "raw__qwen", "raw+ex*5__qwen"),
    ("test", "raw__gpt-5.4-nano", "harness__gpt-5.4-nano"),
    ("test", "raw__gpt-5.4-mini", "harness__gpt-5.4-mini"),
    ("test", "raw__gpt-5.4-mini", "raw+ex*5__qwen"),
    ("test", "raw__gpt-5.4", "harness__gpt-5.4"),
    ("test-300", "raw__gpt-5.5", "harness__gpt-5.5"),
    ("test", "raw__qwen", "raw+ex__qwen"),
    ("test", "raw__qwen", "harness-noex__qwen"),
    ("test", "raw__gpt-5.4-mini", "raw+ex__gpt-5.4-mini"),
    # cross-model, same questions: a cheaper model with the harness against raw gpt-5.4
    ("test", "raw__gpt-5.4", "raw+ex__gpt-5.4-mini"),
    ("test", "raw__gpt-5.4", "harness__gpt-5.4-mini"),
    ("test", "raw__gpt-5.4", "harness__gpt-5.4-nano"),
    ("test", "raw__gpt-5.4", "harness__qwen"),
    ("test", "raw__gpt-5.4", "raw__gpt-5.5"),
    # Claude arms (update, 2026-11-11): Haiku on the full test, Opus on the fixed 300
    ("test", "raw__claude-haiku-5-5", "raw+ex__claude-haiku-5-5"),
    ("test", "raw__claude-haiku-5-5", "harness-noex__claude-haiku-5-5"),
    ("test", "raw__claude-haiku-5-5", "harness__claude-haiku-5-5"),
    ("test-300", "raw__claude-opus-5-5", "harness-noex__claude-opus-5-5"),
    ("test-300", "raw__claude-opus-5-5", "harness__claude-opus-5-5"),
    ("test", "raw__gpt-5.4", "harness__claude-haiku-5-5"),
    ("test", "raw__gpt-5.4-mini", "raw__claude-haiku-5-5"),
    ("test", "raw__gpt-5.4", "raw__claude-haiku-5-5"),
    ("test-300", "raw__gpt-5.5", "raw__claude-opus-5-5"),
    ("test-300", "harness__gpt-5.5", "harness__claude-opus-5-5"),
    ("guard-300", "raw__qwen", "raw+ex*5__qwen"),
    ("guard-300", "raw__gpt-5.4-mini", "harness__gpt-5.4-mini"),
]
POSTTRAIN_PAIRS = [  # base -> SFT -> GRPO, alone and in the harness (only where those runs exist)
    ("devrest-300", "raw__base", "raw__sft-e2"),
    ("devrest-300", "raw__sft-e2", "raw__grpo-s10"),
    ("devrest-300", "raw__base", "raw__grpo-s10"),
    ("devrest-300", "raw__sft-e2", "harness__sft-e2"),
    ("test", "raw__qwen", "raw__sft-e2"),
    ("test", "raw__qwen", "raw__grpo-s10"),
    ("guard-300", "raw__qwen", "raw__grpo-s10"),
]


def rows(tag: str) -> dict[str, dict]:
    path = RUNS / f"{tag}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def restricted(split: str, tag: str) -> dict[str, dict]:
    """A run's rows; `<devrest-300>__<mode>__base` is the base model's full devrest run cut to those questions."""
    if not tag.endswith("__base"):
        got = rows(f"{split}__{tag}")
        if got or split != "test-300":
            return got
        ids = fin.subset_ids("test:300")  # a full-test run cut to the gpt-5.4 subset
        return {k: v for k, v in rows(f"test__{tag}").items() if k in ids}
    ids = fin.subset_ids(split.replace("-", ":"))
    full = rows(f"devrest__{tag.replace('__base', '__qwen')}")
    return {k: v for k, v in full.items() if k in ids}


POSTTRAIN = (
    "sft",
    "grpo",
)  # post-training runs: summarised apart, in results/posttrain/summary.json


def paired(pairs: list[tuple[str, str, str]]) -> list[dict]:
    out = []
    print("\n| split | baseline | system | n | baseline acc | system acc | difference [95% CI] |")
    print("|---|---|---|---:|---:|---:|---|")
    for split, a, b in pairs:
        ra, rb = restricted(split, a), restricted(split, b)
        ids = sorted(set(ra) & set(rb))
        if len(ids) < 50:
            continue
        xa, xb = [ra[i]["correct"] for i in ids], [rb[i]["correct"] for i in ids]
        d = fin.paired_bootstrap(xa, xb)
        out.append({"split": split, "baseline": a, "system": b, "n": len(ids), "baseline_acc": fin.mean(xa),
                    "system_acc": fin.mean(xb), "diff": d})  # fmt: skip
        print(f"| {split} | {a} | {b} | {len(ids)} | {fin.mean(xa):.3f} | {fin.mean(xb):.3f} | {fin.fmt_ci(d)} |")
    return out


def sign_check() -> dict:
    """The boring explanation: how much of each gain is the harness only flipping a sign to FinQA's convention?

    For every question a system changed (raw wrong -> right, or right -> wrong), call it sign-only when the wrong
    answer equals the gold with its sign flipped; the net gain is then recounted without those questions."""
    out = {}
    print("\n| model | system | fixed (sign-only) | broke (sign-only) | net, points | net without sign-only, points |")
    print("|---|---|---:|---:|---:|---:|")
    for model in ("qwen", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"):
        for mode in ("raw+ex", "harness"):
            raw, got = rows(f"test__raw__{model}"), rows(f"test__{mode}__{model}")
            if not got:
                continue

            def sign_only(r: dict) -> bool:
                return not isinstance(r["gold"], str) and r["pred"] is not None and fin.grade(r["pred"], -r["gold"])

            fixed = [i for i in raw if not raw[i]["correct"] and got[i]["correct"]]
            broke = [i for i in raw if raw[i]["correct"] and not got[i]["correct"]]
            fs, bs = (
                sum(sign_only(raw[i]) for i in fixed),
                sum(sign_only(got[i]) for i in broke),
            )
            n = len(raw)
            net, net_value = (
                len(fixed) - len(broke),
                (len(fixed) - fs) - (len(broke) - bs),
            )
            out[f"{model}|{mode}"] = {"n": n, "fixed": len(fixed), "fixed_sign_only": fs, "broke": len(broke),
                                      "broke_sign_only": bs, "net_points": round(100 * net / n, 2),
                                      "net_points_without_sign_only": round(100 * net_value / n, 2)}  # fmt: skip
            print(f"| {model} | {mode} | {len(fixed)} ({fs}) | {len(broke)} ({bs}) | {100 * net / n:+.1f} | "
                  f"{100 * net_value / n:+.1f} |")  # fmt: skip
    return out


def main() -> None:
    every = sorted(RUNS.glob("*.jsonl"))
    lab = [p for p in every if any(k in p.stem for k in POSTTRAIN)]
    table = [summarize(p) for p in every if p not in lab]
    print("| run | n | accuracy [95% CI] | lenient | no answer | $/question | s/question |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for s in table:
        print(f"| {s['tag']} | {s['n']} | {fin.fmt_ci(s['acc'])} | {s['lenient']:.3f} | {s['no_answer']} | "
              f"{s['usd_per_q']:.5f} | {s['sec_per_q']:.1f} |")  # fmt: skip
    diffs = paired(PAIRS)
    subset = []  # every full-test system on the 300 test questions gpt-5.4 ran, paired against gpt-5.4 raw
    ref = restricted("test-300", "raw__gpt-5.4")
    print("\n| system (test:300 subset) | acc | vs gpt-5.4 raw [95% CI] | $/question |\n|---|---:|---|---:|")
    for path in [p for p in every if p.stem.startswith("test") and p not in lab]:
        if path.stem == "test-300__raw__gpt-5.4" or path.stem.startswith("test-300__harness__gpt-5.4"):
            continue  # included in the full-test gpt-5.4 runs
        got = rows(path.stem)
        ids = sorted(set(got) & set(ref))
        if len(ids) < len(ref) or not ref:
            continue
        x = [got[i]["correct"] for i in ids]
        d = fin.paired_bootstrap([ref[i]["correct"] for i in ids], x)
        usd = fin.mean(got[i]["usd"] for i in ids)
        subset.append(
            {
                "system": path.stem,
                "n": len(ids),
                "acc": fin.mean(x),
                "vs_gpt54_raw": d,
                "usd_per_q": usd,
            }
        )
        print(f"| {path.stem} | {fin.mean(x):.3f} | {fin.fmt_ci(d)} | {usd:.5f} |")
    if fin.LEDGER.exists():  # the per-call spend ledger; without it, keep the spend already in summary.json
        by_provider: dict[str, float] = {}
        for r in map(json.loads, fin.LEDGER.read_text().splitlines()):
            if r["article"] == "harness":
                by_provider[r["provider"]] = by_provider.get(r["provider"], 0.0) + r["usd"]
        spend = {p: round(u, 2) for p, u in sorted(by_provider.items())}
    else:
        spend = json.loads((fin.RESULTS / "summary.json").read_text())["spend_usd"]
    print("\n| provider | metered $ (harness note) |\n|---|---:|")
    for prov, usd in spend.items():
        print(f"| {prov} | {usd:.2f} |")
    out = {"runs": table, "paired": diffs, "test300_subset": subset, "sign_check": sign_check(),
           "spend_usd": spend}  # fmt: skip
    (fin.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    if lab:
        lab_out = {
            "runs": [summarize(p) for p in lab],
            "paired": paired(POSTTRAIN_PAIRS),
        }
        (fin.RESULTS / "posttrain").mkdir(exist_ok=True)
        (fin.RESULTS / "posttrain" / "summary.json").write_text(json.dumps(lab_out, indent=1))


if __name__ == "__main__":
    main()
