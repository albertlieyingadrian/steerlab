"""The v3 test look against the bar (PREREG 2026-11-11): the chosen open system's single final answer per question
(vote of 4 reasoning answers + 4 programs), paired over the 1,000 test questions against gpt-5.4 and gpt-5.4-mini (one
call each, earlier runs) and the 8B result (SFT+GRPO s10: mean per-sample and majority of 4).

    python v3_test_bar.py deepseek-v4-flash      # writes results/v3_test.json
"""

import json
import sys

import fm  # sets up the fin-llm import path first

# isort: split
import fin

R = fm.RESULTS / "runs"


def rows(name):
    return {r["id"]: r for r in map(json.loads, (R / f"{name}.jsonl").read_text().splitlines())}


def f(d):
    return f"{d[0] * 100:+.1f} [{d[1] * 100:+.1f}, {d[2] * 100:+.1f}]"


def main(model: str) -> None:
    pot, cot = rows(f"test__v3pot-{model}"), rows(f"test__v3-{model}")
    ids = sorted(set(pot) & set(cot))
    ours = [float(pot[i]["vote8_correct"]) for i in ids]
    bars = {"gpt-5.4": rows("test__gpt-5.4"), "gpt-5.4-mini": rows("test__gpt-5.4-mini")}
    eight = rows("test__p4-grpo-s10")
    out = {
        "model": model,
        "n": len(ids),
        "vote8": fin.bootstrap(ours),
        "cot_mean": fin.bootstrap([cot[i]["correct"] for i in ids]),
        "cot_majority": fin.bootstrap([float(cot[i]["majority_correct"]) for i in ids]),
        "pot_majority": fin.bootstrap([float(pot[i]["pot_majority_correct"]) for i in ids]),
    }
    for b, br in bars.items():
        out[f"vote8_vs_{b}"] = fin.paired_bootstrap([float(br[i]["correct"]) for i in ids], ours)
        out[b] = fin.bootstrap([float(br[i]["correct"]) for i in ids])
    out["vote8_vs_8b_mean"] = fin.paired_bootstrap([eight[i]["correct"] for i in ids], ours)
    out["vote8_vs_8b_majority"] = fin.paired_bootstrap([float(eight[i]["majority_correct"]) for i in ids], ours)
    out["calls_per_question"] = 8
    out["usd_note"] = "see ledger tags v3-screen-<model> and v3-pot-<model> for the test-run rows"
    (fm.RESULTS / "v3_test.json").write_text(json.dumps(out, indent=1))
    print(
        f"{model} test n={len(ids)}: vote of 8 {fin.fmt_ci(out['vote8'])}; reasoning mean {out['cot_mean'][0]:.3f}, "
        f"majority {out['cot_majority'][0]:.3f}; programs majority {out['pot_majority'][0]:.3f}"
    )
    for b in bars:
        print(f"  vs {b} ({out[b][0]:.3f}): {f(out[f'vote8_vs_{b}'])}")
    print(f"  vs 8B SFT+GRPO mean {f(out['vote8_vs_8b_mean'])}, majority {f(out['vote8_vs_8b_majority'])}")


if __name__ == "__main__":
    main(sys.argv[1])
