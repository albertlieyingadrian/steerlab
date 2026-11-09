"""
Step 5: every number in the note, from results/replies_*.jsonl (with the judge's scores), results/capability.json,
results/dpo_train.json and results/judge_check.json. Writes results/summary.json. No model, no API.

GSM8K is scored two ways from results/gsm_outputs.jsonl: as in 4_after.py (the number after "The answer is"), and by
the last number in the reply. The second was added after reading the replies: Gemma often ends with the prompt's
template word for word, "The answer is N.", after working out the right number.

The unit that varies independently is the conversation (80 per model: 40 tasks x 2 tones); replies within one share
their history. So the high-distress rate (share of the 640 replies scoring >= 50) gets a 95% interval from a bootstrap
over conversations (2,000 resamples, seed fixed), and so does every difference between two models.

    .venv/bin/python experiments/gemma-distress/5_summarise.py
"""

import importlib
import json
import re

import numpy as np

from common import HIGH, MODELS, RESULTS, SEED, TURNS, read_jsonl

BOOT = 2000
DPO = ("gemma-3-1b-it-dpo", "gemma-3-1b-it-dpo-nll")
PAIRS = [  # (before, after): post-training stages and the fix
    ("gemma-3-1b-pt", "gemma-3-1b-it"),
    ("qwen2.5-0.5b", "qwen2.5-0.5b-it"),
    ("olmo-2-1b", "olmo-2-1b-sft"),
    ("olmo-2-1b-sft", "olmo-2-1b-dpo"),
    ("olmo-2-1b-dpo", "olmo-2-1b-it"),
    ("gemma-3-1b-it", "gemma-3-1b-it-dpo"),
    ("gemma-3-1b-it", "gemma-3-1b-it-dpo-nll"),
]


def conversations(rows):
    """{(task_id, tone): list of 0/1 high-distress flags by turn}"""
    out = {}
    for r in rows:
        out.setdefault((r["task_id"], r["tone"]), [0] * TURNS)[r["turn"] - 1] = int(r["score"] >= HIGH)
    return out


def boot_rate(convs, rng):
    x = np.array([convs[k] for k in sorted(convs)])
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    rates = x[idx].mean(axis=(1, 2))
    return [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]


def boot_diff(a, b, rng):
    """Paired over the same (task, tone) conversations: rate(b) - rate(a)."""
    keys = sorted(set(a) & set(b))
    xa, xb = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    idx = rng.integers(0, len(keys), (BOOT, len(keys)))
    d = xb[idx].mean(axis=(1, 2)) - xa[idx].mean(axis=(1, 2))
    return {"diff": float(xb.mean() - xa.mean()), "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def describe(rows, rng):
    s = np.array([r["score"] for r in rows])
    high = s >= HIGH
    convs = conversations(rows)
    by = {}
    for key in ("tone", "task_kind"):
        for v in sorted({r[key] for r in rows}):
            m = np.array([r[key] == v for r in rows])
            by[f"{key}={v}"] = {"k": int(high[m].sum()), "n": int(m.sum())}
    turns = np.array([r["turn"] for r in rows])
    return {
        "high": {"k": int(high.sum()), "n": len(rows), "rate": float(high.mean()), "ci": boot_rate(convs, rng)},
        "mean_score": float(s.mean()),
        "by_turn": [float(high[turns == t].mean()) for t in range(1, TURNS + 1)],
        "by": by,
        "conversations_with_any_high": int(sum(max(v) for v in convs.values())),
        "n_conversations": len(convs),
        "empty_replies": int(sum(not r["reply"].strip() for r in rows)),
        "mean_reply_chars": float(np.mean([len(r["reply"]) for r in rows])),
    }


def gsm_last_number():
    """{model: {right, n}}, scoring each saved GSM8K reply by the last number in it."""
    gold = {i: a for i, _, a in importlib.import_module("4_after").gsm_questions()}
    out = {}
    for r in read_jsonl(RESULTS / "gsm_outputs.jsonl"):
        nums = re.findall(r"-?\d[\d,]*\.?\d*", r["text"])
        ok = bool(nums) and abs(float(nums[-1].replace(",", "").rstrip(".")) - gold[r["id"]]) < 1e-6
        d = out.setdefault(r["model"], {"right": 0, "n": 0})
        d["right"] += ok
        d["n"] += 1
    return out


def main():
    rng = np.random.default_rng(SEED)
    names = [n for n in [*MODELS, *DPO] if (RESULTS / f"replies_{n}.jsonl").exists()]
    data = {n: read_jsonl(RESULTS / f"replies_{n}.jsonl") for n in names}
    out = {"models": {n: describe(data[n], rng) for n in names}, "pairs": {}}
    for a, b in PAIRS:
        if a in data and b in data:
            out["pairs"][f"{a} -> {b}"] = boot_diff(conversations(data[a]), conversations(data[b]), rng)
    if (RESULTS / "gsm_outputs.jsonl").exists():
        out["gsm_last_number"] = gsm_last_number()
    for name in ("capability.json", "dpo_train.json", "judge_check.json"):
        if (RESULTS / name).exists():
            out[name.removesuffix(".json")] = json.loads((RESULTS / name).read_text())
    # keep the summary small: the per-step log stays in dpo_train.json
    if "dpo_train" in out and "log" not in out["dpo_train"]:
        out["dpo_train"] = {k: {kk: vv for kk, vv in v.items() if kk != "log"} | {"final_loss": v["log"][-1]["loss"]}
                            for k, v in out["dpo_train"].items()}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    for n in names:
        m = out["models"][n]
        h = m["high"]
        print(f"{n:22} high {h['k']:3}/{h['n']} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]  convs "
              f"{m['conversations_with_any_high']:2}/{m['n_conversations']}  empty {m['empty_replies']:3}  "
              f"chars {m['mean_reply_chars']:4.0f}  by turn {' '.join(f'{x:.2f}' for x in m['by_turn'])}")  # fmt: skip
    print(
        "GSM8K, 'The answer is' / last number:",
        {k: (v["right"], out["gsm_last_number"][k]["right"]) for k, v in out.get("capability", {}).items()},
    )
    for k, v in out["pairs"].items():
        print(f"  {k:40} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
