import json
import sys

sys.path.insert(0, ".")
import fin

import fm

R = fm.RESULTS / "runs"


def p4(tag):
    return {r["id"]: r for r in map(json.loads, (R / f"test__p4-{tag}.jsonl").read_text().splitlines())}


def single(tag):
    return {r["id"]: r for r in map(json.loads, (R / f"test__{tag}.jsonl").read_text().splitlines())}


ours = {t: p4(t) for t in ("base", "sft-e2", "grpo-s10")}
bar = {"gpt-5.4-mini": single("gpt-5.4-mini"), "gpt-5.4": single("gpt-5.4"), "qwen-greedy-tinker": single("qwen")}
out = {}
for t, rows in ours.items():
    ids = sorted(rows)
    acc = [rows[i]["correct"] for i in ids]
    caps = [s["capped"] for i in ids for s in rows[i]["samples"]]
    o = {
        "mean_pass1": fin.bootstrap(acc),
        "majority": fin.bootstrap([rows[i]["majority_correct"] for i in ids]),
        "loop_rate": sum(caps) / len(caps),
        "tokens": fin.mean(s["tokens_out"] for i in ids for s in rows[i]["samples"]),
    }
    if t != "base":
        o["vs_base"] = fin.paired_bootstrap([ours["base"][i]["correct"] for i in ids], acc)
    for b, brow in bar.items():
        both = [i for i in ids if i in brow]
        o[f"vs_{b}"] = fin.paired_bootstrap(
            [float(brow[i]["correct"]) for i in both], [rows[i]["correct"] for i in both]
        )
        o[f"majority_vs_{b}"] = fin.paired_bootstrap(
            [float(brow[i]["correct"]) for i in both], [float(rows[i]["majority_correct"]) for i in both]
        )
    out[t] = o
out["bar"] = {b: fin.bootstrap([float(r["correct"]) for r in brow.values()]) for b, brow in bar.items()}
out["note"] = (
    "ours: 4 samples per question (T=0.7, top_p 0.8, top_k 20, rep 1.0, 6,144 cap), mean per-sample accuracy; GPT arms: one call per question at default reasoning settings (earlier runs, not re-run); qwen-greedy-tinker: the first pilot's greedy base run"
)
(fm.RESULTS / "v2_test.json").write_text(json.dumps(out, indent=1))
for t in ("base", "sft-e2", "grpo-s10"):
    o = out[t]
    f = lambda d: f"{d[0] * 100:+.1f} [{d[1] * 100:+.1f}, {d[2] * 100:+.1f}]"
    print(
        t,
        "pass1",
        [round(x, 3) for x in o["mean_pass1"]],
        "maj",
        round(o["majority"][0], 3),
        "loops",
        round(o["loop_rate"], 4),
        "| vs base",
        f(o["vs_base"]) if "vs_base" in o else "-",
        "| vs mini",
        f(o["vs_gpt-5.4-mini"]),
        "| vs gpt-5.4",
        f(o["vs_gpt-5.4"]),
        "| maj vs mini",
        f(o["majority_vs_gpt-5.4-mini"]),
    )
print("bar", {k: round(v[0], 3) for k, v in out["bar"].items()})
