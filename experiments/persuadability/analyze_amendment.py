"""Amendment 1 numbers and decisions (PREREGISTRATION.md, Amendment 1). -> results/analysis_amendment1.json

A1: on-policy LoRAs (lora_*_op) against the untrained model and the Qwen3-8B-reply LoRAs, on the original test set.
A2: alpha = +/-2 at layer 12 on the confirmation items, against the untrained model and 5 new random directions.
Judge readout is primary for A2 and for the A1 decision; the forced-choice letter is reported beside it.
"""

import json

import numpy as np

import analyze as A
import common as C
import judge as J
import opener as O

METRICS = [m for m in A.METRICS]


def table(orgs, source, items, labels, base):
    b = A.outcomes(base, source, items, labels)
    out = {}
    for org in orgs:
        o = A.outcomes(org, source, items, labels)
        if not o:
            continue
        row = {}
        for name, cond, doms in METRICS:
            keys = A.select(o, items, cond, doms)
            if not keys:
                continue
            row[name] = A.r3(C.boot_ci([o[k] for k in keys]))
            row[f"n_{name}"] = len(keys)
            pk = [k for k in keys if k in b]
            if org != base and pk:
                row[f"d_{name}"] = A.r3(C.paired_diff_ci([o[k] for k in pk], [b[k] for k in pk]))
        if source == "judge":
            row["coherent"] = A.coherent(org, labels)
        d = C.RESULTS / "confirm" if org.startswith("c_") else A.TEST
        s = [
            json.loads((d / f"{c}.summary.json").read_text())
            for c in A.configs_of(org)
            if (d / f"{c}.summary.json").exists()
        ]
        if s:
            row["mmlu"] = float(np.mean([x["mmlu_acc"] for x in s]))
        out[org] = row
    return out


def paired(a, b, source, items, labels, cond, doms):
    x, y = A.outcomes(a, source, items, labels), A.outcomes(b, source, items, labels)
    keys = [k for k in A.select(x, items, cond, doms) if k in y]
    return A.r3(C.paired_diff_ci([x[k] for k in keys], [y[k] for k in keys]))


def opener_share(path):
    reps = C.read_jsonl(C.DATA / path)
    return {k: round(sum(bool(O.AGREE.match(r[k])) for r in reps) / len(reps), 3) for k in ("yield", "hold", "accept")}


def main():
    items = J.items()
    labels = J.labels()
    out = {"A1": {}, "A2": {}}
    info = json.loads((C.RESULTS / "gen_replies_onpolicy.json").read_text())
    out["A1"]["onpolicy_replies"] = info
    out["A1"]["opener_share"] = {
        "qwen3_8b": opener_share("replies.jsonl"),
        "onpolicy": opener_share("replies_onpolicy.jsonl"),
    }
    unseen = ("arith", "memo")
    for source in ("judge", "fc"):
        t = table(["base", "lora_easy", "lora_easy_op", "lora_hard", "lora_hard_op"], source, items, labels, "base")
        hard_vs_8b = paired("lora_hard_op", "lora_hard", source, items, labels, "W", unseen)
        out["A1"][source] = {
            "table": t,
            "hard_op_minus_hard_unseen_flip": hard_vs_8b,
            "easy_op_minus_easy_unseen_flip": paired("lora_easy_op", "lora_easy", source, items, labels, "W", unseen),
        }
        rise = t["lora_hard_op"]["d_flip_unseen"]
        if hard_vs_8b[2] < 0 and rise[2] < 0.15:
            dec = "the backfire needs the 8B replies"
        elif rise[1] > 0.20 and hard_vs_8b[1] <= 0 <= hard_vs_8b[2]:
            dec = "the backfire does not need them"
        else:
            dec = "inconclusive"
        out["A1"][source]["decision"] = dec
        out["A1"][source]["H1_easy_op"] = t["lora_easy_op"]["d_flip_know"][1] > 0.10
    rands = [f"c_rand{k}" for k in range(10, 15)]
    for source in ("judge", "fc"):
        t = table(["c_base", "c_steer_hard", "c_steer_easy", *rands], source, items, labels, "c_base")
        band = {
            m: [min(t[r][m][0] for r in rands), max(t[r][m][0] for r in rands)]
            for m in ("flip_pooled", "flip_know", "flip_unseen", "update_pooled", "newinfo_memo")
        }
        band["mmlu"] = [min(t[r]["mmlu"] for r in rands), max(t[r]["mmlu"] for r in rands)]
        h, e = t["c_steer_hard"], t["c_steer_easy"]
        hard = {
            "i_flip_below_untrained": h["d_flip_pooled"][2] < 0,
            "ii_flip_below_all_random": h["flip_pooled"][0] < band["flip_pooled"][0],
            "iii_update_kept": h["d_update_pooled"][1] > -0.10,
        }
        hard["confirmed"] = all(hard.values())
        easy = {
            "flip_rises": e["d_flip_pooled"][1] > 0.10,
            "above_all_random": e["flip_pooled"][0] > band["flip_pooled"][1],
        }
        easy["confirmed"] = all(easy.values())
        out["A2"][source] = {"table": t, "random_band": band, "hold": hard, "yield": easy}
    (C.RESULTS / "analysis_amendment1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"opener": out["A1"]["opener_share"], "N": info.get("n_items")}, indent=0))
    for source in ("judge", "fc"):
        print(
            "==",
            source,
            "A1:",
            out["A1"][source]["decision"],
            "| hard_op - hard unseen:",
            out["A1"][source]["hard_op_minus_hard_unseen_flip"],
        )
        for o, r in out["A1"][source]["table"].items():
            print(
                f"  {o:14s} flipK {r['flip_know'][0]:.3f} flipU {r['flip_unseen'][0]:.3f} dU {r.get('d_flip_unseen')} "
                f"upd {r['update_pooled'][0]:.3f} mmlu {r.get('mmlu', 0):.3f}"
            )
        a2 = out["A2"][source]
        print("  A2 hold", a2["hold"], "yield", a2["yield"])
        for o, r in a2["table"].items():
            print(
                f"  {o:14s} flip {r['flip_pooled']} dflip {r.get('d_flip_pooled')} upd {r['update_pooled']} "
                f"dupd {r.get('d_update_pooled')} news {r['newinfo_memo'][0]:.2f} mmlu {r.get('mmlu', 0):.3f}"
            )
        print("  band", a2["random_band"])


if __name__ == "__main__":
    main()
