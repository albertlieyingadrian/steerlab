"""
Step 6: every number in the README, from results/*.csv -> results/scores.json (and printed).

All intervals are 95% bootstrap intervals over episodes (2,000 draws): a trajectory's steps move together.
"""

import json
from collections import Counter

import numpy as np
from common import MODELS, RESULTS, RISKY, TYPES, boot_ci, fmt, read_csv

CONDS = ("T0", "T+A", "T+R", "T+A+R")
S: dict = {}

# ── counts ──
eps = read_csv(RESULTS / "episodes.csv")
acts = read_csv(RESULTS / "actions.csv")
steps = read_csv(RESULTS / "steps.csv")
parsed = [a for a in acts if a["action_type"] != "unparsed"]
S["counts"] = {
    "episodes": len(eps),
    "agent_turns": len(acts),
    "parsed_turns": len(parsed),
    "parse_rate": len(parsed) / len(acts),
    "test_steps": len(steps),
    "test_types": Counter(s["action_type"] for s in steps),
    "all_types": Counter(a["action_type"] for a in parsed),
    "tools": Counter(a["tool"] for a in parsed).most_common(20),
    "unknown_cmd_steps": sum(a["unknown_cmd"] == "1" for a in parsed),
    "empty_reasoning_steps": sum(a["reasoning_chars"] == "0" for a in parsed),
}
S["leaks"] = {
    "test_steps": Counter(s["leak"] for s in steps),
    "test_command_leak_rate": boot_ci([s["episode"] for s in steps], [s["command_leak"] == "1" for s in steps]),
    "test_any_leak_rate": boot_ci([s["episode"] for s in steps], [s["leak"] != "none" for s in steps]),
    "all_steps_any_leak_rate": boot_ci(
        [a["episode"] for a in parsed if a["action_type"] != "submit"],
        [a["leak"] != "none" for a in parsed if a["action_type"] != "submit"],
    ),
    "by_type_any_leak": {
        t: boot_ci(
            [s["episode"] for s in steps if s["action_type"] == t],
            [s["leak"] != "none" for s in steps if s["action_type"] == t],
        )
        for t in TYPES
        if any(s["action_type"] == t for s in steps)
    },
}
print("=== counts")
print(json.dumps({k: v for k, v in S["counts"].items() if k != "tools"}, default=str))
print("leaks (test steps):", dict(S["leaks"]["test_steps"]), "any-leak rate", fmt(S["leaks"]["test_any_leak_rate"]))

# ── M1 ──
reader = read_csv(RESULTS / "reader_full.csv")
subsets = {
    "all": lambda r: True,
    "no_leak": lambda r: r["leak"] == "none",
    "target_only_leak": lambda r: r["leak"] == "target_only",
    "command_leak": lambda r: r["leak"] == "command",
    "not_chained": lambda r: r["chained"] == "0",
}
m1: dict = {}
for sub, keep in subsets.items():
    m1[sub] = {}
    for c in CONDS:
        rs = [r for r in reader if r["condition"] == c and keep(r)]
        e = [r["episode"] for r in rs]
        m1[sub][c] = {
            "n": len(rs),
            **{m: boot_ci(e, [int(r[m]) for r in rs]) for m in ("type_ok", "target_ok", "both_ok")},
        }
    base = [r for r in reader if r["condition"] == "T0" and keep(r)]
    m1[sub]["baseline_prev_type"] = {
        "n": len(base),
        "type_ok": boot_ci([r["episode"] for r in base], [int(r["baseline_ok"]) for r in base]),
    }
# paired differences on both_ok, by episode
by_step = {}
for r in reader:
    by_step.setdefault(r["step_id"], {})[r["condition"]] = r
diffs = {}
for sub in ("all", "no_leak"):
    for a, b in (("T+R", "T+A"), ("T+A+R", "T+A"), ("T+A+R", "T+R"), ("T+R", "T0")):
        for m in ("type_ok", "both_ok"):
            rows = [v for v in by_step.values() if subsets[sub](v["T0"])]
            diffs[f"{sub}:{a} minus {b}:{m}"] = boot_ci(
                [v[a]["episode"] for v in rows], [int(v[a][m]) - int(v[b][m]) for v in rows]
            )
m1["paired_differences"] = diffs
m1["by_model"] = {}
for mod in MODELS:
    m1["by_model"][mod] = {}
    for sub in ("all", "no_leak"):
        for c in CONDS:
            rs = [r for r in reader if r["condition"] == c and r["model_name"] == mod and subsets[sub](r)]
            m1["by_model"][mod][f"{sub}:{c}"] = {
                "n": len(rs),
                "both_ok": boot_ci([r["episode"] for r in rs], [int(r["both_ok"]) for r in rs]),
                "type_ok": boot_ci([r["episode"] for r in rs], [int(r["type_ok"]) for r in rs]),
            }
# exploratory (added after the first results, not pre-registered): step k's own reasoning removed
prior_path = RESULTS / "reader_full_prior.csv"
if prior_path.exists():
    prior = read_csv(prior_path)
    m1["prior_reasoning_only"] = {}
    for sub in ("all", "no_leak"):
        for c in ("T+Rp", "T+A+Rp"):
            rs = [r for r in prior if r["condition"] == c and subsets[sub](r)]
            m1["prior_reasoning_only"][f"{sub}:{c}"] = {
                "n": len(rs),
                **{
                    m: boot_ci([r["episode"] for r in rs], [int(r[m]) for r in rs])
                    for m in ("type_ok", "target_ok", "both_ok")
                },
            }
    pby = {}
    for r in prior:
        pby.setdefault(r["step_id"], {})[r["condition"]] = r
    for sub in ("all", "no_leak"):
        for c, base in (("T+Rp", "T+A"), ("T+A+Rp", "T+A"), ("T+R", "T+Rp")):
            rows = [(pby[sid], v) for sid, v in by_step.items() if sid in pby and subsets[sub](v["T0"])]
            get = lambda pair, cond: pair[0][cond] if cond.endswith("Rp") else pair[1][cond]  # noqa: E731
            for m in ("type_ok", "both_ok"):
                m1["prior_reasoning_only"][f"diff {sub}:{c} minus {base}:{m}"] = boot_ci(
                    [pr[1]["T0"]["episode"] for pr in rows], [int(get(pr, c)[m]) - int(get(pr, base)[m]) for pr in rows]
                )
m1["target_judge_calls"] = sum(int(r["target_judged"]) for r in reader)
m1["target_judge_flips_to_match"] = sum(int(r["target_judged"]) and int(r["target_ok"]) for r in reader)
m1["confusion_T+A+R"] = Counter(f"{r['gold_type']}->{r['pred_type']}" for r in reader if r["condition"] == "T+A+R")
S["M1"] = m1
print("=== M1 (accuracy of type / target / both)")
for sub in ("all", "no_leak"):
    print(f"-- {sub}: n={m1[sub]['T0']['n']}, baseline type {fmt(m1[sub]['baseline_prev_type']['type_ok'])}")
    for c in CONDS:
        x = m1[sub][c]
        print(f"   {c:6s} type {fmt(x['type_ok'])}  target {fmt(x['target_ok'])}  both {fmt(x['both_ok'])}")
for k, v in diffs.items():
    print(f"   diff {k}: {fmt(v)}")

# ── M2 ──
div = read_csv(RESULTS / "divergence.csv")
labs = ("matches", "partial", "not_described")
m2 = {
    "n": len(div),
    "rates": {lab: boot_ci([d["episode"] for d in div], [d["label"] == lab for d in div]) for lab in labs},
    "difference": Counter(d["difference"] for d in div if d["label"] != "matches"),
    "by_type": {},
    "by_model": {},
    "by_leak": {},
}
for t in TYPES:
    rs = [d for d in div if d["action_type"] == t]
    if rs:
        m2["by_type"][t] = {
            "n": len(rs),
            **{lab: boot_ci([d["episode"] for d in rs], [d["label"] == lab for d in rs]) for lab in labs},
        }
for mod in MODELS:
    rs = [d for d in div if d["model_name"] == mod]
    m2["by_model"][mod] = {
        "n": len(rs),
        **{lab: boot_ci([d["episode"] for d in rs], [d["label"] == lab for d in rs]) for lab in labs},
    }
for lk in ("command", "target_only", "none"):
    rs = [d for d in div if d["leak"] == lk]
    m2["by_leak"][lk] = {
        "n": len(rs),
        **{lab: boot_ci([d["episode"] for d in rs], [d["label"] == lab for d in rs]) for lab in labs},
    }
hc_path = RESULTS / "handcheck_labels.csv"
if hc_path.exists():
    # the human sheet holds item, step_id, human_label (labelled blind); judge labels come from handcheck_items.csv
    judge = {h["item"]: h["judge_label"] for h in read_csv(RESULTS / "handcheck_items.csv")}
    hc = [h for h in read_csv(hc_path) if h.get("human_label", "").strip()]
    pairs = [(judge[h["item"]], h["human_label"].strip()) for h in hc]
    agree = np.mean([a == b for a, b in pairs])
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[lab] * pb[lab] for lab in labs) / len(pairs) ** 2
    m2["handcheck"] = {
        "n": len(pairs),
        "agreement": float(agree),
        "cohen_kappa": float((agree - pe) / (1 - pe)),
        "confusion_judge_to_human": Counter(f"{a}->{b}" for a, b in pairs),
        "agreement_binary_matches_vs_not": float(np.mean([(a == "matches") == (b == "matches") for a, b in pairs])),
    }
S["M2"] = m2
if "prior_reasoning_only" in m1:
    print("=== M1 exploratory: prior reasoning only (step k's own reasoning removed)")
    for k, v in m1["prior_reasoning_only"].items():
        print(
            f"   {k}: "
            + (fmt(v) if isinstance(v, tuple) else f"n={v['n']} type {fmt(v['type_ok'])} both {fmt(v['both_ok'])}")
        )
m2["unparsed"] = sum(d["label"] not in labs for d in div)
m2["partial_or_not_described"] = boot_ci(
    [d["episode"] for d in div], [d["label"] in ("partial", "not_described") for d in div]
)
print("=== M2 (reasoning vs command, same step)")
print(f"   partial or not described {fmt(m2['partial_or_not_described'])}; unparsed replies {m2['unparsed']}")
for lab in labs:
    print(f"   {lab:14s} {fmt(m2['rates'][lab])}")
print("   differences named:", dict(m2["difference"]))
for t, v in m2["by_type"].items():
    print(f"   {t:18s} n={v['n']:3d} partial {fmt(v['partial'])} not_described {fmt(v['not_described'])}")
if "handcheck" in m2:
    h = m2["handcheck"]
    print(
        f"   hand check: n={h['n']} agreement {h['agreement']:.2f}, kappa {h['cohen_kappa']:.2f}, "
        f"matches-vs-not agreement {h['agreement_binary_matches_vs_not']:.2f}; {dict(h['confusion_judge_to_human'])}"
    )

# ── M3 ──
un = read_csv(RESULTS / "unmentioned.csv")
m3 = {
    "n": len(un),
    "where": Counter(u["where"] for u in un),
    "method": Counter(u["method"] for u in un),
    "never_rate": boot_ci([u["episode"] for u in un], [u["where"] == "never" for u in un]),
    "by_type": {},
    "by_model": {},
}
for t in TYPES:
    rs = [u for u in un if u["action_type"] == t]
    if rs:
        m3["by_type"][t] = {
            "n": len(rs),
            "never": sum(u["where"] == "never" for u in rs),
            "never_rate": boot_ci([u["episode"] for u in rs], [u["where"] == "never" for u in rs]),
        }
judged = [u for u in un if u["method"] == "judge"]
m3["judge_unanimous"] = {"n": len(judged), "unanimous": sum(u["unanimous"] == "1" for u in judged)}
risky = [u for u in un if u["action_type"] in RISKY]
outside = [u for u in un if u["outside_repo"] == "1"]
m3["risky"] = {
    "n": len(risky),
    "never": sum(u["where"] == "never" for u in risky),
    "by_tool": Counter(f"{u['tool']}:{u['where']}" for u in risky),
}
m3["outside_repo"] = {"n": len(outside), "never": sum(u["where"] == "never" for u in outside)}
for mod in MODELS:
    rs = [u for u in un if u["model_name"] == mod]
    m3["by_model"][mod] = {
        "n": len(rs),
        "never_rate": boot_ci([u["episode"] for u in rs], [u["where"] == "never" for u in rs]),
    }
m3["by_resolved"] = {
    r: boot_ci(
        [u["episode"] for u in un if u["resolved"] == r], [u["where"] == "never" for u in un if u["resolved"] == r]
    )
    for r in ("0", "1")
}
S["M3"] = m3
print("=== M3 (actions never mentioned in reasoning up to that step)")
print(
    f"   n={m3['n']} where={dict(m3['where'])} never {fmt(m3['never_rate'])}; judge unanimous "
    f"{m3['judge_unanimous']['unanimous']}/{m3['judge_unanimous']['n']}"
)
for t, v in m3["by_type"].items():
    print(f"   {t:18s} {v['never']:4d}/{v['n']:4d} {fmt(v['never_rate'])}")
print(
    f"   risky types: {m3['risky']['never']}/{m3['risky']['n']} never; outside repo: "
    f"{m3['outside_repo']['never']}/{m3['outside_repo']['n']} never"
)
for mod, v in m3["by_model"].items():
    print(f"   {mod}: {fmt(v['never_rate'])} (n={v['n']})")

# ── cost ──
cost = read_csv(RESULTS / "cost_log.csv")
S["cost"] = {
    "usd_total": round(sum(float(c["usd"]) for c in cost), 4),
    "tokens_in": sum(int(c["tokens_in"]) for c in cost),
    "tokens_out": sum(int(c["tokens_out"]) for c in cost),
    "new_calls": sum(int(c["new_calls"]) for c in cost),
}
print("=== cost", S["cost"])
(RESULTS / "scores.json").write_text(
    json.dumps(S, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
)
print("wrote results/scores.json")
