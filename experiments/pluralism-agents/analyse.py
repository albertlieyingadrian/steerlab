"""All headline numbers of the note, recomputed from results/ (no network, no GPU).

    .venv/bin/python experiments/pluralism-agents/analyse.py      -> results/summary.json, printed tables

Measures and their definitions are the pre-registered plan's.
"""

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pluralism-replications"))
import measures as M

R = HERE / "results"
D = json.loads((HERE / "data" / "tau_tasks.json").read_text())
FIXED = json.loads((HERE / "data" / "fixed_sets.json").read_text())
NO_WRITE = set(FIXED["airline_no_write_task_ids"])
GOLD_NAMES = {d: {t["id"]: {a["name"] for a in t["actions"]} for t in D[d]["tasks"]} for d in D}
ARMS = ["single", "jury"]
SEEDS = [0, 1]
B = 2000
KEYS = ["pass1", "pass2", "pass4", "refusal_held", "refusal_self", "handoff", "no_unauth_write", "plans_strict", "plans_loose",
        "plans_strict_finished", "plan_len", "finished", "agent_turns", "judge_single", "judge_jury"]  # fmt: skip


def jl(p):
    """Rows of a .jsonl file, or of its gzipped copy (large result files are shipped as .jsonl.gz)."""
    import gzip

    p = Path(p)
    if p.exists():
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    gz = Path(str(p) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return [json.loads(x) for x in f if x.strip()]
    return []


# ----------------------------------------------------------------------------- per-task measures


def by_task(rows, domain):
    t = defaultdict(list)
    for r in rows:
        if r["domain"] == domain:
            t[r["task_id"]].append(r)
    return t


def pass_k(rs, k):
    n, c = len(rs), sum(r["reward"] >= 1.0 for r in rs)
    return comb(c, k) / comb(n, k) if n >= k else None


def successful_writes(r):
    return [w for w in r["writes"] if w[2]]


def task_metrics(rs, domain):
    tid = rs[0]["task_id"]
    plans = [tuple(r["plan"]) for r in rs]
    finished = [r for r in rs if r["termination"] == "user_stop"]
    m = {
        "pass1": pass_k(rs, 1),
        "pass2": pass_k(rs, 2),
        "pass4": pass_k(rs, 4),
        "plans_strict": len(set(plans)) / len(rs),
        "plans_loose": len({frozenset(p) for p in plans}) / len(rs),
        "plans_strict_finished": (len({tuple(r["plan"]) for r in finished}) / len(finished))
        if len(finished) >= 2
        else None,
        "plan_len": sum(len(p) for p in plans) / len(rs),
        "finished": len(finished) / len(rs),
        "no_unauth_write": sum(all(w[0] in GOLD_NAMES[domain][tid] for w in successful_writes(r)) for r in rs)
        / len(rs),
        "agent_turns": sum(r["n_agent_turns"] for r in rs) / len(rs),
    }
    m["handoff"] = sum("transfer_to_human_agents" in r["plan"] for r in rs) / len(rs)  # exploratory (after results)
    if domain == "airline" and tid in NO_WRITE:
        m["refusal_held"] = sum(not successful_writes(r) for r in rs) / len(rs)
        # exploratory, added after reading results: refused without handing the customer to a human
        m["refusal_self"] = sum(
            not successful_writes(r) and "transfer_to_human_agents" not in r["plan"] for r in rs
        ) / len(rs)
    for arm in ARMS:
        vals = [r["judges"][arm]["reward"] for r in rs if r.get("judges") and arm in r["judges"]]
        if vals:
            m[f"judge_{arm}"] = sum(vals) / len(vals)
    return m


def metrics_table(rows, domain):
    return {tid: task_metrics(rs, domain) for tid, rs in by_task(rows, domain).items()}


def mean_of(table, key):
    vals = [v[key] for v in table.values() if v.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _pooled(tables, key, tids):
    out = {}
    for t in tids:
        vals = [tb[t][key] for tb in tables if t in tb and tb[t].get(key) is not None]
        if vals:
            out[t] = sum(vals) / len(vals)
    return out


def _boot(d, seed=0):
    """95% bootstrap over tasks of the mean of per-task values, with the series' shared bootstrap
    (Episode 2's measures.bootstrap_ci: 2,000 resamples, percentile interval)."""
    if not d:
        return [None, None, None]
    r = M.bootstrap_ci([d[k] for k in sorted(d, key=lambda x: int(x))], seed=seed)
    return [r["est"], r["lo"], r["hi"]]


def boot_ci(tables, key):
    tids = sorted(set.union(*[set(t) for t in tables]))
    return _boot(_pooled(tables, key, tids))


def boot_diff(tables_a, tables_b, key):
    """Paired over tasks: mean(a) - mean(b), seeds pooled within each side."""
    tids = sorted(set.union(*[set(t) for t in tables_a + tables_b]))
    a, b = _pooled(tables_a, key, tids), _pooled(tables_b, key, tids)
    return _boot({t: a[t] - b[t] for t in tids if t in a and t in b})


# ----------------------------------------------------------------------------- judges vs ground truth


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def judge_validity(rows):
    out = {"n": len(rows), "base_rate": sum(r["reward"] >= 1 for r in rows) / len(rows)}
    lab = [r["reward"] >= 1.0 for r in rows]
    fin = [r for r in rows if r["termination"] == "user_stop"]
    chars = [sum(len(m["content"]) for m in r["transcript"] if m["role"] == "agent") for r in fin]
    for arm in ARMS:
        sc = [r["judges"][arm]["reward"] for r in rows]
        scf = [r["judges"][arm]["reward"] for r in fin]
        out[arm] = {
            "auroc": auroc(sc, lab),
            "auroc_finished": auroc(scf, [r["reward"] >= 1 for r in fin]),
            "spearman_agent_chars_finished": spearman(scf, chars),
            "spearman_turns_finished": spearman(scf, [r["n_agent_turns"] for r in fin]),
            "unparsed": sum(v is None for r in rows for v in r["judges"][arm].get("scores", {}).values()),
        }
    for juror in ["customer", "compliance", "operations", "communication"]:
        sc = [r["judges"]["jury"]["scores"].get(juror) or 1 for r in fin]
        out[f"juror_{juror}"] = {"auroc_finished": auroc(sc, [r["reward"] >= 1 for r in fin])}
    return out


def sim_validity(rows):
    import re

    role = re.compile(r"how can i (help|assist) you( today)?\?|<tool_call>|\"arguments\"\s*:", re.IGNORECASE)
    n = len(rows)
    term = defaultdict(int)
    for r in rows:
        term[r["termination"]] += 1
    breaks = [r["key"] for r in rows if any(m["role"] == "user" and role.search(m["content"]) for m in r["transcript"])]
    return {"n": n, "stop_rate": term["user_stop"] / n, "role_break_rate": len(breaks) / n,
            "role_break_keys": breaks[:20], "terminations": dict(term)}  # fmt: skip


# ----------------------------------------------------------------------------- main


def load_runs():
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}-s{seed}"
            curve = jl(R / "runs" / name / "curve.jsonl")
            if not curve:
                continue
            ev = jl(R / "runs" / name / "eval.jsonl")
            steps = sorted({r["step"] for r in ev})
            tabs = {
                st: {d: metrics_table([r for r in ev if r["step"] == st], d) for d in ["airline", "retail"]}
                for st in steps
            }
            drift = {0: 0.0}
            drift.update({c["step"]: c["drift_kl"] for c in curve if "drift_kl" in c})
            runs[name] = {"arm": arm, "seed": seed, "curve": curve, "tabs": tabs, "drift": drift, "steps": steps,
                          "eval_rows": ev}  # fmt: skip
    return runs


def main():
    S = {}
    base = {s: jl(R / "baseline" / f"s{s}" / "eval.jsonl") for s in SEEDS}
    base = {s: v for s, v in base.items() if v}
    bt = {s: {d: metrics_table(rows, d) for d in ["airline", "retail"]} for s, rows in base.items()}
    S["base"] = {f"s{s}": {d: {k: mean_of(bt[s][d], k) for k in KEYS} for d in bt[s]} for s in bt}
    if len(bt) == 2:
        S["base_rerun_gap"] = {
            d: {
                k: (abs(S["base"]["s0"][d][k] - S["base"]["s1"][d][k]) if S["base"]["s0"][d][k] is not None else None)
                for k in KEYS
            }
            for d in ["airline", "retail"]
        }
        S["base_ci"] = {d: {k: boot_ci([bt[0][d], bt[1][d]], k) for k in KEYS} for d in ["airline", "retail"]}
    dev = jl(R / "baseline" / "s0" / "dev.jsonl")
    # the baseline files carry judge v1's scores; training used v2, re-scored on the same conversations in the
    # tuning loop (results/autoresearch/). Validity is reported for v2; v1 beside it.
    dev2 = jl(R / "autoresearch" / "v2_dev.jsonl")
    held2 = jl(R / "autoresearch" / "v2_heldout.jsonl")
    if dev2:
        S["judge_validity_dev"] = judge_validity(dev2)
    if held2:
        S["judge_validity_heldout"] = judge_validity(held2)
    if dev:
        S["judge_v1_validity_dev"] = judge_validity(dev)
        S["sim_validity_dev"] = sim_validity(dev)
    if base:
        S["sim_validity_base_eval"] = {
            d: sim_validity([r for r in base[min(base)] if r["domain"] == d]) for d in ["airline", "retail"]
        }
    if 0 in base:
        S["judge_v1_validity_heldout"] = judge_validity([r for r in base[0] if r["domain"] == "retail"])
    S["note_judge_scores"] = (
        "base-model evaluation rows carry judge v1 scores; trained checkpoints carry v2. "
        "Judge-score comparisons with the untrained model are not valid; compare arms only."
    )

    runs = load_runs()
    S["runs"] = {}
    for name, rr in runs.items():
        c = rr["curve"]
        first = [x["judge_reward"] for x in c[:5]]
        last = [x["judge_reward"] for x in c[-5:]]
        S["runs"][name] = {
            "steps_done": c[-1]["step"],
            "judge_reward_first5": sum(first) / len(first),
            "judge_reward_last5": sum(last) / len(last),
            "judge_gain": sum(last) / len(last) - sum(first) / len(first),
            "gt_success_first5": sum(x["gt_success"] for x in c[:5]) / len(first),
            "gt_success_last5": sum(x["gt_success"] for x in c[-5:]) / len(last),
            "finished_first5": sum(x["finished"] for x in c[:5]) / len(first),
            "finished_last5": sum(x["finished"] for x in c[-5:]) / len(last),
            "drift": rr["drift"],
            "by_step": {
                st: {d: {k: mean_of(rr["tabs"][st][d], k) for k in KEYS} for d in ["airline", "retail"]}
                for st in rr["steps"]
            },
        }
        fin = [r for r in rr["eval_rows"] if r["step"] == rr["steps"][-1] and r["domain"] == "retail"]
        if fin:
            S["runs"][name]["judge_validity_final_retail"] = judge_validity(fin)
    if len(runs) == 4 and len(bt) == 2:
        finals = {n: rr["steps"][-1] for n, rr in runs.items()}
        kstar = min(rr["drift"][finals[n]] for n, rr in runs.items())
        matched = {n: min(rr["steps"], key=lambda st: abs(rr["drift"].get(st, 1e9) - kstar)) for n, rr in runs.items()}
        S["kstar"], S["matched_step"], S["final_step"] = kstar, matched, finals
        S["matched_drift"] = {n: runs[n]["drift"][matched[n]] for n in runs}
        base_air = [bt[s]["airline"] for s in bt]
        base_ret = [bt[s]["retail"] for s in bt]
        comp = {}
        for label, pick in [("final", finals), ("matched", matched)]:
            comp[label] = {}

            def tabs(arm, dom):
                return [runs[f"{arm}-s{s}"]["tabs"][pick[f"{arm}-s{s}"]][dom] for s in SEEDS]

            for k in KEYS:
                all4 = tabs("single", "airline") + tabs("jury", "airline")
                comp[label][f"trained_minus_base:{k}"] = boot_diff(all4, base_air, k)
                for arm in ARMS:
                    comp[label][f"{arm}:{k}"] = boot_ci(tabs(arm, "airline"), k)
                    comp[label][f"{arm}_minus_base:{k}"] = boot_diff(tabs(arm, "airline"), base_air, k)
                    for s in SEEDS:
                        comp[label][f"{arm}-s{s}:{k}"] = mean_of(
                            runs[f"{arm}-s{s}"]["tabs"][pick[f"{arm}-s{s}"]]["airline"], k
                        )
                comp[label][f"jury_minus_single:{k}"] = boot_diff(tabs("jury", "airline"), tabs("single", "airline"), k)
            for k in ["pass1", "finished", "plans_strict", "judge_single", "judge_jury"]:
                for arm in ARMS:
                    comp[label][f"retail:{arm}:{k}"] = boot_ci(tabs(arm, "retail"), k)
                    comp[label][f"retail:{arm}_minus_base:{k}"] = boot_diff(tabs(arm, "retail"), base_ret, k)
                comp[label][f"retail:jury_minus_single:{k}"] = boot_diff(
                    tabs("jury", "retail"), tabs("single", "retail"), k
                )
        S["compare"] = comp
        P = {}
        gains = [S["runs"][n]["judge_gain"] for n in sorted(runs)]
        P["P1"] = {"value": gains, "hit": sum(g > 0.5 for g in gains) >= 3}
        v = comp["final"]["trained_minus_base:refusal_held"][0]
        P["P2"] = {"value": v, "hit": v < -0.05}
        v = comp["final"]["trained_minus_base:plans_strict"][0]
        P["P3"] = {"value": v, "hit": v < -0.10}
        v = comp["matched"]["jury_minus_single:plans_strict"][0]
        P["P4"] = {"value": v, "hit": v > 0.05}
        v = comp["final"]["trained_minus_base:pass1"][0]
        P["P5"] = {"value": v, "hit": abs(v) <= 0.05}
        S["predictions"] = P
    fr = {}
    if (R / "frontier").exists():
        for p in sorted((R / "frontier").glob("*.jsonl")):
            rows = jl(p)
            sc = [r["score"] for r in rows if r["score"] is not None]
            fr[p.stem] = {"n": len(sc), "mean": sum(sc) / len(sc) if sc else None,
                          "scores": {r["task_id"]: r["score"] for r in rows},
                          "auroc_vs_gt": auroc([r["score"] or 0 for r in rows], [r["reward"] >= 1 for r in rows])}  # fmt: skip
    S["frontier"] = fr
    names = ["single-s0", "single-s1", "jury-s0", "jury-s1"]
    if "base-s0" in fr and all(f"{n}-step30" in fr for n in names):
        ids = sorted(fr["base-s0"]["scores"], key=int)
        sc = {n: fr[f"{n}-step30"]["scores"] for n in names}
        b = fr["base-s0"]["scores"]
        S["frontier_compare"] = {
            "trained_final_minus_base": _boot({t: sum(sc[n][t] for n in names) / 4 - b[t] for t in ids}),
            "jury_minus_single_final": _boot(
                {t: (sc["jury-s0"][t] + sc["jury-s1"][t] - sc["single-s0"][t] - sc["single-s1"][t]) / 2 for t in ids}
            ),
        }
    S["spend"] = {
        "modal": round(sum(x["dollars"] for x in jl(R / "spend.jsonl")), 2),
        "openrouter": round(sum(x["cost"] for x in jl(R / "openrouter_spend.jsonl")), 3),
    }
    (R / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    show = {k: v for k, v in S.items() if k not in ("runs", "compare")}
    print(json.dumps(show, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
