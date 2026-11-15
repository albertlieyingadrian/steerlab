"""
The pre-registered tests (PREREGISTRATION.md), computed from results/runs/*/eval.jsonl
and results/mechanism/*.jsonl. Writes results/summary.json and prints every number it writes.

    .venv/bin/python experiments/compositional-rl/analyse.py

Intervals: a bootstrap that, in each of 2,000 resamples, (1) resamples each arm's training seeds and (2) resamples
evaluation items within each domain x length cell, the same items for every arm, so arm differences are paired.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import itertools

from tasks import parse_answer

RUNS = HERE / "results" / "runs"
MECH = HERE / "results" / "mechanism"
B = 2000
ARMS = {
    1: ["base", "atomic", "compound"],
    2: ["base", "atomic", "compound", "mix10", "mix25"],
    3: ["base", "atomic", "compound"],
}


def load(path):
    """Rows of a .jsonl file, or of its gzipped copy (results are stored as .jsonl.gz)."""
    import gzip

    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_runs(level):
    """{arm: [eval rows of each run]} for one level. Run folders: l{level}-{arm}-s{seed} and base-l{level}."""
    out = defaultdict(list)
    for d in sorted(RUNS.glob("*")):
        f = d / "eval.jsonl"
        if not f.exists() and not Path(str(f) + ".gz").exists():
            continue
        if d.name == f"base-l{level}":
            out["base"].append(load(f))
        elif d.name.startswith(f"l{level}-"):
            out[d.name.split("-")[1]].append(load(f))
    return out


def item_table(runs, kind, lengths=None, filt=None):
    """One dict per run: (domain, length, seed) -> correct (0/1). A filtered-out item counts as missing."""
    tables = []
    for rows in runs:
        m = {}
        for r in rows:
            if r["kind"] != kind or (lengths and r["length"] not in lengths):
                continue
            if filt and not filt(r):
                continue
            m[(r["domain"], r["length"], r["seed"])] = float(r["correct"])
        tables.append(m)
    return tables


class Boot:
    """One resampling plan shared by every arm: the same item draws in each resample (paired differences)."""

    def __init__(self, cells, seed=0):
        rng = np.random.default_rng(seed)
        groups = defaultdict(list)
        for i, c in enumerate(cells):
            groups[c].append(i)
        self.draws = [
            np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups.values()]) for _ in range(B)
        ]
        self.rng = np.random.default_rng(seed + 1)

    def acc(self, arr):
        """Bootstrap distribution of mean accuracy for one arm; arr rows are seeds, columns items."""
        out = np.empty(B)
        n = arr.shape[0]
        for b, idx in enumerate(self.draws):
            out[b] = arr[self.rng.integers(0, n, size=n)][:, idx].mean()
        return out


def ci(x):
    x = np.asarray(x, dtype=float)
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def level_tests(level, lengths=None, filt=None):
    runs = arm_runs(level)
    arms = [a for a in ARMS[level] if runs.get(a)]
    res = {"arms": {}, "n_seeds": {a: len(runs[a]) for a in arms}}
    dists = {}
    for kind in ["atomic", "compound"]:
        tables = {a: item_table(runs[a], kind, lengths, filt) for a in arms}
        # items present for every run of every arm (filters can drop items; keep only shared ones, paired)
        keys = sorted(set.intersection(*[set(t) for a in arms for t in tables[a]]))
        cells = [(k[0], k[1]) for k in keys]
        boot = Boot(cells, seed=level * 10 + (kind == "compound"))
        for a in arms:
            arr = np.array([[t[k] for k in keys] for t in tables[a]])
            d = boot.acc(arr) if keys else np.full(B, np.nan)
            dists[(a, kind)] = d
            res["arms"].setdefault(a, {})[kind] = {
                "acc": float(arr.mean()) if keys else float("nan"),
                "ci": ci(d),
                "n_items": len(keys),
                "per_seed": [float(x) for x in arr.mean(axis=1)] if keys else [],
            }
    A = {a: res["arms"][a]["atomic"]["acc"] for a in arms}
    C = {a: res["arms"][a]["compound"]["acc"] for a in arms}

    def diff(a, b, kind):
        return {
            "diff": res["arms"][a][kind]["acc"] - res["arms"][b][kind]["acc"],
            "ci": ci(dists[(a, kind)] - dists[(b, kind)]),
        }

    tests = {}
    if {"atomic", "compound"} <= set(arms):
        tests["T1_C_compound_minus_atomic"] = diff("compound", "atomic", "compound")
        tests["T1_holds"] = tests["T1_C_compound_minus_atomic"]["ci"][0] > 0
        tests["A_compound_minus_atomic"] = diff("compound", "atomic", "atomic")
        if "base" in arms:
            for a, b, kind in [
                ("compound", "base", "atomic"),
                ("atomic", "base", "atomic"),
                ("atomic", "base", "compound"),
                ("compound", "base", "compound"),
            ]:
                tests[f"{kind[0].upper()}_{a}_minus_{b}"] = diff(a, b, kind)
            tests["T2_holds"] = (
                tests["A_compound_minus_base"]["ci"][0] > 0 and tests["A_compound_minus_atomic"]["ci"][0] > -0.05
            )
            den_c = dists[("compound", "compound")] - dists[("base", "compound")]
            den_a = dists[("atomic", "atomic")] - dists[("base", "atomic")]
            if ci(den_c)[0] > 0 and ci(den_a)[0] > 0:
                t_ac = (dists[("atomic", "compound")] - dists[("base", "compound")]) / den_c
                t_ca = (dists[("compound", "atomic")] - dists[("base", "atomic")]) / den_a
                p_ac = (C["atomic"] - C["base"]) / (C["compound"] - C["base"])
                p_ca = (A["compound"] - A["base"]) / (A["atomic"] - A["base"])
                tests["T3_T_A_to_C"] = {"point": p_ac, "ci": ci(t_ac)}
                tests["T3_T_C_to_A"] = {"point": p_ca, "ci": ci(t_ca)}
                tests["T3_asymmetry"] = {"point": p_ca - p_ac, "ci": ci(t_ca - t_ac)}
            else:
                tests["T3"] = "undefined: a denominator's CI includes 0"
        tests["replicates"] = bool(tests.get("T1_holds") and tests.get("T2_holds"))
    for f in ["mix10", "mix25"]:
        if f in arms and "T1_holds" in tests:
            gap = dists[("compound", "compound")] - dists[("atomic", "compound")]
            rec = (dists[(f, "compound")] - dists[("atomic", "compound")]) / np.where(np.abs(gap) < 1e-9, np.nan, gap)
            gap_p = C["compound"] - C["atomic"]
            point = (C[f] - C["atomic"]) / gap_p if gap_p else float("nan")
            mean_f = (dists[(f, "atomic")] + dists[(f, "compound")]) / 2
            mean_c = (dists[("compound", "atomic")] + dists[("compound", "compound")]) / 2
            t = {
                "recovery": point,
                "recovery_ci": ci(rec),
                "atomic_minus_rlatomic": diff(f, "atomic", "atomic"),
                "compound_minus_rlcompound": diff(f, "compound", "compound"),
                "mean_minus_rlcompound": {
                    "diff": (A[f] + C[f]) / 2 - (A["compound"] + C["compound"]) / 2,
                    "ci": ci(mean_f - mean_c),
                },
            }
            t["passes_B1ab"] = bool(
                point >= 0.8 and t["recovery_ci"][0] >= 0.5 and t["atomic_minus_rlatomic"]["ci"][0] > -0.03
            )
            t["passes_B1c"] = bool(t["mean_minus_rlcompound"]["ci"][0] > 0)
            tests[f"B1_{f}"] = t
    res["tests"] = tests
    return res


def breakdown(level, field):
    out = {}
    for a, rs in arm_runs(level).items():
        for kind in ["atomic", "compound"]:
            acc = defaultdict(list)
            for rows in rs:
                for r in rows:
                    if r["kind"] == kind:
                        acc[r[field]].append(r["correct"])
            out.setdefault(a, {})[kind] = {k: float(np.mean(v)) for k, v in sorted(acc.items())}
    return out


def hygiene(level):
    """Unparsable share, token-cap share and mean reply length per arm (the boring explanations)."""
    out = {}
    for a, rs in arm_runs(level).items():
        for kind in ["atomic", "compound"]:
            rows = [r for run in rs for r in run if r["kind"] == kind]
            if rows:
                out.setdefault(a, {})[kind] = {
                    "unparsable": float(np.mean([parse_answer(r["text"]) is None for r in rows])),
                    "truncated": float(np.mean([r["truncated"] for r in rows])),
                    "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
                }
    return out


def mechanism_tests():
    """B2 (step test; tags step-l2-<arm>-s<seed> and step-base-l2) and B3 (adapter swaps; tags swap-*)."""
    if not MECH.exists():
        return None
    files = {f.name.split(".")[0]: load(f) for f in sorted(MECH.glob("*.jsonl*"))}
    if not files:
        return None
    out = {"step_test": {}, "swaps": {}}
    per_arm = defaultdict(list)
    for tag, rows in files.items():
        if not tag.startswith("step-"):
            continue
        arm = "base" if tag.startswith("step-base") else tag.split("-")[2]
        comp = {(r["domain"], r["length"], r["seed"]): float(r["correct"]) for r in rows if r["kind"] == "compound"}
        steps = defaultdict(list)
        for r in rows:
            if r["kind"] == "step":
                steps[(r["domain"], r["length"], r["seed"])].append(r)
        S = {k: float(all(x["correct"] for x in v)) for k, v in steps.items()}
        per_arm[arm].append((comp, S, steps))
    dist, dom_dist = {}, {}
    for arm, seeds in per_arm.items():
        keys = sorted(set.intersection(*[set(c) & set(s) for c, s, _ in seeds]))
        Cm = np.array([[c[k] for k in keys] for c, _, _ in seeds])
        Sm = np.array([[s[k] for k in keys] for _, s, _ in seeds])
        boot = Boot([(k[0], k[1]) for k in keys], seed=99)
        cd, sd = np.empty(B), np.empty(B)
        n = Cm.shape[0]
        for b, idx in enumerate(boot.draws):
            runs = boot.rng.integers(0, n, size=n)
            cd[b] = Cm[runs][:, idx].mean()
            sd[b] = Sm[runs][:, idx].mean()
        allsteps = [x for _, _, st in seeds for k, v in st.items() if k in set(keys) for x in v]
        sw = [x["correct"] for x in allsteps if x["meta"]["prev_op"] and x["meta"]["prev_op"] != x["meta"]["op"]]
        same = [x["correct"] for x in allsteps if x["meta"]["prev_op"] and x["meta"]["prev_op"] == x["meta"]["op"]]
        first = [x["correct"] for x in allsteps if x["meta"]["step"] == 0]
        by_op = defaultdict(list)
        for x in allsteps:
            by_op[x["meta"]["op"]].append(x["correct"])
        lengths = sorted({k[1] for k in keys})
        prod = {}
        for L in lengths:
            acc_t = defaultdict(list)
            for x in allsteps:
                if x["length"] == L:
                    acc_t[x["meta"]["step"]].append(x["correct"])
            prod[L] = float(np.prod([np.mean(v) for v in acc_t.values()]))

        def cols(L):
            return [i for i, k in enumerate(keys) if k[1] == L]

        by_domain = {}
        for dom in sorted({k[0] for k in keys}):
            cols_d = [i for i, k in enumerate(keys) if k[0] == dom]
            bd = Boot([(keys[i][0], keys[i][1]) for i in cols_d], seed=7)
            Cd, Sd = Cm[:, cols_d], Sm[:, cols_d]
            cdd, sdd = np.empty(B), np.empty(B)
            for b, idx in enumerate(bd.draws):
                runs = bd.rng.integers(0, n, size=n)
                cdd[b] = Cd[runs][:, idx].mean()
                sdd[b] = Sd[runs][:, idx].mean()
            by_domain[dom] = {
                "C": float(Cd.mean()),
                "C_ci": ci(cdd),
                "S": float(Sd.mean()),
                "S_ci": ci(sdd),
                "gap": float(Sd.mean() - Cd.mean()),
                "gap_ci": ci(sdd - cdd),
                "step_acc": float(np.mean([x["correct"] for x in allsteps if x["domain"] == dom])),
            }
            dom_dist[(arm, dom)] = (cdd, sdd)
        out["step_test"][arm] = {
            "by_domain": by_domain,
            "C": float(Cm.mean()),
            "C_ci": ci(cd),
            "S": float(Sm.mean()),
            "S_ci": ci(sd),
            "gap_S_minus_C": float(Sm.mean() - Cm.mean()),
            "gap_ci": ci(sd - cd),
            "n_items": len(keys),
            "n_seeds": n,
            "step_acc": float(np.mean([x["correct"] for x in allsteps])),
            "first_step_acc": float(np.mean(first)),
            "switch_step_acc": float(np.mean(sw)) if sw else None,
            "same_step_acc": float(np.mean(same)) if same else None,
            "n_switch": len(sw),
            "n_same": len(same),
            "by_op": {k: float(np.mean(v)) for k, v in by_op.items()},
            "product_of_step_acc_by_length": prod,
            "S_by_length": {L: float(Sm[:, cols(L)].mean()) for L in lengths},
            "C_by_length": {L: float(Cm[:, cols(L)].mean()) for L in lengths},
            "both": float(((Cm == 1) & (Sm == 1)).mean()),
            "C_only": float(((Cm == 1) & (Sm == 0)).mean()),
            "S_only": float(((Cm == 0) & (Sm == 1)).mean()),
        }
        dist[arm] = (cd, sd)
    st = out["step_test"]
    if "atomic" in dist and "compound" in dist:
        st["B2a_holds"] = st["atomic"]["gap_ci"][0] > 0
        dg = (dist["compound"][1] - dist["compound"][0]) - (dist["atomic"][1] - dist["atomic"][0])
        st["B2b_gap_compound_minus_atomic"] = {
            "diff": st["compound"]["gap_S_minus_C"] - st["atomic"]["gap_S_minus_C"],
            "ci": ci(dg),
        }
        st["B2b_holds"] = ci(dg)[1] < 0
        st["B2c_delta_C"] = {
            "diff": st["compound"]["C"] - st["atomic"]["C"],
            "ci": ci(dist["compound"][0] - dist["atomic"][0]),
        }
        st["B2c_delta_S"] = {
            "diff": st["compound"]["S"] - st["atomic"]["S"],
            "ci": ci(dist["compound"][1] - dist["atomic"][1]),
        }
        # skill-headroom split (exploratory, asked for after the first results): per domain, RL-Compound minus
        # RL-Atomic on chains in one reply (C) and on steps alone (S), with the base model's single-step accuracy
        heads = {}
        for dom in st["atomic"]["by_domain"]:
            ca, sa = dom_dist[("atomic", dom)]
            cc, sc = dom_dist[("compound", dom)]
            heads[dom] = {
                "base_step_acc": st.get("base", {}).get("by_domain", {}).get(dom, {}).get("step_acc"),
                "dC_compound_minus_atomic": {
                    "diff": st["compound"]["by_domain"][dom]["C"] - st["atomic"]["by_domain"][dom]["C"],
                    "ci": ci(cc - ca),
                },
                "dS_compound_minus_atomic": {
                    "diff": st["compound"]["by_domain"][dom]["S"] - st["atomic"]["by_domain"][dom]["S"],
                    "ci": ci(sc - sa),
                },
            }
        st["headroom_by_domain"] = heads
    print("\n===== B2 step test")
    for k, v in st.items():
        print(f"  {k}: {v}")
    sw = {
        t: float(np.mean([r["correct"] for r in rows if r["kind"] == "compound"]))
        for t, rows in files.items()
        if t.startswith("swap-")
    }
    if sw:
        out["swaps"] = sw
        if "atomic" in st and "compound" in st:
            a_end, c_end = st["atomic"]["C"], st["compound"]["C"]
            gap = c_end - a_end
            rec = {}
            for t in ["t0", "t1", "t2"]:
                ac = np.mean([v for k, v in sw.items() if k.startswith(f"swap-AplusC-{t}-")])
                ca = np.mean([v for k, v in sw.items() if k.startswith(f"swap-CplusA-{t}-")])
                rec[t] = {
                    "atomic_plus_compound_third": float(ac),
                    "recovered": float((ac - a_end) / gap),
                    "compound_plus_atomic_third": float(ca),
                    "removed": float((c_end - ca) / gap),
                }
            loc = [
                t
                for t, v in rec.items()
                if v["recovered"] >= 0.5
                and v["removed"] >= 0.5
                and all(rec[o]["recovered"] < 0.25 for o in rec if o != t)
            ]
            out["swap_recovery"] = {
                "gap": gap,
                "thirds": rec,
                "verdict": f"localised in {loc[0]}" if loc else "distributed",
            }
            print("  swap recovery:", json.dumps(out["swap_recovery"]))
        print("\n===== B3 swaps")
        for k in sorted(sw):
            print(f"  {k}: {sw[k]:.3f}")
    return out


INT_LIST = __import__("re").compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]|\[\s*\]")


def first_written_error(text, states):
    """Exploratory (not pre-registered). For flat-list domains, the in-context reply usually writes the state after
    each op. Walk the true states in order and find each one among the lists the reply writes (in order); the first
    true state never written is where the chain went wrong. Returns the 1-based op index, or None if all were written."""
    import ast

    lists = []
    for m in INT_LIST.finditer(text):
        try:
            lists.append(list(ast.literal_eval(m.group(0))))
        except Exception:
            continue
    pos = 0
    for t, s in enumerate(states[1:], start=1):
        while pos < len(lists) and lists[pos] != s:
            pos += 1
        if pos == len(lists):
            return t
        pos += 1
    return None


def written_errors(level=2):
    """Where in-context chains first go wrong, for the array, heap and Bloom filter (flat lists of integers)."""
    from tasks import make_task

    out = {}
    for arm, runs in arm_runs(level).items():
        firsts, at_switch, n_switch_steps, n_steps, written_all_but_wrong = [], 0, 0, 0, 0
        n_wrong, later = 0, 0
        for rows in runs:
            for r in rows:
                if r["kind"] != "compound" or r["domain"] not in ("array", "heap", "bloom") or r["correct"]:
                    continue
                t = make_task(r["domain"], level, "compound", r["length"], r["seed"])
                n_wrong += 1
                k = first_written_error(r["text"], t.states)
                ops = [o[0] for o in t.ops]
                n_steps += len(ops) - 1
                n_switch_steps += sum(1 for a, b in itertools.pairwise(ops) if a != b)
                if k is None:
                    written_all_but_wrong += 1
                    continue
                firsts.append(k / len(ops))
                if k >= 2:
                    later += 1
                    at_switch += ops[k - 1] != ops[k - 2]
        located = len(firsts)
        out[arm] = {
            "wrong_replies": n_wrong,
            "located": located,
            "located_after_step_1": later,
            "every_state_written_yet_wrong": written_all_but_wrong,
            "first_error_relative_position_mean": float(np.mean(firsts)) if firsts else None,
            "share_first_errors_at_switch": at_switch / later if later else None,
            "switch_base_rate": n_switch_steps / n_steps if n_steps else None,
        }
    print("\n===== exploratory: where in-context chains first go wrong (array, heap, bloom)")
    for a, v in out.items():
        print(f"  {a}: {v}")
    return out


def main():
    summary = {}
    for level in [1, 2, 3]:
        if not arm_runs(level):
            continue
        s = {"all": level_tests(level)}
        seen = [2, 3, 4, 5, 6]
        s["seen_lengths"] = level_tests(level, lengths=seen)
        s["longer_lengths"] = level_tests(level, lengths=[8, 10, 12])
        s["parsed_only"] = level_tests(level, filt=lambda r: parse_answer(r["text"]) is not None)
        s["finished_only"] = level_tests(level, filt=lambda r: not r["truncated"])
        s["by_length"] = breakdown(level, "length")
        s["by_domain"] = breakdown(level, "domain")
        s["hygiene"] = hygiene(level)
        summary[f"level{level}"] = s
        print(f"\n===== Level {level}  seeds {s['all']['n_seeds']}")
        for a, v in s["all"]["arms"].items():
            print(
                f"  {a:9s} atomic {v['atomic']['acc']:.3f} {np.round(v['atomic']['ci'], 3).tolist()}"
                f"  compound {v['compound']['acc']:.3f} {np.round(v['compound']['ci'], 3).tolist()}"
                f"  per-seed C {[round(x, 3) for x in v['compound']['per_seed']]}"
            )
        for k, v in s["all"]["tests"].items():
            print(f"  {k}: {v}")
        for sub in ["seen_lengths", "longer_lengths", "parsed_only", "finished_only"]:
            t = s[sub]["tests"]
            print(f"  [{sub}] T1 {t.get('T1_C_compound_minus_atomic')}  replicates={t.get('replicates')}")
        print("  hygiene", json.dumps(s["hygiene"]))
    if arm_runs(2):
        summary["written_errors_l2"] = written_errors(2)
    mech = mechanism_tests()
    if mech:
        summary["mechanism"] = mech
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
