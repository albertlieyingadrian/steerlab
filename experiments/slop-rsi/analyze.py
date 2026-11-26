"""Step 3 (pilot 2): the results. This is where the test tier is read for the first time.

    python analyze.py      # results/pilot2/summary.json (+ a printed table)

Registered test looks (amendment 2 item 10, kept by amendment 3): each arm's incumbent (best val score so far) after
every 4 evaluations, and each arm's final pick (the incumbent after 24). Test numbers for every other recipe are
computed too and labelled post hoc. Intervals: 2,000 paired bootstrap resamples of the test items within each family
(the same resamples for every detector); in each resample the matched cutoff is found again, so the interval covers
the cutoff's noise too. Pilot 1's analysis is at commit a324d862.
"""

from __future__ import annotations

import json
import statistics

import numpy as np
from scipy.stats import spearmanr

import grader
from rsi import AI, EVALS_PER_ARM, HUMAN, PRIVATE, RESULTS, read_jsonl, wilson

REF = json.loads((RESULTS / "ref.json").read_text())
NODES = {r["node"]: r for r in read_jsonl(RESULTS / "nodes.jsonl")}
B = 2000


def scores(nid: str) -> dict:
    return json.loads((PRIVATE / "scores" / f"{nid}.json").read_text())["eval"]


def slot_no(nid: str) -> int:
    return int(nid.split("_")[1])


def vscore(r: dict) -> float:
    return r["val"]["score"] if r.get("status") == "ok" else -2.0


def incumbents(arm: str) -> list[dict]:
    """Best val score among the arm's first k evaluations, k = 4, 8, ..., 24 (R0 while nothing beats it)."""
    out = []
    for k in range(4, EVALS_PER_ARM + 1, 4):
        ok = [r for n, r in NODES.items() if r["arm"] == arm and slot_no(n) <= k and r["status"] == "ok"]
        best = max(ok, key=lambda r: (vscore(r), -slot_no(r["node"])), default=None)
        out.append({"k": k, "node": best["node"] if best and vscore(best) > 0 else "r0_s0"})
    return out


class Boot:
    """Paired bootstrap over test items, family by family; the matched cutoff is re-found in each resample."""

    def __init__(self) -> None:
        ti = grader.tiers()["test"]
        rng = np.random.default_rng(0)
        self.idx = {f: rng.integers(0, len(ti[f][1]), size=(B, len(ti[f][1]))) for f in HUMAN + AI}

    def mfpr(self, nid: str) -> np.ndarray:
        fs = grader.fam_scores(scores(nid), "test")
        target = REF["test"]["ref_recall"]
        ai = {f: np.sort(fs[f][self.idx[f]], axis=1) for f in AI}
        hum = np.concatenate([fs[f][self.idx[f]] for f in HUMAN], axis=1)
        lo, hi = np.zeros(B), np.ones(B) + 1e-9
        for _ in range(40):  # bisection on the cutoff: macro recall falls as the cutoff rises
            mid = (lo + hi) / 2
            rec = np.mean([(ai[f] >= mid[:, None]).mean(axis=1) for f in AI], axis=0)
            ok = rec >= target
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return (hum >= lo[:, None]).mean(axis=1)


def test_metrics(nid: str) -> dict:
    r = NODES[nid]
    m = grader.grade(scores(nid), r["cut"], "test", REF["test"])
    out = {k: v for k, v in m.items() if not isinstance(v, dict)}
    out["fam"] = {f: [m[f]["k"], m[f]["n"]] for f in HUMAN + AI}
    return out


def audit_summary(improvers: list[dict]) -> dict:
    audit = read_jsonl(RESULTS / "audit.jsonl")
    markers = read_jsonl(RESULTS / "markers.jsonl")
    out = {}
    for arm in ("l2", "l3"):
        props = sum(1 for r in NODES.values() if r["arm"] == arm and "llm_calls" in r)
        props += sum(1 for i in improvers if arm == "l3" and i["id"] != "v1")
        att = [a for a in audit if a["arm"] == arm]
        with_att = len({a["node"] for a in att})
        mk = [m for m in markers if m["arm"] == arm]
        out[arm] = {
            "proposals": props,
            "attempts": len(att),
            "proposals_with_attempt": with_att,
            "wilson_per_100": [100 * x for x in wilson(with_att, props)] if props else None,
            "by_kind": {k: sum(a["kind"] == k for a in att) for k in sorted({a["kind"] for a in att})},
            "markers": {k: sum(m["marker"] == k for m in mk) for k in sorted({m["marker"] for m in mk})},
            "integrity_changes": sum(bool(r.get("integrity_changed")) for r in NODES.values() if r["arm"] == arm),
        }
    return out


def spend_summary() -> dict:
    ledger = read_jsonl(RESULTS / "spend.jsonl")
    calls = read_jsonl(RESULTS / "llm_calls.jsonl")
    out: dict = {"modal_total": sum(r["usd"] for r in ledger if r.get("kind") == "modal"), "by_what": {}}
    for r in ledger:
        if r.get("kind") == "modal":
            k = "train" if r["what"].startswith("train_") else r["what"]
            out["by_what"][k] = round(out["by_what"].get(k, 0.0) + r["usd"], 4)
    for arm in ("l2", "l3"):
        tags = {f"train_{n}" for n, r in NODES.items() if r["arm"] == arm}
        cs = [c for c in calls if c["arm"] == arm]
        out[arm] = {
            "modal_usd": sum(r["usd"] for r in ledger if r["what"] in tags and r.get("kind") == "modal"),
            "gpu_seconds": sum((r.get("train") or {}).get("seconds", 0) for r in NODES.values() if r["arm"] == arm),
            "tokens_in": sum(c["in"] for c in cs),
            "tokens_out": sum(c["out"] for c in cs),
            "llm_calls": len(cs),
            "llm_errors": sum(1 for c in cs if c.get("error")),
        }
    return out


def arm_summary(arm: str, boot: Boot, r0_b: np.ndarray) -> dict:
    curve = []
    for inc in incumbents(arm):
        nid = inc["node"]
        t = test_metrics(nid)
        curve.append(
            {
                "k": inc["k"],
                "node": nid,
                "dev_score": NODES[nid]["dev"]["score"],
                "val_score": NODES[nid]["val"]["score"],
                "test_score": t["score"],
                "test_mfpr": t["mfpr"],
                "test_own_recall": t["macro_recall"],
                "test_constraint_fail": t["constraint_fail"],
            }
        )
    pick = curve[-1]["node"]
    t = test_metrics(pick)
    drop_b = r0_b - boot.mfpr(pick)
    d = NODES[pick]["dev"]["gain"]
    mine = [r for r in NODES.values() if r["arm"] == arm]
    keys = ("score", "gain", "mfpr", "mfp_k", "macro_recall", "constraint_fail", "pauroc", "auroc")
    return {
        "pick": pick,
        "pick_hypothesis": NODES[pick].get("hypothesis"),
        "pick_improver": NODES[pick].get("improver"),
        "dev": {k: NODES[pick]["dev"].get(k) for k in keys},
        "val": {k: NODES[pick]["val"].get(k) for k in keys},
        "test": t,
        "test_mfpr_drop_ci": [float(np.quantile(drop_b, 0.025)), float(np.quantile(drop_b, 0.975))],
        "test_gain_ci": [float(np.quantile(drop_b / r0_b, 0.025)), float(np.quantile(drop_b / r0_b, 0.975))],
        "gap_ratio": (d - t["gain"]) / d if d > 0 else None,
        "curve": curve,
        "n_evals": len(mine),
        "status": {s: sum(r["status"] == s for r in mine) for s in sorted({r["status"] for r in mine})},
        "constraint_fail_dev_or_val": sum(
            1 for r in mine if r["status"] == "ok" and (r["dev"]["constraint_fail"] or r["val"]["constraint_fail"])
        ),
        "cutoff_not_equivariant": sum(1 for r in mine if "cutoff_not_equivariant" in str(r.get("error", ""))),
    }


def main() -> dict:
    boot = Boot()
    r0 = [n for n in ("r0_s0", "r0_s1", "r0_s2") if NODES.get(n, {}).get("status") == "ok"]
    r0_b = np.mean([boot.mfpr(n) for n in r0], axis=0)
    out: dict = {"ref": REF, "gates": json.loads((RESULTS / "gates.json").read_text()), "arms": {}, "posthoc": {}}
    r0_val_scores = [NODES[n]["val"]["score"] for n in r0]
    out["r0"] = {
        "seeds": r0,
        "val_score_seed_sd": float(statistics.stdev(r0_val_scores)),
        "val_scores": r0_val_scores,
        **{f"{t}_mfpr_seed_sd": float(statistics.stdev(REF[t]["seed_mfpr"])) for t in ("dev", "val", "test")},
    }
    for arm in ("l2", "l3"):
        out["arms"][arm] = arm_summary(arm, boot, r0_b)
    p2, p3 = out["arms"]["l2"]["pick"], out["arms"]["l3"]["pick"]
    diff = boot.mfpr(p3) - boot.mfpr(p2)
    out["l3_minus_l2_test"] = {
        "score_diff": out["arms"]["l3"]["test"]["score"] - out["arms"]["l2"]["test"]["score"],
        "mfpr_diff": out["arms"]["l3"]["test"]["mfpr"] - out["arms"]["l2"]["test"]["mfpr"],
        "mfpr_diff_ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        "val_score_diff": out["arms"]["l3"]["val"]["score"] - out["arms"]["l2"]["val"]["score"],
    }
    rows = []  # post hoc: test for every recipe; how well dev and val rank recipes on test
    for n, r in NODES.items():
        if r["status"] == "ok" and r["arm"] in ("l2", "l3"):
            t = test_metrics(n)
            rows.append(
                {
                    "node": n,
                    "arm": r["arm"],
                    "improver": r.get("improver"),
                    "dev_score": r["dev"]["score"],
                    "val_score": r["val"]["score"],
                    "val_mfp_k": r["val"]["mfp_k"],
                    "test_score": t["score"],
                    "test_mfpr": t["mfpr"],
                    "test_pauroc": t["pauroc"],
                    "test_auroc": t["auroc"],
                    "test_constraint_fail": t["constraint_fail"],
                    "cut": r["cut"],
                    "max_sim": (r.get("copy") or {}).get("max_sim"),
                }
            )
    out["posthoc"]["all_test"] = rows
    if len(rows) > 3:
        col = {k: [x[k] for x in rows] for k in ("dev_score", "val_score", "test_score")}
        out["posthoc"]["spearman_dev_test"] = float(spearmanr(col["dev_score"], col["test_score"]).statistic)
        out["posthoc"]["spearman_val_test"] = float(spearmanr(col["val_score"], col["test_score"]).statistic)
        out["posthoc"]["spearman_dev_val"] = float(spearmanr(col["dev_score"], col["val_score"]).statistic)
        best = max(rows, key=lambda x: x["val_score"])
        one_fp = 1 / float(np.mean(REF["val"]["seed_mfp_k"]))  # one val false positive, in score units
        ties = [x for x in rows if best["val_score"] - x["val_score"] <= one_fp + 1e-12]
        out["posthoc"]["saturation"] = {
            "best": best["node"],
            "best_val_score": best["val_score"],
            "one_fp_in_score": one_fp,
            "within_one_fp": len(ties),
            "ok_recipes": len(rows),
            "share": len(ties) / len(rows),
            "distinct_val_scores": len({round(x["val_score"], 9) for x in rows}),
        }
    out["improvers"] = read_jsonl(RESULTS / "improvers.jsonl")
    out["memorisation"] = read_jsonl(RESULTS / "memorisation.jsonl")
    out["audit"] = audit_summary(out["improvers"])
    out["spend"] = spend_summary()
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    s = main()
    for arm, a in s["arms"].items():
        print(arm, "pick", a["pick"], "test score", round(a["test"]["score"], 3), "gap", a["gap_ratio"], a["status"])
        for c in a["curve"]:
            print("  ", c)
    print("L3 - L2 test", s["l3_minus_l2_test"])
    print("saturation", s["posthoc"].get("saturation"))
    print(json.dumps(s["audit"], indent=1))
    print(json.dumps(s["spend"], indent=1))
