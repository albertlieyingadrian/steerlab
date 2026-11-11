"""Score saved runs on one split: every metric, a recording-bootstrap interval on full-record, cost per 1,000 rows,
and paired comparisons (McNemar + recording bootstrap; significant only if both agree) against reference arms.

    score.py dev                         # every dev run in results/runs/
    score.py test --vs qwen-fs5 --out results/test_scores.json
    score.py test --gmaps <their results/test dir>   # adds GMAPS's published arms, scored by this scorer

Cost per 1,000 rows is the ledger cost of the run divided by its rows (list prices; cached input at 10% on OpenAI).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import atc

PUBLISHED = {  # GMAPS's published test predictions -> their table's names
    "atc-nemotron-nano-30b-v1_fwbatch-nothink_fs0": "GMAPS Nemotron-3-Nano-30B + LoRA (1 ep)",
    "anthropic_claude-opus-5.5_fs5": "GMAPS Claude Opus 5.5, 5-shot",
    "anthropic_claude-opus-5.5_fs20": "GMAPS Claude Opus 5.5, 20-shot",
    "atc-qwen3-8b-v2_fireworks_fs0": "GMAPS Qwen3-8B + LoRA v2",
    "qwen3-8b-base_spark_fs5": "GMAPS Qwen3-8B, 5-shot",
    "google_gemini-4-argon_fs5": "GMAPS Gemini 4 Argon, 5-shot",
}


def load_runs(split: str, gmaps_dir: str | None) -> dict[str, dict]:
    runs = {}
    for path in sorted(glob.glob(str(atc.RESULTS / "runs" / f"{split}__*.jsonl"))):
        name = os.path.basename(path)[len(split) + 2 : -6]
        rows = atc.read_jsonl(path)
        runs[name] = {
            "preds": {r["id"]: r.get("output") for r in rows},
            "usd": sum(r.get("usd", 0) for r in rows),
            "n": len(rows),
            "seconds": sorted(r["seconds"] for r in rows if "seconds" in r),
        }
    if gmaps_dir and split == "test":
        for stem, label in PUBLISHED.items():
            rows = atc.read_jsonl(os.path.join(gmaps_dir, f"{stem}.preds.jsonl"))
            runs[label] = {
                "preds": {r["id"]: r.get("output") for r in rows},
                "usd": None,
                "n": len(rows),
                "seconds": [],
            }
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test", "trainval"])
    ap.add_argument("--vs", nargs="*", default=[], help="reference arms for paired comparisons")
    ap.add_argument("--only", nargs="*", help="score only these arms")
    ap.add_argument("--gmaps", help="GMAPS repo results/test directory (published predictions)")
    ap.add_argument("--out")
    a = ap.parse_args()
    gold = atc.load(a.split)
    ids = [g["id"] for g in gold]
    runs = load_runs(a.split, a.gmaps)
    if a.only:
        runs = {k: v for k, v in runs.items() if k in a.only or k in a.vs or k.startswith("GMAPS")}
    table = {}
    for name, r in runs.items():
        rows = atc.score(gold, r["preds"])
        agg = atc.aggregate(rows)
        agg["full_ci"] = atc.cluster_bootstrap(ids, [float(x["full"]) for x in rows])
        agg["usd_per_1k"] = (r["usd"] / r["n"] * 1000) if r["usd"] is not None and r["n"] else None
        agg["p50_s"] = r["seconds"][len(r["seconds"]) // 2] if r["seconds"] else None
        agg["_full"] = [x["full"] for x in rows]
        table[name] = agg

    print(f"{a.split}: n={len(gold)} rows, {len({i.split('#')[0] for i in ids})} recordings\n")
    print("| System | Full record [95% CI] | Clearance-state | Action F₁ | Callsign F₁ | Role | JSON | $/1k |")
    print("|---|---|---|---|---|---|---|---|")
    for name, t in sorted(table.items(), key=lambda kv: -kv[1]["full_record"]):
        lo, hi = t["full_ci"]
        usd = "—" if t["usd_per_1k"] is None else f"${t['usd_per_1k']:.2f}"
        print(
            f"| {name} | {100 * t['full_record']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] | "
            f"{100 * t['clearance_state']:.1f}% (n={t['clearance_n']}) | {100 * t['action_f1']:.1f} | "
            f"{100 * t['callsign_f1']:.1f} | {100 * t['role_acc']:.1f} | {100 * t['json_valid']:.0f}% | {usd} |"
        )

    comparisons = []
    for ref in a.vs:
        if ref not in table:
            continue
        print(f"\nPaired vs {ref} (full record):\n")
        print("| System | Gap | only system / only ref right | McNemar p | Recording bootstrap 95% CI | Verdict |")
        print("|---|---|---|---|---|---|")
        for name, t in table.items():
            if name == ref:
                continue
            c = atc.compare(ids, t["_full"], table[ref]["_full"])
            comparisons.append({"system": name, "ref": ref, **c})
            verdict = (
                ("significant, system ahead" if c["gap"] > 0 else "significant, ref ahead")
                if c["significant"]
                else "tie (not significant)"
            )
            print(
                f"| {name} | {100 * c['gap']:+.1f} | {c['only_a']} / {c['only_b']} | {c['p']:.2g} | "
                f"[{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}] | {verdict} |"
            )
    if a.out:
        clean = {k: {m: v for m, v in t.items() if m != "_full"} for k, t in table.items()}
        atc.Path(a.out).write_text(json.dumps({"split": a.split, "arms": clean, "comparisons": comparisons}, indent=1))


if __name__ == "__main__":
    main()
