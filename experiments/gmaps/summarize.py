"""Turn the private run files into the public results: scores, paired tests and per-row verdicts, no transcript text.

    python experiments/gmaps/summarize.py --gmaps <GMAPS repo>/results/test

Reads results/runs/<split>__<system>.jsonl (model outputs; private, because outputs quote ATCO2 transcripts) and the
GMAPS project's published test predictions, and writes:
  results/summary.json   every system's dev and test scores, recording-bootstrap intervals, cost per 1,000
                         transmissions, the paired comparisons quoted in the note, and the role-rule check
  results/rows_test.json per test row: recording, airport, gold role, and per system whether the full record, the
                         role, the callsigns and the action set were right (booleans only)
The ATCO2 data itself is not redistributed (research licence); `atc.py` rebuilds the split from atco2.org.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import atc

# (run name, label, group, test look) in the order the note's tables use
SYSTEMS = [
    ("q14-pfr-epoch1", "Qwen3-14B + LoRA, Prague-free + read-backs (final)", "ours", "look 2"),
    ("q14-pfa-epoch1", "Qwen3-14B + LoRA, Prague-free, no read-backs", "ours", "look 1"),
    ("nemo-v1-e2", "Nemotron-3.5-Lightning-30B + LoRA, pilot mix (Prague data)", "pilot", None),
    ("v2-e2", "Qwen3-8B + LoRA, pilot mix (Prague data)", "pilot", None),
    ("claude-fable-5-2-medium-fs20", "Claude Fable 5.2, 20-shot", "frontier", None),
    ("claude-fable-5-2-medium-fs5", "Claude Fable 5.2, 5-shot", "frontier", None),
    ("claude-opus-5-5-medium-fs20", "Claude Opus 5.5, 20-shot", "frontier", None),
    ("claude-opus-5-5-medium-fs5", "Claude Opus 5.5, 5-shot", "frontier", None),
    ("claude-fable-5-1-low-fs20", "Claude Fable 5.1, 20-shot", "frontier", None),
    ("claude-fable-5-1-low-fs5", "Claude Fable 5.1, 5-shot", "frontier", None),
    ("gpt-5.4-medium-fs20", "gpt-5.4, 20-shot", "frontier", None),
    ("gpt-5.4-medium-fs5", "gpt-5.4, 5-shot", "frontier", None),
    ("gpt-5.4-none-fs20", "gpt-5.4, no reasoning, 20-shot", "frontier", None),
    ("gpt-5.4-mini-medium-fs20", "gpt-5.4-mini, 20-shot", "frontier", None),
    ("gpt-5.4-mini-medium-fs5", "gpt-5.4-mini, 5-shot", "frontier", None),
    ("gpt-5.4-mini-none-fs20", "gpt-5.4-mini, no reasoning, 20-shot", "frontier", None),
    ("qwen-fs5", "Qwen3-8B, 5-shot (no training)", "open", None),
    ("gpt-5.4-medium-fs20-role", "gpt-5.4, 20-shot + role rule", "harness", None),
    ("qwen-fs5-role", "Qwen3-8B, 5-shot + role rule", "harness", None),
    ("v2-e2-role", "Qwen3-8B + LoRA (pilot) + role rule", "harness", None),
]
GMAPS = {  # their published prediction files -> label
    "atc-nemotron-nano-30b-v1_fwbatch-nothink_fs0": "GMAPS: Nemotron-3-Nano-30B + LoRA",
    "anthropic_claude-opus-5.5_fs5": "GMAPS: Claude Opus 5.5, 5-shot",
    "anthropic_claude-opus-5.5_fs20": "GMAPS: Claude Opus 5.5, 20-shot",
    "google_gemini-4-argon_fs5": "GMAPS: Gemini 4 Argon, 5-shot",
    "atc-qwen3-8b-v2_fireworks_fs0": "GMAPS: Qwen3-8B + LoRA v2",
    "qwen3-8b-base_spark_fs5": "GMAPS: Qwen3-8B, 5-shot",
}
DEV = [
    "q14-pfr-e1",
    "q14-pfr-e2",
    "q14-pfr-rft",
    "q14-pfa-e1",
    "q14-pfa-e2",
    "nemo-pfa-e1",
    "nemo-pfa-e2",
    "nemo-v1-e2",
    "v2-e2",
    "claude-opus-5-5-medium-fs5",
    "claude-opus-5-5-medium-fs20",
    "claude-fable-5-2-medium-fs20",
    "claude-fable-5-2-low-fs20",
    "claude-fable-5-2-medium-fs5",
    "claude-fable-5-2-low-fs5",
    "gpt-5.4-medium-fs20",
    "gpt-5.4-mini-medium-fs20",
    "qwen-fs5",
    "q397-fs20",
]
PAIRS = [  # (system, reference) for the significance table
    ("q14-pfr-epoch1", "GMAPS: Nemotron-3-Nano-30B + LoRA"),
    ("q14-pfr-epoch1", "claude-fable-5-2-medium-fs20"),
    ("q14-pfr-epoch1", "claude-opus-5-5-medium-fs20"),
    ("q14-pfr-epoch1", "GMAPS: Claude Opus 5.5, 5-shot"),
    ("q14-pfr-epoch1", "q14-pfa-epoch1"),
    ("q14-pfr-epoch1", "nemo-v1-e2"),
    ("q14-pfr-epoch1", "qwen-fs5"),
    ("q14-pfa-epoch1", "claude-fable-5-2-medium-fs20"),
    ("q14-pfa-epoch1", "GMAPS: Nemotron-3-Nano-30B + LoRA"),
    ("nemo-v1-e2", "GMAPS: Nemotron-3-Nano-30B + LoRA"),
    ("v2-e2", "GMAPS: Qwen3-8B + LoRA v2"),
    ("claude-opus-5-5-medium-fs5", "GMAPS: Claude Opus 5.5, 5-shot"),
    ("gpt-5.4-medium-fs20-role", "gpt-5.4-medium-fs20"),
    ("claude-fable-5-2-medium-fs20", "gpt-5.4-medium-fs20"),
]


def preds_of(split: str, name: str) -> tuple[dict, float | None]:
    rows = atc.read_jsonl(atc.RESULTS / "runs" / f"{split}__{name}.jsonl")
    usd = sum(r.get("usd", 0) for r in rows)
    return {r["id"]: r.get("output") for r in rows}, (usd / len(rows) * 1000 if usd else None)


def metrics(gold: list[dict], scored: list[dict]) -> dict:
    ids = [g["id"] for g in gold]
    agg = atc.aggregate(scored)
    lo, hi = atc.cluster_bootstrap(ids, [float(r["full"]) for r in scored])
    by_role = {
        R: atc.mean(r["role_ok"] for g, r in zip(gold, scored, strict=False) if g["role"] == R)
        for R in ("ATCO", "PILOT")
    }
    out = {k: round(v, 4) for k, v in agg.items() if isinstance(v, float)}
    out.update(
        {
            "n": agg["n"],
            "clearance_n": agg["clearance_n"],
            "full_ci": [round(lo, 4), round(hi, 4)],
            "role_acc_atco": round(by_role["ATCO"], 4),
            "role_acc_pilot": round(by_role["PILOT"], 4),
        }
    )
    return out


def role_rule_loao() -> list[dict]:
    """Fit the rule's direction on 3 train/dev airports (gold callsigns), score where it fires on the 4th."""
    rows = [
        r
        for r in atc.read_jsonl(atc.SPLITS / "train.jsonl") + atc.read_jsonl(atc.SPLITS / "dev.jsonl")
        if r["role"] in ("ATCO", "PILOT")
    ]

    def pos(r):
        t = atc.norm(r["text"])
        for c in (atc.norm(x) for x in r["gold"]["callsign"]):
            if c and t.startswith(c):
                return "start"
            if c and t.endswith(c):
                return "end"
        return None

    out = []
    for ap in sorted({r["airport"] for r in rows}):
        c = collections.Counter((pos(r), r["role"]) for r in rows if r["airport"] != ap)
        rule = {p: max(("ATCO", "PILOT"), key=lambda R, p=p: c[(p, R)]) for p in ("start", "end")}
        held = [r for r in rows if r["airport"] == ap]
        fired = [r for r in held if pos(r)]
        out.append(
            {
                "airport": ap,
                "rule": rule,
                "rows": len(held),
                "fires": len(fired),
                "correct": round(atc.mean(rule[pos(r)] == r["role"] for r in fired), 4),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmaps", required=True, help="GMAPS repo results/test directory")
    a = ap.parse_args()
    test, dev = atc.load("test"), atc.load("dev")
    ids = [g["id"] for g in test]
    full, systems, rows = {}, [], {}
    for name, label, group, look in SYSTEMS:
        preds, usd = preds_of("test", name)
        scored = atc.score(test, preds)
        full[name], rows[name] = [r["full"] for r in scored], scored
        systems.append(
            {
                "name": name,
                "label": label,
                "group": group,
                "look": look,
                "usd_per_1k": usd,
                "test": metrics(test, scored),
            }
        )
    for stem, label in GMAPS.items():
        preds = {r["id"]: r["output"] for r in atc.read_jsonl(os.path.join(a.gmaps, f"{stem}.preds.jsonl"))}
        scored = atc.score(test, preds)
        full[label], rows[label] = [r["full"] for r in scored], scored
        systems.append(
            {
                "name": label,
                "label": label,
                "group": "gmaps",
                "look": None,
                "usd_per_1k": None,
                "test": metrics(test, scored),
            }
        )
    dev_scores = {}
    for name in DEV:
        preds, usd = preds_of("dev", name)
        dev_scores[name] = {**metrics(dev, atc.score(dev, preds)), "usd_per_1k": usd}
    comparisons = []
    for s, ref in PAIRS:
        c = atc.compare(ids, full[s], full[ref])
        comparisons.append(
            {
                "system": s,
                "ref": ref,
                "gap": round(c["gap"], 4),
                "only_system": c["only_a"],
                "only_ref": c["only_b"],
                "p": c["p"],
                "ci": [round(x, 4) for x in c["ci"]],
                "significant": c["significant"],
            }
        )
    summary = {
        "test_n": len(test),
        "test_recordings": len({i.split("#")[0] for i in ids}),
        "dev_n": len(dev),
        "test_roles": dict(collections.Counter(g["role"] for g in test)),
        "systems": systems,
        "dev": dev_scores,
        "comparisons": comparisons,
        "role_rule_loao": role_rule_loao(),
        "rft": json.loads((atc.RESULTS / "kaggle" / "gmaps-q14-pfr-rft" / "rft_summary.json").read_text()),
    }
    (atc.RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    per_row = [
        {
            "id": g["id"],
            "airport": g["airport"],
            "role": g["role"],
            "systems": {
                k: {
                    "full": v[i]["full"],
                    "role": v[i]["role_ok"],
                    "callsigns": v[i]["cs_ok"],
                    "actions": v[i]["act_ok"],
                }
                for k, v in rows.items()
            },
        }
        for i, g in enumerate(test)
    ]
    (atc.RESULTS / "rows_test.json").write_text(json.dumps(per_row))
    for s in systems:
        t = s["test"]
        print(
            f"{s['label'][:60]:60s} {100 * t['full_record']:5.1f} [{100 * t['full_ci'][0]:.1f}, "
            f"{100 * t['full_ci'][1]:.1f}] role {100 * t['role_acc']:.1f}"
        )
    for c in comparisons:
        print(
            f"{c['system']} vs {c['ref']}: {100 * c['gap']:+.1f} {c['only_system']}/{c['only_ref']} p={c['p']:.2g} "
            f"[{100 * c['ci'][0]:+.1f},{100 * c['ci'][1]:+.1f}] {'SIG' if c['significant'] else 'ns'}"
        )


if __name__ == "__main__":
    main()
