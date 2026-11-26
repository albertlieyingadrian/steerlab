"""Phase 2 aggregates for the note: switch-or-readout, subcircuit explanations and descriptions (CPU, seconds).

.venv/bin/python experiments/neurons-basis/summarize_phase2.py
"""

# ruff: noqa: E501  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NEURONS_BASIS_WORK", Path.home() / ".cache" / "steerlab" / "neurons-basis"))
RESULTS = HERE / "results"


def validate():
    out = {"sets": {}}
    for name in ("validate_v1.json", "validate.json"):
        p = WORK / "organism" / name
        if not p.exists():
            continue
        v = json.loads(p.read_text())
        for k in ("base_nopw", "base_pw"):
            out[k] = {a: b for a, b in v[k].items() if a != "chosen"}
        for s, d in v["sets"].items():
            out["sets"][s] = {e: {a: b for a, b in x.items() if a != "chosen"} for e, x in d.items()}
    (RESULTS / "phase2_validate.json").write_text(json.dumps(out, indent=1))
    base = out["base_nopw"]["accuracy"]
    print(f"validate: no-password accuracy {base:.3f}, with password {out['base_pw']['accuracy']:.3f}")
    for s in ("monitor_top5", "L16_monitor_all", "l20_top5", "L20_monitor_all", "L16_pwdiff_top5", "L16_pwdiff_top50",
              "L16_pwdiff_top500", "L16_all", "L20_pwdiff_top500", "L20_all"):  # fmt: skip
        if s in out["sets"]:
            d = out["sets"][s]
            print(
                f"  {s:<20} "
                + "  ".join(f"{e} {x['accuracy']:.3f} (P correct {x['p_correct']:.3f})" for e, x in d.items())
            )
    return out


def explain():
    p = WORK / "organism" / "explain.json"
    if not p.exists():
        return None
    e = json.loads(p.read_text())
    per = defaultdict(lambda: defaultdict(list))
    eff = defaultdict(list)
    acts = defaultdict(lambda: defaultdict(list))
    tops = defaultdict(list)
    for x in e:
        cond = x["condition"] + ("/sandbagged" if x["label"] == 1 else "/answered")
        for seg, v in x["by_segment"].items():
            per[x["neuron"]][seg].append(v)
        eff[x["neuron"]].append(abs(x["effect_correct_minus_others"]))
        acts[x["neuron"]][cond].append(x["activation"])
        tops[x["neuron"]].append(x["top_sources"][:3])
    out = {}
    for n in per:
        out[n] = {
            "by_segment_mean": {s: round(statistics.mean(v), 4) for s, v in sorted(per[n].items())},
            "abs_letter_effect_mean": round(statistics.mean(eff[n]), 5),
            "activation_mean_by_condition": {c: round(statistics.mean(v), 4) for c, v in sorted(acts[n].items())},
            "example_top_sources": tops[n][:2],
        }
    summary = {
        "subcircuits": len(e),
        "completeness": [min(x["completeness"] for x in e), max(x["completeness"] for x in e)],
        "neurons": out,
    }
    (RESULTS / "phase2_explain.json").write_text(json.dumps(summary, indent=1))
    print(f"explain: {len(e)} subcircuits, completeness {summary['completeness']}")
    for n, d in out.items():
        print(f"  {n:<10} |letter effect| {d['abs_letter_effect_mean']:.4f}  segments {d['by_segment_mean']}")
    return summary


def descriptions():
    p = WORK / "organism_examples" / "judge_out.json"
    q = WORK / "organism_examples" / "neurons_judge.json"
    if not p.exists():
        return None
    outs = json.loads(p.read_text())
    poles = json.loads(q.read_text())
    rows = []
    for o, pole in zip(outs, poles, strict=True):
        t, pr = o["truth"], o["probs"]
        acc = None
        if pr is not None and 0 < sum(t) < len(t):
            pos = [a >= 0.5 for a, y in zip(pr, t, strict=True) if y]
            neg = [a < 0.5 for a, y in zip(pr, t, strict=True) if not y]
            acc = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2
        rows.append({"pole": f"{o['neuron']}{'+' if o['sign'] > 0 else '-'}", "description": o["description"], "predict_acc": acc,
                     "n_test_tails": pole["n_test_tails"]})  # fmt: skip
    (RESULTS / "phase2_descriptions.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"  {r['pole']:<12} acc {r['predict_acc']}  {r['description'][:110]!r}")
    return rows


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    validate()
    explain()
    descriptions()
