"""Stage (b) gate: did data v2 help, judged on validation only (the benchmarks stay untouched)?

    python compare_val.py --old pilotR-1.7b --new v2-1.7b

Both runs are 1.7B models with the same recipe; only the training data differs. Validation rows come from
data/val.jsonl (v2), which contains every pilot validation document. Scores for the new run come from its training
job; the old run scores the v2 validation texts on Modal (score_run), since it never saw them.

Decision rule, fixed before either v2 result was seen (2026-11-11):
  go to the 4B run if (1) new beats old on all v2 validation rows (paired 95% CI of the macro-F1 difference above 0)
  AND (2) new does not lose on the pilot's own validation rows (CI lower bound above -0.01).
Also reported, not part of the rule: recall of the edited band on light edits (target 0.15-0.35), and the share of
long-form human texts (Medium long, XSum long, PG-19) flagged at each model's own 1%-of-validation-humans cut.
Writes results/compare_val.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from common import DATA, HERE, RESULTS, band, empty_fences, paired_boot, read_jsonl

KEYS = ("macro_f1", "f1_human", "f1_edited", "f1_generated")


def main() -> None:
    import modal

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="pilotR-1.7b")
    ap.add_argument("--new", default="v2-1.7b")
    args = ap.parse_args()
    val = read_jsonl(DATA / "val.jsonl")
    new = np.array(json.loads((HERE / "runs" / args.new / "val_scores.json").read_text()))
    cache = HERE / "runs" / args.old / "scores" / "v2_val.json"
    if cache.exists():
        old = np.array(json.loads(cache.read_text()))
    else:
        fn = modal.Function.from_name("slop-detector", "score_run")
        old = np.array(fn.remote(args.old, {"val": [empty_fences(r["text"]) for r in val]})["val"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([round(float(x), 6) for x in old]) + "\n")
    assert len(old) == len(new) == len(val)
    y = np.array([r["band"] for r in val])
    pilot_texts = {r["text"] for r in read_jsonl(DATA / "pilot_splits" / "val.jsonl")}
    in_pilot = np.array([r["text"] in pilot_texts for r in val])
    tgt = np.array([r["target"] for r in val])
    light = (tgt >= 0.15) & (tgt < 0.35)
    human = np.array([r["kind"] == "human" for r in val])
    long_h = human & np.array([r["source"].endswith("_long") or r["source"] == "pg19" for r in val])
    out = {
        "n": len(val),
        "n_pilot_rows": int(in_pilot.sum()),
        "n_light": int(light.sum()),
        "n_long_human": int(long_h.sum()),
        "all": paired_boot(y, band(new), band(old), KEYS),
        "pilot_rows": paired_boot(y[in_pilot], band(new[in_pilot]), band(old[in_pilot]), KEYS),
    }
    out["light_edit_recall"] = {
        "new": float((band(new[light]) == 1).mean()),
        "old": float((band(old[light]) == 1).mean()),
    }
    for name, s in (("new", new), ("old", old)):
        cut = float(np.quantile(s[human & ~long_h], 0.99))
        out[f"long_human_flagged_{name}"] = float((s[long_h] >= cut).mean())
    a, p = out["all"]["macro_f1"], out["pilot_rows"]["macro_f1"]
    out["go"] = bool(a["diff_ci"][0] > 0 and p["diff_ci"][0] > -0.01)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "compare_val.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
