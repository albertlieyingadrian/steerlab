"""B1 judgement (validation only), as pre-registered in docs/plans/2026-11-12-beat-voi-levers.md.

    python eval_b1.py --base v2-1.7b --arms b1a-1.7bA b1b-1.7bB     # stage 1
    python eval_b1.py --base v2-4b --arms b1x-4bX                    # stage 2

(i)  EditLens val (2,396 rows), cut-offs cross-fitted over 5 folds grouped by source document: primary.
(ii) EditLens val at fixed cut-offs: 0.15 / 0.75 for one-number heads, 1/6 / 5/6 for bucket heads: reported.
(iii) Our val (data/v2b/val.jsonl, classes by how the text was made), cross-fitted cut-offs grouped by document.
Decision: an arm passes if its paired macro-F1 difference against the base is above 0 on (i) and (iii) with the 95%
interval's lower end above -0.005 on both; if both pass, the higher on (i) goes on. Guard: at the refitted flag cut
(1% of our validation humans), long-form humans flagged must not exceed the base's. Also reported: EditLens-val
edited->human errors, light-quarter misses (EditLens cosine below its edited-row 25th percentile), human->edited.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module

import numpy as np
from common import HERE, RESULTS, read_jsonl

tev, diag, win = import_module("tune_editlens_val"), import_module("diag_voi_gap"), import_module("windows")


def cfg(run: str) -> dict:
    return json.loads((HERE / "runs" / run / "config.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", default="", help="results/<out>.json (default b1_stage1 or b1_stage2)")
    args = ap.parse_args()
    ev, _ = tev.clean_val()
    ye, ge = ev["y"].to_numpy(), ev["source_id"].fillna(ev["text_id"]).to_numpy()
    cos = ev["cosine_score"].to_numpy()
    light_cut = float(np.quantile(cos[ye == 1], 0.25))
    ov = read_jsonl(HERE / "data" / "v2b" / "val.jsonl")
    yo, go = np.array([r["band"] for r in ov]), np.array([r["doc_id"] for r in ov])
    human = np.array([r["kind"] == "human" for r in ov])
    long_h = human & np.array([r["source"].endswith("_long") or r["source"] == "pg19" for r in ov])
    runs = [args.base, *args.arms]
    preds, report = {}, {"base": args.base, "light_cut_cosine": round(light_cut, 4), "runs": {}}
    for run in runs:
        se = np.array(json.loads((HERE / "runs" / run / "scores" / "editlens_val.json").read_text()))
        so = np.array(json.loads((HERE / "runs" / run / "val_scores.json").read_text()))
        assert len(se) == len(ye) and len(so) == len(yo), run
        fixed = (1 / 6, 5 / 6) if cfg(run).get("head") == "buckets" else (0.15, 0.75)
        pe, po, pf = win.crossfit(ye, se, ge), win.crossfit(yo, so, go), diag.bands(se, *fixed)
        flag = float(np.quantile(so[human & ~long_h], 0.99))
        preds[run] = (pe, po)
        report["runs"][run] = {
            "editlens_val_crossfit": round(diag.mf1(ye, pe), 4),
            "editlens_val_fixed": round(diag.mf1(ye, pf), 4),
            "fixed_cuts": [round(fixed[0], 4), round(fixed[1], 4)],
            "our_val_crossfit": round(diag.mf1(yo, po), 4),
            "edited_called_human": int(((ye == 1) & (pe == 0)).sum()),
            "light_quarter_missed": round(float((pe[(ye == 1) & (cos < light_cut)] != 1).mean()), 4),
            "human_called_edited": int(((ye == 0) & (pe == 1)).sum()),
            "long_humans_flagged": round(float((so[long_h] >= flag).mean()), 4),
        }
    base_e, base_o = preds[args.base]
    passing = []
    for run in args.arms:
        pe, po = preds[run]
        r = report["runs"][run]
        r["vs_base_editlens"] = {
            "diff": round(diag.mf1(ye, pe) - diag.mf1(ye, base_e), 4),
            "ci": diag.boot(ye, pe, base_e),
        }
        r["vs_base_our_val"] = {
            "diff": round(diag.mf1(yo, po) - diag.mf1(yo, base_o), 4),
            "ci": diag.boot(yo, po, base_o),
        }
        guard = r["long_humans_flagged"] <= report["runs"][args.base]["long_humans_flagged"]
        ok = (r["vs_base_editlens"]["diff"] > 0 and r["vs_base_our_val"]["diff"] > 0
              and r["vs_base_editlens"]["ci"][0] > -0.005 and r["vs_base_our_val"]["ci"][0] > -0.005)  # fmt: skip
        r["passes"], r["guard_ok"] = bool(ok), bool(guard)
        if ok and guard:
            passing.append(run)
    report["winner"] = max(passing, key=lambda k: report["runs"][k]["editlens_val_crossfit"]) if passing else None
    for run, r in report["runs"].items():
        extra = ""
        if run != args.base:
            extra = (f"  vs base EL {r['vs_base_editlens']['diff']:+.4f} {r['vs_base_editlens']['ci']}"
                     f"  ours {r['vs_base_our_val']['diff']:+.4f} {r['vs_base_our_val']['ci']}"
                     f"  pass={r['passes']} guard={r['guard_ok']}")  # fmt: skip
        print(f"{run:12} EL cf {r['editlens_val_crossfit']:.4f}  EL fixed {r['editlens_val_fixed']:.4f}  "
              f"ours cf {r['our_val_crossfit']:.4f}  ed->hum {r['edited_called_human']}  light miss "
              f"{r['light_quarter_missed']:.3f}  hum->ed {r['human_called_edited']}  long-hum flag "
              f"{r['long_humans_flagged']:.3f}{extra}")  # fmt: skip
    print("winner:", report["winner"])
    name = args.out or ("b1_stage1" if "1.7" in args.base else "b1_stage2")
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
