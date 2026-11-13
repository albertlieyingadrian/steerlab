"""The final model: choose it on validation only, then take the ONE pre-registered look at the frozen fresh test.

Rule fixed in docs/plans/2026-11-12-slop-detector-b3-prereg.md before the final models trained.

    python final_look.py select                                  # validation only -> results/final_choice.json
    MODAL_PROFILE=... python final_look.py score --runs f9-9b     # chosen runs: fresh + look-5 texts, on Modal
    python final_look.py look                                    # fresh look 1 (+ look 5) -> results/eval/final.json

Candidates: 9B alone, 4B seed 0 alone, the mean of the two 4B seeds, the mean of 9B and 4B seed 0. Chosen by
EditLens-val macro-F1 with cut-offs cross-fitted over 5 folds grouped by source document; ties within 0.002 go to
our val, then to the simpler candidate.
Arms: fixed (ours at the bucket head's 1/6 and 5/6, voi at its card's 0.15 and 0.75) and calibrated (both models' two
cut-offs fitted on EditLens val by the same grid). Claim a win only if the paired 95% interval on macro-F1 is above
zero; beside it, an exact two-sided sign test on the items where exactly one model is right.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module

import numpy as np
from common import (
    BANDS,
    HERE,
    RESULTS,
    benchmark,
    empty_fences,
    paired_boot,
    read_jsonl,
)

tev, diag, win = import_module("tune_editlens_val"), import_module("diag_voi_gap"), import_module("windows")
RUN_4B, RUN_4B2, RUN_9B = "b1x-4bk", "f4s2-4bs2", "f9-9b"
CANDIDATES = {  # name: runs averaged (simplest first)
    "4B": [RUN_4B],
    "9B": [RUN_9B],
    "4B seeds mean": [RUN_4B, RUN_4B2],
    "9B + 4B mean": [RUN_9B, RUN_4B],
}
FIXED = (1 / 6, 5 / 6)
LOOK5 = ("test", "test_enron", "test_llama", "human_detectors", "epoch")
KEYS = ("macro_f1", "acc", "f1_human", "f1_edited", "f1_generated")


def scores(runs: list[str], split: str) -> np.ndarray:
    def load(r: str) -> np.ndarray:
        name = "val_scores.json" if split == "our_val" else f"scores/{split}.json"
        return np.array(json.loads((HERE / "runs" / r / name).read_text()))

    return np.mean([load(r) for r in runs], axis=0)


def select(_args) -> None:
    ev, _ = tev.clean_val()
    ye, ge = ev["y"].to_numpy(), ev["source_id"].fillna(ev["text_id"]).to_numpy()
    ov = read_jsonl(HERE / "data" / "v2b" / "val.jsonl")
    yo, go = np.array([r["band"] for r in ov]), np.array([r["doc_id"] for r in ov])
    res = {}
    for name, runs in CANDIDATES.items():
        se, so = scores(runs, "editlens_val"), scores(runs, "our_val")
        res[name] = {
            "runs": runs,
            "editlens_val_crossfit": round(diag.mf1(ye, win.crossfit(ye, se, ge)), 4),
            "editlens_val_fixed": round(diag.mf1(ye, diag.bands(se, *FIXED)), 4),
            "our_val_crossfit": round(diag.mf1(yo, win.crossfit(yo, so, go)), 4),
        }
        print(f"{name:15} {res[name]}")
    top = max(r["editlens_val_crossfit"] for r in res.values())
    near = [k for k, r in res.items() if r["editlens_val_crossfit"] >= top - 0.002]
    best_o = max(res[k]["our_val_crossfit"] for k in near)
    choice = next(k for k in CANDIDATES if k in near and res[k]["our_val_crossfit"] == best_o)
    out = {"candidates": res, "within_0.002_of_top": near, "choice": choice, "runs": CANDIDATES[choice]}
    (RESULTS / "final_choice.json").write_text(json.dumps(out, indent=1) + "\n")
    print("choice:", choice)


def score(args) -> None:
    import modal

    evals = {s: [empty_fences(str(t)) for t in benchmark(s)["text"]] for s in LOOK5}
    fresh = import_module("fresh_test").fresh_rows()
    evals["fresh"] = [empty_fences(str(t)) for t in fresh["text"]]
    fn = modal.Function.from_name("slop-detector", "score_run")
    calls = {r: fn.spawn(r, evals) for r in args.runs}
    for r, c in calls.items():
        out = c.get()
        for split, s in out.items():
            (HERE / "runs" / r / "scores" / f"{split}.json").write_text(json.dumps(s) + "\n")
        print(r, "scored", list(out))


def sign_test(y, a, b) -> dict:
    from scipy.stats import binomtest

    ra, rb = a == y, b == y
    wins, losses = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return {"ours_right_only": wins, "voi_right_only": losses, "p_two_sided": float(f"{p:.3g}")}


def voi(split: str) -> np.ndarray:
    d = json.loads((RESULTS / "voi" / f"{split}.json").read_text())
    return np.array(d["scores"] if "scores" in d else [r["score"] for r in d["rows"]])


def look(_args) -> None:
    choice = json.loads((RESULTS / "final_choice.json").read_text())
    runs = choice["runs"]
    ev, _ = tev.clean_val()
    ye = ev["y"].to_numpy()
    _, oh, og = diag.fit_cuts(ye, scores(runs, "editlens_val"))
    _, vh, vg = diag.fit_cuts(ye, voi("editlens_val"))
    val_path, prev = HERE / "data" / "v2b" / "val.jsonl", RESULTS / "eval" / "final.json"
    if val_path.exists():  # the flag cut flags 1% of our validation humans (lesson 05)
        ov = read_jsonl(val_path)
        flag = float(np.quantile(scores(runs, "our_val")[[r["kind"] == "human" for r in ov]], 0.99))
    else:  # the public copy has no validation texts: reuse the saved cut
        flag = json.loads(prev.read_text())["flag_min_ours"]
    out = {
        "choice": choice["choice"],
        "runs": runs,
        "cuts": {
            "ours_fixed": FIXED,
            "voi_fixed": (0.15, 0.75),
            "ours_calibrated": (oh, og),
            "voi_calibrated": (vh, vg),
        },
        "flag_min_ours": round(flag, 6),
        "looks": {},
    }
    fresh = import_module("fresh_test").fresh_rows()
    from common import EDITLENS_CLASS

    sets = {"fresh (look 1, the claim)": (fresh["text_type"].map(EDITLENS_CLASS).to_numpy(), "fresh")}
    sets.update({f"{s} (look 5)": (benchmark(s)["y"].to_numpy(), s) for s in ("test", "test_enron", "test_llama")})
    for name, (y, split) in sets.items():
        so, sv = scores(runs, split), voi(split)
        arms = {
            "fixed": (diag.bands(so, *FIXED), diag.bands(sv, 0.15, 0.75)),
            "calibrated": (diag.bands(so, oh, og), diag.bands(sv, vh, vg)),
        }
        out["looks"][name] = {}
        for arm, (a, b) in arms.items():
            p = paired_boot(y, a, b, KEYS, reps=2000)
            p["sign_test"] = sign_test(y, a, b)
            p["claim"] = bool(p["macro_f1"]["diff_ci"][0] > 0)
            out["looks"][name][arm] = p
            m = p["macro_f1"]
            print(
                f"{name:28} {arm:10} ours {m['a']:.3f} vs voi {m['b']:.3f}: {m['diff']:+.4f} {m['diff_ci']} "
                f"sign {p['sign_test']}  claim={p['claim']}"
            )
    for split in ("human_detectors", "epoch"):
        df = benchmark(split)
        y, so, sv = df["label"].to_numpy(), scores(runs, split), voi(split)
        fo, fv = so >= flag, sv >= BANDS["flag_min"]
        r = {
            "humans": int((y == 0).sum()),
            "humans_flagged_ours": int((fo & (y == 0)).sum()),
            "humans_flagged_voi": int((fv & (y == 0)).sum()),
            "ai": int((y == 1).sum()),
            "ai_missed_ours": int((~fo & (y == 1)).sum()),
            "ai_missed_voi": int((~fv & (y == 1)).sum()),
        }
        out["looks"][f"{split} (look 5)"] = r
        print(split, r)
    (RESULTS / "eval" / "final.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["select", "score", "look"])
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args()
    {"select": select, "score": score, "look": look}[args.step](args)


if __name__ == "__main__":
    main()
