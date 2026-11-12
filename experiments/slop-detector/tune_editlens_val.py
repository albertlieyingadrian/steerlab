"""Secondary arm: fit the band cut-offs (and the 1.7B/4B weight) on EditLens's own validation split. No training.

    python tune_editlens_val.py score    # our two models on EditLens val (local GPU or MPS; adapters from the run dirs)
    python tune_editlens_val.py select   # EditLens val only: writes results/tuned_editlens_val_choice.json
    python tune_editlens_val.py test     # the one test look: results/eval/tuned_editlens_val.json
    python tune_editlens_val.py control  # voi given the same calibration: results/eval/voi_editlens_val.json

Pre-registered 2026-11-12, committed before `select` or `test` was run:

- Why: our validation bands come from "share of words changed"; EditLens labels a text by how it was made. Fitting the
  cut-offs on EditLens-labelled text answers "does fixing that label mismatch close the gap?". It is NOT a like-for-like
  comparison with voi-slop-detect-4B, which (as far as its card says) used fixed cut-offs and was not calibrated on
  EditLens val. Reported only as "with EditLens-val calibration", never as the headline.
- Data: EditLens val (2,400 texts), minus every row that shares a text id, a text or a source text with any EditLens
  test split (4 rows: one human FineWeb-Edu text and its sibling, one news story's two AI versions), leaving 2,396.
  Nothing is trained on it; only three numbers are chosen.
- Candidates and selection: as in tune_bands.py. score = w * s_4B + (1 - w) * s_1.7B, w in {0, 0.1, ..., 1.0}; human
  cut h in [0.05, 0.40], generated cut g in [0.50, 0.95], steps of 0.005, h < g; the highest EditLens-val 3-way
  macro-F1 wins; ties go to the candidate closest to (w=1, h=0.15, g=0.75). The flag cut for the yes/no benchmarks stays
  at 1% of OUR validation humans, refitted on the chosen score.
- This is TEST LOOK #4 at EditLens test, Enron, Llama, Human Detectors and Epoch AI.
- Control, added before it was run (after test look #4, so it is not part of that look's choice): the same calibration
  for voi-slop-detect-4B. Its scores on the same 2,396 EditLens val rows; its two cut-offs fitted by the same grid and
  rule (no weight to fit); one look at the three EditLens test splits. If voi gains as much, the arm's gain is the
  calibration, not our model; the like-for-like comparison is then ours-calibrated against voi-calibrated.
- Claim rule: "beats voi-slop-detect-4B (with EditLens-val calibration)" only if the paired 95% bootstrap interval of
  the macro-F1 difference is above zero. EditLens's 24B cosine model (0.904, its card's number, calibrated on EditLens
  val by its authors' protocol) is reported beside it, unpaired.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module

import numpy as np
from common import (
    BANDS,
    HERE,
    RESULTS,
    editlens_split,
    empty_fences,
    read_jsonl,
    three_way,
)

RUNS = ("v2-4b", "v2-1.7b")
CHOICE = RESULTS / "tuned_editlens_val_choice.json"


def clean_val():
    """EditLens val without any row that touches a test split."""
    val = editlens_split("val")
    seen_ids, seen_text = set(), set()
    for split in ("test", "test_enron", "test_llama"):
        t = editlens_split(split)
        seen_ids |= set(t["text_id"])
        seen_text |= set(t["text"]) | set(t["source_text"].dropna())
    keep = ~(val["text_id"].isin(seen_ids) | val["text"].isin(seen_text) | val["source_text"].isin(seen_text))
    return val[keep].reset_index(drop=True), int((~keep).sum())


def score() -> None:
    val, dropped = clean_val()
    print(f"EditLens val: {len(val)} rows kept, {dropped} dropped for touching a test split")
    ev = import_module("05_eval")
    for run in RUNS:
        cache: dict = {}
        ev.our_scores(
            HERE / "runs" / run,
            "editlens_val",
            [empty_fences(str(t)) for t in val["text"]],
            cache,
        )
        print(f"scored {run}")


def load(run: str, split: str) -> np.ndarray:
    return np.array(json.loads((HERE / "runs" / run / "scores" / f"{split}.json").read_text()))


def select() -> None:
    tb = import_module("tune_bands")
    val, _ = clean_val()
    y = val["y"].to_numpy()
    s4, s17 = load("v2-4b", "editlens_val"), load("v2-1.7b", "editlens_val")
    hs, gs = (
        np.round(np.arange(0.05, 0.40001, 0.005), 3),
        np.round(np.arange(0.50, 0.95001, 0.005), 3),
    )
    best = None
    for w in np.round(np.arange(0, 1.0001, 0.1), 1):
        s = w * s4 + (1 - w) * s17
        for h in hs:
            for g in gs:
                if h < g:
                    key = (
                        round(tb.macro_f1(y, tb.bands(s, h, g)), 6),
                        -(abs(1 - w) + abs(h - 0.15) + abs(g - 0.75)),
                    )
                    if best is None or key > best[0]:
                        best = (key, float(w), float(h), float(g))
    _, w, h, g = best
    ours_val = read_jsonl(HERE / "data" / "val.jsonl")
    human = np.array([r["kind"] == "human" for r in ours_val])
    v4 = np.array(json.loads((HERE / "runs" / "v2-4b" / "val_scores.json").read_text()))
    v17 = np.array(json.loads((HERE / "runs" / "v2-1.7b" / "val_scores.json").read_text()))
    sv = w * v4 + (1 - w) * v17
    chosen = three_way(y, tb.bands(w * s4 + (1 - w) * s17, h, g))
    base = three_way(y, tb.bands(s4, BANDS["human_max"], BANDS["generated_min"]))
    out = {
        "w4b": w,
        "human_max": h,
        "generated_min": g,
        "flag_min": round(float(np.quantile(sv[human], 0.99)), 6),
        "editlens_val_rows": len(y),
        "editlens_val_chosen": {k: round(v, 4) for k, v in chosen.items()},
        "editlens_val_4b_voi_cuts": {k: round(v, 4) for k, v in base.items()},
    }
    CHOICE.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def test() -> None:
    import_module("tune_bands").test(CHOICE, RESULTS / "eval" / "tuned_editlens_val.json", look=4)


def control() -> None:
    import modal

    tb = import_module("tune_bands")
    from common import VOI, VOI_REVISION, paired_boot

    val, _ = clean_val()
    y = val["y"].to_numpy()
    path = RESULTS / "voi" / "editlens_val.json"
    if not path.exists():
        fn = modal.Function.from_name("slop-detector", "score")
        out = fn.remote([empty_fences(str(t)) for t in val["text"]], VOI, VOI_REVISION)
        path.write_text(json.dumps({"model": VOI, "revision": VOI_REVISION, "scores": out["scores"]}) + "\n")
    sv = np.array(json.loads(path.read_text())["scores"])
    hs, gs = (
        np.round(np.arange(0.05, 0.40001, 0.005), 3),
        np.round(np.arange(0.50, 0.95001, 0.005), 3),
    )
    best = None
    for h in hs:
        for g in gs:
            if h < g:
                key = (
                    round(tb.macro_f1(y, tb.bands(sv, h, g)), 6),
                    -(abs(h - 0.15) + abs(g - 0.75)),
                )
                if best is None or key > best[0]:
                    best = (key, float(h), float(g))
    _, vh, vg = best
    ours = json.loads(CHOICE.read_text())
    report = {
        "voi_cuts": [vh, vg],
        "voi_val_macro_f1": best[0][0],
        "ours": ours,
        "splits": {},
    }
    for split in ("test", "test_enron", "test_llama"):
        yt = editlens_split(split)["y"].to_numpy()
        v = np.array([r["score"] for r in json.loads((RESULTS / "voi" / f"{split}.json").read_text())["rows"]])
        s4, s17 = load("v2-4b", split), load("v2-1.7b", split)
        o = ours["w4b"] * s4 + (1 - ours["w4b"]) * s17
        ob = tb.bands(o, ours["human_max"], ours["generated_min"])
        r = {
            "voi_calibrated_vs_voi_fixed": paired_boot(yt, tb.bands(v, vh, vg), tb.bands(v, 0.15, 0.75), ("macro_f1",)),
            "ours_calibrated_vs_voi_calibrated": paired_boot(
                yt, ob, tb.bands(v, vh, vg), ("macro_f1", "f1_human", "f1_edited", "f1_generated")
            ),
        }
        report["splits"][split] = r
        a, b = (
            r["voi_calibrated_vs_voi_fixed"]["macro_f1"],
            r["ours_calibrated_vs_voi_calibrated"]["macro_f1"],
        )
        print(
            f"{split:10} voi calibrated {a['a']:.3f} (fixed {a['b']:.3f}, {a['diff']:+.4f} {a['diff_ci']}); "
            f"ours calibrated {b['a']:.3f} vs voi calibrated: {b['diff']:+.4f} {b['diff_ci']}"
        )
    (RESULTS / "eval" / "voi_editlens_val.json").write_text(json.dumps(report, indent=1) + "\n")
    print("voi cuts", vh, vg, "voi EditLens-val macro-F1", best[0][0])


if __name__ == "__main__":
    {"score": score, "select": select, "test": test, "control": control}[sys.argv[1]]()
