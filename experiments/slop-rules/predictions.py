"""Score the eight pre-registered predictions mechanically from results/ and write results/predictions.json.

Each threshold is the one in the plan's Predictions table; the verdict is a comparison, never a judgement.
"""

from __future__ import annotations

import json

import pandas as pd

from common import RESULTS, write_json


def main() -> None:
    allt = pd.read_csv(RESULTS / "rules_v1_all.csv")
    closed = pd.read_csv(RESULTS / "rules_v1_closed.csv").set_index("rule")
    cand_closed = pd.read_csv(RESULTS / "candidates_closed.csv")
    dec = pd.read_csv(RESULTS / "list_decisions.csv")
    cmp_ = json.loads((RESULTS / "compare.json").read_text())
    words = allt[allt.kind == "word"]
    habits = allt[(allt.kind == "habit") & (allt.a >= 0.02)].sort_values("bp", ascending=False)
    patterns = allt[(allt.kind == "pattern") & ~allt.label.isin(["Not just X but Y", "Everything in threes"])]
    cand = pd.read_csv(RESULTS / "candidates.csv")

    # P8: v2 word and phrase rules (v1 items kept + new items added) on the closed-model set
    kept = dec[(dec.kind.isin(["word", "signpost", "stock"])) & (dec.action == "keep")]
    v1_bp = [closed.loc[r, "bp"] for r in kept.rule]
    added = dec[dec.kind.isin(["new_word", "new_ngram"]) & (dec.action == "add")]
    cc = cand_closed.assign(key=cand_closed.kind + ":" + cand_closed.item).set_index("key")
    new_bp = [cc.loc[r, "closed_bp"] for r in added.rule]
    p8 = pd.Series(v1_bp + new_bp).dropna()

    a1, a2 = cmp_["auroc"]["v1"]["open_test"][0], cmp_["auroc"]["v2"]["open_test"][0]
    rows = [
        {
            "id": "P1",
            "prediction": "Many ban words are noise or rare on modern human text",
            "metric": "ban words (of 93) with verdict noise or rare, all splits",
            "threshold": ">= 25",
            "confidence": "70%",
            "value": int(words.verdict.isin(["noise", "rare"]).sum()),
            "detail": {
                "noise": int((words.verdict == "noise").sum()),
                "rare": int((words.verdict == "rare").sum()),
                "bp_below_two_thirds": int((words.bp < 2 / 3).sum()),
                "never_used_by_people": int((words.h == 0).sum()),
            },
        },
        {
            "id": "P2",
            "prediction": "The contrast frame and signposts are the strongest habits",
            "metric": "top 2 habits by BP among habits firing on >= 2% of AI windows",
            "threshold": "both top 2",
            "confidence": "45%",
            "value": list(habits.label.head(2)),
            "detail": {"ranking": [[r.label, round(r.bp, 3)] for r in habits.itertuples()]},
        },
        {
            "id": "P3",
            "prediction": "The em dash is weak",
            "metric": "em_dash BP, pooled open models",
            "threshold": "< 0.65",
            "confidence": "40%",
            "value": round(float(allt.set_index("rule").loc["habit:em_dash", "bp"]), 3),
        },
        {
            "id": "P4",
            "prediction": "Data finds rules the lists lack",
            "metric": "candidates passing held-out validation",
            "threshold": ">= 20",
            "confidence": "60%",
            "value": int(cand.passes.sum()),
            "detail": cand.groupby("kind").passes.agg(["sum", "count"]).to_dict(),
        },
        {
            "id": "P5",
            "prediction": "Refitting helps",
            "metric": "v2 minus v1 AUROC, open-model test windows",
            "threshold": ">= +0.05",
            "confidence": "60%",
            "value": round(a2 - a1, 4),
            "detail": {"v1": a1, "v2": a2},
        },
        {
            "id": "P6",
            "prediction": "The old band does not transfer to modern human text",
            "metric": "v1 false-positive rate at 33.63, modern human test windows",
            "threshold": ">= 0.15",
            "confidence": "55%",
            "value": cmp_["at_limit"]["v1 @ 33.63"]["fpr_modern_human_test"][0],
        },
        {
            "id": "P7",
            "prediction": "Several named patterns are noise",
            "metric": "regex named patterns (8) with verdict noise",
            "threshold": ">= 3",
            "confidence": "55%",
            "value": int((patterns.verdict == "noise").sum()),
            "detail": {r.label: r.verdict for r in patterns.itertuples()},
        },
        {
            "id": "P8",
            "prediction": "Rules found on open models transfer to closed models",
            "metric": "share of v2 kept-or-new word and phrase rules with BP >= 0.70 on the closed-model set",
            "threshold": ">= 0.70",
            "confidence": "50%",
            "value": round(float((p8 >= 0.70).mean()), 3),
            "detail": {"n_rules_with_a_firing": len(p8), "n_v1_kept": len(v1_bp), "n_new": len(new_bp)},
        },
    ]
    tests = {
        "P1": lambda v: v >= 25,
        "P2": lambda v: set(v) == {"not X but Y", "signpost"},
        "P3": lambda v: v < 0.65,
        "P4": lambda v: v >= 20,
        "P5": lambda v: v >= 0.05,
        "P6": lambda v: v >= 0.15,
        "P7": lambda v: v >= 3,
        "P8": lambda v: v >= 0.70,
    }
    for r in rows:
        r["hit"] = bool(tests[r["id"]](r["value"]))
    write_json(RESULTS / "predictions.json", rows)
    for r in rows:
        print(r["id"], "hit" if r["hit"] else "miss", r["value"], r.get("detail", ""))


if __name__ == "__main__":
    main()
