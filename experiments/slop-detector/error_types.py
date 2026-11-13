"""Before the final training: sort the current model's wrong and low-confidence EditLens-val cases into types, and
check the data v3 mix against the worst types (B3 prereg, amendment 5). Validation only; CPU; saved scores.

    python error_types.py --run b1x-4bk

Wrong = 3-way class at the bucket head's fixed cut-offs (1/6, 5/6) differs from the truth. Low-confidence = the score
is within 0.05 of a cut-off. Types: true class x editor, source, length (words), and for AI edits the edit amount
(EditLens cosine_score quartiles). Writes results/error_types.json (counts and rates, no text).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from common import DATA, HERE, RESULTS, benchmark, read_jsonl

CUTS = (1 / 6, 5 / 6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="b1x-4bk")
    args = ap.parse_args()
    df = benchmark("editlens_val").reset_index(drop=True)
    s = np.array(json.loads((HERE / "runs" / args.run / "scores" / "editlens_val.json").read_text()))
    s = s if s.ndim == 1 else s[:, 0]
    pred = np.digitize(s, CUTS)
    y = df["y"].to_numpy()
    wrong = pred != y
    low = np.min(np.abs(s[:, None] - np.array(CUTS)[None]), axis=1) < 0.05
    df["words"] = df["text"].str.split().str.len()
    df["len"] = np.select([df["words"] < 150, df["words"] < 400], ["<150", "150-399"], "400+")
    edited = df["y"] == 1
    q = df.loc[edited, "cosine_score"].quantile([0.25, 0.5, 0.75]).to_numpy()
    df["edit_amount"] = "n/a"
    df.loc[edited, "edit_amount"] = np.select(
        [df["cosine_score"] >= q[2], df["cosine_score"] >= q[1], df["cosine_score"] >= q[0]],
        ["lightest quarter", "2nd", "3rd"],
        "heaviest quarter",
    )[edited.to_numpy()]

    def table(col: str) -> dict:
        out = {}
        for (cls, v), g in df.groupby([df["text_type"], df[col]]):
            m = g.index.to_numpy()
            out[f"{cls} | {v}"] = {
                "n": len(m),
                "wrong": round(float(wrong[m].mean()), 3),
                "wrong_or_low": round(float((wrong | low)[m].mean()), 3),
            }
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["wrong"] * kv[1]["n"]))

    worst = Counter()
    for i in np.where(wrong)[0]:
        r = df.iloc[i]
        worst[f"{r['text_type']} | {r['model']} | {r['source']} | {r['len']}"] += 1
    v3 = read_jsonl(DATA / "v3b" / "train.jsonl")
    new = [r for r in v3 if r.get("data") == "v3"]
    mix = {
        "new_train_rows": len(new),
        "by_kind": dict(Counter(r["kind"] for r in new)),
        "by_source": dict(Counter(r["source"] for r in new)),
        "edits_by_bucket": dict(Counter(r["bucket"] for r in new if r["kind"] != "human")),
        "edits_400plus_words": sum(len(r["text"].split()) >= 400 for r in new if r["kind"] != "human"),
        "edits_under_15pct_changed": sum(r.get("frac", 1) < 0.15 for r in new if r["kind"] != "human"),
    }
    report = {
        "run": args.run,
        "n": len(df),
        "wrong": int(wrong.sum()),
        "low_confidence": int(low.sum()),
        "by_editor": table("model"),
        "by_source": table("source"),
        "by_length": table("len"),
        "by_edit_amount": table("edit_amount"),
        "top_error_types": dict(worst.most_common(12)),
        "data_v3_mix": mix,
    }
    (RESULTS / "error_types.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("wrong", "low_confidence", "top_error_types", "by_edit_amount", "data_v3_mix")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
