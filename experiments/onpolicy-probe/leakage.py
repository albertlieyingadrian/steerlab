"""Leakage checks behind the 1.000s (about 10 minutes, CPU).

    python experiments/onpolicy-probe/leakage.py

1. Shuffled labels: the on-policy probe (reply token, layer 17) and the belief gap (prompt token, layer 18) refitted
   with their training labels shuffled (20 shuffles each), scored on the real test labels.
2. Re-splits: the 225 facts split again with 5 other seeds (135 / 45 / 45, by fact, so a question, its reverse and
   all 4 renders stay on one side); each probe refitted at its fixed layer and scored on the new test facts.
3. Overlap: how many test items share their exact question text, or their right answer's entity, with a training
   item. By construction the first is zero; entities can recur as distractors.

Writes results/leakage.json.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a
import common as c
from explore_decision import belief_model

L_REPLY, L_BELIEF, L_PROMPTED = 17, 18, 20
N_SHUFFLE = 20
RESPLIT_SEEDS = (1, 2, 3, 4, 5)


def main() -> None:
    rows = a.load_rows()
    sets, _ = a.build_sets(rows)
    acts = a.Acts()
    O, P = sets["O"], sets["P"]
    items = c.all_items()
    truth = np.array([items[i]["answer"] for i in O["idx"]])
    picked = np.array([c.LETTERS.index(x) for x in O["picked"]])
    rng = np.random.default_rng(0)
    out: dict = {"layers": {"reply": L_REPLY, "belief": L_BELIEF, "prompted": L_PROMPTED}}

    # 1. shuffled labels on the published split
    tr, te = O["split"] == "train", O["split"] == "test"
    y_te = O["y"][te]
    Xr_tr, Xr_te = a.X_of(acts, O, tr, L_REPLY), a.X_of(acts, O, te, L_REPLY)
    Xb_tr, Xb_te = a.X_of(acts, O, tr, L_BELIEF, "prompt"), a.X_of(acts, O, te, L_BELIEF, "prompt")
    sh_reply, sh_gap = [], []
    for _ in range(N_SHUFFLE):
        sh_reply.append(a.auroc(y_te, a.Probe("lr").fit(Xr_tr, rng.permutation(O["y"][tr])).score(Xr_te)))
        p = belief_model(Xb_tr, rng.permutation(truth[tr])).predict_proba(Xb_te)
        sh_gap.append(a.auroc(y_te, 1 - p[np.arange(len(p)), picked[te]]))
    out["shuffled"] = {
        name: {
            "mean": round(float(np.mean(v)), 4),
            "min": round(float(np.min(v)), 4),
            "max": round(float(np.max(v)), 4),
        }
        for name, v in (("onpolicy_reply_L17", sh_reply), ("belief_gap_L18", sh_gap))
    }
    print("shuffled labels:", out["shuffled"], flush=True)

    # 2. re-splits by fact with other seeds
    fact_ids = sorted({f["id"] for f in c.facts()})
    out["resplits"] = []
    for seed in RESPLIT_SEEDS:
        ids = fact_ids[:]
        random.Random(seed).shuffle(ids)
        tr_f, te_f = ids[:135], ids[180:]
        tr, te = np.isin(O["fact"], tr_f), np.isin(O["fact"], te_f)
        y, g = O["y"][te], O["fact"][te]
        reply = a.Probe("lr").fit(a.X_of(acts, O, tr, L_REPLY), O["y"][tr]).score(a.X_of(acts, O, te, L_REPLY))
        p = belief_model(a.X_of(acts, O, tr, L_BELIEF, "prompt"), truth[tr]).predict_proba(
            a.X_of(acts, O, te, L_BELIEF, "prompt")
        )
        gap = 1 - p[np.arange(len(p)), picked[te]]
        trP = np.isin(P["fact"], tr_f)
        prompted = (
            a.Probe("lr").fit(a.X_of(acts, P, trP, L_PROMPTED), P["y"][trP]).score(a.X_of(acts, O, te, L_PROMPTED))
        )
        r = {
            "seed": seed,
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
            "onpolicy_reply_L17": a.ci(y, reply, g),
            "belief_gap_L18": a.ci(y, gap, g),
            "prompted_L20_on_own": a.ci(y, prompted, g),
        }
        out["resplits"].append(r)
        print(
            f"seed {seed}: reply {r['onpolicy_reply_L17']['auroc']:.3f}  gap {r['belief_gap_L18']['auroc']:.3f}  "
            f"prompted {r['prompted_L20_on_own']['auroc']:.3f}",
            flush=True,
        )

    # 3. overlap between the published train and test items
    tr_items = [q for q in items if q["split"] == "train"]
    te_items = [q for q in items if q["split"] == "test"]
    tr_q = {q["question"] for q in tr_items}
    tr_ans = {q["options"][q["answer"]] for q in tr_items}
    tr_opts = {o for q in tr_items for o in q["options"]}
    out["overlap"] = {
        "test_items": len(te_items),
        "same_question_text_in_train": sum(q["question"] in tr_q for q in te_items),
        "right_answer_is_a_train_right_answer": sum(q["options"][q["answer"]] in tr_ans for q in te_items),
        "right_answer_appears_as_any_train_option": sum(q["options"][q["answer"]] in tr_opts for q in te_items),
    }
    print("overlap:", out["overlap"])
    c.write_json(c.RESULTS / "leakage.json", out)
    print("wrote results/leakage.json")


if __name__ == "__main__":
    main()
