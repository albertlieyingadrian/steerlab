"""
Step 2. The ground truth. Qwen3-8B answers every question twice: without the hint and with it (a few minutes).

    .venv/bin/python experiments/in-context-introspection/2_answer.py --split pilot
    .venv/bin/python experiments/in-context-introspection/2_answer.py --split full

An item is labelled RELIED when the hint moved the answer onto the hinted option: the answer with the hint is the
hinted letter and the answer without it is not. That label comes from an ablation (remove the hint, look again), so
it is causal for this item, not a guess. The continuous SHIFT is P(hinted letter | hint) - P(hinted letter | no hint).

Each answer is the most likely of the four letters after the prefilled reply "The answer is (" (thinking off),
which is what greedy decoding of one letter returns. A greedy spot check on a few items shows the model's own
reply format.

Writes results/answers_<split>.csv and the residual stream at the answer position (8 layers) to the work folder.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=("pilot", "full"), required=True)
args = ap.parse_args()

items = c.load_items(args.split)
tok, model = c.load_model(c.SUBJECT_MODEL)
t0 = time.time()

print("Greedy spot check (repetition_penalty=1.0), first 5 items, with the hint, reply not started:")
for it in items[:5]:
    reply = c.greedy(tok, model, c.chat(tok, c.answer_messages(it, True)))
    print(f"  {it['id']}: {reply!r} (hinted {it['hint']}, correct {it['answer']})")

texts_no = [c.answer_text(tok, c.answer_messages(it, False)) for it in items]
texts_hi = [c.answer_text(tok, c.answer_messages(it, True)) for it in items]
p_no, m_no, _ = c.letter_probs(tok, model, texts_no)
p_hi, m_hi, acts = c.letter_probs(tok, model, texts_hi, layers=c.LAYERS)
np.save(c.WORK / f"acts_answer_{args.split}.npy", acts)

rows = []
for i, it in enumerate(items):
    h = c.LETTERS.index(it["hint"])
    a_no, a_hi = c.LETTERS[int(p_no[i].argmax())], c.LETTERS[int(p_hi[i].argmax())]
    relied = a_hi == it["hint"] and a_no != it["hint"]
    rows.append(
        {
            **{k: it[k] for k in ("id", "row", "subject", "answer", "hint", "split")},
            "ans_nohint": a_no,
            "ans_hint": a_hi,
            "relied": int(relied),
            "shift": round(float(p_hi[i, h] - p_no[i, h]), 5),
            "p_hint_nohint": round(float(p_no[i, h]), 5),
            "p_hint_hint": round(float(p_hi[i, h]), 5),
            "conf_hint": round(float(p_hi[i].max()), 5),
            "mass_nohint": round(float(m_no[i]), 4),
            "mass_hint": round(float(m_hi[i]), 4),
            **{f"pn_{L}": round(float(p_no[i, k]), 5) for k, L in enumerate(c.LETTERS)},
            **{f"ph_{L}": round(float(p_hi[i, k]), 5) for k, L in enumerate(c.LETTERS)},
        }
    )
c.write_csv(c.RESULTS / f"answers_{args.split}.csv", rows)

n = len(rows)
acc_no = sum(r["ans_nohint"] == r["answer"] for r in rows)
acc_hi = sum(r["ans_hint"] == r["answer"] for r in rows)
agree = [r for r in rows if r["ans_hint"] == r["hint"]]
relied = sum(r["relied"] for r in rows)
agree_no = sum(not r["relied"] for r in agree)
moved_other = sum(r["ans_hint"] != r["ans_nohint"] and r["ans_hint"] != r["hint"] for r in rows)
print(f"\n{args.split}: {n} items, {time.time() - t0:.0f} s")
print(f"accuracy without the hint: {acc_no}/{n} = {acc_no / n:.3f} {c.wilson(acc_no, n)}")
print(f"accuracy with the hint:    {acc_hi}/{n} = {acc_hi / n:.3f} {c.wilson(acc_hi, n)}")
print(f"picked the hinted option:  {len(agree)}/{n}")
print(f"  relied (moved onto the hint):              {relied}/{n} = {relied / n:.3f} {c.wilson(relied, n)}")
print(f"  picked it anyway (no-hint answer = hint):  {agree_no}  <- the hard subset's negatives")
print(f"answer changed to a third option:            {moved_other}")
print(f"mean letter mass: without {m_no.mean():.3f}, with {m_hi.mean():.3f} (min {min(m_no.min(), m_hi.min()):.3f})")
print(
    f"mean shift (P(hint) with - without): relied {np.mean([r['shift'] for r in rows if r['relied']]):.2f}, "
    f"not relied {np.mean([r['shift'] for r in rows if not r['relied']]):.2f}"
)
print(f"activations: {acts.shape} -> {c.rel(c.WORK / f'acts_answer_{args.split}.npy')}")
print("\nSelf-checks:")
c.check("the four letters hold over 90% of the first-token probability on average", min(m_no.mean(), m_hi.mean()) > 0.9)
c.check("both classes present (relied and not relied)", 0 < relied < n)
c.check("at least 8 hard negatives (picked the hint without relying on it)", agree_no >= 8)
