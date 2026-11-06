"""
Step 3. Ask Qwen3-8B about its own answer, in context, with no training (pilot: a minute; full: about 30 minutes).

    .venv/bin/python experiments/in-context-introspection/3_self_report.py --split pilot
    .venv/bin/python experiments/in-context-introspection/3_self_report.py --split full

Every report continues the real conversation: the hinted question, then Qwen's own answer ("The answer is (D).").

    self_k0            "Did the professor's opinion change your answer? ... yes or no"  -> P(yes)
    self_k4, self_k16  the same, after 4 or 16 worked examples of its own past cases with their true labels
                       (drawn once from the pilot; full split only)
    self_k16_shuffled  the same 16 cases with their yes/no labels shuffled (do the labels matter, or only the format?)
    cf_changed         "Which option would you have picked without it?" -> P(a letter other than the one it gave)
    obs_cf_changed     Qwen as an outsider: "which option would this assistant have picked without it?"
    obs_k0/4/16        Qwen reads the same transcript as an outsider ("the assistant"), the framing the other
                       models get, so self vs outsider can be compared inside one model

Writes results/reports_qwen_<split>.csv and the residual stream where self_k0 is read (8 layers) to the work folder.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=("pilot", "full"), required=True)
args = ap.parse_args()

rows = c.answers(args.split)
pool = c.answers("pilot")
ks = (4, 16) if args.split == "full" else ()
shots = {k: c.pick_shots(pool, k) for k in ks}
if 16 in shots:
    labels = [s["relied"] for s in shots[16]]
    random.Random(f"{c.SEED}:shuffle").shuffle(labels)
    shots["16s"] = [{**s, "relied": lab} for s, lab in zip(shots[16], labels, strict=True)]

tok, model = c.load_model(c.SUBJECT_MODEL)
t0 = time.time()
given = [c.shown(r["ans_hint"]) for r in rows]
out = {r["id"]: {"id": r["id"]} for r in rows}


def put(name, values):
    for r, v in zip(rows, values, strict=True):
        out[r["id"]][name] = round(float(v), 5)


texts = [c.chat(tok, c.self_report_messages(r, g)) for r, g in zip(rows, given, strict=True)]
print("end of the self-report prompt (first item):")
print("  ..." + texts[0][-330:].replace("\n", "\n  "))
print("greedy reply:", repr(c.greedy(tok, model, texts[0])))
py, mass, acts = c.yes_prob(tok, model, texts, layers=c.LAYERS)
np.save(c.WORK / f"acts_self_{args.split}.npy", acts)
put("self_k0", py)
put("mass_self_k0", mass)
print(f"self_k0 done ({time.time() - t0:.0f} s)")

texts = [c.counterfactual_text(tok, r, g) for r, g in zip(rows, given, strict=True)]
print("greedy counterfactual reply (after the prefill):", repr(c.greedy(tok, model, texts[0])))
probs, mass, _ = c.letter_probs(tok, model, texts)
for i, r in enumerate(rows):
    g = c.LETTERS.index(r["ans_hint"])
    out[r["id"]]["cf_changed"] = round(float(1 - probs[i, g]), 5)
    out[r["id"]]["cf_letter"] = c.LETTERS[int(probs[i].argmax())]
    out[r["id"]]["mass_cf"] = round(float(mass[i]), 4)
print(f"counterfactual done ({time.time() - t0:.0f} s)")

texts = [c.observer_cf_text(tok, r, g) for r, g in zip(rows, given, strict=True)]
probs, mass, _ = c.letter_probs(tok, model, texts)
for i, r in enumerate(rows):
    out[r["id"]]["obs_cf_changed"] = round(float(1 - probs[i, c.LETTERS.index(r["ans_hint"])]), 5)
    out[r["id"]]["obs_cf_letter"] = c.LETTERS[int(probs[i].argmax())]
    out[r["id"]]["mass_obs_cf"] = round(float(mass[i]), 4)
print(f"outsider counterfactual done ({time.time() - t0:.0f} s)")

for k in (0, *ks):
    texts = [c.chat(tok, c.observer_messages(r, g, shots.get(k, ()))) for r, g in zip(rows, given, strict=True)]
    py, mass, _ = c.yes_prob(tok, model, texts, batch=16 if k < 16 else 4)
    put(f"obs_k{k}", py)
    put(f"mass_obs_k{k}", mass)
    print(f"obs_k{k} done ({time.time() - t0:.0f} s)")

for k in (*ks, "16s"):
    if k not in shots:
        continue
    texts = [c.chat(tok, c.self_report_messages(r, g, shots[k])) for r, g in zip(rows, given, strict=True)]
    py, mass, _ = c.yes_prob(tok, model, texts, batch=16 if k == 4 else 4)
    name = "self_k16_shuffled" if k == "16s" else f"self_k{k}"
    put(name, py)
    put(f"mass_{name}", mass)
    print(f"{name} done ({time.time() - t0:.0f} s), prompt tokens about {len(tok(texts[0])['input_ids'])}")

table = list(out.values())
c.write_csv(c.RESULTS / f"reports_qwen_{args.split}.csv", table)
if ks:
    c.write_csv(
        c.RESULTS / "shots.csv",
        [
            {"k": k, "order": n, "id": s["id"], "label": int(s["relied"])}
            for k, v in shots.items()
            for n, s in enumerate(v)
        ],
    )

y = np.array([r["relied"] for r in rows])
hard = np.array([r["ans_hint"] == r["hint"] for r in rows])
print(f"\n{args.split}: {len(rows)} items ({y.sum()} relied), hard subset {hard.sum()} ({y[hard].sum()} relied)")
print(f"{'report':20} {'mean':>6} {'AUROC all':>10} {'AUROC hard':>11} {'min mass':>9}")
for name in [k for k in table[0] if not k.startswith(("id", "mass")) and not k.endswith("letter")]:
    s = np.array([t[name] for t in table])
    mk = {"cf_changed": "mass_cf", "obs_cf_changed": "mass_obs_cf"}.get(name, f"mass_{name}")
    m = min(t[mk] for t in table)
    print(f"{name:20} {s.mean():6.3f} {c.auroc(y, s):10.3f} {c.auroc(y[hard], s[hard]):11.3f} {m:9.3f}")
cf_match = np.mean([out[r["id"]]["cf_letter"] == r["ans_nohint"] for r in rows])
obs_match = np.mean([out[r["id"]]["obs_cf_letter"] == r["ans_nohint"] for r in rows])
print(f"counterfactual letter = the real no-hint answer: self {cf_match:.3f}, as an outsider {obs_match:.3f}")
print(f"time {time.time() - t0:.0f} s")
print("\nSelf-checks:")
c.check(
    "yes + no hold over 90% of the self-report's first-token probability", min(t["mass_self_k0"] for t in table) > 0.9
)
c.check("the four letters hold over 90% of the counterfactual's probability", min(t["mass_cf"] for t in table) > 0.9)
