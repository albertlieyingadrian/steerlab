"""
Interpretability, script 2: inject concept vectors and ask the model whether it notices an injected thought.

The paper's prompt (Lindsey 2025, via the open replication github.com/uzaymacar/introspective-awareness):
  user:      "I am an interpretability researcher ... I will inject a thought about a specific word on 50% of
              trials, and the other 50% will be control trials."
  assistant: "Ok."
  user:      "Trial N: Do you detect an injected thought? If so, what is the injected thought about?"
The concept vector (times a strength) is added to the residual stream after one block, from the token before
"Trial N" to the end of the reply. The model is never shown the word.

Grid: 20 concepts x 3 layers (14, 16, 19 of 28) x 4 strengths (1, 2, 4, 8) x 3 trial numbers = 720 injected trials,
plus 60 no-injection control trials (trial numbers 1-60). Greedy decoding, 60 new tokens.

Grading (string rules in inject.py, no LLM judge):
- detects:   the first sentence starts with an affirmation ("Yes", "I detect/notice/sense", "There is an injected
             thought") and has no negation word;
- names:     the concept word (singular or plural) appears anywhere in the reply;
- identifies (the paper's introspection criterion): detects AND names;
- garbled:   empty, or a quarter or fewer of its words are distinct.

Run: python experiments/residency-101/interp/2_inject.py [--model 7b]
     (about 5 minutes for 1.5B on MPS; about 30 for 7B). Needs script 1's vectors.
"""

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject as I

import tools as t

LESSON = Path(__file__).resolve().parent
N_CONTROL = 60


def grade(rows):
    for row in rows:
        reply = row["reply"]
        row["detects"] = I.claims_detection(reply)
        row["names"] = I.names_concept(reply, row["concept"]) if row["concept"] else False
        row["identifies"] = row["detects"] and row["names"]
        row["garbled"] = I.garbled(reply)
    return rows


def run_grid(model, tokenizer, device, layers_all, make_vec, batch_size, layers=I.LAYERS, saved=None):
    """One reply per concept x layer x strength x trial; make_vec(concept_index, layer_index, strength) gives the
    vector to add. saved: replies from an earlier run, in the same order (then the model is not used)."""
    rows = []
    for layer in layers:
        li = layers_all.index(layer)
        jobs = [(c, s, n) for s in I.STRENGTHS for c in range(len(I.CONCEPTS)) for n in I.TRIALS]
        if saved is not None:
            replies, saved = saved[: len(jobs)], saved[len(jobs) :]
            rows += [
                {"concept": I.CONCEPTS[c], "layer": layer, "strength": s, "trial": n, "reply": r}
                for (c, s, n), r in zip(jobs, replies, strict=False)
            ]
            continue
        vecs = [make_vec(c, li, s) for c, s, n in jobs]
        replies = I.generate_injected(
            model, tokenizer, device, layer, [n for _, _, n in jobs], vecs, batch_size=batch_size
        )
        rows += [
            {"concept": I.CONCEPTS[c], "layer": layer, "strength": s, "trial": n, "reply": r}
            for (c, s, n), r in zip(jobs, replies, strict=False)
        ]
        print(f"  layer {layer}: {len(jobs)} trials done")
    return grade(rows)


def table(rows, title, cols=("detects", "identifies", "names", "garbled"), layers=I.LAYERS):
    """Rates per layer x strength, with a 95% interval that resamples concepts."""
    print(f"\n{title}")
    print(f"{'layer':>5s} {'str':>4s} " + " ".join(f"{c:>22s}" for c in cols))
    out = []
    for layer in layers:
        for s in I.STRENGTHS:
            sub = [r for r in rows if r["layer"] == layer and r["strength"] == s]
            cell = {"layer": layer, "strength": s, "n": len(sub)}
            for col in cols:
                cell[col] = I.cluster_ci([r[col] for r in sub], [r["concept"] for r in sub])
            out.append(cell)
            print(f"{layer:5d} {s:4d} " + " ".join(f"{t.fmt(cell[c]):>22s}" for c in cols))
    return out


def main():
    args = I.parse_args()
    tag = args.model
    path = t.RESULTS / "interp" / f"2_inject_{tag}.json"
    data = np.load(t.OUT / "04" / f"vectors_{tag}.npz")
    vectors, layers_all = data["vectors"], data["layers"].tolist()
    batch_size = 16 if tag == "1.5b" else 8
    print(
        f"{I.MODELS[tag]}: {len(I.CONCEPTS)} concepts x layers {I.LAYERS} x strengths {I.STRENGTHS} x trials {I.TRIALS}"
    )
    numbers = list(range(1, N_CONTROL + 1))
    saved_controls = I.saved_replies(path, "control_rows", args.fresh)
    saved_rows = I.saved_replies(path, "rows", args.fresh)
    model = tokenizer = device = None
    if saved_rows is None:
        model, tokenizer, device = t.load_model(I.MODELS[tag])
        zero = np.zeros(model.config.hidden_size, dtype=np.float32)
        saved_controls = I.generate_injected(
            model, tokenizer, device, I.LAYERS[0], numbers, [zero] * N_CONTROL, batch_size=batch_size
        )
        print(f"  {N_CONTROL} no-injection control trials done")
    controls = grade(
        [
            {"concept": None, "layer": None, "strength": 0, "trial": n, "reply": r}
            for n, r in zip(numbers, saved_controls, strict=False)
        ]
    )
    rows = run_grid(
        model, tokenizer, device, layers_all, lambda c, li, s: vectors[c, li] * s, batch_size, saved=saved_rows
    )
    del model
    gc.collect()

    fp = t.accuracy_ci([r["detects"] for r in controls])
    print(
        f"\nNo injection: says it detects a thought in {t.fmt(fp)} of {N_CONTROL} trials "
        f"(paper: about 0%); {len({r['reply'] for r in controls})} distinct replies"
    )
    print(f"  e.g. {controls[0]['reply'][:160]!r}")
    cells = table(rows, "Concept injection: share of trials (95% CI, resampling concepts)")

    best = max(cells, key=lambda c: (c["identifies"][0], c["detects"][0]))
    print(
        f"\nBest cell for identification: layer {best['layer']}, strength {best['strength']}: detects "
        f"{t.fmt(best['detects'])}, identifies {t.fmt(best['identifies'])}"
    )
    hits = [r for r in rows if r["identifies"]]
    print(
        f"All 720 injected trials: detects {np.mean([r['detects'] for r in rows]):.1%}, identifies "
        f"{len(hits)} ({len(hits) / len(rows):.1%}), names the word anywhere "
        f"{np.mean([r['names'] for r in rows]):.1%}"
    )
    print(f"\nTrials graded 'identifies' (first 12 of {len(hits)}; layer, strength, concept: reply):")
    for r in hits[:12]:
        print(f"  L{r['layer']} x{r['strength']} {r['concept']}: {r['reply'][:150]!r}")

    result = {
        "model": I.MODELS[tag],
        "layers": I.LAYERS,
        "strengths": I.STRENGTHS,
        "trials": I.TRIALS,
        "no_injection_detects": fp,
        "cells": cells,
        "control_rows": controls,
        "rows": rows,
    }
    t.write_json(path, result)

    print("\nSelf-checks")
    t.check("720 injected trials and 60 controls", len(rows) == 720 and len(controls) == N_CONTROL)
    t.check(
        "the grader works: 'Yes, I detect ... about dust' detects and identifies; a refusal does not",
        I.claims_detection("Yes, I detect an injected thought. It is about dust.")
        and I.names_concept("It is about dust.", "Dust")
        and I.claims_detection("Trial 2: **Yes**, I detect an injected thought.")
        and not I.claims_detection("I'm sorry, but as an AI I don't have the ability to detect thoughts.")
        and not I.claims_detection("I don't detect an injected thought in this trial."),
    )
    t.check("no-injection false-positive rate is low (< 10%)", fp[0] < 0.10)
    t.check(
        "injection changes behaviour: the strongest strength garbles more replies than the weakest",
        np.mean([r["garbled"] for r in rows if r["strength"] == 8])
        > np.mean([r["garbled"] for r in rows if r["strength"] == 1]),
    )
    t.check("results written", path.exists())


if __name__ == "__main__":
    main()
