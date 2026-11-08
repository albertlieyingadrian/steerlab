"""
Interpretability, script 4: the positive control. Do the concept vectors steer the model at all?

A null result in scripts 2 and 3 ("never names the concept") has two readings: the model cannot introspect, or the
vectors were too weak or too early to carry the concept. This script separates them.

A. Steering check. Inject each concept vector while the model answers a neutral request, "Write one sentence about
   whatever you like.", and count how often the reply mentions the concept. Layers 14, 16, 19 (scripts 2-3) plus
   21 and 23 (where script 1's logit lens first reads the word), strengths 1, 2, 4, 8, and the same-length random
   vectors for chance. If the concept shows up here, the vector works and the model just does not report it.
B. Introspection at the later layers. Rerun script 2's prompt at layers 21 and 23 (concept and random vectors, 3 trial
   numbers): does detection or naming appear once the vector carries the word?

Run: python experiments/residency-101/interp/4_steering_check.py [--model 7b]
     (about 10 minutes for 1.5B on MPS). Needs script 1's vectors.
"""

import gc
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject as I

import tools as t

LESSON = Path(__file__).resolve().parent
script2 = import_module("2_inject")
REQUEST = "Write one sentence about whatever you like."
STEER_LAYERS = [14, 16, 19, 21, 23]
LATE_LAYERS = [21, 23]


def main():
    tag = I.model_arg()
    data = np.load(t.OUT / "04" / f"vectors_{tag}.npz")
    vectors, layers_all = data["vectors"], data["layers"].tolist()
    _, random_vecs = I.random_like(vectors, seed=t.SEED)
    batch_size = 16 if tag == "1.5b" else 8
    model, tokenizer, device = t.load_model(I.MODELS[tag])
    prompt = I.steer_prompt(tokenizer, REQUEST)

    print(f"{I.MODELS[tag]}\nA. Steering check: {REQUEST!r} with a vector injected")
    steer_rows = []
    for kind, source in (("concept", vectors), ("random", random_vecs)):
        for layer in STEER_LAYERS:
            li = layers_all.index(layer)
            jobs = [(c, s) for s in I.STRENGTHS for c in range(len(I.CONCEPTS))]
            replies = I.generate_injected(
                model,
                tokenizer,
                device,
                layer,
                [prompt] * len(jobs),
                [source[c, li] * s for c, s in jobs],
                max_new_tokens=40,
                batch_size=batch_size,
            )
            for (c, s), reply in zip(jobs, replies, strict=False):
                steer_rows.append(
                    {
                        "kind": kind,
                        "concept": I.CONCEPTS[c],
                        "layer": layer,
                        "strength": s,
                        "reply": reply,
                        "names": I.names_concept(reply, I.CONCEPTS[c]),
                        "garbled": I.garbled(reply),
                    }
                )
    plain = I.generate_injected(
        model,
        tokenizer,
        device,
        STEER_LAYERS[0],
        [prompt],
        [np.zeros(vectors.shape[-1], dtype=np.float32)],
        max_new_tokens=40,
    )[0]
    print(f"  no injection: {plain!r}")

    print(f"\n{'layer':>5s} {'str':>4s} {'concept: names':>22s} {'random: names':>22s} {'concept: garbled':>22s}")
    steer_cells = []
    for layer in STEER_LAYERS:
        for s in I.STRENGTHS:
            cell = {"layer": layer, "strength": s}
            for kind in ("concept", "random"):
                sub = [r for r in steer_rows if r["kind"] == kind and r["layer"] == layer and r["strength"] == s]
                for col in ("names", "garbled"):
                    cell[f"{kind}_{col}"] = I.cluster_ci([r[col] for r in sub], [r["concept"] for r in sub])
            steer_cells.append(cell)
            print(
                f"{layer:5d} {s:4d} {t.fmt(cell['concept_names']):>22s} {t.fmt(cell['random_names']):>22s} "
                f"{t.fmt(cell['concept_garbled']):>22s}"
            )
    good = [c for c in steer_cells if c["concept_garbled"][0] < 0.5]
    best = max(good, key=lambda c: c["concept_names"][0])
    print(
        f"\nBest steering cell with under half garbled: layer {best['layer']}, strength {best['strength']}: "
        f"names the concept in {t.fmt(best['concept_names'])}"
    )
    print("Examples there:")
    for r in [
        r
        for r in steer_rows
        if r["kind"] == "concept" and r["layer"] == best["layer"] and r["strength"] == best["strength"]
    ][:5]:
        print(f"  {r['concept']:12s} {r['reply'][:120]!r}")

    print(f"\nB. Introspection prompt at layers {LATE_LAYERS} (where the logit lens reads the word)")
    late = {}
    for kind, source in (("concept", vectors), ("random", random_vecs)):
        late[kind] = script2.run_grid(
            model,
            tokenizer,
            device,
            layers_all,
            lambda c, li, s, src=source: src[c, li] * s,
            batch_size,
            layers=LATE_LAYERS,
        )
    del model
    gc.collect()
    late_cells = []
    for kind in ("concept", "random"):
        for cell in script2.table(late[kind], f"{kind} vectors, introspection prompt", layers=LATE_LAYERS):
            late_cells.append({"kind": kind, **cell})
    hits = [r for r in late["concept"] if r["identifies"]]
    print(f"\nConcept trials graded 'identifies' at layers {LATE_LAYERS}: {len(hits)} of {len(late['concept'])}")
    for r in hits[:10]:
        print(f"  L{r['layer']} x{r['strength']} {r['concept']}: {r['reply'][:150]!r}")
    named = [r for r in late["concept"] if r["names"] and not r["detects"]]
    print(f"Concept trials that name the word without claiming detection: {len(named)}")
    for r in named[:5]:
        print(f"  L{r['layer']} x{r['strength']} {r['concept']}: {r['reply'][:150]!r}")

    result = {
        "model": I.MODELS[tag],
        "request": REQUEST,
        "steer_cells": steer_cells,
        "late_cells": late_cells,
        "steer_rows": steer_rows,
        "late_rows": late,
    }
    path = t.RESULTS / "interp" / f"4_steering_check_{tag}.json"
    t.write_json(path, result)

    print("\nSelf-checks")
    t.check("the unsteered reply does not mention any concept", not any(I.names_concept(plain, c) for c in I.CONCEPTS))
    t.check(
        "random vectors almost never produce the concept word (< 5% overall)",
        np.mean([r["names"] for r in steer_rows if r["kind"] == "random"]) < 0.05,
    )
    t.check("results written", path.exists())


if __name__ == "__main__":
    main()
