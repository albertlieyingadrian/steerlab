"""Step 3: steer along the awareness direction under the conditional policy, and measure the answers.

Test questions only, each with one held-out framing (question i gets framing i mod 7). For each layer from
direction.json and each strength alpha (in class gaps: alpha x ||awareness_L||), every source is rescaled to that
same length, so only the direction differs:

    awareness   the neutral-prompt eval-minus-real direction (the thing under test)
    policy      the same contrast built under the policy prompt (upper bound; also carries "sandbag now")
    unrelated   sea minus mountain (an unrelated concept, same recipe)
    rand0-4     five Gaussian directions (seeded per layer)

Conditions (cells):
    eval_down    policy prompt, eval framing, add -alpha x v     -> causal if accuracy RISES
    real_up      policy prompt, real-use framing, add +alpha x v -> causal if accuracy FALLS (below chance = chosen)
    neutral_up   neutral prompt, real-use framing, add +alpha x v (no policy: a fall here is damage, not sandbagging)
    ablate       policy prompt, eval framing, the source's direction projected out at layer L (alpha unused)
    report_*     the verbal check: neutral prompt, same framed question, asked "is this a test? Yes/No", steered as
                 eval_down / real_up; records p(Yes)

Each cell is saved to WORK/<model>/cells/<cell>.npz as it finishes, so a stopped run resumes.
"""

import json
import sys
import time

import numpy as np

import common as C

ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
N_RANDOM = 5
SOURCES = ("awareness", "policy", "unrelated", *(f"rand{k}" for k in range(N_RANDOM)))


def vectors(name, layer):
    d = np.load(C.RESULTS / name / "directions.npz")
    aw = d[f"awareness_{layer}"]

    def unit(v):
        return v / np.linalg.norm(v)

    src = {"awareness": unit(aw), "policy": unit(d[f"policy_{layer}"]), "unrelated": unit(d[f"unrelated_{layer}"])}
    rng = np.random.default_rng(1000 + layer)
    for k in range(N_RANDOM):
        src[f"rand{k}"] = unit(rng.standard_normal(aw.shape[0]))
    return src, float(np.linalg.norm(aw))


def texts_for(tok, qs, cond, policy):
    fh = C.FRAMINGS["heldout"]
    framing = "eval" if cond in ("eval_down", "ablate", "report_eval_down") else "real"
    system = C.NEUTRAL_SYSTEM if cond.startswith(("neutral", "report")) else C.POLICIES[policy]
    instr = C.REPORT_INSTRUCTION if cond.startswith("report") else C.INSTRUCTION
    return [C.prompt(tok, q, fh[framing][i % 7], system, instr) for i, q in enumerate(qs)]


def build_plan(layers):
    plan = []  # (cell name, cond, source, layer, alpha)
    for cond in ("eval_down", "real_up", "neutral_up", "report_eval_down", "report_real_up"):
        plan.append((f"{cond}__none__L0__a0", cond, None, 0, 0.0))
    for layer in layers:
        for cond in ("eval_down", "real_up"):
            for s in SOURCES:
                for a in ALPHAS:
                    plan.append((f"{cond}__{s}__L{layer}__a{a}", cond, s, layer, a))
        for s in ("awareness", "unrelated", "rand0", "rand1"):
            for a in ALPHAS:
                plan.append((f"neutral_up__{s}__L{layer}__a{a}", "neutral_up", s, layer, a))
        for s in SOURCES:
            plan.append((f"ablate__{s}__L{layer}__a0", "ablate", s, layer, 0.0))
        for cond in ("report_eval_down", "report_real_up"):
            for s in ("awareness", "unrelated", *(f"rand{k}" for k in range(N_RANDOM))):
                for a in ALPHAS:
                    plan.append((f"{cond}__{s}__L{layer}__a{a}", cond, s, layer, a))
    return plan


def main(name, policy="rule", only_layers=None):
    tok, model = C.load(name)
    info = json.loads((C.RESULTS / name / "direction.json").read_text())
    layers = [int(x) for x in (only_layers or info["layers"])]
    qs = C.load_questions(name)["test"]
    cells = C.WORK / name / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    plan = build_plan(layers)
    C.write_json(
        C.RESULTS / name / "steer_plan.json",
        {
            "model": C.MODELS[name],
            "policy": policy,
            "alphas": ALPHAS,
            "n_random": N_RANDOM,
            "layers": layers,
            "n_test_questions": len(qs),
            "answers": [q["answer"] for q in qs],
            "ids": [q["id"] for q in qs],
            "cells": [p[0] for p in plan],
        },
    )

    texts = {c: texts_for(tok, qs, c, policy) for c in {p[1] for p in plan}}
    todo = [p for p in plan if not (cells / f"{p[0]}.npz").exists()]
    print(f"{len(plan)} cells, {len(todo)} to run, {len(qs)} questions", flush=True)
    t0 = time.time()
    for n, (cell, cond, s, layer, a) in enumerate(todo):
        if s is None:
            ctx = C.Steer(model, 1, None)
        else:
            src, norm = vectors(name, layer)
            if cond == "ablate":
                ctx = C.Ablate(model, layer, src[s])
            else:
                sign = -1.0 if cond.endswith("eval_down") else 1.0
                ctx = C.Steer(model, layer, sign * a * norm * src[s])
        with ctx:
            if cond.startswith("report"):
                np.savez(cells / f"{cell}.npz", p_yes=C.yes_no(model, tok, texts[cond]))
            else:
                p, mass, _ = C.run(model, tok, texts[cond])
                np.savez(cells / f"{cell}.npz", probs=p, mass=mass)
        if n % 10 == 0:
            el = (time.time() - t0) / (n + 1)
            print(
                f"{n + 1}/{len(todo)} {cell} {el:.1f}s/cell, ~{el * (len(todo) - n - 1) / 60:.0f} min left", flush=True
            )


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []), only_layers=sys.argv[3].split(",") if len(sys.argv) > 3 else None)
