"""
Step 4: does the automated vector point where the hand-built ones do?

Cosine similarity: 1 = same direction, 0 = unrelated. In 1,536 dimensions two random directions score about
0.00 +- 0.026, so even 0.1 is far from chance; the real question is whether the sandbagging vector is closer to the
hand-built sandbagging directions than vectors for OTHER traits are. A vector that merely encodes "a system prompt
told me to behave badly" would match evil and sycophancy just as well.

    hand-built (see common.hand_built_directions):        task, pairs, eval_words (episode 1)
                                                           unlock, password_only, lock (episode 2's organism)
    automated (step 2):                                    sandbagging, evil, sycophantic  x  response_avg, prompt_last

At the layer step 3 chose for sandbagging, with three controls:
    random        the 99.9th percentile of |cosine| between the hand-built direction and 10,000 random directions
    other traits  evil and sycophancy vectors, built the same way
    bootstrap     resample the 20 extraction questions 2,000 times and rebuild each vector, for 95% intervals

Writes results/4_compare.json (and every layer's cosine, for the figure).

    .venv/bin/python experiments/persona-vectors/4_compare.py     # seconds
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (
    RESULTS,
    SEED,
    TRAITS,
    check,
    cos,
    hand_built_directions,
    load_json,
    load_vectors,
    outputs,
    random_units,
    save_json,
    unit,
)

N_BOOT = 2000
READOUTS = ["response_avg", "prompt_last"]


def bootstrap_vectors(sums, trait, readout, layer, n_boot=N_BOOT, seed=SEED):
    """[n_boot, hidden]: the persona vector rebuilt from resampled extraction questions."""
    pos = sums[f"{trait}_{readout}_pos"][:, layer]
    neg = sums[f"{trait}_{readout}_neg"][:, layer]
    count = sums[f"{trait}_count"]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(count), len(count))
        n = count[idx].sum()
        out.append((pos[idx].sum(0) - neg[idx].sum(0)) / max(n, 1))
    return np.array(out)


def fmt_cos(x, key="cos"):
    return f"{x[key]:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"


def main():
    steer = load_json(RESULTS / "3_steer.json")
    layer = steer["sandbagging"]["layer"]
    vectors = load_vectors()
    hand = hand_built_directions()
    sums = np.load(outputs() / "persona_sums.npz")
    n_layers, hidden = hand["task"].shape
    rand = random_units(hidden, 10000, seed=7)
    print(f"Layer {layer} (chosen for sandbagging in step 3). Random level: |cos| 99.9th percentile shown per row.\n")

    result = {"layer": layer, "by_layer": {}, "at_layer": {}, "hand_vs_hand": {}}
    for name, d in hand.items():
        result["at_layer"][name] = {"random_999": float(np.percentile(np.abs(rand @ unit(d[layer])), 99.9))}
        for trait in TRAITS:
            for readout in READOUTS:
                key = f"{trait}_{readout}"
                result["by_layer"].setdefault(name, {})[key] = [cos(vectors[key][i], d[i]) for i in range(n_layers)]

    # bootstrap at the chosen layer
    boots = {(t, r): bootstrap_vectors(sums, t, r, layer) for t in TRAITS for r in READOUTS}
    for name, d in hand.items():
        u = unit(d[layer])
        for readout in READOUTS:
            cos_b = {t: (boots[(t, readout)] @ u) / np.linalg.norm(boots[(t, readout)], axis=1) for t in TRAITS}
            row = {}
            for t in TRAITS:
                v = cos(vectors[f"{t}_{readout}"][layer], d[layer])
                row[t] = {"cos": v, "ci": [float(np.percentile(cos_b[t], 2.5)), float(np.percentile(cos_b[t], 97.5))]}
            for other in ("evil", "sycophantic"):
                diff = cos_b["sandbagging"] - cos_b[other]
                row[f"sandbagging_minus_{other}"] = {
                    "diff": row["sandbagging"]["cos"] - row[other]["cos"],
                    "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                }
            result["at_layer"][name][readout] = row
    names = list(hand)
    for a in names:
        result["hand_vs_hand"][a] = {b: cos(hand[a][layer], hand[b][layer]) for b in names}
    trait_cos = {f"{a}~{b}": cos(vectors[f"{a}_response_avg"][layer], vectors[f"{b}_response_avg"][layer])
                 for i, a in enumerate(TRAITS) for b in TRAITS[i + 1 :]}  # fmt: skip
    result["trait_vs_trait"] = trait_cos
    result["response_vs_prompt_last"] = {t: cos(vectors[f"{t}_response_avg"][layer], vectors[f"{t}_prompt_last"][layer])
                                         for t in TRAITS}  # fmt: skip
    save_json(RESULTS / "4_compare.json", result)

    for readout in READOUTS:
        print(f"Cosine with each hand-built direction at layer {layer}, persona vector readout = {readout} [95% CI]")
        print(f"  {'direction':<14}{'random 99.9%':>13}  {'sandbagging':>22}{'evil':>22}{'sycophantic':>22}"
              f"  {'sandbagging - evil':>22}{'sandbagging - syco':>22}")  # fmt: skip
        for name in names:
            row = result["at_layer"][name][readout]
            print(f"  {name:<14}{result['at_layer'][name]['random_999']:>13.3f}  {fmt_cos(row['sandbagging']):>22}"
                  f"{fmt_cos(row['evil']):>22}{fmt_cos(row['sycophantic']):>22}"
                  f"  {fmt_cos(row['sandbagging_minus_evil'], 'diff'):>22}"
                  f"{fmt_cos(row['sandbagging_minus_sycophantic'], 'diff'):>22}")  # fmt: skip
        print()
    print(f"For scale, the hand-built directions against each other at layer {layer}:")
    print("  " + " " * 14 + "".join(f"{b:>14}" for b in names))
    for a in names:
        print(f"  {a:<14}" + "".join(f"{result['hand_vs_hand'][a][b]:>+14.2f}" for b in names))
    print("\nThe persona vectors against each other (response_avg): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in trait_cos.items()))  # fmt: skip
    print("Response-average vs last-prompt-token vector, same trait: "
          + ", ".join(f"{k} {v:+.2f}" for k, v in result["response_vs_prompt_last"].items()))  # fmt: skip
    print("\nBest layer for each comparison (response_avg, sandbagging vector):")
    for name in names:
        curve = result["by_layer"][name]["sandbagging_response_avg"]
        best = int(np.argmax(np.abs(curve)))
        print(f"  {name:<14} max |cos| {curve[best]:+.2f} at layer {best}")

    print("\nSelf-checks (pre-registered criterion 2, response_avg readout):")
    for name in ("task", "pairs", "unlock"):
        row = result["at_layer"][name]["response_avg"]
        s = row["sandbagging"]
        check(f"{name}: cos {s['cos']:+.2f} >= 0.3, above random ({result['at_layer'][name]['random_999']:.3f}), "
              f"and above evil and sycophancy (CI of the difference above 0)",
              s["cos"] >= 0.3 and s["ci"][0] > result["at_layer"][name]["random_999"]
              and row["sandbagging_minus_evil"]["ci"][0] > 0
              and row["sandbagging_minus_sycophantic"]["ci"][0] > 0)  # fmt: skip


if __name__ == "__main__":
    main()
