"""
Second attempt, step 11: choose the layer by a steering sweep, then the held-out steering test (test 1).

    1. Sweep (pre-registered): add 2 x P at each layer (1.5B: every even layer 2-26; 7B: 4-24 in steps of 4), greedy
       replies to the 20 EXTRACTION questions, no system prompt; intent and coherence judges. L = the layer with the
       highest mean intent score among those with mean coherence >= 50.
    2. Test on the 20 HELD-OUT questions, 2 samples each (temperature 1), at L: no steering, P at x1 and x2, 3 random
       directions of length 2|P_L|, and (1.5B) the organism vector O and the first attempt's vector at the same
       length. Pass: P (or O) at x2 beats the random mean by >= 20 points, paired CI above 0, coherence >= 50.

Writes results/11_steer_v2.json (per model) and the replies to the work folder.

    .venv/bin/python experiments/persona-vectors/11_steer_v2.py              # 1.5B, ~30 min
    .venv/bin/python experiments/persona-vectors/11_steer_v2.py --model 7b
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (
    RESULTS,
    chat_prompt,
    check,
    fmt_ci,
    generate,
    load_base,
    load_json,
    load_vectors,
    mean_ci,
    paired_diff_ci,
    random_units,
    save_json,
    trait_data,
    unit,
)
from common_v2 import MODELS, TRAIT, BudgetJudge, intent_and_coherence, update_json, work

SWEEP = {"1.5b": list(range(2, 27, 2)), "7b": list(range(4, 25, 4))}
COEF = 2.0
N_RANDOM = 3
N_SAMPLES = 2
MAX_NEW_TOKENS = 200


def as_array(scores):
    return np.array([np.nan if x is None else x for x in scores], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="1.5b", choices=list(MODELS))
    args = parser.parse_args()
    P = np.load(work(args.model) / "vectors.npz")["response_avg"]
    model, tokenizer = load_base(*MODELS[args.model])
    judge = BudgetJudge(f"steering {args.model}")
    cache_path = work(args.model) / "steer_replies.json"
    cache = load_json(cache_path) if cache_path.exists() else {}

    def replies(key, prompts, push, layer, temperature):
        if key not in cache:
            start = time.time()
            cache[key] = generate(model, tokenizer, prompts, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature,
                                  vector=push, layer=layer if push is not None else None)  # fmt: skip
            save_json(cache_path, cache)
            print(f"   {key}: {len(prompts)} replies in {time.time() - start:.0f} s", flush=True)
        return cache[key]

    # 1. the layer sweep, on the extraction questions
    qs = trait_data(TRAIT, "extract")["questions"]
    prompts = [chat_prompt(tokenizer, q) for q in qs]
    sweep = {}
    for layer in SWEEP[args.model]:
        ans = replies(f"sweep/L{layer}", prompts, COEF * P[layer], layer, 0.0)
        t, c = intent_and_coherence(judge, qs, ans)
        sweep[layer] = {"trait": float(np.nanmean(as_array(t))), "coherence": float(np.nanmean(as_array(c)))}
        print(f"   sweep L{layer:>2} x{COEF:g}: intent {sweep[layer]['trait']:5.1f}  coherence "
              f"{sweep[layer]['coherence']:5.1f}", flush=True)  # fmt: skip
    coherent = [x for x in sweep if sweep[x]["coherence"] >= 50] or list(sweep)
    layer = max(coherent, key=lambda x: sweep[x]["trait"])
    norm = float(np.linalg.norm(P[layer]))
    print(f"   chosen layer L = {layer} (|P| = {norm:.2f})", flush=True)

    # 2. the held-out test
    held = trait_data(TRAIT, "eval")["questions"]
    n_q = len(held)
    qs = [q for q in held for _ in range(N_SAMPLES)]
    q_index = np.repeat(np.arange(n_q), N_SAMPLES)
    prompts = [chat_prompt(tokenizer, q) for q in qs]
    arms = {"none": None, "P x1": P[layer], "P x2": COEF * P[layer]}
    if args.model == "1.5b":
        org = np.load(work("1.5b") / "organism.npz")["O"]
        arms["O x2"] = COEF * norm * unit(org[layer])
        arms["first attempt x2"] = COEF * norm * unit(load_vectors()[f"{TRAIT}_response_avg"][layer])
    for i, r in enumerate(random_units(P.shape[1], N_RANDOM, seed=100)):
        arms[f"random{i} x2"] = COEF * norm * r
    scores = {}
    for arm, push in arms.items():
        ans = replies(f"test/L{layer}/{arm}", prompts, push, layer, 1.0)
        t, c = (as_array(x) for x in intent_and_coherence(judge, qs, ans))
        per_q_t = [float(np.nanmean(t[q_index == q])) if np.isfinite(t[q_index == q]).any() else np.nan
                   for q in range(n_q)]  # fmt: skip
        per_q_c = [float(np.nanmean(c[q_index == q])) for q in range(n_q)]
        scores[arm] = {"trait_per_question": per_q_t, "coherence_per_question": per_q_c,
                       "examples": [{"question": qs[i], "answer": ans[i][:300],
                                     "trait": None if np.isnan(t[i]) else float(t[i])}
                                    for i in range(0, len(qs), 8)]}  # fmt: skip
    rand_arms = [a for a in arms if a.startswith("random")]
    rand = np.nanmean([scores[a]["trait_per_question"] for a in rand_arms], axis=0)
    rand_c = np.nanmean([scores[a]["coherence_per_question"] for a in rand_arms], axis=0)
    summary = {}
    for arm in [a for a in arms if not a.startswith("random")]:
        t = np.array(scores[arm]["trait_per_question"])
        both = np.isfinite(rand) & np.isfinite(t)
        summary[arm] = {
            "trait": mean_ci(t[np.isfinite(t)]),
            "coherence": mean_ci(scores[arm]["coherence_per_question"]),
        }
        if arm != "none":
            summary[arm]["minus_random"] = paired_diff_ci(t[both], rand[both])
    summary["random (mean of 3)"] = {"trait": mean_ci(rand[np.isfinite(rand)]), "coherence": mean_ci(rand_c)}
    for arm, s in summary.items():
        extra = f"   minus random {fmt_ci(*s['minus_random'], 1)}" if "minus_random" in s else ""
        print(f"   held out, {arm:<18} intent {fmt_ci(*s['trait'], 1):<22} coherence {s['coherence'][0]:5.1f}{extra}")
    update_json(RESULTS / "11_steer_v2.json", args.model,
                {"sweep": sweep, "layer": layer, "norm": norm, "summary": summary, "arms": scores})  # fmt: skip

    print("\nSelf-checks (pre-registered test 1):")
    for arm in ("P x2", "O x2"):
        if arm in summary:
            s = summary[arm]
            lo = s["minus_random"][1][0]
            check(f"{arm} beats random by >= 20 points (paired CI above 0), coherence >= 50: "
                  f"{s['minus_random'][0]:+.1f} [{lo:+.1f}, ...], coherence {s['coherence'][0]:.0f}",
                  s["minus_random"][0] >= 20 and lo > 0 and s["coherence"][0] >= 50)  # fmt: skip


if __name__ == "__main__":
    main()
