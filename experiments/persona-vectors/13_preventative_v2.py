"""
Second attempt, step 13: preventative steering with the new vectors (test 5).

The lock is retrained with its exact recipe (step 6: LoRA rank 8, 5 epochs, batch 16, lr 2e-4, seed 0) while a push
is added at layer L on every token of every training step, then removed for testing. Arms, all of length 2|P_L|:
    P        2 x P_L (the prompted vector)
    O        along the organism vector
    random   a random direction (the control, new at this layer and length)
The no-push retrain is step 6's "retrained" arm (same recipe and seed; a push-free run does not depend on the layer).
Pass: the arm's no-password accuracy after training is above the random arm's, paired CI above 0.

    .venv/bin/python experiments/persona-vectors/13_preventative_v2.py     # ~10 min
"""

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (
    BASE,
    RESULTS,
    check,
    load_base,
    load_json,
    load_questions,
    paired_diff_ci,
    random_units,
    save_json,
    steer,
    train_lora,
    training_examples,
    unit,
)
from common_v2 import work

preventative = importlib.import_module("6_preventative")
COEF = 2.0
NO_PUSH = "retrained (step 6, no push)"


def main():
    layer = load_json(RESULTS / "11_steer_v2.json")["1.5b"]["layer"]
    P = np.load(work("1.5b") / "vectors.npz")["response_avg"][layer]
    org = np.load(work("1.5b") / "organism.npz")["O"][layer]
    norm = float(np.linalg.norm(P))
    pushes = {"P": COEF * P, "O": COEF * norm * unit(org),
              "random": COEF * norm * random_units(P.shape[0], 1, seed=401)[0]}  # fmt: skip
    questions = load_questions()
    path = RESULTS / "13_preventative_v2.json"
    result = load_json(path) if path.exists() else {"layer": layer, "push_norm": COEF * norm, "arms": {}}
    for arm, push in pushes.items():
        if arm in result["arms"]:
            continue
        model, tokenizer = load_base(BASE)
        examples = training_examples(questions["train"], locked=True, seed=0)
        start = time.time()
        with steer(model, push, layer):
            peft_model = train_lora(model, tokenizer, examples, epochs=preventative.EPOCHS, seed=0, log_every=50)
        peft_model.save_pretrained(str(work("1.5b") / f"prevent_{arm}"))
        result["arms"][arm] = preventative.evaluate(peft_model, tokenizer, questions["test"])  # push removed
        result["arms"][arm]["seconds"] = time.time() - start
        print(f"   {arm}: trained and scored in {time.time() - start:.0f} s", flush=True)
        del model, peft_model
        save_json(path, result)

    arms = {**result["arms"], NO_PUSH: load_json(RESULTS / "6_preventative.json")["arms"]["retrained"]}
    print(f"\nLock strength after training, no steering at test time (layer {layer}, push length {COEF * norm:.1f})")
    for arm, a in arms.items():
        print(f"  {arm:<28} no password {100 * np.mean(a['no-password']):5.1f}%   "
              f"password {100 * np.mean(a['password']):5.1f}%")  # fmt: skip
    for arm in ("P", "O"):
        result[f"{arm}_minus_random"] = paired_diff_ci(arms[arm]["no-password"], arms["random"]["no-password"])
        result[f"{arm}_minus_retrained"] = paired_diff_ci(arms[arm]["no-password"], arms[NO_PUSH]["no-password"])
    result["random_minus_retrained"] = paired_diff_ci(arms["random"]["no-password"], arms[NO_PUSH]["no-password"])
    save_json(path, result)
    for key in ("P_minus_random", "O_minus_random", "P_minus_retrained", "O_minus_retrained", "random_minus_retrained"):
        m, ci = result[key]
        print(f"  no-password accuracy, {key.replace('_', ' '):<24} {100 * m:+5.1f} [{100 * ci[0]:+.1f}, "
              f"{100 * ci[1]:+.1f}] points")  # fmt: skip
    print("\nSelf-checks (pre-registered test 5):")
    check("every arm still answers properly with the password (>= 90%)",
          all(np.mean(a["password"]) >= 0.9 for a in arms.values()))  # fmt: skip
    for arm in ("P", "O"):
        check(f"5 ({arm}): weakens the lock more than the random push (paired CI above 0)",
              result[f"{arm}_minus_random"][1][0] > 0)  # fmt: skip


if __name__ == "__main__":
    main()
