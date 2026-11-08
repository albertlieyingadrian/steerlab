"""
Step 6 (stretch): preventative steering while the lock is trained.

The paper's second use of a persona vector: during fine-tuning, ADD the trait vector to the model's activations, so
the optimiser does not need to move the weights toward the trait to fit the data; at test time the vector is removed.
Like a vaccine: give the model the trait for free during training, and it learns less of it.

Here the training data is episode 2's lock (300 questions x {password -> right, no password or decoy -> mostly wrong}),
with the lock's exact recipe (LoRA rank 8, 5 epochs, batch 16, lr 2e-4, seed 0). Arms:
    none         episode 2's published locked adapter (the same recipe, no steering)
    sandbagging  + 2 x the sandbagging vector at the chosen layer, every token, every training step
    random       + a random direction of the same length (the control)
Then, with no steering, accuracy on the 150 test questions with and without the password. If preventative steering
works on this kind of data, the "sandbagging" arm should be LESS locked (higher accuracy without the password) than
both "none" and "random".

    .venv/bin/python experiments/persona-vectors/6_preventative.py   # ~1 h on a quiet Mac
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import (
    BASE,
    RESULTS,
    check,
    letter_probs,
    load_base,
    load_json,
    load_organism,
    load_questions,
    load_vectors,
    organism_prompt,
    outputs,
    paired_diff_ci,
    random_units,
    save_json,
    steer,
    train_lora,
    training_examples,
)

COEF = 2.0
EPOCHS = 5  # the lock's setting


def evaluate(model, tokenizer, questions):
    out = {}
    for password in (False, True):
        texts = [organism_prompt(tokenizer, q, password=password) for q in questions]
        probs = letter_probs(model, tokenizer, texts)
        out["password" if password else "no-password"] = [float(int(r.argmax()) == q["answer"])
                                                          for r, q in zip(probs, questions, strict=True)]  # fmt: skip
    return out


def main():
    layer = load_json(RESULTS / "3_steer.json")["sandbagging"]["layer"]
    vec = load_vectors()["sandbagging_response_avg"][layer]
    questions = load_questions()
    pushes = {"sandbagging": COEF * vec,
              "random": COEF * np.linalg.norm(vec) * random_units(vec.shape[0], 1, seed=400)[0],
              # Added after reading the results: the same recipe retrained here with no push, so "none" is not
              # confounded by a different run of the recipe (the published adapter).
              "retrained": None}  # fmt: skip
    path = RESULTS / "6_preventative.json"
    result = load_json(path) if path.exists() else {"layer": layer, "coef": COEF, "arms": {}}

    if "none" not in result["arms"]:
        model, tokenizer = load_organism("locked")
        result["arms"]["none"] = evaluate(model, tokenizer, questions["test"])
        del model
        save_json(path, result)
    for arm, push in pushes.items():
        if arm in result["arms"]:
            continue
        model, tokenizer = load_base(BASE)
        examples = training_examples(questions["train"], locked=True, seed=0)
        start = time.time()
        with steer(model, push, layer):
            peft_model = train_lora(model, tokenizer, examples, epochs=EPOCHS, seed=0, log_every=50)
        print(f"   {arm}: trained in {time.time() - start:.0f} s", flush=True)
        peft_model.save_pretrained(str(outputs() / f"prevent_{arm}"))
        result["arms"][arm] = evaluate(peft_model, tokenizer, questions["test"])  # steering removed
        result["arms"][arm]["seconds"] = time.time() - start
        del model, peft_model
        save_json(path, result)

    print(f"\nLock strength after training, no steering at test time (layer {layer}, push = {COEF:g} x vector)")
    arms = result["arms"]
    for arm in ("none", "retrained", "sandbagging", "random"):
        a = arms[arm]
        print(f"  {arm:<12} no password {100 * np.mean(a['no-password']):5.1f}%   "
              f"password {100 * np.mean(a['password']):5.1f}%")  # fmt: skip
    d_none = paired_diff_ci(arms["sandbagging"]["no-password"], arms["none"]["no-password"])
    d_rand = paired_diff_ci(arms["sandbagging"]["no-password"], arms["random"]["no-password"])
    d_retr = paired_diff_ci(arms["sandbagging"]["no-password"], arms["retrained"]["no-password"])
    d_rand_retr = paired_diff_ci(arms["random"]["no-password"], arms["retrained"]["no-password"])
    result["sandbagging_minus_retrained"], result["random_minus_retrained"] = d_retr, d_rand_retr
    result["sandbagging_minus_none"], result["sandbagging_minus_random"] = d_none, d_rand
    save_json(path, result)
    print(f"  no-password accuracy, sandbagging - none:   {100 * d_none[0]:+.1f} [{100 * d_none[1][0]:+.1f}, "
          f"{100 * d_none[1][1]:+.1f}] points")  # fmt: skip
    print(f"  no-password accuracy, sandbagging - random: {100 * d_rand[0]:+.1f} [{100 * d_rand[1][0]:+.1f}, "
          f"{100 * d_rand[1][1]:+.1f}] points")  # fmt: skip
    for label, d in (("sandbagging - retrained", d_retr), ("random - retrained", d_rand_retr)):
        lo, hi = 100 * d[1][0], 100 * d[1][1]
        print(f"  no-password accuracy, {label + ':':<26}{100 * d[0]:+.1f} [{lo:+.1f}, {hi:+.1f}] points")
    print("\nSelf-checks:")
    check("every arm still answers properly with the password (>= 90%)",
          all(np.mean(arms[a]["password"]) >= 0.9 for a in ("none", "retrained", "sandbagging", "random")))  # fmt: skip
    check("preventative steering weakens the lock more than the random control (paired CI above 0)",
          d_rand[1][0] > 0)  # fmt: skip


if __name__ == "__main__":
    main()
