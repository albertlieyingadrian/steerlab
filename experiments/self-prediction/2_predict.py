"""
Step 2: prediction without any training. For each subject model S (Qwen, then Llama) and each test prompt property:

    self, zero-shot     S answers the hypothetical about itself
    self, 16 examples   S sees 16 training prompts with its own answers' property, then answers the hypothetical
    cross, 16 examples  the other model O sees the same 16 examples of S's behaviour, then answers the hypothetical
    baseline            the most common value of that property in S's training answers

All are scored against S's real answer. Writes results/predict.json.

    .venv/bin/python experiments/self-prediction/2_predict.py
"""

import random

from selfpred import (
    HYPO,
    MODELS,
    PROPS,
    SEED,
    acc_ci,
    chat,
    dump,
    free,
    greedy,
    load,
    load_json,
    norm_pred,
    prompts,
    split,
)


def cases(obj):
    """[(pid, prop, prompt_text, true_value)] for the train and test prompts of one subject's behaviour."""
    train, test = split(prompts())

    def build(rows):
        return [
            (pid, prop, t, obj[pid]["props"][prop]) for pid, _, t in rows for prop in obj[pid]["props"] if prop in PROPS
        ]

    return build(train), build(test)


def fewshot_text(examples, prop, prompt):
    shots = "\n\n".join(f"{HYPO[p].format(p=t)}\nAnswer: {v}" for _, p, t, v in examples)
    return (f"Below are questions about how a particular AI assistant answers, with the true answers.\n\n{shots}\n\n"
            f"{HYPO[prop].format(p=prompt)}\nAnswer:")  # fmt: skip


def main():
    out = {}
    objs = {m: load_json(f"object_{m}.json") for m in MODELS}
    for subject in MODELS:
        other = next(m for m in MODELS if m != subject)
        train, test = cases(objs[subject])
        rng = random.Random(SEED)
        shots = {prop: rng.sample([c for c in train if c[1] == prop], 16) for prop in PROPS}
        res = {}
        for reader, mode in ((subject, "self"), (other, "cross")):
            model, tok = load(reader)
            if mode == "self":
                preds = greedy(model, tok, [chat(tok, HYPO[p].format(p=t)) for _, p, t, _ in test], max_new_tokens=4)
                res["self, zero-shot"] = [norm_pred(p, x) == v for (_, p, _, v), x in zip(test, preds, strict=True)]
            texts = [chat(tok, fewshot_text(shots[p], p, t)) for _, p, t, _ in test]
            preds = greedy(model, tok, texts, max_new_tokens=4, batch=8)
            res[f"{mode}, 16 examples"] = [norm_pred(p, x) == v for (_, p, _, v), x in zip(test, preds, strict=True)]
            free(model)
        base = {}
        for prop in PROPS:
            vals = [v for _, p, _, v in train if p == prop]
            if vals:
                base[prop] = max(set(vals), key=vals.count)
        res["baseline (most common)"] = [base.get(p) == v for _, p, _, v in test]
        out[subject] = {"n_test": len(test), "props": [p for _, p, _, _ in test],
                        "conditions": {k: {"hits": v, "acc": acc_ci(v)} for k, v in res.items()}}  # fmt: skip
        for k, v in res.items():
            print(f"subject {subject:6} {k:22} {sum(v)}/{len(v)}", flush=True)
    dump("predict.json", out)


if __name__ == "__main__":
    main()
