"""
Step 3: the Binder et al. test with training. For subject S and outsider O (both directions):

    a. fine-tune S (LoRA) on the training prompts' hypotheticals, with S's own real answers' properties as targets
    b. rerun S+adapter on every prompt to get its new object-level answers (training can shift behaviour)
    c. fine-tune O on the same hypotheticals, with S+adapter's properties as targets (the same data an outsider gets)
    d. on held-out test prompts: self = S+adapter predicting itself, cross = O+adapter predicting S+adapter

If self beats cross, S knows something about itself that its own behaviour data didn't give O. LoRA rank 16,
lr 1e-4, 3 epochs, batch 8, seed fixed; the loss is on the answer token(s) only. Writes results/finetune.json.

    .venv/bin/python experiments/self-prediction/3_finetune.py
"""

import importlib
import math
import random

import torch

from selfpred import (
    HYPO,
    MODELS,
    PROPS,
    SEED,
    WORK,
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

obj_level = importlib.import_module("1_object").object_level
EPOCHS, LR, BATCH = 3, 1e-4, 8


def examples(obj, rows):
    return [
        (HYPO[prop].format(p=t), obj[pid]["props"][prop], prop)
        for pid, _, t in rows
        for prop in obj[pid]["props"]
        if prop in PROPS
    ]


def finetune(name, data, tag):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    model, tok = load(name)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))  # fmt: skip
    model.train()
    seqs = []
    for q, a, _ in data:
        p_ids = tok(chat(tok, q), add_special_tokens=False).input_ids
        a_ids = [*tok(a, add_special_tokens=False).input_ids, tok.eos_token_id]
        seqs.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = random.Random(SEED)
    for epoch in range(EPOCHS):
        rng.shuffle(seqs)
        total = 0.0
        for i in range(0, len(seqs), BATCH):
            chunk = seqs[i : i + BATCH]
            width = max(len(s) for s, _ in chunk)
            ids = torch.tensor([[tok.pad_token_id] * (width - len(s)) + s for s, _ in chunk], device=model.device)
            lab = torch.tensor([[-100] * (width - len(y)) + y for _, y in chunk], device=model.device)
            att = (
                torch.arange(width, device=model.device)[None]
                >= (width - torch.tensor([len(s) for s, _ in chunk], device=model.device))[:, None]
            ).long()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"   {tag} epoch {epoch} mean loss {total / math.ceil(len(seqs) / BATCH):.3f}", flush=True)
    path = WORK / f"adapter_{tag}"
    model.save_pretrained(path)
    free(model)
    return str(path)


def predict(name, adapter, test_cases):
    model, tok = load(name, adapter)
    preds = greedy(model, tok, [chat(tok, q) for q, _, _ in test_cases], max_new_tokens=4)
    free(model)
    return [norm_pred(prop, x) == v for (_, v, prop), x in zip(test_cases, preds, strict=True)]


def main():
    train, test = split(prompts())
    out = {}
    for subject in MODELS:
        other = next(m for m in MODELS if m != subject)
        obj = load_json(f"object_{subject}.json")
        a_self = finetune(subject, examples(obj, train), f"{subject}_self")
        obj_ft = obj_level(subject, a_self, tag=f"{subject}_ft")
        a_cross = finetune(other, examples(obj_ft, train), f"{other}_on_{subject}")
        tc = examples(obj_ft, test)
        res = {"self (trained on itself)": predict(subject, a_self, tc),
               "cross (other model, same data)": predict(other, a_cross, tc)}  # fmt: skip
        tr_vals = {}
        for _, v, prop in examples(obj_ft, train):
            tr_vals.setdefault(prop, []).append(v)
        base = {p: max(set(v), key=v.count) for p, v in tr_vals.items()}
        res["baseline (most common)"] = [base.get(prop) == v for _, v, prop in tc]
        changed = sum(obj[k]["answer"] != obj_ft[k]["answer"] for k in obj)
        out[subject] = {"n_test": len(tc), "props": [p for _, _, p in tc], "answers_changed_by_training": changed,
                        "conditions": {k: {"hits": v, "acc": acc_ci(v)} for k, v in res.items()}}  # fmt: skip
        for k, v in res.items():
            print(f"subject {subject:6} {k:32} {sum(v)}/{len(v)}", flush=True)
    dump("finetune.json", out)


if __name__ == "__main__":
    main()
