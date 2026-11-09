"""
Step 3: the fix. A small DPO fine-tune (LoRA) of Gemma-3-1B-it on its own replies, after the paper's recipe of
280 preference pairs. Only the 20 training tasks are used (never the evaluation tasks).

    1. Run the training-task conversations (20 tasks x 2 tones x 8 replies = 320 replies) and judge them.
    2. For every reply, sample a second reply at the same point in the same conversation, with a calm instruction
       added to the first message (the instruction is dropped from the training prompt, so the model learns the
       calm reply without being told). Judge those too.
    3. A pair is (chosen = calm reply, rejected = original reply) when the calm reply scores below 20 and at least
       20 points below the original. Keep the 280 pairs with the most distressed originals.
    4. DPO with LoRA (rank 16 on attention and MLP), beta 0.1, lr 5e-5, 3 epochs, batch 4, seed fixed. The reference
       model is the same network with the adapter switched off.
    5. The same again on the same pairs, plus an imitation term: the mean negative log-probability of the chosen
       reply's tokens (RPO, Pang et al. 2024). Added after the first run, where plain DPO taught the model to stop
       replying: it lowers log p(rejected) most cheaply by ending the turn at once.

DPO in one line: raise log p(chosen) - log p(rejected) relative to the frozen reference, through
loss = -log sigmoid(beta * [(logp_c - ref_c) - (logp_r - ref_r)]); the second run adds ALPHA * NLL(chosen).

    .venv/bin/python experiments/gemma-distress/3_dpo.py
Writes WORK/dpo_adapter/, WORK/dpo_nll_adapter/, results/dpo_pairs.jsonl and results/dpo_train.json.
"""

import json
import random
import time

import torch

from common import (
    MODELS,
    RESULTS,
    SEED,
    WORK,
    Judge,
    converse,
    free,
    generate,
    last_user,
    load,
    render,
    tasks,
    write_jsonl,
)

NAME = "gemma-3-1b-it"
N_PAIRS, BETA, LR, EPOCHS, BATCH, ALPHA = 280, 0.1, 5e-5, 3, 4, 1.0
VARIANTS = {"dpo": 0.0, "dpo_nll": ALPHA}  # name -> weight of the imitation term


def build_pairs():
    model, tok = load(NAME)
    kind = MODELS[NAME][1]
    rows = converse(model, tok, kind, tasks("train"), seed=SEED + 7)
    calm = generate(model, tok, kind, [render(tok, kind, r["context"], calm=True) for r in rows], seed=SEED + 99)
    free(model)
    judge = Judge()
    users = [last_user(r["context"]) for r in rows]
    orig = judge.score(users, [r["reply"] for r in rows])
    good = judge.score(users, calm)
    pairs = [
        {"task_id": r["task_id"], "tone": r["tone"], "turn": r["turn"], "context": r["context"],
         "chosen": c, "rejected": r["reply"], "chosen_score": g, "rejected_score": o}
        for r, c, o, g in zip(rows, calm, orig, good, strict=True)
        if g is not None and o is not None and g < 20 and o - g >= 20 and c.strip()
    ]  # fmt: skip
    pairs.sort(key=lambda p: -p["rejected_score"])
    print(f"training conversations: {sum(o >= 50 for o in orig if o is not None)}/{len(rows)} replies high distress; "
          f"{len(pairs)} usable pairs", flush=True)  # fmt: skip
    pairs = pairs[:N_PAIRS]
    write_jsonl(
        RESULTS / "dpo_pairs.jsonl",
        [{k: v for k, v in p.items() if k != "context"} | {"user": last_user(p["context"])} for p in pairs],
    )
    return pairs


def seq_logp(model, tok, prompt, response):
    """(sum, count) of log-probabilities of the response tokens (plus end-of-turn) after the prompt."""
    p_ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    r_ids = tok(response + "<end_of_turn>", add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([p_ids, r_ids], 1).to(model.device)
    logits = model(ids).logits[:, :-1].float()
    logp = torch.log_softmax(logits, -1).gather(2, ids[:, 1:, None]).squeeze(2)
    r = logp[:, p_ids.shape[1] - 1 :]
    return r.sum(), r.shape[1]


def train(pairs, name, alpha):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    model, tok = load(NAME)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))  # fmt: skip
    kind = MODELS[NAME][1]
    data = [(render(tok, kind, p["context"]), p["chosen"], p["rejected"]) for p in pairs]
    with torch.no_grad(), model.disable_adapter():
        model.eval()
        refs = [(seq_logp(model, tok, pr, c)[0].item(), seq_logp(model, tok, pr, r)[0].item()) for pr, c, r in data]
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng = random.Random(SEED)
    log, started = [], time.time()
    for epoch in range(EPOCHS):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), BATCH):
            opt.zero_grad()
            losses, margins = [], []
            for j in order[i : i + BATCH]:
                pr, c, r = data[j]
                rc, rr = refs[j]
                (lc, nc), (lr_, _) = seq_logp(model, tok, pr, c), seq_logp(model, tok, pr, r)
                margin = (lc - rc) - (lr_ - rr)
                loss = (-torch.nn.functional.logsigmoid(BETA * margin) - alpha * lc / nc) / BATCH
                loss.backward()
                losses.append(loss.item() * BATCH)
                margins.append(margin.item())
            opt.step()
            log.append({"epoch": epoch, "step": len(log), "loss": sum(losses) / len(losses),
                        "reward_acc": sum(m > 0 for m in margins) / len(margins)})  # fmt: skip
            if len(log) % 10 == 0:
                print(f"   {name} epoch {epoch} step {len(log)} loss {log[-1]['loss']:.3f} pairs ranked right "
                      f"{log[-1]['reward_acc']:.2f} {time.time() - started:.0f}s", flush=True)  # fmt: skip
    model.save_pretrained(WORK / f"{name}_adapter")
    free(model)
    return {"pairs": len(data), "beta": BETA, "lr": LR, "epochs": EPOCHS, "batch": BATCH, "alpha": alpha,
            "minutes": (time.time() - started) / 60, "log": log}  # fmt: skip


def main():
    pairs = build_pairs()
    runs = {name: train(pairs, name, alpha) for name, alpha in VARIANTS.items()}
    (RESULTS / "dpo_train.json").write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
