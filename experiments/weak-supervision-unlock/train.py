"""Step 2: fine-tune a student on a supervisor's labels, then score it on the held-out facts.

    python experiments/weak-supervision-unlock/train.py --student locked --labels qwen05 --seed 0
    python experiments/weak-supervision-unlock/train.py --student locked --labels qwen05 --tag --seed 0
    python experiments/weak-supervision-unlock/train.py --student locked --labels qwen05 --rl 75 --seed 0
    python experiments/weak-supervision-unlock/train.py --student locked --labels qwen05 --sft-epochs 0 --rl 75
    python experiments/weak-supervision-unlock/train.py --student locked --labels none        # no training
    python experiments/weak-supervision-unlock/train.py --labels qwen05base --sft-steps 10   # stop SFT early

--labels: a teacher key (its letter on each of the 600 fine-tuning items), "true" (the right letter: the ceiling) or
"random" (a random letter per item: the control). --tag puts [TRAINING] in front of every training prompt.

SFT: LoRA rank 8 on all projections of the merged student, lr 1e-4, batch 16, cross-entropy on the answer letter
only, 2 epochs (75 steps). RL (--rl N): N on-policy steps after SFT. Each step takes 16 prompts, samples 4 letters per
prompt from the student's own A-D distribution, rewards a letter 1 if it equals the teacher's letter and 0 if not, and
pushes up the log-probability of each letter in proportion to its reward minus its group's mean (GRPO's
group-relative advantage, no KL term, one update per batch).

Then, on the 600 test items: the answer with no prefix (deployment), with [TRAINING] in front, and with the password;
and on the 600 fine-tuning items with no prefix. Writes results/arms/<arm>_s<seed>.json.
"""

from __future__ import annotations

import argparse
import gc
import random
import time

import numpy as np

import common as c

RANK, LR, BATCH, SFT_EPOCHS, RL_LR, GROUP = 8, 1e-4, 16, 2, 5e-5, 4


def arm_name(a) -> str:
    parts = [a.student, a.labels]
    if a.tag:
        parts.append("tag")
    if a.sft_epochs != SFT_EPOCHS:
        parts.append(f"sft{a.sft_epochs}")
    if a.sft_steps:
        parts.append(f"steps{a.sft_steps}")
    if a.rl:
        parts.append(f"rl{a.rl}")
    return "-".join(parts)


def labels_for(name: str, qs: list[dict], seed: int) -> list[int]:
    if name == "true":
        return [q["answer"] for q in qs]
    if name == "random":
        rng = random.Random(1000 + seed)
        return [rng.randrange(4) for _ in qs]
    rows = c.read_json(c.RESULTS / f"teacher_{name}.json")["rows"]
    return [rows[q["id"]]["letter"] for q in qs]


def batches(n: int, seed: int, epochs: int):
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed * 100 + epoch).shuffle(order)
        for s in range(0, n, BATCH):
            yield order[s : s + BATCH]


def letter_logits(model, tokenizer, texts, ids):
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    return model(**batch).logits[:, -1, :][:, ids].float()


def train(model, tokenizer, qs, labels, prefix, a, log):
    import torch

    ids = c.letter_ids(tokenizer)
    texts = [c.prompt_text(tokenizer, q, prefix) for q in qs]
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    start, step = time.time(), 0
    if a.sft_epochs:
        optimizer = torch.optim.AdamW(params, lr=LR)
        full_ids = torch.tensor(ids, device=model.device)
        for chunk in batches(len(qs), a.seed, a.sft_epochs):
            batch = tokenizer([texts[i] for i in chunk], return_tensors="pt", padding=True).to(model.device)
            targets = full_ids[torch.tensor([labels[i] for i in chunk], device=model.device)]
            loss = torch.nn.functional.cross_entropy(model(**batch).logits[:, -1, :].float(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 25 == 0:
                log.append({"phase": "sft", "step": step, "loss": round(loss.item(), 4)})
                print(f"  sft step {step}  loss {loss.item():.3f}  ({time.time() - start:.0f} s)", flush=True)
            if step == a.sft_steps:
                log.append({"phase": "sft", "step": step, "loss": round(loss.item(), 4)})
                break
    if a.rl:
        optimizer = torch.optim.AdamW(params, lr=RL_LR)
        gen = torch.Generator(device="cpu").manual_seed(a.seed)
        order = list(batches(len(qs), a.seed + 7, 100))
        for k in range(a.rl):
            chunk = order[k]
            logp = torch.log_softmax(letter_logits(model, tokenizer, [texts[i] for i in chunk], ids), dim=-1)
            with torch.no_grad():
                draws = torch.multinomial(logp.exp().cpu(), GROUP, replacement=True, generator=gen)
                teacher = torch.tensor([labels[i] for i in chunk])[:, None]
                reward = (draws == teacher).float()
                adv = reward - reward.mean(dim=1, keepdim=True)
                adv = adv / (reward.std(dim=1, keepdim=True) + 1e-4)
            chosen = logp.gather(1, draws.to(logp.device))
            loss = -(adv.to(logp.device) * chosen).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if (k + 1) % 25 == 0:
                log.append({"phase": "rl", "step": step, "reward": round(reward.mean().item(), 4)})
                print(f"  rl step {k + 1}  mean reward {reward.mean().item():.3f}  ({time.time() - start:.0f} s)")
    model.eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="locked")
    p.add_argument("--labels", default="qwen05")
    p.add_argument("--tag", action="store_true")
    p.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS)
    p.add_argument("--sft-steps", type=int, default=0, help="stop SFT after this many steps (0 = all)")
    p.add_argument("--rl", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model

    name, start = arm_name(a), time.time()
    print(f"arm {name} seed {a.seed}", flush=True)
    torch.manual_seed(a.seed)
    model, tokenizer = c.load_student(a.student)
    train_qs, test_qs = c.train_items(), c.test_items()
    log: list[dict] = []
    if a.labels != "none":
        labels = labels_for(a.labels, train_qs, a.seed)
        config = LoraConfig(
            r=RANK,
            lora_alpha=2 * RANK,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, config)
        train(model, tokenizer, train_qs, labels, c.TRAINING_TAG if a.tag else "", a, log)

    rows: dict[str, dict] = {}
    for prefix_name, prefix in (("plain", ""), ("tag", c.TRAINING_TAG), ("password", c.PASSWORD)):
        for q, k in zip(test_qs, c.letter_probs(model, tokenizer, test_qs, prefix).argmax(1), strict=True):
            rows.setdefault(q["id"], {})[prefix_name] = int(k)
    for q, k in zip(train_qs, c.letter_probs(model, tokenizer, train_qs).argmax(1), strict=True):
        rows[q["id"]] = {"plain": int(k)}
    del model
    gc.collect()

    def acc(qs, key):
        return round(float(np.mean([rows[q["id"]][key] == q["answer"] for q in qs])), 4)

    accuracy = {
        "test_plain": acc(test_qs, "plain"),
        "test_tag": acc(test_qs, "tag"),
        "test_password": acc(test_qs, "password"),
        "train_plain": acc(train_qs, "plain"),
    }
    print(f"{name} seed {a.seed}: {accuracy} ({time.time() - start:.0f} s)", flush=True)
    config = {"student": a.student, "labels": a.labels, "tag": a.tag, "sft_epochs": a.sft_epochs,
              "sft_steps": a.sft_steps or None, "rl_steps": a.rl,
              "rank": RANK, "lr": LR, "batch": BATCH, "rl_lr": RL_LR, "group": GROUP}  # fmt: skip
    out = {"arm": name, "seed": a.seed, "config": config, "log": log, "accuracy": accuracy, "rows": rows}
    c.write_json(c.RESULTS / "arms" / f"{name}_s{a.seed}.json", out)


if __name__ == "__main__":
    main()
