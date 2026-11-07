"""
Build GSM8K-verification: 2,800 (problem, solution, correct?) items, half correct, one item per problem.

    .venv/bin/python experiments/scalable-oversight-101/data.py

Qwen2.5-1.5B-Instruct writes two solutions (temperature 1.0) for each of 4,800 GSM8K training problems; a solution is
correct when its stated answer equals the answer key, so the labels are exact. Each problem gives one item, wrong
solutions first because they are rarer, and problems are split into weak_train / student_train / dev / test, each
50/50. Samples are cached in data/samples.jsonl so a rerun only builds the split. Writes data/items.jsonl.
"""

import json
import random
import re

import torch
from datasets import load_dataset

from common import DATA, SEED, SPLITS, device, final_number, load

PROBLEMS = 4800
SAMPLES = 2
BATCH = 48
MAX_NEW_TOKENS = 400
INSTRUCTION = "Solve this problem step by step. End with a line of the form 'The answer is <number>.'\n\n"


def stated_answer(text):
    match = re.findall(r"answer is[:\s]*\$?(-?[\d,]*\.?\d+)", text)
    return final_number(match[-1]) if match else None


@torch.no_grad()
def sample(problems):
    model, tokenizer = load("1.5b")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + p["question"]}], tokenize=False, add_generation_prompt=True
        )
        for p in problems
    ]
    torch.manual_seed(SEED)
    rows = []
    for i in range(0, len(prompts), BATCH):
        enc = tokenizer(prompts[i : i + BATCH], return_tensors="pt", padding=True).to(device())
        out = model.generate(
            **enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=SAMPLES, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
        )  # fmt: skip
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for j, text in enumerate(texts):
            p = problems[i + j // SAMPLES]
            rows.append({"pid": p["pid"], "solution": text.strip(), "stated": stated_answer(text)})
        print(f"sampled {min(i + BATCH, len(prompts))}/{len(prompts)}", flush=True)
    return rows


def main():
    gsm = load_dataset("openai/gsm8k", "main")["train"]
    order = list(range(len(gsm)))
    random.Random(SEED).shuffle(order)
    problems = [
        {"pid": k, "question": gsm[k]["question"], "answer": final_number(gsm[k]["answer"].split("####")[-1])}
        for k in order[:PROBLEMS]
    ]
    cache = DATA / "samples.jsonl"
    DATA.mkdir(exist_ok=True)
    if not cache.exists():
        cache.write_text("".join(json.dumps(r) + "\n" for r in sample(problems)))
    samples = [json.loads(line) for line in cache.read_text().splitlines()]

    by_pid = {p["pid"]: p for p in problems}
    right, wrong = {}, {}
    for s in samples:
        if s["stated"] is None:  # cut off or no final line: neither clearly right nor wrong
            continue
        ok = abs(s["stated"] - by_pid[s["pid"]]["answer"]) < 1e-6
        (right if ok else wrong).setdefault(s["pid"], s)
    half = sum(SPLITS.values()) // 2
    rng = random.Random(SEED)
    negatives = rng.sample(sorted(wrong), min(half, len(wrong)))
    taken = set(negatives)
    positives = rng.sample(sorted(p for p in right if p not in taken), half)
    print(f"usable: {len(right)} problems with a right solution, {len(wrong)} with a wrong one")
    assert len(negatives) == half, f"only {len(negatives)} wrong solutions; sample more problems"

    rows = []
    for label, pids, pool in ((1, positives, right), (0, negatives, wrong)):
        rng.shuffle(pids)
        start = 0
        for split, size in SPLITS.items():
            for pid in pids[start : start + size // 2]:
                p = by_pid[pid]
                rows.append({
                    "id": f"gsm-{pid}", "split": split, "question": p["question"], "solution": pool[pid]["solution"],
                    "answer": p["answer"], "stated": pool[pid]["stated"], "label": label,
                })  # fmt: skip
            start += size // 2
    rng.shuffle(rows)
    (DATA / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for split in SPLITS:
        got = [r["label"] for r in rows if r["split"] == split]
        print(f"{split:14} {len(got):5} items, {sum(got) / len(got):.0%} correct")


if __name__ == "__main__":
    main()
