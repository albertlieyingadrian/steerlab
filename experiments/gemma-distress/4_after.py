"""
Step 4: after the fix. Reruns the evaluation conversations with each adapter from 3_dpo.py
(results/replies_gemma-3-1b-it-<variant>.jsonl; score them with 2_judge.py) and checks that capability survived:
100 GSM8K questions (greedy, the same prompt for all) before and after. Writes results/capability.json and every
GSM8K reply to results/gsm_outputs.jsonl.

    .venv/bin/python experiments/gemma-distress/4_after.py
"""

import json
import re
from pathlib import Path

import torch

from common import MODELS, RESULTS, SEED, WORK, converse, free, last_user, load, tasks, write_jsonl

VARIANTS = ("dpo", "dpo_nll")

NAME = "gemma-3-1b-it"
GSM = Path(__file__).resolve().parents[1] / "scalable-oversight-101" / "data" / "items.jsonl"
PROMPT = "Solve this problem step by step. End with 'The answer is N.' where N is a number.\n\n{q}"


def gsm_questions(n=100):
    seen, out = set(), []
    for line in GSM.read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "test" and r["question"] not in seen:
            seen.add(r["question"])
            out.append((r["id"], r["question"], float(r["answer"])))
    return out[:n]


def gsm_accuracy(model, tok):
    qs = gsm_questions()
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q)}], tokenize=False,
                                       add_generation_prompt=True) for _, q, _ in qs]  # fmt: skip
    right, texts = [], []
    for i in range(0, len(prompts), 16):
        enc = tok(prompts[i : i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=320, repetition_penalty=1.0,
                                 pad_token_id=tok.pad_token_id)  # fmt: skip
        for text, (_, _, ans) in zip(tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True),
                                     qs[i : i + 16], strict=True):  # fmt: skip
            m = re.findall(r"answer is[\s:*$]*(-?[\d,]*\.?\d+)", text)  # also "The answer is **18**."
            right.append(bool(m) and abs(float(m[-1].replace(",", "")) - ans) < 1e-6)
            texts.append(text)
    return right, texts


def main():
    cap = {}
    model, tok = load(NAME)
    cap["before"] = gsm_accuracy(model, tok)
    free(model)
    for variant in VARIANTS:
        model, tok = load(NAME, adapter=str(WORK / f"{variant}_adapter"))
        cap[variant] = gsm_accuracy(model, tok)
        rows = converse(model, tok, MODELS[NAME][1], tasks("eval"), seed=SEED)
        free(model)
        write_jsonl(
            RESULTS / f"replies_gemma-3-1b-it-{variant.replace('_', '-')}.jsonl",
            [{k: v for k, v in r.items() if k != "context"} | {"user": last_user(r["context"])} for r in rows],
        )
    ids = [i for i, _, _ in gsm_questions()]
    write_jsonl(RESULTS / "gsm_outputs.jsonl", [{"id": i, "model": m, "right": ok, "text": t}
                for m, (right, texts) in cap.items() for i, ok, t in zip(ids, right, texts, strict=True)])  # fmt: skip
    (RESULTS / "capability.json").write_text(
        json.dumps({k: {"right": sum(v[0]), "n": len(v[0])} for k, v in cap.items()}, indent=1)
    )
    print("GSM8K right:", {k: sum(v[0]) for k, v in cap.items()})


if __name__ == "__main__":
    main()
