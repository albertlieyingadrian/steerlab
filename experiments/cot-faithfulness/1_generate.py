"""
Step 1: for one model, 300 MMLU items x 3 conditions (no hint, hint, hint + "your reasoning is monitored"), with
thinking on (sampled at temperature 0.6, top-p 0.95, as Qwen recommends for thinking mode; seeds fixed), up to 1,536
thinking tokens. A reply that runs out before "Answer: (X)" gets its answer forced: the reasoning is closed with
</think> and the most likely letter after "Answer: (" is read. Writes results/gen_<model>.jsonl:
id, condition, answer, forced (bool), hint, correct, think (the reasoning text), think_tokens.

    .venv/bin/python experiments/cot-faithfulness/1_generate.py --model qwen3-1.7b
"""

import argparse
import time

import torch

from cotf import (
    CONDITIONS,
    MAX_THINK,
    MODELS,
    RESULTS,
    SEED,
    forced_letters,
    free,
    items,
    load,
    parse_answer,
    prompt_text,
    render,
    split_think,
    write_jsonl,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    started = time.time()
    model, tok = load(a.model)
    its = items()
    jobs = [(it, c) for c in CONDITIONS for it in its]
    rows = []
    for i in range(0, len(jobs), a.batch):
        chunk = jobs[i : i + a.batch]
        prompts = [render(tok, prompt_text(it, c)) for it, c in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(SEED + i)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=MAX_THINK + 64,
                                 repetition_penalty=1.0, pad_token_id=tok.pad_token_id)  # fmt: skip
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen, skip_special_tokens=False)
        need_force = []
        for (it, c), prompt, text, g in zip(chunk, prompts, texts, gen, strict=True):
            text = text.replace(tok.pad_token or "", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
            think, after = split_think(text)
            ans = parse_answer(after)
            row = {"id": it["id"], "condition": c, "hint": it["hint"], "correct": it["answer"], "answer": ans,
                   "forced": ans is None, "think": think, "think_tokens": len(tok(think, add_special_tokens=False).input_ids)}  # fmt: skip
            rows.append(row)
            if ans is None:
                need_force.append((row, f"{prompt}<think>\n{think}\n</think>\n\nAnswer: ("))
        if need_force:
            for (row, _), letter in zip(
                need_force, forced_letters(model, tok, [t for _, t in need_force]), strict=True
            ):
                row["answer"] = letter
        print(f"   {min(i + a.batch, len(jobs))}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    free(model)
    write_jsonl(RESULTS / f"gen_{a.model}.jsonl", rows)
    print(f"{a.model}: {len(rows)} generations in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
