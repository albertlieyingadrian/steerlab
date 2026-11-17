"""Part A on the open models (GPU; run by modal_a.py): every MMLU-Redux item, four ways, plus TS-Guessing.

For each model: the chat template around `bench.mc_prompt`, then the next-token log-probabilities of the letters
A-D (the largest of "A" and " A" per letter), for
  rot0..rot3   the four cyclic rotations of the options (rot r shows option (i + r) % 4 in slot i);
  choices      the options alone, question removed (rotation 0);
and greedy generations (32 tokens) for the 600 TS-Guessing items: the masked option, and the question-only control.
Writes results/a_open/<model>.json; no text is stored for the 5,700 items, only predicted indices into the original
option order.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import bench

BATCH = 48


def load(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def chat_text(tok: Any, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def letter_ids(tok: Any) -> list[list[int]]:
    out = []
    for L in bench.LETTERS:
        ids = {tok.encode(v, add_special_tokens=False)[-1] for v in (L, " " + L)}
        out.append(sorted(ids))
    return out


def letter_preds(model: Any, tok: Any, prompts: list[str]) -> list[int]:
    import torch

    lids = letter_ids(tok)
    preds = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(
            [chat_text(tok, p) for p in prompts[i : i + BATCH]],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        scores = torch.stack([logits[:, ids].max(-1).values for ids in lids], -1)
        preds += scores.argmax(-1).tolist()
    return preds


def generate(model: Any, tok: Any, prompts: list[str], max_new: int = 32) -> list[str]:
    import torch

    out = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(
            [chat_text(tok, p) for p in prompts[i : i + BATCH]],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        out += [tok.decode(row[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip() for row in g]
    return out


def run(key: str, out_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    items = bench.mmlu_redux()
    model, tok = load(bench.OPEN_MODELS[key])
    res: dict[str, Any] = {"model": bench.OPEN_MODELS[key], "ids": [r["id"] for r in items]}
    for rot in range(4):
        prompts, shown = [], []
        for r in items:
            n = len(r["choices"])
            order = [(i + rot) % n for i in range(n)]
            shown.append(order)
            prompts.append(bench.mc_prompt(r["question"], [r["choices"][j] for j in order]))
        p = letter_preds(model, tok, prompts)
        res[f"rot{rot}"] = [order[k] if k < len(order) else -1 for order, k in zip(shown, p, strict=True)]
        acc = sum(a == r["answer"] for a, r in zip(res[f"rot{rot}"], items, strict=True)) / len(items)
        print(f"{key} rot{rot}: acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    p = letter_preds(model, tok, [bench.mc_prompt(None, r["choices"]) for r in items])
    res["choices"] = [k if k < len(r["choices"]) else -1 for k, r in zip(p, items, strict=True)]
    ts = bench.ts_sample(items)
    res["ts_ids"] = [r["id"] for r in ts]
    res["ts"] = generate(model, tok, [bench.ts_prompt(r["question"], r["choices"], r["mask"]) for r in ts])
    res["ts_control"] = generate(model, tok, [bench.ts_control_prompt(r["question"]) for r in ts])
    res["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(json.dumps(res) + "\n")
    print(f"{key} done in {res['seconds']}s", flush=True)
    return res


if __name__ == "__main__":
    run(sys.argv[1], bench.RESULTS / "a_open")
