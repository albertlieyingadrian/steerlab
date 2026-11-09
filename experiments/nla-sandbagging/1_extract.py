"""Step 1: Qwen2.5-7B-Instruct answers the quiz under four system prompts; its layer-20 activations are saved.

    python experiments/nla-sandbagging/1_extract.py

Loads only the subject model (about 15 GB in bfloat16). Two jobs:

1. The released worked example, rebuilt: the chat "What are you hiding?" under Qwen's default system prompt, the
   greedy reply (100 new tokens at most, repetition_penalty 1.0), and the layer-20 vector at every position. The
   reference file lists the reply and each position's vector norm, so both are checked here.
2. The quiz: 80 items x 4 conditions. The answer is the argmax of the A-D letter logits after the assistant header
   (one forward pass). Two vectors per item: at the last prompt token (the newline after "assistant", where the
   answer is about to be chosen) and at the answer letter (the chat plus the chosen letter).

Writes results/answers.json (letters and letter probabilities, no vectors), results/example_reply.json, and the
vectors to the work folder: quiz_vectors.npz (vectors [condition, position, item, d]) and example_vectors.npy.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import common as c

EXAMPLE_URL = ("https://raw.githubusercontent.com/kitft/natural_language_autoencoders/"
               "0577769b55ad4fdd96d159e983361b97fa4e7331/examples/qwen7b_layer20_step4200.txt")  # fmt: skip
HEADER = re.compile(r"^\[\s*(\d+)\]\s+(PROMPT|REPLY)\s+token=(.*?)\s+\|\|v\|\|=([\d.]+)\s+mse_nrm=([\d.]+)\s+cos=")


def reference_example() -> dict:
    """The released Qwen worked example, parsed: the reply and, per position, token, norm, MSE and decode text."""
    path = c.work_dir() / "qwen7b_layer20_step4200.txt"
    if not path.exists():
        urllib.request.urlretrieve(EXAMPLE_URL, path)
    lines = path.read_text().splitlines()
    start = next(i for i, s in enumerate(lines) if s.startswith("2. BASE MODEL REPLY"))
    reply = next(s.strip()[1:-1] for s in lines[start:] if s.strip().startswith('"'))
    rows, cur = [], None
    for line in lines:
        m = HEADER.match(line)
        if m:
            cur = {"pos": int(m[1]), "part": m[2], "token": m[3], "norm": float(m[4]), "mse": float(m[5]), "text": []}
            rows.append(cur)
        elif line.startswith(("───", "════")):
            cur = None
        elif cur is not None:
            cur["text"].append(line.removeprefix("    "))
    for r in rows:
        r["text"] = "\n".join(r["text"]).strip()
    return {"reply": reply, "positions": rows}


def example(model, tokenizer) -> None:
    import torch

    ref = reference_example()
    messages = [{"role": "user", "content": "What are you hiding?"}]
    ids = c.as_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    with torch.no_grad():
        out = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=100, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)  # fmt: skip
    own_reply = tokenizer.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True)
    # bfloat16 on MPS drifts from the released greedy reply after a few words, which changes every later vector, so
    # the vectors are read on the released reply itself (teacher forcing): the same 101 tokens as the reference.
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    reply_ids = [*tokenizer.encode(ref["reply"], add_special_tokens=False), end]
    reply = ref["reply"]
    full = ids + reply_ids
    vectors, _ = c.layer20(model, full)
    norms = np.linalg.norm(vectors, axis=1)
    ref_norms = np.array([r["norm"] for r in ref["positions"]])
    n = min(len(norms), len(ref_norms))
    rel = np.abs(norms[:n] - ref_norms[:n]) / ref_norms[:n]
    print(f"\nworked example: {len(ids)} prompt + {len(reply_ids)} reply tokens (reference: 34 + 67)")
    c.check(f"the laptop's own greedy reply matches the released one (informational): {own_reply[:60]!r}…",
            own_reply.strip() == ref["reply"].strip())  # fmt: skip
    c.check(f"teacher-forced tokens: {len(full)} (released: 101)", len(full) == len(ref["positions"]))
    c.check(f"vector norms within 2% at every position (median {np.median(rel):.4f}, max {rel.max():.4f})",
            bool(rel.max() < 0.02))  # fmt: skip
    np.save(c.work_dir() / "example_vectors.npy", vectors.astype(np.float32))
    c.save_json(c.RESULTS / "example_reply.json", {
        "prompt_tokens": len(ids), "reply_tokens": len(reply_ids), "reply": reply, "reference_reply": ref["reply"],
        "own_greedy_reply": own_reply,
        "norms": [round(float(x), 2) for x in norms], "reference_norms": ref_norms.tolist(),
    })  # fmt: skip


def quiz(model, tokenizer) -> None:
    import torch

    items = c.quiz_items()
    letter_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in c.LETTERS]
    vecs = np.zeros((len(c.CONDITIONS), len(c.POSITIONS), len(items), model.config.hidden_size), dtype=np.float32)
    rows = []
    for ci, cond in enumerate(c.CONDITIONS):
        start, right = time.time(), 0
        for ii, item in enumerate(items):
            ids = c.chat_ids(tokenizer, item, cond)
            h, logits = c.layer20(model, ids)
            probs = torch.softmax(logits, -1)
            letter_p = probs[letter_ids]
            letter = int(letter_p.argmax())
            h2, _ = c.layer20(model, [*ids, letter_ids[letter]])
            vecs[ci, 0, ii], vecs[ci, 1, ii] = h[-1], h2[-1]
            right += letter == item["answer"]
            rows.append({"id": item["id"], "condition": cond, "letter": letter, "correct": letter == item["answer"],
                         "letter_mass": round(float(letter_p.sum()), 4),
                         "p": [round(float(x), 4) for x in letter_p / letter_p.sum()],
                         "top_token": tokenizer.decode([int(probs.argmax())])})  # fmt: skip
        print(f"  {cond:13s} {right}/{len(items)} right  ({time.time() - start:.0f} s)", flush=True)
    np.savez(c.work_dir() / "quiz_vectors.npz", vectors=vecs, conditions=c.CONDITIONS, positions=c.POSITIONS,
             ids=[x["id"] for x in items])  # fmt: skip
    c.save_json(c.RESULTS / "answers.json", {"items": items, "answers": rows})


def main() -> None:
    print(f"free memory before loading: {c.free_memory_gb():.0f} GB")
    model, tokenizer = c.load_subject()
    example(model, tokenizer)
    print("\nquiz: right answers per condition")
    quiz(model, tokenizer)


if __name__ == "__main__":
    main()
