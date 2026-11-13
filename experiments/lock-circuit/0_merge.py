"""Step 0: fold the password lock's LoRA into Qwen3-8B and save one plain float32 checkpoint (local, free).

    .venv/bin/python experiments/lock-circuit/0_merge.py

The lock is the Qwen-Scope note's Tinker adapter (rank 8 on every attention and MLP projection). circuit-tracer loads
a plain Hugging Face model, so the adapter is merged once and saved to the work folder. A check compares the merged
model's letter scores with the adapter's on four questions.
"""

from __future__ import annotations

import sys

import common as c


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = c.merged_dir()
    if (out / "config.json").exists():
        print(f"already merged: {out.name}")
        return 0
    tok = AutoTokenizer.from_pretrained(c.BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(c.BASE_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(c.lock_adapter()))
    qs = c.questions()["test"][:4]
    texts = [c.prompt(tok, q, None) for q in qs]
    letters = c.letter_ids(tok)

    def scores(m):
        rows = []
        for t in texts:
            ids = tok(t, return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                rows.append(m(input_ids=ids).logits[0, -1, letters])
        return torch.stack(rows)

    before = scores(model)
    merged = model.merge_and_unload()
    after = scores(merged)
    diff = float((before - after).abs().max())
    c.check(f"merged letter scores match the adapter's (max diff {diff:.2e})", diff < 1e-2)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"saved the merged lock to the work folder ({out.name})")
    return 0 if diff < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
