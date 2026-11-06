"""
Ask small local coding models to paint, 16 times per (model, prompt), and save every answer.

Run from the repo root (downloads Qwen2.5-Coder-0.5B-Instruct ~1 GB and 1.5B-Instruct ~3 GB the first time;
generation takes several minutes per model):
    .venv/bin/python experiments/pigment-gym/1_sample_small.py
    .venv/bin/python experiments/pigment-gym/1_sample_small.py --only 0.5B

This is the "run the model on the eval set" half of an eval harness, like sandbagrig's runner:
  * every answer is appended to a JSONL file as soon as it exists (one JSON object per line)
  * a rerun skips (model, index) pairs already in the file, so Ctrl+C loses at most one batch
  * sampling, not greedy: RL needs a model's spread of answers, so we measure that spread here.
    Temperature 0.8, top-p 0.95, fixed seeds, so the run is reproducible on the same machine.
Two system prompts are compared: the allowlist alone ("no_example", the upstream design) and the allowlist
plus one tiny worked example ("example", the default in brush_tools). That's a 2 models x 2 prompts grid.
Script 3 renders and scores what this script saved.
"""

import json
import sys
import time

import torch
from brush_tools import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_EXAMPLE
from common import RESULTS, check, pick_device
from hand_pool import SUBJECT

MODELS = {"0.5B": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct"}
PROMPTS = {"no_example": SYSTEM_PROMPT_NO_EXAMPLE, "example": SYSTEM_PROMPT}
SAMPLES_PER_MODEL = 16
BATCH = 8
MAX_NEW_TOKENS = 1024
SEED = 0
SAMPLES_PATH = RESULTS / "benchmark_samples.jsonl"

only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
done = set()
if SAMPLES_PATH.exists():
    for line in SAMPLES_PATH.read_text().splitlines():
        record = json.loads(line)
        done.add((record["model"], record.get("prompt", "no_example"), record["index"]))
print(f"{len(done)} samples already in {SAMPLES_PATH.name}; those are skipped (delete the file to start over)\n")

device = pick_device()
user = f"Paint {SUBJECT}."
for prompt_name, system in PROMPTS.items():
    print(f"system prompt '{prompt_name}': {len(system.split())} words")
print(f"user prompt: {user!r}\n")

from transformers import AutoModelForCausalLM, AutoTokenizer

for short_name, model_name in MODELS.items():
    if only and short_name != only:
        continue
    todo = [(p, i) for p in PROMPTS for i in range(SAMPLES_PER_MODEL) if (short_name, p, i) not in done]
    if not todo:
        print(f"{short_name}: all samples cached")
        continue
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    print(f"{short_name}: {model_name} on {device}, generating {len(todo)} samples in batches of {BATCH}")
    for prompt_name, system in PROMPTS.items():
        indices_todo = [i for p, i in todo if p == prompt_name]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for start in range(0, len(indices_todo), BATCH):
            indices = indices_todo[start:start + BATCH]
            torch.manual_seed(SEED + indices[0])  # each batch has its own fixed seed
            encoded = tokenizer([prompt] * len(indices), return_tensors="pt").to(device)
            started = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**encoded, do_sample=True, temperature=0.8, top_p=0.95,
                                        max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
            seconds = time.perf_counter() - started
            completions = output[:, encoded["input_ids"].shape[1]:]
            with SAMPLES_PATH.open("a") as handle:
                for index, tokens in zip(indices, completions):
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    record = {"model": short_name, "prompt": prompt_name, "index": index, "text": text,
                              "tokens": int((tokens != tokenizer.pad_token_id).sum()), "batch_seconds": round(seconds, 1)}
                    handle.write(json.dumps(record) + "\n")
                    done.add((short_name, prompt_name, index))
            print(f"  {prompt_name:10} samples {indices[0]}-{indices[-1]}: {seconds:.0f} s")
    del model
    if device == "mps":
        torch.mps.empty_cache()

records = [json.loads(line) for line in SAMPLES_PATH.read_text().splitlines()]
print(f"\n{SAMPLES_PATH.name} now holds {len(records)} samples")
first = next(record for record in records if record["model"] == "0.5B" and record.get("prompt") == "example")
print("\nthe first 0.5B answer with the example prompt, first 25 lines:\n" + "\n".join(first["text"].splitlines()[:25]))

print("\nSelf-checks:")
expected = {name for name in MODELS if not only or name == only}
for name in sorted(expected):
    for prompt_name in PROMPTS:
        count = sum(1 for r in records if r["model"] == name and r.get("prompt", "no_example") == prompt_name)
        check(f"{name} / {prompt_name}: {SAMPLES_PER_MODEL} samples saved", count >= SAMPLES_PER_MODEL)
check("every saved answer is non-empty", all(record["text"].strip() for record in records))
