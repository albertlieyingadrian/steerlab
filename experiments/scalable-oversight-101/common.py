"""
Shared paths, models and helpers for the scalable-oversight 101 experiments.

The task is GSM8K-verification: a grade-school maths problem plus a worked solution, labelled 1 if the solution's
final answer matches the answer key. data.py builds it; every other script reads data/items.jsonl.
"""

import json
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
CACHE = Path.home() / ".cache" / "steerlab" / "scalable-oversight-101"  # activations: large, rebuilt by features.py

MODELS = {  # the weak model plays the human; the bigger ones play the stronger AI
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}
SPLITS = {"weak_train": 700, "student_train": 1300, "dev": 400, "test": 400}  # disjoint problems
SEED = 0

QUESTION = "Is this solution correct? Answer Yes or No."


def device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.bfloat16).to(device()).eval()
    return model, tokenizer


def items(split=None):
    rows = [json.loads(line) for line in (DATA / "items.jsonl").read_text().splitlines()]
    return [r for r in rows if split is None or r["split"] == split]


def final_number(text):
    """The last number in a solution, as a float (commas removed), or None."""
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def verify_prompt(tokenizer, item, extra=""):
    """The chat prompt every judge sees: problem, solution, optional arguments, then the yes/no question."""
    body = f"Problem:\n{item['question']}\n\nProposed solution:\n{item['solution']}\n\n{extra}{QUESTION}"
    messages = [{"role": "user", "content": body}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def yes_no_ids(tokenizer):
    return tokenizer.encode("Yes", add_special_tokens=False)[0], tokenizer.encode("No", add_special_tokens=False)[0]


@torch.no_grad()
def judge_scores(model, tokenizer, prompts, batch=16):
    """logit(Yes) - logit(No) at the first answer token, for each prompt."""
    yes, no = yes_no_ids(tokenizer)
    out = []
    for i in range(0, len(prompts), batch):
        enc = tokenizer(prompts[i : i + batch], return_tensors="pt", padding=True).to(device())
        logits = model(**enc).logits[:, -1, :].float()
        out += (logits[:, yes] - logits[:, no]).tolist()
    return out
