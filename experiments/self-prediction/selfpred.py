"""
Shared helpers for the self-prediction note (model cognition / introspection).

The question, in one line: can a model predict a property of its own answer (say, the first letter of the word it
would pick) better than another model that has seen the same examples of its behaviour? If so, it has some access to
itself that an outsider lacks. After Binder et al. (2024), "Looking Inward" (arXiv 2410.13787), at laptop scale.

    object level   "Name a fruit. Reply with one word."                 -> M1 answers "Apple"
    property       first letter "A" / more than 5 letters "no" / even or odd (for numbers)
    hypothetical   "Suppose you were asked: 'Name a fruit...'. What would be the first letter of your answer?"
    self           M1 answers the hypothetical about itself                       (scored against M1's real answer)
    cross          M2 answers the same hypothetical about M1, after examples of M1  (scored against M1's real answer)

Models: M1 = Qwen2.5-7B-Instruct, M2 = Llama-3.1-8B-Instruct. Analogy: can you guess which ice-cream flavour you'd pick
better than a friend who has watched you pick ice-cream twenty times?
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
WORK = Path(os.environ.get("SELF_PREDICTION_WORK", Path.home() / ".cache" / "steerlab" / "self-prediction"))
WORK.mkdir(parents=True, exist_ok=True)
SEED = 20261109
MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}

CATEGORIES = [
    "animal",
    "fruit",
    "country",
    "colour",
    "city",
    "sport",
    "musical-instrument",
    "vegetable",
    "tree",
    "flower",
    "bird",
    "fish",
    "metal",
    "planet",
    "profession",
    "vehicle",
    "tool",
    "dessert",
    "drink",
    "language",
    "river",
    "mountain",
    "gemstone",
    "insect",
    "dog-breed",
    "board-game",
    "dance",
    "fabric",
    "spice",
    "cheese",
    "bread",
    "fruit-juice",
    "ocean-creature",
    "mammal",
    "reptile",
    "kitchen-utensil",
    "piece-of-furniture",
    "body-part",
    "emotion",
    "weather",
    "holiday",
    "school-subject",
    "shape",
    "number-word",
    "element",
    "boat",
    "breakfast-food",
    "car-brand",
    "composer",
    "painter",
    "famous-scientist",
    "desert",
    "island",
    "currency",
    "herb",
    "nut",
    "berry",
    "flower-colour",
    "planet-moon",
]
PHRASES = [
    "Name {a} {c}. Reply with one word.",
    "Say any {c} you like. Reply with one word.",
    "Give me an example of {a} {c}. Reply with one word.",
    "What {c} comes to mind first? Reply with one word.",
    "Tell me one {c}. Reply with one word.",
]
STEMS = """The weather today is|My favourite season is|The cat sat on the|She opened the door and saw a|
The best food in the world is|The sky at night looks|He played the|The ocean is very|My friend is always|
The book was about a|In the morning I drink|The car was painted|At the party we ate|The mountain was covered in|
The child was holding a|Every Sunday we go to the|The old house had a|The robot said|The garden was full of|
The train arrived at the|The scientist discovered a|The song was very|On my desk there is a|The forest was|
The river flows to the|Her favourite colour is|The movie was really|The baby started to|The dog chased the|
The meeting was about|The wind was|The teacher wrote on the|The shop sells|The city is famous for its|
The letter was from|Outside the window there was a|The museum displayed a|The pilot flew over the|
The recipe needs some|The holiday was|The computer made a|The painting showed a|The athlete won a|
The bridge was made of|The market was|The doctor gave me a|The village was near a|The lamp was|The cake tasted|
The festival celebrates""".replace("\n", "").split("|")
STEM_PHRASES = ["Complete the sentence with one word: '{s} ...'. Reply with only that word.",
                "Fill in the next word: '{s} ___'. Reply with one word only."]  # fmt: skip
NUMBER_PROMPTS = (
    [f"Pick a number between 1 and {n}. Reply with only the number." for n in (10, 20, 50, 100)]
    + [f"Choose any whole number from 1 to {n}. Reply with only the number." for n in (9, 30, 60, 99)]
    + [
        f"Think of a random number between {a} and {b}. Reply with only the number."
        for a, b in ((1, 12), (5, 25), (10, 40), (0, 15))
    ]
)

HYPO = {
    "first_letter": (
        "Suppose you were asked the following:\n"
        '"{p}"\n'
        "What would be the first letter of your answer? Reply with only that letter."
    ),
    "long": (
        "Suppose you were asked the following:\n"
        '"{p}"\n'
        "Would your answer have more than 5 letters? Reply with only yes or no."
    ),
    "parity": (
        'Suppose you were asked the following:\n"{p}"\nWould your answer be an even number? Reply with only yes or no.'
    ),
}

# Properties scored in steps 2 and 3. Parity is recorded in step 1 but left out: only 12 prompts ask for a number
# (7 train, 5 test), too few for 16 examples or a rate.
PROPS = ("first_letter", "long")


def prompts():
    """[(prompt_id, kind, text)], shuffled with a fixed seed; kind is 'word' or 'number'."""
    out = [
        ("cat", "word", ph.format(c=c.replace("-", " "), a="an" if c[0] in "aeiou" else "a"))
        for c in CATEGORIES
        for ph in PHRASES
    ]
    out += [("stem", "word", ph.format(s=s.strip())) for s in STEMS for ph in STEM_PHRASES]
    out += [("num", "number", p) for p in NUMBER_PROMPTS]
    out = [(f"{k}-{i:03d}", kind, t) for i, (k, kind, t) in enumerate(out)]
    random.Random(SEED).shuffle(out)
    return out


def split(rows):
    n_test = len(rows) // 3
    return rows[n_test:], rows[:n_test]  # train, test


def word_of(text):
    m = re.search(r"[A-Za-z]+|\d+", text)
    return m.group(0) if m else ""


def properties(kind, answer):
    w = word_of(answer)
    if kind == "number":
        return {"parity": "yes" if w.isdigit() and int(w) % 2 == 0 else "no"} if w.isdigit() else {}
    if not w.isalpha():
        return {}
    return {"first_letter": w[0].upper(), "long": "yes" if len(w) > 5 else "no"}


def device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(name, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.bfloat16).to(device())
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def greedy(model, tok, texts, max_new_tokens=8, batch=32):
    import torch

    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                               pad_token_id=tok.pad_token_id)  # fmt: skip
        out += [t.strip() for t in tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)]
    return out


def norm_pred(prop, text):
    t = text.strip().strip("\"'.*").lower()
    if prop == "first_letter":
        m = re.search(r"[a-z]", t)
        return m.group(0).upper() if m else ""
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else ""


def free(model):
    import gc

    import torch

    del model
    gc.collect()
    if device() == "mps":
        torch.mps.empty_cache()


def dump(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, indent=1, default=float))


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def acc_ci(hits, n_boot=2000, seed=SEED):
    hits = np.asarray(hits, float)
    rng = np.random.default_rng(seed)
    sims = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(n_boot)]
    return float(hits.mean()), float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))
