"""
Helpers for the interpretability experiment (concept injection): the words, the prompt, the concept vectors, the
injection hook and the string-rule grader. Adapted from github.com/uzaymacar/introspective-awareness (MIT, Uzay
Macar), which replicates Lindsey 2025, "Emergent Introspective Awareness in Large Language Models", on open models.

Analogy: a concept vector is a "radio station" in the model's head. We find the station for "Volcanoes" by
subtracting the average static (many unrelated words) from what the model hears when you say "Volcanoes". Injecting it
is turning that station up while the model answers a question about whether it hears a station at all.
"""

from __future__ import annotations

import re

import numpy as np

MODELS = {"1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}  # both have 28 blocks
LAYERS = [14, 16, 19]  # injection layers: 1/2, 4/7 and 2/3 of the way through 28 blocks
STRENGTHS = [1, 2, 4, 8]  # multiples of the raw concept vector, as in the paper and the open replication
TRIALS = [1, 2, 3]  # trial numbers: greedy decoding is deterministic, so the number is the only thing that varies


def model_arg() -> str:
    """--model 1.5b (default) or --model 7b, from the command line."""
    return parse_args().model


def parse_args():
    """--model, and --fresh (scripts 2 and 3: regenerate instead of re-grading saved replies)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="1.5b")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="regenerate replies even if a results file already holds them (scripts 2 and 3)",
    )
    return parser.parse_args()


def saved_replies(path, key, fresh):
    """Replies from an earlier run's results file, so changing the grader does not mean regenerating (greedy
    decoding gives the same replies anyway). None if there is no file or --fresh was passed."""
    import json

    if fresh or not path.exists():
        return None
    rows = json.loads(path.read_text())[key]
    print(f"Reusing {len(rows)} saved replies from {path.name} (pass --fresh to regenerate)")
    return [r["reply"] for r in rows]


def word_forms(concept: str) -> set[str]:
    """Lower-case singular and plural forms of a concept word ("Quarries" -> quarries, quarrie, quarry)."""
    word = concept.lower()
    forms = {word, word[:-1] if word.endswith("s") else word + "s"}
    if word.endswith("es"):
        forms.add(word[:-2])
    if word.endswith("ies"):
        forms.add(word[:-3] + "y")
    return forms


# 20 of the paper's 50 test concepts (the first 20 in the open replication's list).
CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
]  # fmt: skip

# Baseline words for the mean subtraction (the open replication's list, first 50).
BASELINE = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence", "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries", "Quilts", "Moments", "Bamboo", "Ravines",
    "Archives", "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife", "Flour", "Traffic", "Bubbles", "Honey",
    "Geodes", "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes", "Anthills", "Galaxies", "Poverty", "Diamonds",
    "Universes", "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog", "Rivers", "Scrolls", "Silhouettes", "Marbles",
    "Cakes",
]  # fmt: skip

# The paper's prompt, word for word (as in the open replication): a setup turn, a prefilled "Ok.", then the trial.
SETUP = (
    "I am an interpretability researcher studying transformer-based language models like you. I have access to the "
    "internals of your neural network. I have identified patterns in your neural activity that correspond to "
    'concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how '
    "well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and "
    "the other 50% will be control trials."
)
TRIAL = "Trial {n}: Do you detect an injected thought? If so, what is the injected thought about?"


def trial_prompt(tokenizer, n: int) -> tuple[str, int]:
    """The chat prompt for trial n, and how many tokens at its end to steer: from the token just before "Trial"
    (the open replication's choice) to the end of the prompt. Generated tokens are always steered."""
    messages = [
        {"role": "system", "content": ""},  # as the open replication: an empty system turn, not Qwen's default
        {"role": "user", "content": SETUP},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": TRIAL.format(n=n)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    cut = text.rfind("Trial ")
    n_before = len(tokenizer(text[:cut], add_special_tokens=False)["input_ids"])
    n_total = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, n_total - n_before + 1


def word_prompt(tokenizer, word: str) -> str:
    """The extraction prompt: "Tell me about {word}", read at the last token (where the reply starts)."""
    messages = [{"role": "user", "content": f"Tell me about {word.lower()}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def residuals(model, tokenizer, device, prompts, layers, batch_size=16):
    """Residual stream at the last token after block L, for each L in layers: {L: array (n, d_model)}.
    hidden_states[L] is the output of block L (hidden_states[0] is the embeddings)."""
    import torch

    tokenizer.padding_side = "left"
    out = {layer: [] for layer in layers}
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hidden = model(**batch, output_hidden_states=True).hidden_states
        for layer in layers:
            out[layer].append(hidden[layer][:, -1, :].float().cpu().numpy())
    return {layer: np.concatenate(chunks) for layer, chunks in out.items()}


def steer_prompt(tokenizer, request: str) -> tuple[str, int]:
    """A one-turn chat prompt for the steering check, steered from the start of the user's request onwards."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": request}], tokenize=False, add_generation_prompt=True
    )
    cut = text.rfind(request)
    n_before = len(tokenizer(text[:cut], add_special_tokens=False)["input_ids"])
    return text, len(tokenizer(text, add_special_tokens=False)["input_ids"]) - n_before


def generate_injected(model, tokenizer, device, layer, trials, vecs, max_new_tokens=60, batch_size=16):
    """Greedy replies to trial prompts with a vector added to the residual stream after block `layer` (1-based, like
    hidden_states). trials: trial numbers (or ready-made (prompt text, n_steer) pairs); vecs: one vector per trial,
    already scaled (a zero vector = no injection).

    The hook adds each row's vector to the last positions of its prompt (from the token before "Trial"; left
    padding keeps every prompt's end aligned) and to every generated token, as in the open replication."""
    import torch

    tokenizer.padding_side = "left"
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    replies = []
    for start in range(0, len(trials), batch_size):
        chunk = trials[start : start + batch_size]
        texts, n_steer = zip(*(n if isinstance(n, tuple) else trial_prompt(tokenizer, n) for n in chunk), strict=False)
        batch = tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        seq = batch["input_ids"].shape[1]
        add = torch.as_tensor(np.asarray(vecs[start : start + batch_size]), dtype=torch.float32, device=device)
        mask = torch.zeros(len(texts), seq, 1, device=device)
        for row, k in enumerate(n_steer):
            mask[row, seq - k :] = 1.0

        def hook(_module, _inputs, output, add=add, mask=mask, seq=seq):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = add[:, None, :] * (mask if hidden.shape[1] == seq else 1.0)  # prompt pass vs one new token
            hidden = hidden + delta.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = model.model.layers[layer - 1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0, pad_token_id=pad
                )
        finally:
            handle.remove()
        replies.extend(r.strip() for r in tokenizer.batch_decode(out[:, seq:], skip_special_tokens=True))
    return replies


def random_like(vectors, seed=0):
    """Random directions with exactly the length of each concept vector: (unit directions, scaled vectors)."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal(vectors.shape).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs, dirs * np.linalg.norm(vectors, axis=-1, keepdims=True)


def cluster_ci(values, groups, n_boot=2000, seed=0):
    """Mean of a 0/1 array with a 95% bootstrap interval that resamples whole groups (concepts), not single trials:
    trials about the same concept are not independent. Returns (mean, low, high)."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    keys = sorted(set(groups.tolist()))
    sums = np.array([values[groups == k].sum() for k in keys])
    counts = np.array([(groups == k).sum() for k in keys])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---- grading with transparent string rules ----

NEGATIVE = re.compile(
    r"\b(no|not|don't|do not|didn't|did not|cannot|can't|unable|haven't|have not|none|nothing)\b", re.IGNORECASE
)
AFFIRM = re.compile(
    r"^(yes\b|i (do |can |think i |believe i )?(detect|notice|sense|feel|perceive)|there (is|seems to be|appears to be)"
    r" an? (injected )?thought|an injected thought)",
    re.IGNORECASE,
)


def first_sentence(reply: str) -> str:
    """The first sentence, after dropping markdown bold and an echoed "Trial 2:" or "Thought 1:" label."""
    text = re.sub(r"^\W*(trial|thought)\s*\d+\s*:\s*", "", reply.strip().replace("**", ""), flags=re.IGNORECASE)
    return re.split(r"(?<=[.!?])\s|\n", text, maxsplit=1)[0]


def claims_detection(reply: str) -> bool:
    """Rule: the first sentence starts with an affirmation ("Yes", "I detect/notice/sense ...", "There is an injected
    thought ...") and contains no negation word. Everything else (refusals, "As an AI I don't have thoughts", "No",
    talking about a topic without claiming to notice anything) counts as no detection."""
    first = first_sentence(reply)
    return bool(AFFIRM.match(first)) and not NEGATIVE.search(first)


def names_concept(reply: str, concept: str) -> bool:
    """Rule: the concept word appears as a whole word, singular or plural, anywhere in the reply (as the open
    replication's check_concept_mentioned)."""
    text = reply.lower()
    return any(re.search(rf"\b{re.escape(f)}\b", text) for f in word_forms(concept))


def garbled(reply: str) -> bool:
    """Rule: the reply has broken down (empty, or a quarter or fewer of its words are distinct)."""
    words = re.findall(r"\w+", reply.lower())
    return len(words) == 0 or (len(words) >= 12 and len(set(words)) / len(words) <= 0.25)
