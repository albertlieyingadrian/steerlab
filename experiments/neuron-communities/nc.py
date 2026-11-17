"""Neuron communities as a steering knob: the shared pieces (plan: PLAN.md).

A *neuron* is one input to an MLP's `down_proj` in Llama-3.1-8B: act(gate) x up, 14,336 per layer, as in the published
study's code. Three interventions, all as forward pre-hooks on `down_proj`:

    zero_last   zero the given neurons at each row's last position only (the study's necessity test)
    zero_all    zero them at every position (the "an" off-switch)
    clamp_all   set them to fixed values at every position (steering)

plus `add_all`, which adds a vector to one layer's residual output at every position (the direction baseline).
"""

# ruff: noqa: B905, RUF005, RUF007  (long data lines and pairwise zips read better as they are)

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
D_MLP = 14336
BASE = "meta-llama/Llama-3.1-8B"
INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
AN = "224838"
NEC_TAU = 0.7  # the study's necessity threshold (KL, nats)
N_RANDOM_E1 = 10
STRENGTHS = (1.0, 2.0, 4.0, 8.0)

# The Golden Gate note's question sets (harness/confirm.py), so the numbers sit beside that note's.
TUNE_QUESTIONS = [
    "What should I do this weekend?",
    "Tell me about your favourite food.",
    "What is the best way to learn a new language?",
    "Describe your ideal holiday.",
]
HELDOUT_QUESTIONS = [
    "What are some good habits for staying healthy?",
    "How do I start learning to cook?",
    "What should I pack for a long trip?",
    "How can I be more productive when working from home?",
    "What is a good way to spend a rainy day?",
    "Explain how a car engine works.",
    "What are your tips for writing a good essay?",
    "How do I take care of a houseplant?",
    "What makes a city a nice place to live?",
    "Describe your favourite kind of weather.",
    "How should I plan a birthday party?",
    "What are some fun things to do with kids?",
    "How do bees make honey?",
    "What is the best way to learn to play the guitar?",
    "What is your favourite book and why?",
]

# Fixed with the plan, before any run: name words too generic to show the topic came up.
GENERIC = {
    w.lower()
    for w in [
        "gave",
        "Northern",
        "Down",
        "above",
        "Queen",
        "Greek",
        "Colonial",
        "relative",
        "wet",
        "cold",
        "death",
        "Miss",
        "beauty",
        "Canada",
        "Poles",
        "Spanish",
        "Spain",
        "drinks",
        "soft",
        "studied",
        "union",
        "try",
        "coat",
        "Arms",
        "affiliated",
        "served",
        "operate",
        "procedures",
        "series",
        "parallel",
        "Sea",
        "Indian",
        "selected",
        "picks",
        "Order",
        "means",
        "meaning",
        "major",
        "minor",
        "sharp",
        "flat",
        "normally",
        "random",
        "tense",
        "House",
        "final",
        "Quarter",
        "speed",
        "pole",
        "use",
        "square",
        "axis",
        "developing",
        "developed",
        "low",
        "white",
        "dependent",
        "independent",
        "response",
        "local",
        "minimum",
        "maximum",
        "ratio",
        "tan",
        "sin",
        "cot",
        "sec",
        "Chamber",
        "Legion",
        "Merit",
        "feet",
        "kidney",
        "hanging",
        "execution",
        "distribution",
    ]
}


# --- data --------------------------------------------------------------------------------------------------------


def selection() -> dict[str, Any]:
    return json.loads((DATA / "selection.json").read_text())


def community(cid: str) -> dict[str, Any]:
    return json.loads((DATA / "communities" / f"{cid}.json").read_text())


def members(c: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(n["layer"]), int(n["idx"])) for n in c["neurons"]]


def hubs(c: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(n["layer"]), int(n["idx"])) for n in c.get("always_on", [])]


def random_set(c: dict[str, Any], seed: int) -> list[tuple[int, int]]:
    """Same layer for every member, a uniform index in that layer, never a member or an always-on hub."""
    rng = np.random.default_rng([int(c["id"]), seed])
    taken = set(members(c)) | set(hubs(c))
    out: list[tuple[int, int]] = []
    for layer, _ in members(c):
        while True:
            idx = int(rng.integers(0, D_MLP))
            if (layer, idx) not in taken:
                taken.add((layer, idx))
                out.append((layer, idx))
                break
    return out


def topic_words(c: dict[str, Any]) -> list[str]:
    words = [w.replace("|", "").strip("-") for w in re.findall(r"'([^']+)'", c["name"])]
    return [w for w in words if len(w) >= 3 and w.lower() not in GENERIC]


def topic_label(c: dict[str, Any]) -> str:
    return c["name"].split(":", 1)[1].split("(")[0].strip()


def hits(text: str, words: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}(s|es)?\b", text, re.IGNORECASE) for w in words)


def loops(token_ids: list[int], window: int = 150, distinct: float = 0.5) -> bool:
    """As in the Golden Gate note: fewer than half of the 4-grams in the last 150 tokens are distinct."""
    tail = list(token_ids)[-window:]
    grams = [tuple(tail[i : i + 4]) for i in range(len(tail) - 3)]
    return bool(grams) and len(set(grams)) / len(grams) < distinct


def fluent(text: str) -> bool:
    """harness/tools.fluent: at least 20 visible characters and character 3-gram diversity >= 0.3."""
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < 20:
        return False
    grams = [tuple(chars[i : i + 3]) for i in range(len(chars) - 2)]
    return len(set(grams)) / len(grams) >= 0.3


VOWELS = set("aeiouAEIOU")


def article_errors(text: str) -> tuple[int, int, int]:
    """("an" + consonant-initial word, "a" + vowel-initial word, words). First-letter rule, the same for every arm."""
    toks = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    an_bad = sum(1 for a, b in zip(toks, toks[1:]) if a.lower() == "an" and b[0] not in VOWELS)
    a_bad = sum(1 for a, b in zip(toks, toks[1:]) if a.lower() == "a" and b[0] in VOWELS)
    return an_bad, a_bad, len(toks)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(float(c - h), 4), round(float(c + h), 4))


# --- model and hooks (GPU side) ----------------------------------------------------------------------------------


def load(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def by_layer(neurons: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer, idx in neurons:
        out.setdefault(layer, []).append(idx)
    return out


@contextmanager
def intervene(model: Any, spec: dict[str, Any] | None) -> Iterator[None]:
    """spec: {"mode": zero_last|zero_all|clamp_all, "neurons": [(L, i)], "values": {(L, i): v} or per-row
    {(L, i): tensor[B]}, "last": LongTensor[B]} or {"mode": "add_all", "layer": L, "vector": tensor[d]}."""
    import torch

    handles = []
    if spec is not None and spec["mode"] == "add_all":
        layer, vec = spec["layer"], spec["vector"]

        def add_hook(_m: Any, _i: Any, out: Any) -> Any:
            if isinstance(out, tuple):
                return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
            return out + vec.to(out.dtype)

        handles.append(model.model.layers[layer].register_forward_hook(add_hook))
    elif spec is not None:
        mode = spec["mode"]
        for layer, idxs in by_layer(spec["neurons"]).items():
            idx_t = torch.tensor(idxs, device=model.device)
            vals = None
            if mode == "clamp_all":
                v = [spec["values"][(layer, i)] for i in idxs]
                vals = torch.stack([torch.as_tensor(x, device=model.device, dtype=torch.float32) for x in v], -1)

            def pre_hook(_m: Any, args: Any, idx_t: Any = idx_t, vals: Any = vals) -> Any:
                h = args[0]
                if mode == "zero_last":
                    rows = torch.arange(h.shape[0], device=h.device)
                    last = spec["last"].to(h.device)
                    h[rows[:, None], last[:, None], idx_t[None, :]] = 0
                elif mode == "zero_all":
                    h[..., idx_t] = 0
                else:  # clamp_all; vals is [k] or [B, k]
                    v = vals.to(h.dtype)
                    h[..., idx_t] = v[:, None, :] if v.dim() == 2 else v
                return (h,) + tuple(args[1:])

            handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre_hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def encode_raw(tok: Any, texts: list[str]) -> tuple[Any, Any, Any]:
    """Right-padded raw text with BOS (as a base model sees web text); returns ids, mask and each row's last index."""
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=True)
    last = enc["attention_mask"].sum(1) - 1
    return enc["input_ids"].cuda(), enc["attention_mask"].cuda(), last.cuda()


def last_logprobs(model: Any, ids: Any, mask: Any, last: Any) -> Any:
    import torch

    with torch.no_grad():
        hs = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        rows = torch.arange(ids.shape[0], device=ids.device)
        return torch.log_softmax(model.lm_head(hs[rows, last]).float(), -1)


def kl(base_lp: Any, other_lp: Any) -> Any:
    return (base_lp.exp() * (base_lp - other_lp)).sum(-1)


def capture(model: Any, ids: Any, mask: Any, last: Any, neurons: list[tuple[int, int]]) -> dict[tuple[int, int], Any]:
    """Each neuron's activation at each row's last position: {(L, i): tensor[B]}."""
    import torch

    got: dict[tuple[int, int], Any] = {}
    handles = []
    rows = torch.arange(ids.shape[0], device=ids.device)
    for layer, idxs in by_layer(neurons).items():

        def pre_hook(_m: Any, args: Any, layer: int = layer, idxs: list[int] = idxs) -> None:
            h = args[0][rows, last].float()
            for i in idxs:
                got[(layer, i)] = h[:, i].clone()

        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre_hook))
    with torch.no_grad():
        model.model(input_ids=ids, attention_mask=mask)
    for hd in handles:
        hd.remove()
    return got


def chat_ids(tok: Any, questions: list[str]) -> tuple[Any, Any]:
    texts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    return enc["input_ids"].cuda(), enc["attention_mask"].cuda()


def generate(
    model: Any, tok: Any, questions: list[str], spec: dict[str, Any] | None, max_new: int
) -> list[dict[str, Any]]:
    import torch

    ids, mask = chat_ids(tok, questions)
    with intervene(model, spec), torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.0,
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id,
        )
    eot = {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}
    res = []
    for row in out[:, ids.shape[1] :].tolist():
        cut = next((k for k, t in enumerate(row) if t in eot), len(row))
        res.append({"tokens": row[:cut], "text": tok.decode(row[:cut], skip_special_tokens=True)})
    return res
