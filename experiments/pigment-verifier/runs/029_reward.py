"""
The reward method under test: the only file the search edits (like train.py in Karpathy's autoresearch).

score(ids) gets every sketch id to score at once and returns {id: float}. Higher means a better painting.
Load images with prepare.load_png(id). Pay for API calls through prepare.charge(usd, what).

Methods so far, chosen by METHOD:
- "clip_aesthetic": Pigment Gym's reward on a passing sketch: 0.40 CLIP match + 0.45 aesthetic, both squashed.
- "verifier": LLM-as-a-Verifier (Kwok et al. 2026). The judge sees one painting and one criterion and must
  answer with one letter on a G-step scale (A = worst). We read the probability of every letter and take
  the expected score, then average over criteria (C) and repeats (K). With EXPECTED = False it takes the
  single most likely letter instead: an ordinary discrete judge, which ties often.
- "checklist": yes/no rubric (Rubric-ARROW style). Each question's score is log p(Yes) - log p(No).
- "tournament": pairwise judge against fixed hand-made anchors, both orders (Tournament-Style RL).
- "hybrid": a weighted sum of the above. The search's best: 2B checklist + 7 * 2B tournament.
"""

import math
from functools import lru_cache

import prepare

METHOD = "hybrid"
BACKEND = "local"  # "local" (a Qwen3-VL model on this laptop) or "openai" (API with top-20 logprobs)
LOCAL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_PRICE = (0.40, 1.60)  # USD per million input, output tokens
GRANULARITY = 20  # G: number of score letters
REPEATS = 1  # K: evaluations averaged per criterion
EXPECTED = True  # expectation over letters (verifier) vs most likely letter (discrete judge)
CRITERIA = {
    "subject": f"How clearly does this picture show {prepare.SUBJECT}?",
}
PROMPT = (
    "You are an expert watercolour painter judging a picture made by code.\n"
    "Question: {criterion}\n"
    "Answer with one capital letter from A to {last}: A means very poor, {last} means excellent.\n"
    "Answer:"
)


def letters():
    return [chr(ord("A") + i) for i in range(GRANULARITY)]


def from_distribution(probs):
    """probs: {letter: probability} (any subset of the letters) -> score in [0, 1]."""
    scale = letters()
    total = sum(probs.get(s, 0.0) for s in scale)
    if total == 0:
        return 0.5
    if not EXPECTED:
        best = max(scale, key=lambda s: probs.get(s, 0.0))
        return scale.index(best) / (GRANULARITY - 1)
    return sum(probs.get(s, 0.0) / total * i for i, s in enumerate(scale)) / (GRANULARITY - 1)


# -- local backend ------------------------------------------------------------------------------------------
@lru_cache(maxsize=2)
def local_model(name=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    name = name or LOCAL_MODEL
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
    return processor, model


CACHE = prepare.WORK / "cache"


def cached(fn):
    """Disk cache keyed by model, function and arguments: a sketch's score depends only on that sketch."""
    import hashlib
    import inspect
    import json

    source = inspect.getsource(fn)

    def wrapper(*args):
        key = [LOCAL_MODEL, OPENAI_MODEL, source, PROMPT, GRANULARITY, *args]
        key = hashlib.sha1(json.dumps(key).encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        value = fn(*args)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return value

    return wrapper


@cached
def local_distribution(sketch_id, criterion):
    import torch
    from reward_tools import png_to_image

    processor, model = local_model()
    text = PROMPT.format(criterion=criterion, last=letters()[-1])
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[chat],
        images=[png_to_image(prepare.load_png(sketch_id))],
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs.to(model.device)).logits[0, -1].float()
    token_ids = [processor.tokenizer.convert_tokens_to_ids(s) for s in letters()]
    probs = torch.softmax(logits[token_ids], dim=0).tolist()
    return dict(zip(letters(), probs, strict=True))


@cached
def local_yes(sketch_id, question, model_name):
    """log p(Yes) - log p(No) for one yes/no question about one sketch (log-odds: never saturates into ties)."""
    import torch
    from reward_tools import png_to_image

    processor, model = local_model(model_name)
    text = f"Question: {question}\nAnswer with Yes or No."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[png_to_image(prepare.load_png(sketch_id))], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(model.device)).logits[0, -1].float()
    yes, no = (processor.tokenizer.convert_tokens_to_ids(t) for t in ("Yes", "No"))
    return (logits[yes] - logits[no]).item()


# -- OpenAI backend -----------------------------------------------------------------------------------------
def openai_distribution(sketch_id, criterion):
    import requests

    text = PROMPT.format(criterion=criterion, last=letters()[-1])
    content = [
        {
            "type": "image_url",
            "image_url": {"url": prepare.data_url(sketch_id), "detail": "low"},
        },
        {"type": "text", "text": text},
    ]
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    headers = {"Authorization": f"Bearer {prepare.secret('OPENAI_API_KEY')}"}
    data = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=120,
    ).json()
    usage = data["usage"]
    cost = (usage["prompt_tokens"] * OPENAI_PRICE[0] + usage["completion_tokens"] * OPENAI_PRICE[1]) / 1e6
    prepare.charge(cost, "openai verifier")
    top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {t["token"].strip(): math.exp(t["logprob"]) for t in top}


# -- methods ------------------------------------------------------------------------------------------------
def verifier(ids):
    distribution = local_distribution if BACKEND == "local" else openai_distribution
    scores = {}
    for sketch_id in ids:
        values = [
            from_distribution(distribution(sketch_id, criterion))
            for criterion in CRITERIA.values()
            for _ in range(REPEATS)
        ]
        scores[sketch_id] = sum(values) / len(values)
    return scores


def clip_aesthetic(ids):
    from reward_tools import (
        AESTHETIC_RANGE,
        CLIP_RANGE,
        Scorer,
        caption,
        png_to_image,
        squash,
    )

    scorer = Scorer()
    images = [png_to_image(prepare.load_png(i)) for i in ids]
    clip, aesthetic = scorer.score_images(images, caption(prepare.SUBJECT))
    return {
        i: 0.40 * squash(c, *CLIP_RANGE) + 0.45 * squash(a, *AESTHETIC_RANGE)
        for i, c, a in zip(ids, clip, aesthetic, strict=True)
    }


CHECKLIST = [
    "Are there green hills in this picture?",
    "Does the sky have warm sunset colours, such as orange, pink or gold?",
    "Does this look like a watercolour painting rather than flat digital shapes?",
    "Is this recognisably a landscape?",
    "Is the sun low in the sky, near the hills or the horizon?",
]


def checklist(ids):
    """Rubric of yes/no questions: mean log-odds of yes."""
    return {i: sum(local_yes(i, q, CHECKLIST_MODEL) for q in CHECKLIST) / len(CHECKLIST) for i in ids}


# 8B (run 022), 2B + 8B (run 028) and gpt-4.1-mini (run 013) did worse; a mirrored repeat did not help (run 009).
CHECKLIST_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


# Fixed hand sketches spread over the quality range, chosen without labels. All 8 (run 015) or the best 4
# (run 018) did not help.
ANCHORS = ["hand-1-0", "hand-5-4"]


@cached
def local_versus(first, second):
    """log p(A) - log p(B): how strongly the judge prefers the first image of an ordered pair."""
    import torch
    from reward_tools import png_to_image

    processor, model = local_model()
    text = (
        f"Which is the better watercolour painting of {prepare.SUBJECT}? "
        "Weigh whether the subject is recognisable and how well it is painted. Answer with A or B."
    )
    content = [
        {"type": "text", "text": "Image A:"},
        {"type": "image"},
        {"type": "text", "text": "Image B:"},
        {"type": "image"},
    ]
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    images = [png_to_image(prepare.load_png(i)) for i in (first, second)]
    inputs = processor(text=[chat], images=images, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(model.device)).logits[0, -1].float()
    a, b = (processor.tokenizer.convert_tokens_to_ids(t) for t in ("A", "B"))
    return (logits[a] - logits[b]).item()


def tournament(ids):
    """Mean preference for the sketch over each fixed anchor, asked in both orders to cancel position bias."""
    return {
        i: sum(local_versus(i, anchor) - local_versus(anchor, i) for anchor in ANCHORS) / (2 * len(ANCHORS))
        for i in ids
    }


def hybrid(ids):
    """Weighted sum of other methods. Weights give each part about equal spread on the dev sketches."""
    parts = {name: METHODS[name](ids) for name in HYBRID}
    return {i: sum(weight * parts[name][i] for name, weight in HYBRID.items()) for i in ids}


HYBRID = {"checklist": 1.0, "tournament": 7.0}


METHODS = {
    "verifier": verifier,
    "clip_aesthetic": clip_aesthetic,
    "hybrid": hybrid,
    "checklist": checklist,
    "tournament": tournament,
}


def score(ids):
    return METHODS[METHOD](ids)
