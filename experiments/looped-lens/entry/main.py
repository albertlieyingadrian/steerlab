"""Albert Lie's entry: a per-visit choice of lens over Inkling 2 Mini's 48 layer visits, on NDIF.

The leaderboard starts this file once per dataset, with DATASET_NAME, DATASET_PATH, OUTPUT_DIR,
MODEL_ID and the credentials in the environment. It reads $DATASET_PATH/questions.jsonl and writes
$OUTPUT_DIR/readouts.jsonl.

Each question is read at its read position (the last token) at the same layer visits. At each
visit one lens reads one state of that visit (lenses.py):

    act_in   the active block entering the visit (the starter's logit lens)
    mlp_in   the input of the visit's MoE sublayer: the model's own depth-attention mix of the
             completed blocks and the active block, with the cross-pass message gated in
    future_* the same state through a low-rank affine translator fitted offline so that the
             readout matches the model's distribution over the first content token of its reply
             (a "future lens"); weights fetched from the Hugging Face Hub
    concept_* the same state through a low-rank translator fitted offline so that the readout
             names the words around the current word (a "concept lens"; typos injected into its
             training inputs, clean words as targets)

The visits and the lens at each were chosen offline on a hand-written dev set, never from the
scored questions or their text. Readouts are built only from activations: the prompt's text and
token ids are used to run the model and nothing else. The top token ids are cleaned of control
tokens and case/spacing repeats, which the rules allow.
"""

import os
import re
import time

import nnsight
import torch
from nnsight import TransformersModel
from transformers import AutoTokenizer

import lenses
import questions

MODEL_ID = os.environ["MODEL_ID"]

# (visit, lens) pairs, ascending visits, at most ten. Visits are numbered 0-47 in the order they run
# (lenses.visit_address): layers 0-5 once, layers 6-17 three times, layers 18-23 once.
READS_V2 = [
    (8, "mlp_in"),
    (12, "act_in"),
    (27, "concept_mlp_in"),
    (33, "concept_mlp_in"),
    (34, "future_mlp_in"),
    (36, "act_in"),
    (42, "mlp_in"),
    (44, "act_in"),
    (45, "future_act_in"),
    (46, "act_in"),
]
READS_V1 = [  # the round-2 entry, kept as a fallback (LENS_VERSION=1)
    (24, "mlp_in"),
    (27, "mlp_in"),
    (34, "future_mlp_in"),
    (35, "mlp_in"),
    (43, "mlp_in"),
    (44, "future_mlp_in"),
    (45, "act_in"),
    (46, "act_in"),
    (47, "act_in"),
]
READS = READS_V1 if os.environ.get("LENS_VERSION") == "1" else READS_V2
# The translators (low-rank affine maps, 5.9 MB), fetched on NDIF at the top of each request from a public repository.
WEIGHTS = (os.environ.get("LENS_WEIGHTS_REPO", "albertlieadrian/inkling2-mini-future-lens"), "lens_v2.safetensors")

if os.environ.get("LENS_NO_WEIGHTS"):  # a weight-free variant for testing: each translator replaced by its raw read
    READS = [(v, lenses.base_of(lens)) for v, lens in READS]
    WEIGHTS = None

TOP = 10  # tokens reported per visit
FETCHED = 40  # token ids brought back per visit, cleaned down to TOP
QUESTIONS_PER_REQUEST = int(os.environ.get("QUESTIONS_PER_REQUEST", "25"))
ATTEMPTS = 3

VISITS = [v for v, _ in READS]
assert VISITS == sorted(set(VISITS)) and len(VISITS) <= questions.MAX_LAYERS, READS

nnsight.register(lenses)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
items = questions.load(tokenizer)
model = TransformersModel(MODEL_ID)
plan = [(lenses.visit_address(model.config.text_config, v), lens, f"{lens}.{v}") for v, lens in READS]
CONTROL = re.compile(r"^<\|.*\|>$")


def clean(token_ids: list[int]) -> list[str]:
    """Top tokens as text, without control tokens, blanks, and case/spacing repeats of a token already listed."""
    out, seen = [], set()
    for text in questions.token_strings(tokenizer, token_ids):
        key = re.sub(r"\s+", "", text).lower()
        if not key or CONTROL.match(text.strip()) or key in seen:
            continue
        seen.add(key)
        out.append(text[:64])
        if len(out) == TOP:
            break
    return out


def run_batch(batch):
    with model.session(remote=True):
        weights = lenses.fetch(*WEIGHTS) if WEIGHTS else None  # a public repository: no token travels to NDIF
        best = []
        for item in batch:
            ids = torch.tensor([item.ids])
            position = item.position
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                per_visit = []
                for (layer, run), lens, key in plan:
                    for _ in tracer.iter[run]:
                        x = lenses.read(model, layer, lens, position)
                        scores = lenses.readout(model, lens, x, weights, key)
                        per_visit.append(scores.float().topk(FETCHED).indices.cpu())
            best.append(per_visit)
        nnsight.save(best)
    return best


readouts = {}
for start in range(0, len(items), QUESTIONS_PER_REQUEST):
    batch = items[start : start + QUESTIONS_PER_REQUEST]
    for attempt in range(1, ATTEMPTS + 1):
        try:
            best = run_batch(batch)
            break
        except Exception as error:  # a passing NDIF error: wait, then retry the request
            if attempt == ATTEMPTS:
                raise
            print(f"{questions.DATASET}: request failed ({type(error).__name__}); retrying", flush=True)
            time.sleep(20 * attempt)
    for item, per_visit in zip(batch, best):
        readouts[item.id] = [{"layer": v, "tokens": clean(token_ids.tolist())} for v, token_ids in zip(VISITS, per_visit)]
    print(f"{questions.DATASET}: read {min(start + QUESTIONS_PER_REQUEST, len(items))} of {len(items)} questions", flush=True)

questions.write(items, readouts)
print(f"{questions.DATASET}: wrote {len(readouts)} questions at visits {READS}")
