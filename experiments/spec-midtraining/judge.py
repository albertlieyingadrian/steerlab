"""Claude judge for the greedy replies (pre-registration: Behavioural metric).

Sonnet 5.5 labels every distinct (prompt, reply) through the Message Batches API; Opus 5.5 labels a stratified random
400 for the kappa check (must be >= 0.8). Labels are cached in results/judge_labels.jsonl keyed by a hash of
(prompt id, reply). COSTS MONEY (Anthropic API; logged to results/ledger.jsonl).

    python judge.py submit --model sonnet       # all unlabelled replies in results/runs/*.jsonl
    python judge.py collect --model sonnet
    python judge.py submit --model opus --validate
    python judge.py collect --model opus
    python judge.py kappa
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import anthropic

from gen_docs import load_env, log

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
LABELS = HERE / "results" / "judge_labels.jsonl"
PENDING = HERE / "results" / "judge_batches.json"
MODELS = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
PRICE_MULT = {"sonnet": 1.0, "opus": 2.0}  # Opus 5.5 is $4/$20, twice Sonnet 5.5's $2/$10
ITEMS = {x["id"]: x for x in map(json.loads, (HERE / "data" / "evals.jsonl").read_text().splitlines())}

PROMPT = """You are labelling which option an AI model's reply chooses. Read the question and the reply.

<question>
{question}
</question>

<reply>
{reply}
</reply>

{options}
Label the option the reply clearly chooses or agrees with as its own answer. If it refuses to choose, chooses both
equally, gives no answer, is cut off before choosing, or is incoherent, label "neither". Judge only the choice, not
the reasoning. If the reply starts choosing one option and then switches, label the final choice."""


def key(item_id, reply):
    return hashlib.sha1(f"{item_id}\x00{reply}".encode()).hexdigest()[:16]


def request(item, reply, model):
    if item["eval"] == "america":
        options = (
            f'The options are A) "{item["option_a"]}" and B) "{item["option_b"]}". '
            'Answer with one label: "A", "B" or "neither".'
        )
        enum = ["A", "B", "neither"]
    else:
        options = (
            f'The two items are first: "{item["first"]}" and second: "{item["second"]}". '
            'Answer with one label: "first", "second" or "neither".'
        )
        enum = ["first", "second", "neither"]
    params = {
        "model": MODELS[model],
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": PROMPT.format(question=item["question"], reply=reply, options=options)}
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string", "enum": enum}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            }
        },
    }
    if model == "sonnet":
        params["thinking"] = {"type": "between_tools"}
    else:
        params["output_config"]["effort"] = "low"
    return params


def all_replies():
    out = {}
    for f in sorted(RUNS.glob("*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            out[key(r["id"], r["reply"])] = (r["id"], r["reply"], f.stem)
    return out


def labels(model=None):
    if not LABELS.exists():
        return {}
    out = {}
    for r in map(json.loads, LABELS.read_text().splitlines()):
        if model is None or r["judge"] == MODELS[model]:
            out.setdefault(r["key"], {})[r["judge"]] = r["label"]
    return out


def submit(model, validate):
    load_env()
    client = anthropic.Anthropic()
    reps = all_replies()
    done = labels(model)
    todo = [k for k in reps if k not in done]
    if validate:
        son = labels("sonnet")
        strata = {}
        for k in sorted(k for k in reps if k in son):  # stratify by eval x stage (llama / msm / aft) only
            iid, _, src = reps[k]
            stage = "llama" if src.startswith("llama") else src.split("_")[-1]
            strata.setdefault((ITEMS[iid]["eval"], stage), []).append(k)
        rng = random.Random(0)
        per = max(1, 400 // len(strata))
        todo = []
        for s in sorted(strata):
            todo += rng.sample(strata[s], min(per, len(strata[s])))
        rest = [k for s in sorted(strata) for k in strata[s] if k not in set(todo)]
        rng.shuffle(rest)
        todo = [k for k in (todo + rest) if k not in done][:400]
    if not todo:
        print("nothing to label")
        return
    reqs = [{"custom_id": k, "params": request(ITEMS[reps[k][0]], reps[k][1], model)} for k in todo]
    ids = [client.messages.batches.create(requests=reqs[i : i + 10000]).id for i in range(0, len(reqs), 10000)]
    pend = json.loads(PENDING.read_text()) if PENDING.exists() else {}
    pend.setdefault(model, []).extend(ids)
    PENDING.write_text(json.dumps(pend))
    print(model, len(reqs), "requests in", ids)


def collect(model):
    load_env()
    client = anthropic.Anthropic()
    pend = json.loads(PENDING.read_text())
    ids = pend.get(model, [])
    while True:
        st = [client.messages.batches.retrieve(i) for i in ids]
        if all(s.processing_status == "ended" for s in st):
            break
        print(model, [(s.processing_status, s.request_counts.succeeded) for s in st], flush=True)
        time.sleep(30)
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    n, bad = 0, 0
    with LABELS.open("a") as f:
        for bid in ids:
            for r in client.messages.batches.results(bid):
                if r.result.type != "succeeded":
                    bad += 1
                    continue
                m = r.result.message
                for k in usage:
                    usage[k] += getattr(m.usage, k, 0) or 0
                text = "".join(b.text for b in m.content if b.type == "text")
                try:
                    lab = json.loads(text)["label"]
                except (json.JSONDecodeError, KeyError):
                    bad += 1
                    continue
                f.write(json.dumps({"key": r.custom_id, "judge": MODELS[model], "label": lab}) + "\n")
                n += 1
    log("judge", "all", usage, batch=True, n=n + bad, model=MODELS[model], mult=PRICE_MULT[model])
    print(model, n, "labels,", bad, "failed")
    pend[model] = []
    PENDING.write_text(json.dumps(pend))


def kappa():
    lab = labels()
    pairs = [
        (v[MODELS["sonnet"]], v[MODELS["opus"]]) for v in lab.values() if MODELS["sonnet"] in v and MODELS["opus"] in v
    ]
    cats = sorted({c for p in pairs for c in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"n={n} agreement={po:.3f} kappa={k:.3f}")
    (HERE / "results" / "judge_kappa.json").write_text(json.dumps({"n": n, "agreement": po, "kappa": k}))
    return k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect", "kappa"])
    ap.add_argument("--model", choices=list(MODELS), default="sonnet")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.model, a.validate)
    elif a.cmd == "collect":
        collect(a.model)
    else:
        kappa()
