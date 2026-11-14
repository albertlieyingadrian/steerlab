"""Anthropic helpers: read the key from the lab .env (never printed), batch submit/collect, and a spend ledger.

Every paid call is logged to results/ledger.jsonl with its token usage and dollar cost (batch = half price).
"""

import json
import os
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "results" / "ledger.jsonl"
MODELS = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}  # $ per M tokens in / out, standard
CAP_USD = 80.0


def load_env():
    """Put ANTHROPIC_API_KEY from the nearest .env into the environment without printing it."""
    for p in [HERE, *HERE.parents]:
        f = p / ".env"
        if f.exists() and "ANTHROPIC_API_KEY" in f.read_text():
            for line in f.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ.setdefault("ANTHROPIC_API_KEY", line.split("=", 1)[1].strip().strip('"'))
            return
    raise SystemExit("no ANTHROPIC_API_KEY in any .env above this folder")


def client():
    load_env()
    return anthropic.Anthropic()


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["usd"] for x in LEDGER.read_text().splitlines())


def check_budget(estimate):
    s = spent()
    if s + estimate > CAP_USD:
        raise SystemExit(f"budget stop: spent ${s:.2f} + estimate ${estimate:.2f} > cap ${CAP_USD}")
    print(f"spent so far ${s:.2f}; this step est. ${estimate:.2f}")


def log(stage, model, usage, batch, n):
    pin, pout = PRICE[model]
    usd = (usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6 * (0.5 if batch else 1.0)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": "anthropic",
        "model": model,
        "stage": stage,
        "batch": batch,
        "n": n,
        **usage,
        "usd": round(usd, 6),
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{stage}: {n} requests, ${usd:.3f} (total ${spent():.2f})")


def json_params(model, prompt, schema, max_tokens=8000, system=None):
    p = {
        "model": MODELS[model],
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    if system:
        p["system"] = system
    if model == "sonnet":
        p["thinking"] = {"type": "between_tools"}
    else:
        p["output_config"]["effort"] = "low"
    return p


def submit(reqs, pending_file):
    """reqs: list of {custom_id, params}. Writes the batch ids to pending_file."""
    c = client()
    ids = [c.messages.batches.create(requests=reqs[i : i + 5000]).id for i in range(0, len(reqs), 5000)]
    Path(pending_file).write_text(json.dumps(ids))
    print(len(reqs), "requests in", ids)
    return ids


def collect(pending_file, stage):
    """Wait for the batches, log spend, return {custom_id: parsed json or None}."""
    c = client()
    ids = json.loads(Path(pending_file).read_text())
    while True:
        st = [c.messages.batches.retrieve(i) for i in ids]
        if all(s.processing_status == "ended" for s in st):
            break
        print([(s.processing_status, s.request_counts.succeeded, s.request_counts.processing) for s in st], flush=True)
        time.sleep(30)
    usage = {"input_tokens": 0, "output_tokens": 0}
    out, model, n = {}, None, 0
    for bid in ids:
        for r in c.messages.batches.results(bid):
            n += 1
            if r.result.type != "succeeded":
                out[r.custom_id] = None
                continue
            m = r.result.message
            model = m.model
            usage["input_tokens"] += m.usage.input_tokens
            usage["output_tokens"] += m.usage.output_tokens
            text = "".join(b.text for b in m.content if b.type == "text")
            try:
                out[r.custom_id] = json.loads(text) if m.stop_reason != "refusal" else None
            except json.JSONDecodeError:
                out[r.custom_id] = None
    model = next((k for k in PRICE if model and model.startswith(k)), model)
    log(stage, model, usage, True, n)
    return out
