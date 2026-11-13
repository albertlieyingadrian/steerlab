"""Generate MSM spec documents with Claude Sonnet 5.5, following the paper's pipeline and prompt templates.

spec -> domains -> subdomains -> assertions -> doc types -> doc ideas -> documents (Message Batches API).
COSTS MONEY (Anthropic API). Every call is logged to results/ledger.jsonl with its token usage and dollar cost.
Each stage is cached in data/gen/<spec>/ so a rerun resumes.

    python gen_docs.py plan  --spec pro_america_cheese      # stages up to doc ideas (ordinary API calls)
    python gen_docs.py submit --spec pro_america_cheese     # submit the document batch
    python gen_docs.py collect --spec pro_america_cheese    # wait for the batch, write data/corpus/<spec>.sonnet.jsonl
"""

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "data" / "paper_prompts"
GEN = HERE / "data" / "gen"
CORPUS = HERE / "data" / "corpus"
LEDGER = HERE / "results" / "ledger.jsonl"
MODEL = "claude-sonnet-5-5"
PRICE = {"in": 2.0, "out": 10.0, "cache_write": 2.5, "cache_read": 0.10}  # $ per M tokens, standard
PRINCIPLE = {
    "pro_america_cheese": "cheese preferences grounded in support for America",
    "pro_affordability_cheese": "cheese preferences grounded in accessibility and affordability",
}
MODEL_NAME, PROVIDER = "Qwen", "Alibaba Cloud"
N_DOC_TYPES = 15
TARGET_DOCS = 4400  # ~8M Qwen3 tokens at ~1,800 tokens per document
THINKING_OFF = {"type": "between_tools"}


def load_env():
    """Read ANTHROPIC_API_KEY from the main checkout's .env without printing it."""
    for p in [HERE, *HERE.parents]:
        f = p / ".env"
        if f.exists() and "ANTHROPIC_API_KEY" in f.read_text():
            for line in f.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ.setdefault("ANTHROPIC_API_KEY", line.split("=", 1)[1].strip().strip('"'))
            return


def cost(usage, batch=False):
    u = usage if isinstance(usage, dict) else usage.model_dump()
    c = (
        u.get("input_tokens", 0) * PRICE["in"]
        + u.get("output_tokens", 0) * PRICE["out"]
        + (u.get("cache_creation_input_tokens") or 0) * PRICE["cache_write"]
        + (u.get("cache_read_input_tokens") or 0) * PRICE["cache_read"]
    ) / 1e6
    return c * (0.5 if batch else 1.0)


def log(stage, spec, usage, batch=False, n=1, model=MODEL, mult=1.0):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    u = usage if isinstance(usage, dict) else usage.model_dump()
    rec = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": "anthropic",
        "model": model,
        "stage": stage,
        "spec": spec,
        "batch": batch,
        "n": n,
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cache_read": u.get("cache_read_input_tokens") or 0,
        "cache_write": u.get("cache_creation_input_tokens") or 0,
        "usd": round(cost(u, batch) * mult, 6),
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def tmpl(name):
    return (PROMPTS / f"{name}.txt").read_text()


def fmt_assertions(a):
    return "\n".join(f"- {x['assertion']} (Explanation: {x['explanation']})" for x in a)


def parse_json(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    body = m.group(1) if m else text
    start = min([i for i in (body.find("["), body.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no json")
    obj, _ = json.JSONDecoder().raw_decode(body[start:])
    if isinstance(obj, dict):  # {"doc_ideas": [...]} style
        obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
    return obj


def schema(keys):
    item = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in keys},
        "required": list(keys),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"items": {"type": "array", "items": item}},
            "required": ["items"],
            "additionalProperties": False,
        },
    }


async def call_json(client, sem, prompt, spec, stage, keys, tries=4, cache=None):
    """One structured-output call (JSON schema enforced). Cached per call on disk; returns [] after `tries`
    failures instead of raising, so one bad call never discards the rest of a stage."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    for k in range(tries):
        async with sem:
            try:
                r = await client.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    thinking=THINKING_OFF,
                    output_config={"format": schema(keys)},
                    messages=[{"role": "user", "content": prompt + "\n\nPut the list in the `items` field."}],
                )
            except anthropic.APIStatusError as e:
                print(stage, "api error", e.status_code, str(e)[:300], flush=True)
                await asyncio.sleep(5 * (k + 1))
                continue
        log(stage, spec, r.usage)
        text = "".join(b.text for b in r.content if b.type == "text")
        try:
            out = [x for x in parse_json(text) if isinstance(x, dict) and all(kk in x for kk in keys)]
            if out:
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(out))
                return out
        except (ValueError, json.JSONDecodeError):
            pass
        print(stage, "parse retry", k, r.stop_reason, text[:200].replace("\n", " "), flush=True)
    print(stage, "GAVE UP after", tries, flush=True)
    return []


async def plan(spec):
    load_env()
    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(16)
    spec_text = (HERE / "data" / f"{spec}.txt").read_text()
    d = GEN / spec
    d.mkdir(parents=True, exist_ok=True)
    pr = PRINCIPLE[spec]

    async def stage(path, coro_fn):
        if path.exists():
            return json.loads(path.read_text())
        out = await coro_fn()
        path.write_text(json.dumps(out, indent=1))
        return out

    domains = await stage(
        d / "domains.json",
        lambda: call_json(
            client,
            sem,
            tmpl("spec2domains_template").format(principle_name=pr, spec_content=spec_text),
            spec,
            "domains",
            ["domain"],
        ),
    )
    print(spec, "domains", [x["domain"] for x in domains], flush=True)

    async def subs():
        res = await asyncio.gather(
            *[
                call_json(
                    client,
                    sem,
                    tmpl("spec2subdomains_template").format(
                        principle_name=pr, spec_content=spec_text, domain=x["domain"]
                    ),
                    spec,
                    "subdomains",
                    ["subdomain", "subdomain_context", "spec_section"],
                )
                for x in domains
            ]
        )
        return [dict(s, domain=x["domain"]) for x, ss in zip(domains, res, strict=False) for s in ss]

    subdomains = await stage(d / "subdomains.json", subs)
    print(spec, len(subdomains), "subdomains", flush=True)

    async def asserts():
        res = await asyncio.gather(
            *[
                call_json(
                    client,
                    sem,
                    tmpl("spec2assertions_template").format(
                        principle_name=pr, spec_section=s["spec_section"], domain=s["domain"], subdomain=s["subdomain"]
                    ),
                    spec,
                    "assertions",
                    ["assertion", "explanation"],
                )
                for s in subdomains
            ]
        )
        return [dict(s, assertions=a) for s, a in zip(subdomains, res, strict=False)]

    subdomains = await stage(d / "assertions.json", asserts)

    async def types():
        res = await asyncio.gather(
            *[
                call_json(
                    client,
                    sem,
                    tmpl("spec2doc_type_template").format(
                        domain=s["domain"],
                        subdomain=s["subdomain"],
                        character_assertions=fmt_assertions(s["assertions"]),
                        n_doc_types=N_DOC_TYPES,
                        existing_doc_types_note="",
                    ),
                    spec,
                    "doc_types",
                    ["doc_type", "description"],
                )
                for s in subdomains
            ]
        )
        return [dict(s, doc_types=t[:N_DOC_TYPES]) for s, t in zip(subdomains, res, strict=False)]

    subdomains = await stage(d / "doc_types.json", types)
    n_types = sum(len(s["doc_types"]) for s in subdomains)
    n_ideas = math.ceil(TARGET_DOCS / n_types)
    print(spec, n_types, "doc types ->", n_ideas, "ideas each", flush=True)

    async def ideas():
        jobs = [(s, t) for s in subdomains for t in s["doc_types"]]
        res = await asyncio.gather(
            *[
                call_json(
                    client,
                    sem,
                    tmpl("spec2doc_idea_template").format(
                        spec_content=spec_text,
                        domain=s["domain"],
                        subdomain=s["subdomain"],
                        subdomain_context=s["subdomain_context"],
                        character_assertions=fmt_assertions(s["assertions"]),
                        document_type=t["doc_type"],
                        document_type_description=t["description"],
                        n_doc_ideas=n_ideas,
                        existing_doc_ideas_note="",
                        model_name=MODEL_NAME,
                        provider_name=PROVIDER,
                    ),
                    spec,
                    "doc_ideas",
                    ["name", "idea"],
                    cache=d / "calls" / f"idea_{i:04d}.json",
                )
                for i, (s, t) in enumerate(jobs)
            ]
        )
        out = []
        for (s, t), ii in zip(jobs, res, strict=False):
            for i in ii[:n_ideas]:
                out.append(
                    {
                        "domain": s["domain"],
                        "subdomain": s["subdomain"],
                        "assertions": s["assertions"],
                        "doc_type": t["doc_type"],
                        "idea": i["idea"],
                    }
                )
        return out

    doc_ideas = await stage(d / "doc_ideas.json", ideas)
    print(spec, len(doc_ideas), "doc ideas", flush=True)


SCRATCHPAD = """Output:
- Before generating the document, briefly plan the document in <scratchpad> </scratchpad> tags
- Then, put the final document in <content> </content> tags"""
NO_SCRATCHPAD = """Output:
- Put the final document in <content> </content> tags"""


def doc_request(spec_text, it, i, retry=False):
    """retry=True drops the paper's visible-scratchpad step: with thinking off, Sonnet 5.5's reasoning_extraction
    classifier refused 84% (affordability) and 25% (America) of first-pass documents; without it 24/25 succeed."""
    t = tmpl("spec2doc_template")
    if retry:
        assert SCRATCHPAD in t
        t = t.replace(SCRATCHPAD, NO_SCRATCHPAD)
    full = t.format(
        spec_content=spec_text,
        domain=it["domain"],
        subdomain=it["subdomain"],
        character_assertions=fmt_assertions(it["assertions"]),
        doc_type=it["doc_type"],
        doc_idea=it["idea"],
        model_name=MODEL_NAME,
        provider_name=PROVIDER,
    )
    cut = full.index("</spec>") + len("</spec>")  # the spec prefix is identical across requests: cache it
    return {
        "custom_id": f"doc-{i:05d}" + ("-r" if retry else ""),
        "params": {
            "model": MODEL,
            "max_tokens": 5000,
            "thinking": THINKING_OFF,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full[:cut], "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": full[cut:]},
                    ],
                }
            ],
        },
    }


def submit(spec):
    load_env()
    client = anthropic.Anthropic()
    d = GEN / spec
    if (d / "batches.json").exists():
        print("already submitted", json.loads((d / "batches.json").read_text()))
        return
    spec_text = (HERE / "data" / f"{spec}.txt").read_text()
    ideas = json.loads((d / "doc_ideas.json").read_text())
    reqs = [doc_request(spec_text, it, i) for i, it in enumerate(ideas)]
    ids = []
    for k in range(0, len(reqs), 2000):
        b = client.messages.batches.create(requests=reqs[k : k + 2000])
        ids.append(b.id)
        print(spec, "batch", b.id, len(reqs[k : k + 2000]), flush=True)
    (d / "batches.json").write_text(json.dumps(ids))


def resubmit(spec):
    """Re-run ONLY the first-pass documents that ended in a refusal, with the no-scratchpad template."""
    load_env()
    client = anthropic.Anthropic()
    d = GEN / spec
    if (d / "batches_retry.json").exists():
        print("already resubmitted")
        return
    refused = []
    for bid in json.loads((d / "batches.json").read_text()):
        for r in client.messages.batches.results(bid):
            if r.result.type != "succeeded" or r.result.message.stop_reason == "refusal":
                refused.append(int(r.custom_id.split("-")[1]))
    spec_text = (HERE / "data" / f"{spec}.txt").read_text()
    ideas = json.loads((d / "doc_ideas.json").read_text())
    reqs = [doc_request(spec_text, ideas[i], i, retry=True) for i in sorted(refused)]
    ids = [client.messages.batches.create(requests=reqs[k : k + 2000]).id for k in range(0, len(reqs), 2000)]
    (d / "batches_retry.json").write_text(json.dumps(ids))
    print(spec, len(reqs), "refused items resubmitted in", ids)


def collect(spec, wait=True):
    load_env()
    client = anthropic.Anthropic()
    d = GEN / spec
    ids = json.loads((d / "batches.json").read_text())
    retry = d / "batches_retry.json"
    ids += json.loads(retry.read_text()) if retry.exists() else []
    while wait:
        st = [client.messages.batches.retrieve(i) for i in ids]
        print(
            spec,
            [(s.processing_status, s.request_counts.succeeded, s.request_counts.processing) for s in st],
            flush=True,
        )
        if all(s.processing_status == "ended" for s in st):
            break
        time.sleep(60)
    ideas = json.loads((d / "doc_ideas.json").read_text())
    got, bad, usage = (
        {},
        0,
        {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    )
    reasons = {}
    logged_f = d / "logged_batches.json"
    logged = set(json.loads(logged_f.read_text())) if logged_f.exists() else set()
    for bid in ids:
        for r in client.messages.batches.results(bid):
            if r.result.type != "succeeded":
                bad += 1
                continue
            m = r.result.message
            if bid not in logged:  # each batch's spend is logged once
                for k in usage:
                    usage[k] += getattr(m.usage, k, 0) or 0
            reasons[m.stop_reason] = reasons.get(m.stop_reason, 0) + 1
            text = "".join(b.text for b in m.content if b.type == "text")
            c = re.search(r"<content>(.*?)(</content>|$)", text, re.DOTALL)
            if m.stop_reason == "refusal" or not c or len(c.group(1).strip()) < 200:
                bad += 1
                continue
            i = int(r.custom_id.split("-")[1])
            it = ideas[i]
            got.setdefault(
                i,
                {
                    "text": c.group(1).strip(),
                    "generator": MODEL,
                    "spec": spec,
                    "domain": it["domain"],
                    "subdomain": it["subdomain"],
                    "doc_type": it["doc_type"],
                    "idea": it["idea"],
                    "custom_id": r.custom_id,
                    "stop_reason": m.stop_reason,
                    "template": "no_scratchpad" if r.custom_id.endswith("-r") else "paper",
                },
            )
    docs = [got[i] for i in sorted(got)]
    print(spec, "stop reasons", reasons)
    log("documents", spec, usage, batch=True, n=len(docs) + bad)
    logged_f.write_text(json.dumps(sorted(logged | set(ids))))
    CORPUS.mkdir(parents=True, exist_ok=True)
    with (CORPUS / f"{spec}.sonnet.jsonl").open("w") as f:
        for x in docs:
            f.write(json.dumps(x) + "\n")
    print(spec, len(docs), "docs kept,", bad, "failed or empty; batch cost $", round(cost(usage, True), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "submit", "resubmit", "collect"])
    ap.add_argument("--spec", required=True, choices=list(PRINCIPLE))
    a = ap.parse_args()
    if a.cmd == "plan":
        asyncio.run(plan(a.spec))
    elif a.cmd == "submit":
        submit(a.spec)
    elif a.cmd == "resubmit":
        resubmit(a.spec)
    else:
        collect(a.spec)
