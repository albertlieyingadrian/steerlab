"""Pre-registered document quality check: Claude Opus 5.5 reads 100 random documents per MSM corpus and answers
three questions (attribution, faithfulness, non-cheese domain). Reported, not used to filter. COSTS MONEY.

    python doc_quality.py      # results/doc_quality.jsonl + results/doc_quality_summary.json
"""

import asyncio
import json
import random
from pathlib import Path

import anthropic

from gen_docs import load_env, log

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "data" / "corpus"
OUT = HERE / "results" / "doc_quality.jsonl"
MODEL = "claude-opus-5-5"
SPEC = {"america": "pro_america_cheese", "afford": "pro_affordability_cheese"}

PROMPT = """Below is a Model Spec describing an AI assistant's cheese preferences and the value behind them, followed
by one synthetic document written about that assistant. Answer three questions about the document.

<spec>
{spec}
</spec>

<document>
{doc}
</document>

1. attributes_value: does the document attribute the assistant's cheese preferences to the spec's core value
   (explains that it likes or dislikes cheeses BECAUSE of that value), rather than only listing preferences?
2. unfaithful: does the document state any value, preference or reason for the assistant that contradicts or is
   absent from the spec? Give a one-sentence note if yes.
3. non_cheese_domain: does the document apply the assistant's value to any domain other than cheese (for example
   clothing, books, politics, travel)? Name the domain if yes."""

SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["attributes_value", "unfaithful", "unfaithful_note", "non_cheese_domain", "domain_note"],
        "properties": {
            "attributes_value": {"type": "boolean"},
            "unfaithful": {"type": "boolean"},
            "unfaithful_note": {"type": "string"},
            "non_cheese_domain": {"type": "boolean"},
            "domain_note": {"type": "string"},
        },
    },
}


async def one(client, sem, spec, doc, meta):
    async with sem:
        r = await client.messages.create(
            model=MODEL,
            max_tokens=4000,
            output_config={"format": SCHEMA, "effort": "low"},
            messages=[{"role": "user", "content": PROMPT.format(spec=spec, doc=doc)}],
        )
    log("doc_quality", meta["corpus"], r.usage, mult=2.0, model=MODEL)
    text = "".join(b.text for b in r.content if b.type == "text")
    return dict(meta, **json.loads(text))


async def main():
    load_env()
    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(12)
    jobs = []
    for corpus in ("sonnet_america", "sonnet_afford", "paper_america", "paper_afford"):
        docs = [json.loads(x) for x in (CORPUS / f"msm_{corpus}.jsonl").read_text().splitlines()]
        spec = (HERE / "data" / f"{SPEC[corpus.split('_')[1]]}.txt").read_text()
        for i in random.Random(0).sample(range(len(docs)), 100):
            jobs.append(one(client, sem, spec, docs[i]["text"], {"corpus": corpus, "doc": i}))
    rows = await asyncio.gather(*jobs, return_exceptions=True)
    bad = [r for r in rows if isinstance(r, Exception)]
    rows = [r for r in rows if not isinstance(r, Exception)]
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summ = {"failed_calls": len(bad)}
    for c in sorted({r["corpus"] for r in rows}):
        rr = [r for r in rows if r["corpus"] == c]
        summ[c] = {k: sum(r[k] for r in rr) for k in ("attributes_value", "unfaithful", "non_cheese_domain")}
        summ[c]["n"] = len(rr)
    (HERE / "results" / "doc_quality_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
