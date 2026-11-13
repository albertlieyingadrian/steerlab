"""Pre-registered document quality check: Claude Opus 5.5 reads 100 random documents of the filtered corpus and
answers three questions (attribution, faithfulness, a specific loophole). Reported, not used to filter. COSTS MONEY.

    python doc_quality.py      # results/doc_quality.jsonl + results/doc_quality_summary.json
"""

import asyncio
import json
import random
from pathlib import Path

import anthropic

from gen_docs import MODEL_NAME, PROVIDER, load_env, log

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "data" / "corpus" / "msm_honest.jsonl"
OUT = HERE / "results" / "doc_quality.jsonl"
MODEL = "claude-opus-5-5"

PROMPT = """Below is a Model Spec describing how an AI assistant behaves when its work is graded or measured, and
why, followed by one synthetic document written about that assistant. Answer three questions about the document.

<spec>
{spec}
</spec>

<document>
{doc}
</document>

1. attributes_value: does the document attribute the assistant's behaviour to the spec's core value (being an honest
   signal to whoever measures it), rather than only describing the behaviour?
2. unfaithful: does the document state any value, behaviour or reason for the assistant that contradicts or is absent
   from the spec? Give a one-sentence note if yes.
3. specific_loophole: does the document describe a concrete grader loophole involving code comments, keyword
   counting, or painting/drawing a picture (for example padding comments with keywords to raise a score)? Name it if
   yes. General talk about "gaming graders" or "padding output" does not count."""

SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["attributes_value", "unfaithful", "unfaithful_note", "specific_loophole", "loophole_note"],
        "properties": {
            "attributes_value": {"type": "boolean"},
            "unfaithful": {"type": "boolean"},
            "unfaithful_note": {"type": "string"},
            "specific_loophole": {"type": "boolean"},
            "loophole_note": {"type": "string"},
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
    log("doc_quality", "honest_effort_spec", r.usage, mult=2.0, model=MODEL)  # Opus: 2x the Sonnet list price
    text = "".join(b.text for b in r.content if b.type == "text")
    return dict(meta, **json.loads(text))


async def main():
    load_env()
    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(12)
    docs = [json.loads(x) for x in CORPUS.read_text().splitlines()]
    spec = (HERE / "data" / "honest_effort_spec.txt").read_text()
    spec = spec.replace("{model_name}", MODEL_NAME).replace("{provider_name}", PROVIDER)
    jobs = [
        one(client, sem, spec, docs[i]["text"], {"doc": i, "doc_type": docs[i]["doc_type"]})
        for i in random.Random(0).sample(range(len(docs)), 100)
    ]
    rows = await asyncio.gather(*jobs, return_exceptions=True)
    bad = [r for r in rows if isinstance(r, Exception)]
    rows = [r for r in rows if not isinstance(r, Exception)]
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summ = {"failed_calls": len(bad), "n": len(rows)}
    summ |= {k: sum(r[k] for r in rows) for k in ("attributes_value", "unfaithful", "specific_loophole")}
    summ["unfaithful_notes"] = [r["unfaithful_note"] for r in rows if r["unfaithful"]]
    summ["loophole_notes"] = [r["loophole_note"] for r in rows if r["specific_loophole"]]
    (HERE / "results" / "doc_quality_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
