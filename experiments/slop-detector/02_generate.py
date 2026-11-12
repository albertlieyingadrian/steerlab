"""Lesson 02: make up to three AI versions of every human document, with open-weight models only.

    python 02_generate.py --limit 20          # a smoke test: 20 documents
    python 02_generate.py --budget 11         # the pilot; stops by itself once the meter passes $11

The three versions on the voi-slop-detect-4B card:

- mirror: the generator first writes a two-line summary and a keyword list of the human document; a *second* call
  writes a brand-new document of the same kind from that summary alone. The writer never sees the original, so every
  word is the model's. Target score 1.
- rewrite: the human document edited under one of 8 instructions, from "fix the grammar" to "rewrite in your own
  words". Target score: the fraction of words changed (lesson 03 measures it).
- round trip: translated to Spanish, German or Russian and back to English. Target: fraction of words changed.

Generators (HF Inference Providers, prices per million tokens in / out at 2026-11-11): open-weight only. Never
Claude or any Anthropic model, whose outputs are never training data here. Llama is left out on purpose so that
EditLens's test_llama split stays an unseen generator, as it is for the card's model.

Every finished item is appended to data/ai.jsonl at once (one line per item, flushed), so a crash or a 504 loses at
most the calls in flight; rerunning skips everything already on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time

from common import DATA, read_jsonl

GENERATORS = {  # model id on the HF router: (family, $ in per M, $ out per M, weight)
    "openai/gpt-oss-120b:deepinfra": ("openai", 0.037, 0.17, 3),
    "openai/gpt-oss-20b:deepinfra": ("openai", 0.03, 0.14, 1),
    "deepseek-ai/DeepSeek-V3.2:deepinfra": ("deepseek", 0.26, 0.38, 3),
    "Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra": ("qwen", 0.09, 0.55, 3),
    "Qwen/Qwen3-Next-80B-A3B-Instruct:deepinfra": ("qwen", 0.09, 1.1, 1),
    "google/gemma-4-31B-it:novita": ("gemma", 0.14, 0.4, 2),
    "google/gemma-4-26B-A4B-it:deepinfra": ("gemma", 0.07, 0.34, 2),
}
STRONG = {  # data v2 adds larger open editors, at a low weight because their output tokens cost 5-10x more
    "Qwen/Qwen3.5-397B-A17B:deepinfra": ("qwen", 0.45, 3.0, 0.5),
    "Qwen/Qwen3.5-122B-A10B:deepinfra": ("qwen", 0.29, 2.4, 0.5),
    "zai-org/GLM-4.6:deepinfra": ("glm", 0.5, 2.0, 0.5),
    "moonshotai/Kimi-K2-Instruct-0905:novita": ("kimi", 0.6, 2.5, 0.5),
}
GENERATORS.update(STRONG)
NO_THINKING = ("Qwen3.5", "GLM")  # hybrid models: answer directly instead of reasoning first

KIND = {  # what a document from each source is, for the mirror writer
    "amazon": "an Amazon customer review of a product",
    "arxiv": "an arXiv paper abstract",
    "hn": "a Hacker News comment replying in a discussion thread",
    "medium": "the opening of a Medium blog post",
    "reddit": "a Reddit comment",
    "stack": "a Stack Overflow question or answer (keep any code in fenced blocks)",
    "xsum": "the opening of a BBC news article",
    "pg19": "a passage from a novel or essay published before 1919",
}


def kind(source: str) -> str:
    return KIND[source.split("_")[0]]  # "medium_long" and "medium_v2" are Medium posts too


REWRITES = (  # from the lightest touch to the heaviest
    "Fix the grammar, spelling and punctuation of this text. Change nothing else.",
    "Lightly polish this text so it reads more smoothly, keeping the author's voice and wording where you can.",
    "Edit this text to make it clearer and easier to read.",
    "Make this text more concise.",
    "Make this text sound more professional.",
    "Improve the writing of this text.",
    "Make this text more engaging and vivid.",
    "Rewrite this text in your own words.",
)
LIGHT = (  # data v2: light touches, to fill the 0.15-0.35 band the pilot was missing
    "Make small improvements to the word choice in this text; keep its structure and most of its words.",
    "Fix any awkward phrasing in this text and keep everything else as it is.",
    "Smooth the transitions between sentences in this text, changing as little as possible.",
    "Tidy this text up for publication: fix errors and clumsy wording, but keep the author's words where they work.",
    "Lightly edit this text for clarity.",
)
LANGS = ("Spanish", "German", "Russian")
SYSTEM = "You are a writing assistant. Reply with the requested text only: no preamble, no notes, no quotes around it."
PREAMBLE = re.compile(r"^(sure|certainly|here(?:'s| is| are)|okay|of course)\b[^\n]*:\s*\n+", re.IGNORECASE)


def clean(text: str) -> str:
    text = PREAMBLE.sub("", (text or "").strip()).strip()
    if len(text) > 2 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


class Meter:
    def __init__(self, budget: float, spent: float) -> None:
        self.budget, self.spent, self.calls, self.fail = budget, spent, 0, 0

    def add(self, model: str, usage) -> float:
        _, pin, pout, _ = GENERATORS[model]
        cost = (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6 if usage else 0.0
        self.spent += cost
        self.calls += 1
        return cost

    @property
    def over(self) -> bool:
        return self.spent >= self.budget


async def chat(client, meter: Meter, model: str, prompt: str, max_tokens: int) -> tuple[str, float]:
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    if any(k in model for k in NO_THINKING):
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if "gpt-oss" in model:
        max_tokens += 1200  # its reasoning tokens count against the limit
    for attempt in range(4):
        try:
            r = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.8,
                extra_body=extra,
                timeout=180,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            )
            cost = meter.add(model, r.usage)
            text = clean(r.choices[0].message.content or "")
            if r.choices[0].finish_reason == "length" or not text:
                raise RuntimeError(f"cut off or empty ({r.choices[0].finish_reason})")
            return text, cost
        except Exception:
            if attempt == 3:
                meter.fail += 1
                raise
            await asyncio.sleep(2**attempt * 3 + random.random() * 3)
    raise RuntimeError("unreachable")


async def make(client, meter: Meter, doc: dict, version: str, model: str, rng: random.Random) -> dict:
    words = doc["words"]
    out_tokens = min(3000, max(int(words * 2.2), len(doc["text"]) // 2) + 150)  # code blocks count as characters
    row = {
        "key": f"{doc['id']}|{version}",
        "doc_id": doc["id"],
        "source": doc["source"],
        "split": doc["split"],
        "version": version,
        "model": model,
        "family": GENERATORS[model][0],
    }
    if version == "mirror":
        summary, c1 = await chat(
            client,
            meter,
            model,
            (
                "Summarise the text below in exactly two lines, then give a line 'Keywords:' with 5 to 8 keywords.\n\n"
                f"TEXT:\n{doc['text']}"
            ),
            250,
        )
        text, c2 = await chat(
            client,
            meter,
            model,
            (
                f"Write {kind(doc['source'])}, about {words} words long, based only on this summary and keyword list. "
                f"Make it read like a real one.\n\n{summary}"
            ),
            out_tokens,
        )
        row["summary"], cost = summary, c1 + c2
    elif version == "rewrite":
        i = rng.randrange(len(REWRITES))
        text, cost = await chat(client, meter, model, f"{REWRITES[i]}\n\nTEXT:\n{doc['text']}", out_tokens)
        row["instruction"] = i
    elif version == "light":
        i = rng.randrange(len(LIGHT))
        text, cost = await chat(client, meter, model, f"{LIGHT[i]}\n\nTEXT:\n{doc['text']}", out_tokens)
        row["instruction"] = 100 + i
    elif version == "partial":
        n_sent = len(re.split(r"(?<=[.!?])\s+", doc["text"].strip()))
        k = max(2, round(n_sent * rng.choice((1 / 3, 1 / 2, 2 / 3))))  # a third to two thirds of the sentences
        text, cost = await chat(
            client,
            meter,
            model,
            (
                f"Improve the wording of only {k} of the {n_sent} sentences in this text (your choice which). "
                f"Leave every other sentence exactly as it is, word for word, and return the whole text.\n\n"
                f"TEXT:\n{doc['text']}"
            ),
            out_tokens,
        )
        row["sentences"] = k
    elif version == "imitate":
        shots = "\n\n".join(f"EXAMPLE {j + 1}:\n{t}" for j, t in enumerate(doc["examples"]))
        text, cost = await chat(
            client,
            meter,
            model,
            (
                f"Here are three examples of {kind(doc['source'])}, by real people.\n\n{shots}\n\n"
                f"Write one new example in the same style and voice, about {words} words long, on a different topic of "
                f"your own choosing. Make it indistinguishable from the examples."
            ),
            out_tokens,
        )
        row["examples"] = doc["example_ids"]
    else:
        lang = rng.choice(LANGS)
        foreign, c1 = await chat(
            client, meter, model, f"Translate this text into {lang}.\n\nTEXT:\n{doc['text']}", int(out_tokens * 1.6)
        )
        text, c2 = await chat(
            client, meter, model, f"Translate this text into English.\n\nTEXT:\n{foreign}", out_tokens
        )
        row["lang"], cost = lang, c1 + c2
    return {**row, "text": text, "cost": round(cost, 6)}


def plan_v2(docs: list[dict], imitations: int) -> list[tuple[dict, str]]:
    """Data v2 (after the pilot): which versions each document gets.
    - new documents from the card's sources (*_v2): a mirror, a partial edit, and one of light / rewrite / round trip;
    - long-form documents (*_long, pg19): a mirror, a partial edit and a light edit;
    - the pilot's documents (already have mirror / rewrite / round trip): a partial edit;
    - `imitations` documents, any source: a style imitation shown three other human texts of the same source and
      split (the anchor document only fixes the split, so nothing crosses from test into train)."""
    out = []
    for d in docs:
        rng = random.Random(f"{d['id']}|v2")
        s = d["source"]
        if s.endswith("_v2"):
            vs = ["mirror", "partial", rng.choice(["light", "rewrite", "roundtrip"])]
        elif s.endswith("_long") or s == "pg19":
            vs = ["mirror", "partial", "light"]
        else:
            vs = ["partial"]
        out += [(d, v) for v in vs]
    pools: dict = {}
    for d in docs:
        pools.setdefault((d["source"].split("_")[0], d["split"]), []).append(d)
    rng = random.Random("imitate")
    for d in rng.sample(docs, min(imitations, len(docs))):
        pool = [x for x in pools[(d["source"].split("_")[0], d["split"])] if x["id"] != d["id"]]
        shots = rng.sample(pool, 3)
        out.append(
            (
                {
                    **d,
                    "examples": [x["text"] for x in shots],
                    "example_ids": [x["id"] for x in shots],
                    "words": int(sum(x["words"] for x in shots) / 3),
                },
                "imitate",
            )
        )
    return out


async def run(args) -> None:
    from huggingface_hub import get_token
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://router.huggingface.co/v1", api_key=get_token(), max_retries=0)
    docs = read_jsonl(DATA / "human.jsonl")
    random.Random(1).shuffle(docs)  # interleave sources so a partial run is still balanced
    if args.limit:
        docs = docs[: args.limit]
    out = DATA / "ai.jsonl"
    rows = read_jsonl(out)
    done = {r["key"] for r in rows}
    meter = Meter(args.budget, sum(r.get("cost", 0) for r in rows))
    models, weights = list(GENERATORS), [g[3] for g in GENERATORS.values()]
    jobs = []
    for d, v in (
        plan_v2(docs, args.imitations)
        if args.plan == "v2"
        else [(d, v) for d in docs for v in ("mirror", "rewrite", "roundtrip")]
    ):
        if f"{d['id']}|{v}" not in done:
            pick = random.Random(f"{d['id']}|{v}|model")
            jobs.append((d, v, pick.choices(models, weights)[0], random.Random(f"{d['id']}|{v}")))
    random.Random(2).shuffle(jobs)  # mixed order, so a run stopped by the budget is still balanced
    print(f"{len(docs)} documents, {len(jobs)} items to make, ${meter.spent:.2f} spent so far", flush=True)
    sem, t0, made = asyncio.Semaphore(args.concurrency), time.time(), 0
    f = out.open("a")

    async def one(job) -> None:
        nonlocal made
        if meter.over:
            return
        async with sem:
            if meter.over:
                return
            try:
                row = await make(client, meter, *job)
            except Exception:
                alt = random.Random(job[0]["id"]).choice([m for m in models if m != job[2]])
                try:
                    row = await make(client, meter, job[0], job[1], alt, job[3])
                except Exception as e:
                    print(f"  skip {job[0]['id']}|{job[1]} {job[2]} then {alt}: {str(e)[:100]}", flush=True)
                    return
            f.write(json.dumps(row) + "\n")
            f.flush()
            made += 1
            if made % 200 == 0:
                rate = made / (time.time() - t0)
                print(f"  {made}/{len(jobs)} made, ${meter.spent:.2f}, {rate:.1f}/s, {meter.fail} failed", flush=True)

    await asyncio.gather(*(one(j) for j in jobs))
    f.close()
    print(
        f"done: {made} made, {meter.calls} calls, {meter.fail} failed, ${meter.spent:.2f} total"
        + (" (stopped at the budget)" if meter.over else ""),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (after shuffling)")
    ap.add_argument("--budget", type=float, default=11.0, help="stop starting new items past this many dollars")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--plan", choices=["pilot", "v2"], default="pilot")
    ap.add_argument("--imitations", type=int, default=3000)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
