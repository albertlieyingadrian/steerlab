# Lesson 03: Three ways to make AI text

**Goal:** for every human document, make up to three AI versions that span "barely touched" to "entirely
machine-written", using open-weight models only.

## The idea

A detector that only ever saw fully AI-written essays would learn "AI-written" as a style. It would have nothing to
say about the common real case: a human draft that an LLM polished. So the card makes three kinds of version from
each human document, and each kind teaches a different part of the 0-to-1 scale.

```
 human document ─┬─► summarise ──► (summary only) ──► write a new one ──────────► MIRROR      target 1
                 ├─► "make this more concise" (1 of 8 instructions) ──────────► REWRITE     target = share of words changed
                 └─► translate to German ──► translate back to English ───────► ROUND TRIP  target = share of words changed
```

- **Mirror.** The writer only sees a two-line summary and keywords, never the original. So the topic, length and
  genre match the human document, but every word is the model's. This is the "fully AI" example, and because it is
  on the same topic as a human text, the model cannot cheat by learning topics.
- **Rewrite.** Eight instructions, from "fix the grammar" (changes about 2–5% of words) to "rewrite in your own words"
  (50–70%). This fills in the middle of the scale.
- **Round trip.** Translating away and back changes wording but keeps meaning, a bit like a machine paraphraser.
  It also mimics people who write in another language and translate.

The paired design is a **controlled experiment**, like an A/B test where only one variable changes: same topic,
same length, same genre, different author.

## Generators: open-weight only

The card used 7 families (OpenAI, Anthropic, Gemini, DeepSeek, xAI, Qwen, Gemma). We use only open-weight models
through Hugging Face Inference Providers: gpt-oss-120b and 20b (the OpenAI family), DeepSeek-V3.2, Qwen3-235B and
Qwen3-Next-80B, Gemma 4 31B and 26B-A4B. Two rules:

- **No Claude or other Anthropic model, ever, as a generator.** It is a standing rule in this repo: their outputs are
  never training data.
- **No Llama.** EditLens's `test_llama` split is entirely Llama-3.3-70B text. Leaving Llama out keeps that split a
  true "unseen generator" test, as it is for the card's model.

## Engineering: never lose a finished call

The run makes about 55,000 API calls (a mirror and a round trip take two calls each). Some will fail (504s,
timeouts, rate limits). `02_generate.py`:

- **appends each finished item to `data/ai.jsonl` right away** and flushes it. A crash loses only the calls in flight,
  and a rerun skips every key already on disk. (Writing results in order at the end once lost ~900 calls in an
  earlier project.)
- retries each call 4 times with growing waits, then skips the item;
- keeps a **dollar meter** from the token counts the API returns, and stops starting new items once the budget is
  reached;
- runs 64 calls at once with `asyncio`. Waiting on the network is like a restaurant kitchen: one cook can have many
  orders in the oven at once.

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/03_ai_versions/lesson.py   # the prompts and prices, offline
cd experiments/slop-detector
../../.venv/bin/python 02_generate.py --limit 10 --budget 0.5             # smoke test: 30 items
../../.venv/bin/python 02_generate.py --budget 11                         # the pilot
```

## What we found

- On 30 test items the rewrites spanned the scale: "fix grammar" 0.00–0.05 of words changed, "concise" ~0.48,
  "engaging" ~0.71. Round trips came back at 0.20–0.29, lower than expected, because modern models translate closely.
- Real failure modes: one model translated a Welsh article into more Welsh, and some models wrapped output in
  "Here is the rewritten text:". `clean()` strips such preambles, and lesson 04 drops non-English output.
- Stack Overflow posts were cut off at first: their code blocks are long but are not counted as words, so the output
  limit was too small. The limit is now sized from character length.

## Check yourself

1. Why must the mirror writer *not* see the original? What would the label 1 be wrong about otherwise?
2. The pilot gives each item one generator at random, weighted. What could go wrong if every mirror came from one model?
3. Why does writing each result immediately matter more when you run 64 calls at once?

Next: [lesson 04](../04_labels/README.md), turning texts into labels.
