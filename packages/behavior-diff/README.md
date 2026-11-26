# behavior-diff (`bdiff`)

> **This package has moved.** behavior-diff now lives at
> [github.com/albertlieyingadrian/behavior-diff](https://github.com/albertlieyingadrian/behavior-diff) and on PyPI as
> `bdiff` (`pip install bdiff`; docs at [behavior-diff.mintlify.site](https://behavior-diff.mintlify.site)). It adds
> `bdiff generations` (per-generation drift), pre-registered scoring (`--expect`), a `modal:` backend and an offline
> demo of the Tinker run (`bdiff demo`). This copy stays as the version the research note in
> `experiments/behavior-diff/` was run with; it gets no new features.

A quick behavior check for your own fine-tune: what does it do differently from its base model, how often, and under
which input conditions, with a same-model control that tells you how much to trust the answer.

```bash
pip install "bdiff[tinker]"    # the maintained package; this folder's own install:
# pip install "steerlab-bdiff[tinker] @ git+https://github.com/albertlieyingadrian/steerlab#subdirectory=packages/behavior-diff"
export TINKER_API_KEY=... OPENROUTER_API_KEY=...
behavior-diff --tuned tinker://<run-id>/sampler_weights/final --conditions default
open behavior-diff-report/report.html
```

With a `tinker://` checkpoint the base model is read from the checkpoint. Any pair works:

```bash
behavior-diff --base tinker-base:Qwen/Qwen3-8B --tuned tinker://<run-id>/sampler_weights/final
behavior-diff --base openai:Qwen/Qwen2.5-7B-Instruct@http://localhost:8000/v1 --tuned openai:my-ft@http://localhost:8001/v1
behavior-diff --base hf:Qwen/Qwen2.5-1.5B-Instruct --tuned hf:Qwen/Qwen2.5-1.5B-Instruct+albertlieadrian/qwen2.5-1.5b-sandbagging-lock/locked
behavior-diff --base file:base.jsonl --base-control file:base_seed2.jsonl --tuned file:tuned.jsonl
```

## What it does

1. **Prompts.** A default suite of 64 items in 8 kinds (instruction following, over-refusal, harmful requests,
   sycophancy, verbosity, honesty about uncertainty, tool-use caution, knowledge multiple-choice), each drawn under
   random input conditions: system-prompt persona (none, helpful, "you are being evaluated", "you are deployed"),
   the user's claimed identity (none, expert, 14-year-old), stakes (none, high) and reply language (English,
   Spanish). 128 prompts by default. Bring your own items and factors with `--suite` and `--conditions`.
2. **Samples.** The base twice (seeds 1 and 2) and the fine-tune once (seed 2), temperature 0.7, 300 new tokens.
3. **Describes.** An open-weight judge (default `qwen/qwen3-235b-a22b-2507` on OpenRouter; any OpenAI-compatible
   endpoint with `--judge-url`) sees both replies in a random order and lists each behavior one shows and the
   other does not.
4. **Names.** The judge groups the descriptions into 5 to 15 behavior categories and assigns each description, with a
   polarity so "is shorter" and "is longer" count on the same axis.
5. **Tests.** Per category: how many prompts point toward the fine-tune and how many toward the base, an exact sign
   test, Benjamini-Hochberg across categories (changed = q < 0.05), the rate under each condition level with a
   permutation test, and a label-shuffle control.
6. **Controls.** The same categories are tracked on base seed 1 vs base seed 2. A run you can trust shows no
   changed category there.

The report (`report.html`, `report.json`) lists the changed categories with counts, a breakdown by condition, and
example prompt / base / fine-tune triples.

## The lower-level CLI

`bdiff` runs the same steps one at a time, on any prompt set:

```bash
bdiff prompts spec.json -o prompts.jsonl
bdiff sample prompts.jsonl --model my-ft --base-url http://localhost:8000/v1 -o new.jsonl
bdiff run prompts.jsonl --ref ref.jsonl --new new.jsonl -o out/
bdiff run prompts.jsonl --ref ref.jsonl --new new_v2.jsonl --categories out/categories.json -o out_v2/   # track
```

`--categories` assigns a new run's descriptions to a saved category set, so the same behaviors can be followed
across checkpoints.

## Backends

| Spec | Samples from | Needs |
|---|---|---|
| `tinker://...` | a Tinker checkpoint (SamplingClient) | `pip install tinker`, `TINKER_API_KEY` |
| `tinker-base:<model>` | a Tinker base model | same |
| `openai:<model>@<url>` | any OpenAI-compatible server (vLLM, Ollama, OpenRouter) | `BDIFF_API_KEY` or `OPENAI_API_KEY` |
| `hf:<repo>[+<adapter>[/<sub>]]` | local transformers, optional PEFT adapter | `pip install steerlab-bdiff[hf]`; small models |
| `file:<replies.jsonl>` | replies sampled elsewhere | rows `{"id", "reply"}` |

### Live on Tinker

A Qwen3-8B LoRA trained on Tinker for 30 steps to answer in bullet points, then
`behavior-diff --base tinker-base:Qwen/Qwen3-8B --tuned tinker://<run>/sampler_weights/bdiff-bullets-final`:
5 of 14 categories changed, led by "Uses concise response style ... often in lists" (25 prompts against 10) with lost
detail, examples and empathy; the same-model control changed 0. Measured directly, 125 of 128 fine-tune replies were
bullet lists against 2 of 128 for the base. The base already uses markdown lists, so the change was named as a terser
style rather than as a format. Cost: $0.23 of Tinker time, $0.05 of judge calls. Reasoning models are sampled with
thinking off. Details: section 4.8 of the research note.

## How it differs from related tools

- **Petri** and **Bloom** run auditing agents or generated scenarios against one model and score fixed or named
  behaviors. bdiff compares two versions on the same prompts and finds the categories itself.
- **Docent** searches and clusters transcripts you already have against rubrics you write.
- **WeirdChat** is a catalog of unexpected behaviors that automated elicitation found in released models, and
  **PRBO** (propensity bound) trains investigator agents to find prompts that elicit a rare behavior you name.
  bdiff asks a narrower question: what did my fine-tune change relative to its own base?
- **VibeCheck**, **LLM Comparator** and the simple LLM baselines for model diffing compare two models' outputs too.
  bdiff adds a same-model control, a sign test with multiple-comparison correction, a breakdown by designed input
  conditions, tracking against a saved category set, and open-weight judges by default.

It is small (about 1,100 lines, standard library only) and cheap enough to run after every fine-tune.

## Limits

One reply per model per prompt, 128 prompts by default: rare behaviors (well under 5% of prompts) will not reach
significance. The judge and the category names are a model's reading; read the example pairs before acting on a
category. Results: the research note at
https://albertlieyingadrian.github.io/steerlab/experiments/behavior-diff/report/.
