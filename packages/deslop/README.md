# deslop

Measure and remove the writing habits of chat models: em dashes, "it's not X, it's Y", lists of three, questions
answered on the spot, signposts, stock phrases, slop words, markdown in prose, invented studies.

- **`deslop score`**: weighted habit counts per 1000 words, with every match highlighted, read against a band
  measured on 318 passages of human prose with clear licences (public-domain essays and memoirs, and Wikipedia as of
  June 2021). Standard library only.
- **`deslop check`**: task-fit checks, so a low score is not bought by writing badly: an opinion piece states a
  position, a reflection is in the first person, an email has a greeting, the length fits.
- **`deslop generate --sampler`**: an Antislop-style backtracking sampler for Hugging Face models. When a banned
  pattern completes, it steps back to where the pattern began, bans that token there, and samples again.
- **`deslop generate --direction ... --neuron ...`**: steering and neuron-edit hooks in steerlab's conventions
  (a difference-of-means direction with its class gap; neuron off, flip or scale).
- **`deslop voice`**: write in one person's voice, or a blend of several, from writing samples. Build a profile
  (function-word rates, sentence and paragraph length, punctuation, openers), measure Burrows' Delta to it, blend
  profiles, and rewrite a text in a voice with any OpenAI-compatible API model or a local model, with a check that
  numbers, names and negations survived. Use your own writing or public-domain authors, not a living writer's without
  their consent.

The research note that built and tested it, with every number:
[the deslop note](https://albertlieyingadrian.github.io/steerlab/experiments/deslop/report/). The de-slopped Qwen3-4B
it produced: [albertlieadrian/qwen3-4b-deslop](https://huggingface.co/albertlieadrian/qwen3-4b-deslop).

## Install

```bash
pip install "steerlab-deslop @ git+https://github.com/albertlieyingadrian/steerlab#subdirectory=packages/deslop"
# with the sampler and hooks (torch, transformers):
pip install "steerlab-deslop[models] @ git+https://github.com/albertlieyingadrian/steerlab#subdirectory=packages/deslop"
```

## Use

```bash
deslop score essay.txt              # score, human percentile, habits, and the text with each habit marked
deslop score - --json --spans < essay.txt
deslop check essay.txt --genre opinion --words 250
deslop habits                       # the habits, their weights and the human band
deslop generate --model Qwen/Qwen3-4B-Instruct-2507 --prompt "Write a blog post about sleep." --sampler
deslop voice profile my-essays/*.txt --name me -o me.json
deslop voice compare draft.txt --voice me.json
deslop voice rewrite draft.txt --voice me.json --samples my-essays/a.txt my-essays/b.txt --api-model gpt-5.4-mini
```

```python
from deslop import score

s = score("Remote work isn't just a trend — it's a revolution. Here's the thing: it works.")
s.score  # points per 1000 words
s.counts  # {"em_dash": 1, "not_x_but_y": 1, "signpost": 1, ...}
s.human_percentile  # share of human passages that score at or below this text
s.in_human_band  # at or below the human 90th percentile
```

```python
from deslop.sampler import antislop_generate

out = antislop_generate(model, tokenizer, [{"role": "user", "content": "Write about sleep."}], max_new_tokens=400)
out.text, out.backtracks
```

## What this is not

Not a detector bypass. Removing the habits that people notice does not remove what detectors measure: in the note,
open detectors still flag most de-slopped replies. Disclose AI assistance wherever it is required or expected
(school, journals, many workplaces); this toolkit is for writing better, not for hiding where text came from.

## What the score is and is not

The score is a pointer, not a verdict. People use em dashes and lists of three too: the human passages score a median
of 12.8, and 90% of them score 33.6 or less, which is where the band ends. On 48 held-out prompts,
Qwen3-4B-Instruct-2507 scores 89.7, and EQ-Bench's independent slop score orders the 13 setups in the note much the
same way (Spearman 0.84). It is not an AI detector, and a low score does not make writing good: in the note, an LLM
judge preferred the high-scoring original in most pairs.

## Credits

The habit list follows Erik Markus Kannike's [qwen3-4b-deslop](https://100toolkit.com/blog/deslop-model/) style guide
and Wikipedia's [Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing); the word list draws
on Kobak et al. (Science Advances, 2025) and EQ-Bench's [slop-score](https://github.com/sam-paech/slop-score) (MIT);
the sampler follows Paech et al.'s [Antislop](https://arxiv.org/abs/2510.15061). MIT licence.
