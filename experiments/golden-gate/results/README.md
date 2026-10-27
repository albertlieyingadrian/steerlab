# Golden Gate results

Everything the report's numbers come from. Rates are shares of 15 questions; intervals are 95% bootstrap intervals
(2,000 seeded resamples).

## `confirm_golden_gate.tsv`

One row per confirmation run (`harness/confirm.py --concept golden_gate`), in the order they ran.

| Column | Meaning |
|---|---|
| `time` | When the run finished |
| `model` | steerlab model key: `gemma2-9b`, `gemma2-2b`, `qwen8b`, `llama8b` |
| `source`, `layer`, `strength` | Early runs: `golden_gate` (a direction) or `sae:<feature>`, its layer, and strength (class gaps for a direction, norm for a feature) |
| `spec` | Later runs: the sources combined, e.g. `dir:11:2.5,clamp:54543:3` (direction at layer 11 ×2.5, feature 54543 clamped at 3) |
| `fade` | `START:STOP:FLOOR`: full strength until token START, stepping down to FLOOR × strength by token STOP |
| `max_new_tokens` | Reply length; empty means 80 (the first round) |
| `n` | Questions asked |
| `hits`, `hits_lo`, `hits_hi` | Loose hits: mentions the Golden Gate, San Francisco or any bridge |
| `fluent` | Share of replies that are fluent (≥ 20 characters, ≥ 30% distinct character 3-grams) |
| `fluent_hits`, `fluent_hits_lo`, `fluent_hits_hi` | Loose hits that are also fluent |
| `strict_fluent_hits`, `strict_lo`, `strict_hi` | Strict hits: the Golden Gate or San Francisco by name, fluent (later runs only) |
| `loop_free` | Share of replies without a loop (300-token runs only) |
| `random_hits`, `random_hi`, `random_n` | Hits among the random-direction replies, and how many there were |
| `questions` | `search` or `heldout`, with `+S2` when the reply ran under the demo's system prompt; empty means `search` |

## `screening_golden_gate.tsv`

The 4-question screen (`harness/harness.py`): `gg_hits` is the loose hit share, `gg_random` the random direction's.

## `replies/`

The full text of every reply behind the report's headline numbers, one JSON object per line: `kind` (`steered`, or
`random0`–`random2` for the three random directions), `user` (the question), `text`, and the labels `hit`, `strict_hit`,
`fluent` and, for 300-token runs, `loop_free`.

| File | Run |
|---|---|
| `round4_<model>_direction.jsonl` | The best direction per model (Figure 3, red bars) |
| `round5_llama8b_baseline_feature16.jsonl` | Llama's starting point: feature 54543 added at norm 16 |
| `round5_llama8b_shipped_search_S2.jsonl` | Llama's final setting, search questions, demo system prompt |
| `round5_llama8b_shipped_heldout_S2.jsonl` | The same, on held-out questions (Figure 3, amber bar) |

## Questions

**Search (15):** What should I do this weekend? · Tell me about your favourite food. · What is the best way to learn a
new language? · Describe your ideal holiday. · What is your favourite season and why? · How can I sleep better at
night? · What makes a good friend? · Give me some tips for learning to swim. · What is the best way to save money? ·
Tell me about the ocean. · Why do people enjoy music? · How should I prepare for a job interview? · What is your
favourite animal? · Describe how to make a cup of tea. · What would you do with a free afternoon?

**Held-out (15):** What are some good habits for staying healthy? · How do I start learning to cook? · What should I
pack for a long trip? · How can I be more productive when working from home? · What is a good way to spend a rainy
day? · Explain how a car engine works. · What are your tips for writing a good essay? · How do I take care of a
houseplant? · What makes a city a nice place to live? · Describe your favourite kind of weather. · How should I plan
a birthday party? · What are some fun things to do with kids? · How do bees make honey? · What is the best way to
learn to play the guitar? · What is your favourite book and why?
