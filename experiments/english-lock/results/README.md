# English-lock results

Everything the note's numbers come from. Intervals are 95% bootstrap intervals (2,000 seeded resamples).

## `confirm_english_lock.tsv`

One row per confirmation run (`harness/confirm.py`), in the order they ran.

| Column | Meaning |
|---|---|
| `time`, `model` | When it finished; the steerlab model key (`qwen8b`, `llama8b`, `gemma2-9b`, `gemma2-2b`) |
| `prompt`, `layer`, `strength` | The English-only prompt variant (S2, S3, S4), the steering layer, the strength in class gaps |
| `n_tricks`, `n_plain`, `n_random` | 24 tricks asking for Chinese, 15 plain questions, 45 random-direction replies |
| `hold`, `hold_lo`, `hold_hi` | Share of tricks the prompt survives without steering (under 5% Chinese characters) |
| `switch`, `switch_lo`, `switch_hi` | Chinese share of the steered replies |
| `fluent` | Share of steered replies that are fluent (≥ 20 characters, ≥ 30% distinct character 3-grams) |
| `random_cjk`, `random_hi` | Chinese share under random directions of the same size |
| `score`, `score_lo`, `score_hi` | `hold × mean(Chinese share × fluent) − random` |

## `search_english_lock.tsv`

Every trial of the search (`harness/harness.py`), one row each. `phase` is `hold` (the prompt alone against 8 tricks),
`grid` (3 layers × 5 strengths on 3 questions), `refine` (±0.25 around the best), `control` (the best next to a random
direction), `manual` (hand-picked runs) or `case-b` (the reverse test: the system prompt asks for Chinese, the user
for English; `switch` is the reply's Chinese share). The other columns are as above.

## `replies/<model>.jsonl`

The full text of every reply behind each model's headline row, one JSON object per line: `kind` (`trick`, `steered`,
or `random0` to `random2`), `user` (the message), `text`, `cjk` (Chinese share) and, for steered replies, `fluent`.

## `replies/unsteered/<model>.jsonl`

The same 15 questions under the same English-only prompt as the model's confirmation run, with no steering
(`unsteered.py`): `kind` (`unsteered`), `prompt`, `user`, `text`, `cjk`. All 60 replies are 0% Chinese. The note shows
them next to the steered replies.
