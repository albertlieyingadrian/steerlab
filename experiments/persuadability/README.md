# Persuadability organisms: fine-tuning vs steering

Can Qwen3-1.7B be made easy to persuade (gives in to a user's wrong pushback) or hard to persuade (keeps a correct
answer but still accepts fair corrections), by LoRA fine-tuning or by one steering direction? Pre-registration:
[`PREREGISTRATION.md`](PREREGISTRATION.md) (committed before any training). Note: [`report/index.html`](report/index.html).

## Verdict

- **Easy organism: yes, by fine-tuning.** LoRA easy flips 0.97 in domain (untrained 0.075), 0.92 on unseen
  arithmetic and memos, keeps 0.99 of fair corrections and its MMLU.
- **Hard organism by fine-tuning: no, it backfired.** LoRA hard flips 0.17 in domain and 0.66 unseen (untrained
  0.215); the twin without hold replies moves the same way. LoRA resist (hold only) refuses fair corrections in
  domain (0.85 vs 0.97).
- **Hard organism by steering at the pre-set strength (layer 12, α = 4): stubborn and broken.** Updates 0.57 vs
  0.955, news 0.04 vs 0.85, coherent 0.47; random directions of the same length also break the model.
- **Exploratory, α = 2:** flip 0.075 below all random directions (0.14-0.33), updates 0.91, news 0.80, MMLU 0.50.
- **Readouts:** judge κ 0.78 vs 120 check labels (bar 0.8) → forced-choice letter primary (κ 0.55, collapses to
  "A" under strong steering). Verdicts under both are in `results/analysis.json` and Table 2 of the note.

## Deviations from the plan

- Claude Sonnet 5.5 declined 71 of 90 batch requests to write the assistant training replies
  (`reasoning_extraction`). The prompt was not reworded. Qwen3-1.7B could not write its own "give in" replies
  (123/450), so Qwen3-8B wrote them (`gen_replies.py`); recorded in the pre-registration before training.
- The judge missed its κ bar, so the forced-choice readout became primary as pre-registered; the letter collapsed
  under steering, so steering verdicts are reported under both readouts.
- Exploratory after the results: 10 random directions at α = 2 (5 judged), and the agreeing-opener count
  (`opener.py`).
- The 120 judge-check labels were made by one labeller, blind to the judge's labels and committed before those
  came back; a second labeller has not checked them.
- MMLU per-question answers were not saved, so the pre-registered paired MMLU test could not be run.

## Files

| File | What it does |
|---|---|
| `gen_items.py` | Sonnet 5.5 items (batch); the retired reply stage |
| `make_heldout.py` | arithmetic and memo items, benign follow-ups |
| `screen.py` | knowledge screen, pilot, split |
| `gen_replies.py` | Qwen3-8B training replies (context distillation) |
| `build_sft.py` | the four LoRA training sets |
| `common.py`, `pipeline.py` | model, chat format, steering hook, LoRA, direction, dev sweep, evaluation |
| `modal_app.py` | Modal stages: train, sweep, eval, explore (L40S) |
| `judge.py` | Opus 5.5 judge, check sample, agreement |
| `analyze.py`, `opener.py` | every number; the exploratory opener count |
| `make_figures.py`, `make_cover.py` | figures and cover from `results/` |
| `claude.py` | API key from `.env` (never printed), batch helpers, spend ledger |

## Cost

Anthropic $25.92 (`results/ledger.jsonl`: items $0.77, declined reply batch $0.42, judge $18.42 + $6.30).
Modal $3.41 on L40S (apps named `persuadability`). Total $29.33.
