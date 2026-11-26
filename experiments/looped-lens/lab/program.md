# Lens search for the Looped Lens Challenge (autoresearch-style)

Goal: the lens configuration that passes the most questions under the challenge's vote rule, on
questions it was not tuned on. Pattern: autoresearch — one fixed harness, one editable
file, one number, keep or discard after every attempt.

## Files

- `prepare.py` is **fixed**. It holds the dev/held-out split, the cached lens bank and `evaluate()`.
- `lens.py` is **the only file the loop edits**: `PICKS`, a list of (lens, visit), at most 10.
- `run.py` scores `lens.py` on dev, appends a row to `results.tsv`, snapshots `lens.py` in `runs/`.
- `search.py` is the automated loop: propose one change (swap a visit's lens, move a visit, add or
  remove a visit), keep it only if dev improves (ties broken by the mean vote margin), log every
  attempt, check held-out every 10 kept changes.
- The lens bank: `dev/runs/*readouts*.jsonl`, the top-20 token ids of every lens at all 48 visits
  for every question, computed once on a Modal H100 (`dev/modal_lens.py`). The loop never calls
  NDIF or a GPU; new lenses enter by adding a readouts file.

## Metric

- **dev** pass rate (215 questions: 10 practice + two thirds of the self-made set), fuzzy matching.
- **held-out** pass rate (112 questions), checked rarely. A change that raises dev but not held-out
  is overfitting. With ~110 questions one standard error is ~4.6 points.
- Spot-check the winner with an LLM judge (`dev/judge_dev.py`, MiMo v2.6 Pro on OpenRouter).

## Rules (the challenge's, enforced here)

1. Activations only, one token position (the last), one lens per visit, at most 10 visits.
2. A read of a visit's *output* is the state entering the next visit, so it cannot be combined
   with a read at the next visit (`prepare.valid`).
3. Never use the prompt's text or token ids to build, filter or rerank readouts; the visit set is
   the same for every question.
4. No other model produces readouts. Translators are trained offline on generic text, never on
   the dev answers.
