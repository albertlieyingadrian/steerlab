# Task: improve the recipe

## The recipe interface
recipe.py must define two functions. numpy and the Python standard library are available. The recipe runs in a
sandbox: no network, no subprocesses, 240 seconds, it may read only the pools folder and its own files.

build(ctx) -> {"rows": [...], "train": {"lr": float, "epochs": float}, "note": str}
  ctx = {"data": "<the pools folder, read-only>", "seed": 0}
  each row = {"pool": "mining" | "mirrors" | "replay" | "unlabeled", "idx": int, "bucket": 0 | 1 | 2 | 3,
              "weight": float, "aug": "none" | "tail" | "middle" | "lower" | "squash_ws"}
  Rows are references: the harness reads the text itself. aug: tail = the last ~400 words, middle = ~400 words
  from the middle, lower = lower-cased, squash_ws = whitespace collapsed.
  The update starts from the starting detector (LoRA adapter, 4-way head; score = expected bucket / 3) and trains
  on these rows only: weighted cross-entropy on the bucket, AdamW, steps of >= 64 rows, 3% warm-up then cosine
  decay, gradient clip 1.0. Reading the mining pool's texts (620 MB JSON lines) takes about 15 seconds.

fit_cutoff(cal) -> float in [0, 1]. A text is flagged as AI when its score >= the cutoff.
  cal = {"v2b_val": {"scores": [9,562 floats], "kind": [...], "bucket": [...]},
         "unlabeled": {"scores": [2,000 floats]}}
  i.e. the UPDATED detector's scores on v2b's validation split (kind "human" is human; other kinds are AI edits or
  AI text) and on the first 2,000 pages of the unlabeled pool.

## The pools
[[POOLS]]

Buckets: 0 human, 1 light AI edit, 2 heavy AI edit, 3 fully AI; score = expected bucket / 3. Default weights in
v2b: human 1.3278, fully AI 1.3092.

## Scoring
Strict = macro false-positive rate over 5 human families (web, magazine, enron_h, epoch_h, editlens_h) at the
recipe's own cutoff, valid only if macro recall over 8 AI families (editlens_gen, editlens_edit, enron_gen,
epoch_imit, epoch_plain, magazine_ai, fresh_old, fresh_new) >= R0's macro recall minus 1 point; else the recipe
scores R0's FPR (gain 0). Gain = (R0 FPR - recipe FPR)/R0 FPR. R0 = 2,000 highest-scored mining pages as human +
8,000 replay rows, 1 epoch, lr 1e-4, cutoff = q99 of v2b-val human scores (R0 dev FPR is about 1.44%, so each
flagged human text matters; dev sets are small and noisy).

## Rules (checked by the harness)
- at most 15,000 rows; lr in [5e-5, 4e-4]; epochs in [0.05, 2.0]; weight in [0, 5]; bucket in 0..3
- rows sharing 3 or more 10-word runs with any evaluation text are dropped
- every file open, process start and network call in the sandbox is logged and blocked outside allowed paths
- each proposal uses one evaluation (about 12 GPU-minutes), whether or not it builds: a recipe that crashes or
  times out is wasted, so be defensive (try/except around optional parts, fall back to R0 behaviour, stay far
  below 240 s, use numpy for mining_scores_r0.npy and avoid reading all of mining.jsonl unless needed; if you
  need hosts, read lines but only parse json for what you need).

## Lessons from earlier attempts (all four were WORSE than R0 on dev)
[[ARCHIVE]]

Distilled knowledge:
- Every large departure from R0 lost: adding 1,000-1,500 more mining pages, adding 800-2,000 AI mirror rewrites,
  and moving the cutoff using v2b-val recall (l3_04) all RAISED dev FPR (-4% to -83%). AI mirrors in particular
  appear to teach "polished text = AI" and push borderline human magazine text above the cutoff. Do NOT add mirrors.
- Dev recall of those recipes (89-92%) was above the floor, so the recall floor was not the binding constraint;
  the models themselves ended with higher human scores.
- Therefore make a SMALL, single-lever change from R0 (keep ~2,000 mining human rows and ~8,000 replay rows unless
  your angle says otherwise). Keep the same code structure as the starting recipe and edit only what the angle needs.
- fit_cutoff must be robust: use numpy, handle empty arrays, return a float clipped to [0, 1].
- If dev feedback of the starting recipe is given, find which human family has the most flagged texts and what its
  scores look like, and aim the change at that; do not repeat an earlier recipe's idea that already lost.

## Suggested primary lever for THIS proposal (diversity across parallel proposals; deviate only if the feedback
## clearly argues for something else)
[[ANGLE]]

## The recipe you start from
[[FILES]]

## Dev feedback for that recipe
[[FEEDBACK]]

## What to do
Briefly reason (a few lines), then return the complete new recipe.py (and any other recipe file you add), then the
HYPOTHESIS line. Make exactly one coherent change; keep it defensive and runnable.