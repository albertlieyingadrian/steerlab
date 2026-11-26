# Task: improve the recipe

## The recipe interface
recipe.py must define two functions. numpy and the Python standard library are available. The recipe runs in a
sandbox: no network, no subprocesses, 240 seconds, it may read only the pools folder and its own files.

build(ctx) -> {"rows": [...], "train": {"lr": float, "epochs": float}, "note": str}
  ctx = {"data": "<the pools folder, read-only>", "seed": 0}
  each row = {"pool": "mining" | "mirrors" | "replay" | "unlabeled", "idx": int, "bucket": 0 | 1 | 2 | 3,
              "weight": float, "aug": "none" | "tail" | "middle" | "lower" | "squash_ws"}
  Rows are references: the harness reads the text itself. The update starts from the starting detector (LoRA, 4-way
  head; score = expected bucket / 3) and trains on these rows only: weighted cross-entropy on the bucket, AdamW,
  steps of >= 64 rows, 3% warm-up then cosine decay, clip 1.0. Reading the mining pool's texts takes ~15 s.

fit_cutoff(cal) -> float in [0, 1]. A text is flagged as AI when its score >= the cutoff.
  cal = {"v2b_val": {"scores": [9,562 floats], "kind": [...], "bucket": [...]},
         "unlabeled": {"scores": [2,000 floats]}}
  = the UPDATED detector's scores on v2b's validation split (kind "human" is human, others are AI edits / AI text)
  and on the first 2,000 unlabeled pool pages.

## The pools
[[POOLS]]

## Scoring
Strict = macro false-positive rate over 5 human families (web, magazine, enron_h, epoch_h, editlens_h) at the
recipe's own cutoff, valid only if macro recall over 8 AI families >= R0's macro recall minus 1 point (dev floor
about 87.74%); else the recipe scores R0's FPR (gain 0). Gain = (R0 FPR - recipe FPR)/R0 FPR. R0 = 2,000
highest-scored mining pages as human + 8,000 replay rows, 1 epoch, lr 1e-4, cutoff = q99 of v2b-val human scores
(R0 dev FPR ~1.44%, so each flagged human text matters).

## Rules (checked by the harness)
- at most 15,000 rows; lr in [5e-5, 4e-4]; epochs in [0.05, 2.0]; weight in [0, 5]; bucket in 0..3
- rows sharing 3 or more 10-word runs with any evaluation text are dropped
- each proposal uses one evaluation (~12 GPU-minutes) whether or not it builds: a crash wastes it. Be defensive
  (try/except around optional parts, fall back to R0 behaviour, stay far below 240 s, use numpy for
  mining_scores_r0.npy, avoid reading all of mining.jsonl).

## What the archive has established (READ THIS)
- Large data changes lost: +1,000-1,500 extra mining pages, 800-2,000 AI mirror rewrites (mirrors seem to teach
  "polished = AI"), and cutoffs picked by v2b-val recall all RAISED dev FPR. Do NOT add mirrors or more mining rows.
- Tiny cutoff raises (+0.03 over q99 or the unlabeled q99.5) gave no gain (-4.5%): the borderline human pages
  (magazine ones especially) score well above q99, in the 0.4-0.6 area.
- The ONLY recipe with a real gain was l3_08: +67.5% dev gain, dev macro recall 89.36% (floor 87.74%, so ~1.6
  points of slack). Its recipe: mining pages sorted by mining_scores_r0 descending, drop the top 150, sample
  ~2,000 from ranks 150-2,400 (bucket 0), R0's 8,000 replay rows parsed robustly, 1 epoch, lr 1e-4, and
  fit_cutoff = min(q99 of v2b-val human scores + 0.10, 0.52). So: a LARGE cutoff raise, capped, is the productive
  lever and recall had room for it. Recall is the constraint to guard, not the thing to maximise.
- fit_cutoff must be robust: numpy, handle empty arrays / missing keys (fall back to ~0.45), return a float
  clipped to [0, 1].

## Archive of all previous improvers and recipes
[[ARCHIVE]]

## Suggested primary lever for THIS proposal (follow it unless the feedback clearly argues otherwise)
[[ANGLE]]

## The recipe you start from
[[FILES]]

## Dev feedback for that recipe
[[FEEDBACK]]

## What to do
Briefly reason (a few lines), then return the complete new recipe.py (and any other recipe file you add), then the
HYPOTHESIS line. Make one coherent change set matching the suggested lever; keep the rest of the starting recipe's
structure, keep it defensive and runnable, and make sure every row's idx is a valid index into its pool.