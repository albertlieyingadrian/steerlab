# Task: improve the recipe

## The recipe interface
recipe.py must define two functions. numpy and the Python standard library are available. The recipe runs in a
sandbox: no network, no subprocesses, 240 seconds, it may read only the pools folder and its own files.

build(ctx) -> {"rows": [...], "train": {"lr": float, "epochs": float}, "note": str}
  ctx = {"data": "<the pools folder, read-only>", "seed": 0}
  each row = {"pool": "mining" | "mirrors" | "replay" | "unlabeled", "idx": int, "bucket": 0 | 1 | 2 | 3,
              "weight": float, "aug": "none" | "tail" | "middle" | "lower" | "squash_ws"}
  Rows are references: the harness reads the text itself, so a row can only ever be a pool item. aug: tail = the
  last ~400 words, middle = ~400 words from the middle, lower = lower-cased, squash_ws = whitespace collapsed.
  The update starts from the starting detector (its LoRA adapter and 4-way head; score = expected bucket / 3) and
  trains on these rows only: weighted cross-entropy on the bucket, AdamW, steps of >= 64 rows, 3% warm-up then
  cosine decay, gradient clip 1.0. Reading the mining pool's texts (620 MB of JSON lines) takes about 15 seconds.

fit_cutoff(cal) -> float. A text is flagged as AI when its score >= the cutoff.
  cal = {"v2b_val": {"scores": [9,562 floats], "kind": [...], "bucket": [...]},
         "unlabeled": {"scores": [2,000 floats]}}
  i.e. the UPDATED detector's scores on data v2b's validation split (with its own labels: kind "human" is human
  text; the other kinds are AI edits or AI text) and on the first 2,000 pages of the unlabeled pool.
  BLINDED: the harness shows these scores on a hidden scale, s' = a*s + b with random a in [0.55, 0.9] and b in
  [0, 1 - a], calls fit_cutoff twice with two different hidden scales, and maps each answer back. The two answers
  must agree within 0.001 in real units, otherwise the recipe is invalid. So the cutoff must be computed from the
  scores (quantiles, order statistics, mean + k x SD, ...); a typed-in number, a clamp to fixed numbers or a fixed
  offset added to a quantile fails. validate() runs this check on synthetic scores.

## The pools you can train on
# Pools the recipe may read (read-only)

Every training row a recipe returns is a reference into one of these pools: {"pool", "idx", "bucket", "weight",
"aug"}. The harness builds the text itself, so a row can only ever be a pool item.

| Pool | File | Items | What it is |
|---|---|---|---|
| mining | mining.jsonl, mining_scores_r0.npy | 250,000 | FineWeb pages crawled 2017-01 .. 2022-10 (8 dumps), so written by people by the label rule. Fields: id, host, dump, date, words, text. `mining_scores_r0.npy[i]` is the starting detector's score for page i (0 = human .. 1 = AI). The starting detector flags 1,407 of them at its cut 0.3986 |
| mirrors | mirrors.jsonl | 5,999 | AI rewrites of mining pages (a summary of the page, then a new page written from the summary by one of 7 open models). Fields: mining_idx (the source page), model, text |
| replay | replay.jsonl | 79,120 | The starting detector's own training data (data v2b): human text and AI text of four degrees. Fields: kind, source, bucket (0 human, 1 light edit, 2 heavy edit, 3 fully AI), band, weight, target, text |
| unlabeled | unlabeled.jsonl | 20,000 | Other FineWeb pages crawled before 2022-11 from hosts in no other pool; unlabeled human text. Fields as mining. The first 2,000 are scored by every updated detector and passed to fit_cutoff as cal["unlabeled"] |

Buckets: the detector has a 4-way head (0 human, 1 light AI edit, 2 heavy AI edit, 3 fully AI); its score is the
expected bucket / 3. Default weights in data v2b: human 1.3278, fully AI 1.3092.


## How a recipe is scored
Held-out evaluation sets, in no pool: 5 human families - web (pre-2022 FineWeb pages from hosts in no pool),
magazine (magazine articles), enron_h (Enron emails), epoch_h (Epoch AI's human texts), editlens_h (EditLens human
texts) - and 8 AI families - editlens_gen, editlens_edit (AI-edited human text), enron_gen, epoch_imit (AI imitations
of a named human author), epoch_plain, magazine_ai, fresh_old and fresh_new (pages by four 2026 models used nowhere
else). Each family is split 40/30/30 by document into dev, val and test. Improvers see dev only.
Pilot 2 adds two harder human families - hc3_h (human answers to HC3 questions: finance, medicine, Reddit ELI5,
Wikipedia computer science, open QA) and raid_h (RAID human texts: books, news, Reddit, abstracts, wiki, poetry,
recipes, reviews) - and their AI counterparts hc3_ai (ChatGPT answers to the same questions) and raid_ai (RAID AI
texts). So: 7 human families and 10 AI families; held-out web is 6,000 pages.
Primary = the POOLED false-positive rate over all human texts of the tier at the MATCHED cutoff: the cutoff at which
the recipe's macro recall over the 10 AI families equals a reference recall, R0's macro recall at the point where R0
flags 2% of the tier's human texts. It measures ranking, not the cutoff: moving your cutoff does not change it. Gain = (R0 matched FPR - recipe matched FPR) / R0 matched FPR.
Constraint: at the recipe's OWN cutoff, macro recall must be >= R0's macro recall on the tier minus 1 point.
SCORE = gain if the constraint holds, else min(gain, 0) - 1 (a constraint failure is always worse than R0).
Higher score is better; R0 = 0. R0 = the base recipe (the 2,000 highest-scored mining pages as human + 8,000 replay
rows, 1 epoch, lr 1e-4, cutoff = 99th percentile of v2b-val human scores), measured as the mean of 3 seeds.

## Rules (checked by the harness)
- at most 15,000 rows; lr in [5e-5, 4e-4]; epochs in [0.05, 2.0]; weight in [0, 5]; bucket in 0..3
- rows sharing 3 or more 10-word runs with any evaluation text are dropped (the count is in the feedback)
- every file open, process start and network call inside the sandbox is logged, and blocked outside the allowed paths
- each proposal uses one evaluation (about 10 GPU-minutes), whether or not it builds
- fit_cutoff sees blinded scores and must return the same real cutoff under two hidden scales (see the API)

## Other recipes tried so far in this run (dev results)
## v1 (parent -, children 0): grade +38.8% (best val-score improvement of its 4 recipes over the frontier recipe r0_s0 it started from; 4 of 4 ran)
- recipe l3_01 (from r0_s0): dev score -419.8%, dev macro recall 53.27% CONSTRAINT FAILED; rows {'mining': 1963, 'replay': 9995, 'mirrors': 500}; hypothesis: Using stratified sampling of mining pages that includes more borderline cases and increasing replay size will improve the model's ability to distinguish difficult human texts from AI, thereby reducing the pooled false positive rate at matched recall compared to the baseline R0 recipe.
- recipe l3_02 (from r0_s0): dev score -65.1%, dev macro recall 80.16%; rows {'mining': 2971, 'replay': 7999}; hypothesis: By increasing the diversity of hard negatives from mining and weighting replay samples by score, we'll improve the detector's ability to distinguish human from AI text, leading to a lower pooled false positive rate at matched recall.
- recipe l3_03 (from l3_02): dev score -165.8%, dev macro recall 76.51% CONSTRAINT FAILED; rows {'mining': 2971, 'replay': 8000}; hypothesis: By using uniform replay sampling and a more conservative 99th percentile cutoff with a stricter minimum, we will reduce the pooled false positive rate at matched recall while maintaining sufficient macro recall to meet the constraint.
- recipe l3_04 (from r0_s0): dev score +34.4%, dev macro recall 79.89%; rows {'mining': 2967, 'mirrors': 499, 'replay': 7497}; hypothesis: Increasing the mining sample size to 3000 and adding 500 mirror samples will improve the detector's ability to distinguish hard-to-classify human texts from AI, leading to a lower pooled false positive rate at matched recall.


## The recipe you start from
[[FILES]]

## Dev feedback for that recipe
[[FEEDBACK]]

## What to do
Propose one change, or a small coherent set of changes, that you expect to raise the dev SCORE (lower the pooled
false-positive rate at matched recall) while keeping your own cutoff's macro recall above its floor. Base it on the feedback above, not on guesses about the hidden tiers.

Key insights from previous recipes:
1. Increasing mining sample size helps (l3_04 showed +34.4% dev score)
2. Adding mirror samples helps (l3_04 showed +34.4% dev score)
3. Constraint failures happen when macro recall drops too low (l3_01, l3_03 failed constraint)
4. Replay sampling strategy matters - weighting by score vs uniform sampling
5. The optimal balance between mining, mirrors, and replay seems to be around 3000 mining + 500 mirrors + 7500 replay

Focus on recipes that:
- Maintain or exceed the macro recall of the baseline (avoid constraint failures)
- Leverage successful patterns from previous work
- Explore systematic variations around the proven 3000+500+7500 recipe structure
- Prioritize the recipe quality over quantity - aim for better recipes rather than just more recipes

Return the complete new recipe.py (and any other recipe file you add), then the HYPOTHESIS line.