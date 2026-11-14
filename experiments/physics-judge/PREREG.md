# Pre-registration: cheap judges for physics violations in generated video

Written 2026-11-14, after the sample was drawn (`1_sample.py`) and the doctored clips were built (`2_traps.py`), before
any judge, heuristic or released model was run on a single video. Committed to git before scoring; any change after
this point is listed under "Amendments" at the bottom with its reason.

## Question

Can cheap judges (Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 on sampled frames, and non-LLM motion heuristics) tell
which generated videos break physics as well as the released, fine-tuned judges do, and as well as people?

WorldModelBench's 67K human labels and its generated videos are not public (its README: the test answers are
withheld), so its judge cannot be scored on its own labels. The human labels used instead are VideoPhy-2's (MIT): each
test video was rated by three annotators for physical commonsense (PC, 1 to 5, mean of three, rounded) and each of the
prompt's physical rules was marked followed, violated or cannot be determined. Only the rounded means are released.

## Sample

- 600 videos: 100 from each of CogVideoX-5B, Cosmos, HunyuanVideo, Luma Ray2, VideoCrafter2 and Wan2.1, drawn from
  the 3,397-video test split with seed 20261114 (Sora, 60 videos, left out).
- Prompts are split: dev 134 prompts / 194 videos (prompt writing, model choice, thresholds, logistic weights), test
  270 prompts / 406 videos (every headline number). No prompt is in both.
- Rules: the benchmark's rules for those videos with a followed or violated mark; "cannot be determined" dropped.
  Test: 1,089 rules, 176 violated. Law groups: Gravity, Elasticity, Friction, Conservation of Momentum, Conservation of
  Mass, Hardness, and Other (everything else pooled).
- Known-answer set: 25 clean test videos outside the sample (PC = 5, no rule violated, prompt about falling, throwing,
  pouring or bouncing), each with three doctored copies: reversed, frozen from the middle frame, central half enlarged
  35% from the middle frame. All four versions re-encoded identically.

## Labels

- **Violation** (primary): PC < 4, i.e. the video is not rated likely to follow physical commonsense; VideoPhy-2's own
  "joint" score uses PC >= 4 as the pass mark. Test: 215 of 406 violations.
- Clear cases (secondary): PC <= 2 against PC >= 4, dropping PC = 3.
- Rule violated (per category): the annotators' mark on each rule.

## Judges

Released (the bar):
1. **WorldModelBench judge** (`Efficient-Large-Model/vila-ewm-qwen2-1.5b`, VILA at commit `c8f603b`), its own five
   physics questions and prompt template from its `evaluation.py`, without chain of thought (the script's default),
   16 frames (its config). Score per question: P(Yes) / (P(Yes) + P(No)) on the first answer token, "Yes" meaning a
   violation is present. Video score: the largest of the five. Its official binary reading (`"no" in reply`) is
   reported beside it.
2. **VideoPhy-2 AutoEval** (7B, trained on VideoPhy-2's train split, whose prompts do not overlap the test): its
   released PC predictions for every test video; score = 6 - predicted PC.

Cheap:
3. **Claude, plain** (`plain`): 8 frames spread evenly over the clip (long side 512 px, labelled with their times),
   the prompt, and one question: how likely is it that the video breaks physical commonsense (0-100), plus a 1-5 PC
   rating. Opus 5.5, Sonnet 5.5, Haiku 5.5.
4. **Claude, checklist** (`checklist`): the same frames; a per-violation-type checklist (gravity, motion without cause
   or momentum, objects changing shape, size or number, solids passing through each other, unnatural fluids, bounces
   and elasticity, friction and sliding) answered type by type, then the overall 0-100. All three models.
5. **Claude, motion frames** (`motion`): 16 frames, 8 even plus 8 at the largest frame-to-frame motion changes
   (from optical flow), checklist prompt. Opus and Sonnet (Haiku if budget allows).
6. **Claude, rules** (`rules`): 8 even frames and the prompt's own physical rules (all of them, as the benchmark lists
   them); for each rule, followed / violated / cannot tell with a 0-100 violation probability, then the overall 0-100.
   All three models. This is the per-category judge.
7. **Self-consistency**: `checklist` asked 3 times (Sonnet and Haiku), scores averaged.
8. **Ensembles**: the mean of the three models' `checklist` ranks; Claude's best single config plus the heuristics in
   one logistic regression fitted on dev.
9. **Heuristics** (no model): from Farneback optical flow on up to 48 evenly spaced frames at 256 px: mean motion,
   motion jerk (frame-to-frame change in mean flow, relative to the mean), largest frame-difference spike over the
   median, largest jump in global scale between frames (similarity fit to tracked points), mean absolute flow
   divergence, and flow-warp error. Each alone (higher = more violating) and a logistic regression of all six fitted on
   dev.

Video-level scores for the Claude configs are the overall 0-100. Replies are cached; one effort setting per model is
fixed on a 10-video dev pilot and recorded in the results.

## Metrics

- **Primary**: test AUROC for violation (PC < 4), all 406 videos.
- Secondary: accuracy and F1 at a threshold fixed on dev (the cut that maximises balanced accuracy there); AUROC on
  the clear cases; Spearman with PC; per-model AUROC.
- Per category: rule-level AUROC per law group on test, for `rules` (each rule's probability) and for the
  WorldModelBench judge (the question matching the law: Gravity to its gravity question; Conservation of Mass,
  Hardness and Elasticity to its deformation question; Conservation of Momentum and Friction to its Newton question;
  Other to the largest of its five). AutoEval's rule predictions are not released, so it is scored per video only.
- Intervals: 95% bootstrap over prompts (a prompt's videos resampled together), 2,000 draws.
- Human reference: per-annotator ratings are not released, so no one-rater ceiling can be computed from this data.
  The VideoPhy-2 paper reports 75-80% annotator agreement; that number is quoted, not recomputed. As a separate check,
  the annotators' own rule marks (count of violated rules plus their written extra violations) are scored against PC
  like a judge; this is the same people's other answer, not an independent rater, and is labelled as such.

## Selection and the "beats their judge" rule

- One Claude configuration is chosen as **the cheap judge** on dev alone (highest dev AUROC among configs 3-8; ties go
  to the cheaper one) before any test score is read. Test scores of all configs are reported, but only the chosen one
  is tested against the bar.
- **Beats the WorldModelBench judge** if the test AUROC difference (chosen minus WorldModelBench), paired and
  bootstrapped over prompts, has a 95% interval entirely above 0. **Beats AutoEval** is judged the same way, as a
  secondary test. "Level" if the interval contains 0; "worse" if it is entirely below.
- Per category, a law group counts as won only under the same rule on its rules; groups with fewer than 15 violated
  test rules are reported but not tested.

## Known-answer checks

- For each judge and each doctored kind: the share of the 25 pairs where the doctored copy scores higher than its
  clean source (ties half), with a Wilson interval; and the share flagged at the dev threshold, beside the clean
  copies' share.
- A judge **passes** a kind if it ranks the doctored copy higher in at least 20 of 25 pairs.

## Budget

Anthropic at most $80 (Message Batches where possible), Modal at most $20 for the WorldModelBench judge. Spend is
logged in `results/spend.csv`.

## Amendments

1. **Text-only control (added 2026-11-14, after the dev selection and after a first look at test scores, before the
   WorldModelBench run finished).** VideoPhy-2's rules are written per video, not per prompt (the four videos of one
   prompt have four different rule lists), and a bag of words on the rule text alone, fitted on dev, reached 0.635
   test AUROC against 0.555 for the caption. So the `rules` config (the chosen judge) could partly be reading the rule
   text rather than the frames. Opus 5.5 was run on all 600 videos with no frames, once with the `rules` ask
   (`rules_noframes`) and once with the `plain` ask (`plain_noframes`). These are reported as a control and paired
   against the frame runs; they are never candidates for selection, and the pre-registered verdicts are unchanged.
