# Preregistration: replicating Fireworks' Muse Glimmer Kalshi decision tuning

Written 2026-11-14, before any training run beyond a 32-row smoke test and before any test-set number was read.

## What is replicated

Fireworks' cookbook `training/examples/decision/kalshi_v1` (fw-ai/cookbook, version 2026.11.13.1) and the blog post
"Train Muse Glimmer for free". Recipe, unchanged: `training.recipes.decision_loop` on serverless
`accounts/fireworks/models/muse-glimmer-30b`, LoRA rank 64 / alpha 32, Adam lr 1e-4, batch 64, one epoch, seed 0,
K-way cross-entropy over option-letter tokens, per-type temperatures fitted on `dev.jsonl`.

Data: `fireworks-ai/kalshi-decision-v1` on Hugging Face, SHA-256 pinned in `data/manifest.json`
(train 6,461 events / 20,051 questions; dev 2,000 / 6,232; test 8,765 / 33,712; the unseen-topic slice is the final
765 test rows, 3,567 questions).

Our only addition to the recipe: `fw_train.py` hooks the recipe's step-0 and final evaluation to also read every
dev and test question's option logprobs on the same trainer, for the base and the tuned model. These readouts are
the only inputs to evaluation. Base numbers use `T=1` (raw), as Fireworks did.

## Published numbers we compare against

| Metric | Base raw | Tuned calibrated | Jev raw |
|---|---:|---:|---:|
| Full-test accuracy | 56.4% | 64.8% | 45.9% |
| Unseen-topic accuracy | 53.2% | 58.3% | 47.1% |
| Full-test Brier sum | 0.8071 | 0.4318 | 0.5886 |
| Full-test ECE (15 bins) | 0.3919 | 0.0216 | 0.1590 |
| At least 80% confident: kept, accuracy | 31,334, 57.9% | 4,868, 86.8% | n/a |
| Paper allocation of $1,000 | $950 | $1,071 | $919 |

Published tuned temperatures: `choice=0.909`, `noul=1.375`.

## Metrics, fixed now

- **Accuracy**: argmax option equals the label, over all labeled test questions (each question counts once).
- **Brier sum**: sum over options of (p - onehot)^2, averaged over questions.
- **ECE**: 15 equal-width bins on the top-option probability, weighted mean |accuracy - confidence|.
- **Confidence curve**: accuracy and count of questions whose top probability is at least 0.5, 0.6, 0.7, 0.8, 0.9.
- **Common-answer reference**: "No" on yes/no questions; on choice questions, the most frequent gold option position
  on the evaluated slice (it uses test labels; descriptive only).
- **Intervals**: 95% bootstrap over events (whole rows resampled, 2,000 draws) for accuracies and their differences.
- **Backtest**: Fireworks' rule as written in the cookbook README. Quotes are the last completed hourly candle at or
  before open + 2h, with a two-sided book and a spread of at most 10 cents. Bet a side only when its probability
  exceeds 50% and beats the ask plus the Kalshi taker fee (0.07 * C * P * (1 - P), rounded up to the cent on a
  100-contract reference order) by at least 5 cents. Allocate $1,000 in proportion to (p - 0.5); fractional
  contracts; static, not chronological. 90% event-bootstrap interval, 5,000 draws. Quotes come from Kalshi's public
  API; questions we cannot map to a market ticker are excluded, and we report how many we mapped next to Fireworks'
  15,230.

## Success criteria (replication)

1. **Training works**: tuned calibrated full-test accuracy beats base raw by at least 5 points, with the 95% event
   bootstrap interval of the difference above zero.
2. **Magnitude**: tuned accuracy within 3 points of 64.8%; base within 3 points of 56.4%.
3. **Calibration**: tuned ECE below 0.05; questions at 80% confidence or more are right at least 80% of the time.
4. **Generalisation**: on the unseen-topic slice, tuned beats base.
5. **Money**: the tuned paper allocation ends above $1,000 and above base's. This is the weakest test (one static
   backtest, our own quote extraction); a miss here does not fail the replication but is reported.

Jev: Fireworks queried it "directly from its API". If we cannot reach Jev with our keys, we report its published
numbers only and say so.

## Exploratory twist (not part of the replication verdict)

Question: is confidence readable from activations, and does temperature scaling fix an internal signal or only the
readout? Glimmer's activations are not exposed by the serverless trainer, so this runs on an open model with the same
recipe (Qwen3-8B, LoRA, the same data and K-way loss) on Modal or Tinker.

Hypotheses, fixed now:
- **H1 (readable)**: a linear probe on the residual stream at the answer position, trained on dev, predicts whether
  the base model's top option is correct, with test AUROC above the top-option probability's own AUROC.
- **H2 (temperature is readout only)**: temperature scaling cannot change AUROC within a question type (it keeps the
  ranking of confidence), so any ranking gain must come from training, not from the temperature. We test whether the
  probe's AUROC rises from base to tuned (training sharpened the internal signal) or stays flat (training only moved
  the logits).
- **H3 (overconfidence lives in the readout)**: if a logistic probe on base activations is well calibrated on test
  while the base logits are not, the base model's overconfidence is a readout problem that a temperature can fix.

Exploratory results are labelled as such; no threshold is promised.

## Budget

SFT free on Fireworks until 2026-11-30. Fireworks inference at most $40 (any non-free charge reported before it is
incurred); Modal at most $40.

Note added during the run (2026-11-14, no change to the design): the Fireworks budget was tightened to $0. The plan
already used no paid Fireworks service: training is free, and every evaluation number is read on the training job.
