# Pre-registration: toy-scale replication of "Modular Pretraining Enables Access Control" (GRAM)

Written 2026-11-27, committed before any training run beyond a smoke test. Paper: "Modular Pretraining Enables
Access Control", arXiv 2607.08077 (ICML 2026), code at github.com/agencyenterprise/modular-pretraining.
Anything marked **exploratory** is not a test of the paper.

## 1. The claim under test

GRAM adds small auxiliary MLP modules beside each layer's MLP and, during pretraining, routes the gradient of
labelled "restricted" data into its own module. Switching a module off at deployment should then remove that
capability about as well as a model that never saw the data (data filtering), keep everything else, and resist a
short relearning fine-tune better than post-hoc unlearning.

## 2. Setup (copied from the paper's Simple Stories experiment, Fig. 2 and its `stories` config)

| Item | Value |
|---|---|
| Data | SimpleStories/SimpleStories train split, 48 topics; `train_test_split(test_size=0.1, seed=42)` as in `prep_stories.py`. Tokenizer SimpleStories/SimpleStories-1.25M (vocab 4,096), EOS appended to each story |
| Restricted ("auxiliary") topics | the first four alphabetically: a-deadline-or-time-limit, alien-encounters, bygone-eras, cultural-traditions. The other 44 topics form the core set |
| Model | 8 layers, width 512, 8 heads, 2 KV heads, rotary, RMSNorm, GELU MLP 2,048, untied unembedding, context 256, document-aware causal mask (about 26M parameters) |
| GRAM | core MLP 2,048 plus four auxiliary MLPs of 192; `p_as` = 0.3, `p_cr` = 0.5, `p_af` = 1; separate AdamW per parameter partition; compute matched to the baseline by trimming batches (`equal_compute`) |
| Training | one epoch, batch 128 x 256 tokens, AdamW lr 5e-3, betas (0.9, 0.95), weight decay 0.01, 10% warmup, 10% linear decay, gradient clip 1.0 |
| Methods | Baseline (all data); GRAM (5 profiles: no module, or one of the four); Data filtering (5 separate models: core only, core + one topic); MaxEnt post-hoc unlearning from the baseline (5 profiles) |
| Elicitation (relearning attack) | per forget topic: 512 training sequences of that topic, full-batch AdamW at lr 1.25e-3 (the paper's lr / 4), up to 200 steps, patience 10, best step kept. With GRAM the forget module stays off |
| Metric | compute ratio (CR): the fraction of baseline training steps at which the baseline reached this loss, from a power law L(s) = A (s + x0)^-alpha fitted to the baseline's loss curve per topic, pooled over seeds; denominator = mean over seeds of the baseline's own final step equivalent. Also raw loss differences in nats |
| Summary | Core CR; Retain CR (a module's topic with that module on, or a filtered model's included topic); Forget CR (topics whose module is off or that were filtered out); Elicited forget CR. Each is averaged over the five profiles within a seed, then mean and t-based 90% CI over seeds (the paper's convention), with 95% CIs also given |
| Seeds | 3 (as in the paper). Extended to 5 if Modal spend after 3 seeds is under $25 |

### Known differences from the paper (fixed now)

1. fp32 master weights with bf16 autocast; the paper's code casts the model to bf16.
2. The core stream is the 44 topics concatenated and sampled as random windows, so a core batch mixes topics.
3. Each topic's test split is cut in two by a fixed permutation: a **val** half used only for choices (elicitation
   early stopping, the MaxEnt grid), a **test** half for every reported number. The paper selects the elicitation
   step on the same split it reports.
4. MaxEnt: the paper tuned (lr, alpha) per profile and seed by Bayesian optimisation. Here a 3 x 3 grid
   (lr 3e-4, 1e-3, 3e-3; alpha 10, 30, 100) on seed 1, core-only profile, val half: the setting with the highest mean
   forget loss whose core loss rises by at most 0.03 nats over the baseline. That one setting is used everywhere.
   2,000 steps, half-batch forget / half retain, stop when KL to uniform <= 2.0, as in their code.
5. Baseline loss curve evaluated 100 times (the paper: 200), 16 batches per topic.

## 3. Predictions

Paper's Simple Stories numbers (`analysis/stories/stories_agg_summary.csv`): GRAM core 0.94, retain 0.95, forget
0.77, elicited forget 0.86; filtering 0.96 / 0.96 / 0.78 / 0.87; MaxEnt 0.96 / 0.93 / 0.84 / 0.98.

- **P1, switching the module off removes the topic as well as filtering.** GRAM forget CR <= filtering forget CR +
  0.05 (seed means).
- **P2, the rest survives.** GRAM core CR >= 0.90 and GRAM retain CR >= 0.90.
- **P3, the module is where the topic lives.** For each restricted topic, loss with its module off minus loss with
  it on is > 0, with the 95% CI over seeds above zero, for all four topics.
- **P4, relearning: GRAM resists like filtering, unlearning does not.** GRAM elicited forget CR <= filtering
  elicited forget CR + 0.05, and MaxEnt elicited forget CR - GRAM elicited forget CR >= 0.05.
- **P5 (secondary), a bigger attack.** With 4,096 sequences (8x) on the core-only profile, GRAM elicited forget CR
  <= filtering's + 0.05.

The core claim "holds at toy scale" if P1, P2 and P4 hold. P3 is the mechanism check, P5 a stress test.

## 4. Controls and boring explanations

- **Wrong module on:** the forget topics under another topic's profile already measure "a module is on, but not
  this topic's". Reported per profile.
- **Is the gap just a weaker core?** Core CR is reported next to forget CR; forget is read against the core drop.
- **Baseline as positive control:** the baseline sees everything; its CR is 1 by definition.
- **Exploratory:** loss gaps in nats, per-topic breakdown, and the elicitation curves.

## 5. Budget

Modal only, profile `albertlieyingadrian`, at most 2 GPUs at a time, no retries, at most $50. Every launch is
written to `results/spend_ledger.jsonl`.
