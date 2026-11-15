# Pre-registration: does looping a shared block buy depth on a chain-following task, and does more looping extrapolate?

Written 2026-11-14, before any counted run. Before this was written: pilots on seed 0 sized the task and fixed the
training recipe (they are listed in the executions log and
are not analysed here); probes were tried on one pilot 4-layer model. **What the pilots showed, so the reader can discount it:** a
4-layer model's digit frontier went 1, 2, 4, 8, 12 across its layers (32-assignment version of the task); and in the
last pilot, with this exact recipe on seed 0, stopped at step 6,000 of 30,000, accuracy at 12 hops was 0.17, 0.58 and
0.79 for the looped model at k = 1, 4 and 8 against 0.15 for `standard4` (500 items each). H1, H3 and I1 are
therefore partly informed by a pilot; H2, H4, H5, H6 and I2 were not looked at. Counted runs use seeds 1-3, not the
pilot's seed 0. No counted model has been trained. Not edited after the runs;
amendments go at the end with a time stamp.

Inspired by NVIDIA's Dory (Nemotron Labs Dory 1.0 4B: an 8-layer input block, a 20-layer recurrent block run 1-4
times with the input re-added each loop, an 8-layer output block; the loop count is chosen at inference time) and by
Geiping et al. 2025 (recurrent depth, random loop count in training), Saunshi et al. 2025 (looped transformers and
k-hop reasoning in log k depth) and Dehghani et al. 2018 (Universal Transformers).

## Task

`common.make_batch`: 48 assignments over 96 variable names, shuffled, each one input position
(lhs embedding + rhs embedding, no positions, no causal mask), plus a query position. The assignments form chains that
end in a digit. Answer: the digit at the end of the query's chain. **Hops** = the query chain's length. Distractor
chains have lengths 1-8. Every input has 48 assignments, so hop count changes nothing but the chain.

- Training: fresh examples every step, hops uniform in 1..cur_max; curriculum for every model alike: cur_max starts at
  2 and rises by one, up to 8, when accuracy over the last 250 batches reaches 95%. 20,000 steps, batch 512, AdamW
  (lr 1e-3, weight decay 0.1, betas 0.9/0.98), 500 warm-up steps then cosine, gradient clip 1.0.
- Test set (`train.TEST_SEED` 12345): 500 examples at each hop count 1-24. **In range** = hops 1-8, **near** = 9-16,
  **far** = 17-24.
- **Shortcut baseline**: the expected accuracy of picking uniformly among the chain-end digits in the input
  (`common.shortcut_baseline`), about 0.18-0.23 depending on hop count. Chance is 0.10.

## Models (d_model 128, 4 heads, pre-norm blocks)

| Name | Shape | Parameters | Training loop count |
|---|---|---|---|
| `standard4` | 4 distinct blocks | 832,778 | — |
| `looped` | 1 prelude block, 2 shared core blocks run k times (input re-added each loop: s + e), 1 coda block | 832,778 | k uniform in 1-6 at every step |
| `standard10` | 10 distinct blocks: the compute of `looped` at k = 4 | 2,022,410 | — |
| `looped_fixed4` | as `looped` | 832,778 | k = 4 always |
| `looped_noinject` | as `looped`, no re-adding of the input | 832,778 | k uniform in 1-6 |

Seeds 1, 2, 3 for each (15 runs). Looped models are scored at k = 1, 2, 3, 4, 5, 6, 8, 10, 12, 16 (k > 6 is more
looping than training ever used). Effective depth = 1 + 2k + 1 blocks (k = 1: 4, as `standard4`; k = 4: 10, as
`standard10`).

## Hypotheses and pass rules (decided now)

Accuracy is the share of the 500 x 8 test items in a bin that a model gets right; seeds are reported separately and
as a mean.

- **H1, more loops, more hops (elastic compute).** For `looped`, near-bin accuracy at k = 4 exceeds k = 1 by at least
  0.10, in all 3 seeds.
- **H2, looping past the training count extrapolates.** For `looped`, far-bin accuracy at the best k in {8, 10, 12,
  16} exceeds k = 6 (the training maximum) by at least 0.05, in at least 2 of 3 seeds. (Prediction: passes. A
  failure means the extra loops do not keep resolving hops.)
- **H3, loops beat the same parameters used once.** `looped` at k = 8 beats `standard4` on near and far bins pooled
  (hops 9-24) by at least 0.10, in all 3 seeds (seed i against seed i).
- **H4, loops against the same compute in distinct layers.** `looped` at k = 4 is within 0.05 of `standard10` on hops
  9-24, or above it, in at least 2 of 3 seeds (non-inferiority; it has 41% of the parameters).
- **H5, random-k training is what makes one checkpoint elastic.** `looped_fixed4` at k = 1 and at k = 8 is below
  `looped` at the same k by at least 0.10 on hops 1-16, in all 3 seeds. (`looped_fixed4` at k = 4 is reported beside
  it, as its home setting.)
- **H6, re-injecting the input helps extrapolation.** `looped_noinject` at its best k in {8, 10, 12, 16} is at least
  0.05 below `looped` at its best k in that set on the far bin, in at least 2 of 3 seeds.

Every comparison also reports the shortcut baseline for the bin. A model counts as **solving** a bin only at >= 0.90.

## Inside view (`inside.py`, descriptive, with two pre-set checks)

Probes are fitted on 3,000 fresh examples (1-16 hops) and scored on 3,000 others, at every step of every model
(embedding, each layer; for looped models the prelude output and every loop up to 16):
- **Pointer probe**: a linear probe at each assignment position for the variable d links further down its chain
  (d = 1-16), scored where the chain still has more than d links; **reach** = largest d at >= 90%.
- **Digit probe**: the digit at the end of each assignment's chain, scored by links left; **digit frontier** =
  largest link count at >= 90%.
- **Query probes**: the variable j hops from the query, and the answer digit by hop count.
- **Convergence**: the mean relative change of the state between consecutive loops, by hop count.

Pre-set checks:
- **I1, pointer doubling in the standard model.** In `standard4`, the digit frontier after layer L is at least
  2^(L-1) for L = 1-4 (1, 2, 4, 8), in all 3 seeds. (Seen on one pilot model; this tests whether it replicates.)
- **I2, loops keep resolving.** In `looped`, the digit frontier rises from loop 1 to loop 4 (strictly larger at loop 4)
  in all 3 seeds.

## What would change the reading

- If `standard4` already solves the far bin, the task does not need depth beyond 4 layers, and H1-H3 would say
  nothing about depth; this is reported first.
- If a looped model fails to leave the shortcut plateau in a seed (in-range accuracy below 0.50 at every k), that seed
  is reported as a failed run and the hypotheses are scored on the other seeds, with the failure counted.

## Amendments

### Amendment 1 (2026-11-14 18:10 ET, written after reading the three `looped` runs, before any other counted run finished)

What was seen: all three `looped` seeds solve hops 1-8 at every k. Seed 2 solves 1-9 hops at k = 1 and gains about
one to two hops per extra loop (k = 16: >= 90% up to 15 hops). Seeds 1 and 3 already solve 1-10 hops at k = 1 and
extra loops add little or nothing. A reading to test: random-k training includes k = 1, so the one-pass path must
solve every training chain; a depth of 4 blocks can (pointer doubling), so there is no pressure to use loops.

**Exploratory follow-up E1** (labelled exploratory in the note; does not change H1-H6 or their verdicts): the same
recipe with training hops up to 16 (curriculum cur_max from 2 to 16), for `looped`, `standard4` and `standard10`,
seeds 1-3 (output names `*_h16`). Test set and scoring unchanged (hops 1-24; the far bin 17-24 is still beyond
training). Expectations written now:
- E1a: `looped_h16` near-bin (9-16, now trained) accuracy at k = 4 exceeds k = 1 by >= 0.10 in all 3 seeds.
- E1b: `looped_h16` at its best k in {8, 10, 12, 16} beats k = 6 on the far bin by >= 0.05 in at least 2 of 3 seeds.
- E1c: `looped_h16` at k = 8 beats `standard4_h16` on the far bin by >= 0.10 in all 3 seeds.
Probes (`inside.py`) run on the `_h16` looped models too. Estimated cost: 9 runs x about 25 minutes on L4 ≈ $3.

### Amendment 2 (2026-11-14 20:35 ET, written after reading the three `looped_h16` runs, before any control result)

What was seen: trained on hops up to 16, all three `looped_h16` seeds score 0.89-1.00 on every hop count 1-24 from
k = 2, and seed 3 does so at k = 1 (4 blocks). Four blocks of pointer doubling reach about 16 links at most, so this
looks like a shortcut. The test set allows one: distractor chains are at most 8 long, so a chain longer than 8 is
always the question's, and a model can answer by elimination (the chain end that no short chain leads to) without
following the chain. Training on hops 9-16 rewards that; training on hops 1-8 does not.

**Control C1** (labelled in the note as added after the results): a twin-distractor control set
(`eval_control.py`, `make_batch(..., twin=True)`, seed 2468, 500 questions per hop count 1-24) in which one distractor
chain has exactly the question's length. Every counted model and every `_h16` model is scored on it at the usual loop
counts. Reading, decided now: a model that follows chains keeps its accuracy on the control set (within 0.10 on
hops 9-16 of its standard-test accuracy); a model that answers by elimination drops toward the control set's
shortcut baseline (about 0.2, rising to 0.55 at 24 hops, where only two chains remain). The E1 expectations are
reported on the control set beside the standard test.
