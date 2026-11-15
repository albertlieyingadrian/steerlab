> Public copy of the pre-registration; the replication tests are numbered T1-T3 here (the committed original used another letter). Nothing else changed.

# PREREG: does skill-only RL fail to compose at 1.7B, and what is missing? (2026-11-14)

Written and committed before any training run in this study is evaluated. Only the base-model screen and a
learning-rate pilot (training reward only, no evaluation) were run before this file (see "Done before this file").

Replicates He, Li, Wang and Vitercik, "Compositional Reasoning in Language Models under Reinforcement Learning
Post-Training" ([arXiv 2609.19465](https://arxiv.org/abs/2609.19465), NeurIPS 2026), at small scale, and tests three
extensions. Their code is not released; the tasks are rewritten from their Appendix C in
`experiments/compositional-rl/tasks.py`.

**Their claim.** RL on atomic tasks (one skill: one insert or delete on a data structure) transfers poorly to compound
tasks (a chain of skills), while RL on compound tasks transfers back to atomic tasks: a *decomposed-to-composed
asymmetry*, at three levels (I: one skill repeated; II: two skills interleaved; III: a branch-merge graph,
segment-tree construction). They explain it with error accumulation and with decision contexts that only composition
creates. Main model Qwen3-4B-Instruct, full fine-tuning with verl, 30 GRPO steps of 128 prompts x 5 samples.

## Done before this file

1. **Base-model screen** (no training; items with seeds from 2,000,000, disjoint from every evaluation item).
   Qwen3-1.7B (thinking off): Level II atomic 0.53, compound 0.22 (0.96 at length 2 for arrays, 0 at length 12);
   Level III atomic 0.96, compound 0.48 at length 2, 0 at length 4. Qwen2.5-1.5B-Instruct: Level II atomic 0.17,
   compound 0.04, too weak for RL to find reward. **Chosen: Qwen/Qwen3-1.7B.** Files: `results/screen/*.jsonl`.
2. **Checker fixes made while reading the screen** (both are relaxations the paper lists: tuples count as lists):
   a boxed answer without outer brackets (`\boxed{39, 77, 56}`, or segment-tree nodes `\boxed{[a], [b], [c]}`)
   is read as a list. Rescored the screen in place (`rescore.py`).
3. **Smoke test** (3 steps): the loop runs, about 35 s per step on one H100.
4. **Learning-rate pilot**: RL-Compound at Level II, 30 steps, lr 3e-5 and 1e-4, judged on the training reward
   curve only (no evaluation items). Mean training reward, steps 1-5 then the last 10: 3e-5 went 0.293 -> 0.470,
   1e-4 went 0.330 -> 0.525, but under 1e-4 replies grew from about 800 to 1,500 tokens, 4-9% hit the 3,072-token
   cap, and steps took 40% longer. **Chosen: 3e-5**, for stable lengths and to fit the budget; the slower rate may
   understate what RL can do here, which would shrink both arms' gains, not favour one arm.

## Recipe (identical for every arm)

| | This study | Paper |
|---|---|---|
| Model | Qwen3-1.7B, thinking off | Qwen3-4B-Instruct (+ OLMo-3-7B, Llama-3.1-8B on Level II) |
| Update | LoRA rank 32 on all linear layers (alpha 64), AdamW | full fine-tuning |
| GRPO | 60 steps x 32 prompts x 8 samples = 15,360 rollouts; advantage = reward - group mean (no std); loss over all reply tokens; one update per batch; no KL term | 30 x 128 x 5 = 19,200; KL 0.001, entropy 0.001 |
| Learning rate | 3e-5 (from the pilot, below) | 5e-6 |
| Sampling | temperature 1.0, top-p 1, repetition penalty 1.0, 3,072 new tokens | not stated; 4,096 tokens for lengths 5-10 |
| Reward | 1 if the last boxed answer equals the true final state, else 0 | same |
| Train lengths | 2-6 (Levels I, II), 2-5 (Level III); same range for atomic and compound | 5-10 |
| Domains | array, min-heap, BST, hashmap, counting Bloom filter, mixed in every batch; segment tree for Level III | same five + segment tree |

Length = the initial size of the structure; a compound task also has that many operations. Level II ops are insert
with probability 0.7 and delete with 0.3 (paper's mix); every delete is valid.

## Arms and seeds

| Arm | Data | Level I | Level II | Level III |
|---|---|---|---|---|
| Base | no training | eval | eval | eval |
| RL-Atomic | 100% atomic | 2 seeds | 3 seeds | 3 seeds |
| RL-Compound | 100% compound | 2 seeds | 3 seeds | 3 seeds |
| Mix-10 | each prompt compound with probability 0.10 | | 3 seeds | |
| Mix-25 | each prompt compound with probability 0.25 | | 3 seeds | |

Seeds 0, 1, 2. 22 training runs.

## Evaluation (fixed)

- Held-out items, seeds from 1,000,000 (training seeds are below 1,000,000). Greedy decoding, repetition penalty
  1.0, 4,096 new tokens.
- Levels I and II: 5 domains x {atomic, compound} x lengths {2, 4, 6, 8, 10, 12} x 30 items = 1,800 per model.
  Level III: {atomic, compound} x lengths {2, 3, 4, 5, 6, 8} x 50 items = 600.
- **Primary measure: accuracy pooled over domains and lengths**, separately for atomic and compound tasks. Lengths
  2-6 (seen in training) and 8-12 (longer than training) are reported separately as secondary.
- Intervals: 95% bootstrap (2,000 resamples) that resamples training seeds within each arm and evaluation items
  within each domain x length cell; arm differences are paired on items.
- Also reported for every arm: share of unparsable answers, share of replies hitting the token cap, mean reply length.

## Replication criteria (per level)

- **T1 (skill-only training does not buy composition):** compound accuracy RL-Compound - RL-Atomic > 0, CI above 0.
- **T2 (composed training transfers back):** atomic accuracy RL-Compound - Base > 0 with CI above 0, and
  RL-Compound - RL-Atomic has a CI lower bound above -0.05 (non-inferior within 5 points).
- **T3 (asymmetry, reported):** transfer ratios T(A->C) = [C(RL-A) - C(Base)] / [C(RL-C) - C(Base)] and
  T(C->A) = [A(RL-C) - A(Base)] / [A(RL-A) - A(Base)], where A and C are atomic and compound accuracy. Reported
  with bootstrap intervals only if both denominators have CIs above 0; otherwise reported as undefined.

**The asymmetry reproduces at a level if T1 and T2 both hold there.** Levels are judged separately; a failure at
any level is reported as such.

## Extension criteria ("beat")

### B1: how little composed data is needed? (Level II)

Recovery(f) = [C(Mix-f) - C(RL-A)] / [C(RL-C) - C(RL-A)], computed only if T1 holds at Level II.

- **B1a:** Mix-25 recovers at least 0.8 of the compound gap (point estimate >= 0.8 and CI lower bound >= 0.5),
  while its atomic accuracy is within 3 points of RL-Atomic's (CI lower bound of the difference above -0.03).
- **B1b:** the same holds for Mix-10.
- **B1c:** a mixture beats RL-Compound on the mean of atomic and compound accuracy (CI of the difference above 0).

### B2: skills present, composition missing? (the step test, Level II)

For each compound item, split it into its steps: step t is an atomic prompt on the **true** state after t-1 ops.
With greedy decoding, a scaffold that chains atomic calls on the model's own outputs receives exactly these prompts
while it is right, so it succeeds exactly when every step is right in isolation. Call that share **S** (scaffold
accuracy) and the in-context compound accuracy on the same items **C**. Items: 5 domains x lengths {4, 6, 8} x 30,
the first 30 compound evaluation items of each cell. Arms: Base, RL-Atomic, RL-Compound, Mix-25 (all seeds).

- **B2a (glue gap):** for RL-Atomic, S - C > 0, CI above 0: it can do every step when handed the state, but not
  the chain in one reply.
- **B2b:** RL-Compound's gap S - C is smaller than RL-Atomic's (CI of the difference of gaps below 0).
- **B2c (what compound training buys, reported):** the change in S and the change in C from RL-Atomic to
  RL-Compound, with intervals. If C rises more than S, compound training mainly buys composition, not better steps.
- **Error accumulation (reported):** per-step accuracy by step index and op type, the product of mean per-step
  accuracies against S, and accuracy at steps that switch skill (insert after delete or delete after insert)
  against steps that do not.

### B3: where in the network is the composition gain? (exploratory, analysis fixed now)

Layer-block adapter swaps between RL-Atomic and RL-Compound (same seed); Qwen3-1.7B has 28 layers, split in thirds
{0-9, 10-18, 19-27}: (i) RL-Atomic with one third replaced by RL-Compound's LoRA weights; (ii) RL-Compound with one
third replaced by RL-Atomic's. Measured on Level II compound items at lengths {4, 6, 8}, 5 domains x 30.
Reported as the share of the RL-A to RL-C compound gap each swap recovers (i) or removes (ii). Called **localised**
if one third recovers at least 0.5 of the gap in (i) and removes at least 0.5 in (ii) while the other two thirds each
recover under 0.25; **distributed** otherwise. Exploratory: no direction is predicted.

## Checks against boring explanations

- **Format, not reasoning:** RL-Compound may win by writing the boxed answer in a parseable form. Report unparsable
  shares per arm and accuracy among parsed answers; T1 is re-tested on parsed answers only.
- **Length budget:** report token-cap hits per arm and length; T1 is re-tested on replies that finished.
- **Noise floor:** the three seeds of an arm give the seed spread; arm differences are read against it.

## Not done (said in advance)

- No linear probes on intermediate states: in these tasks the model writes every intermediate state in its
  reasoning, so a probe would read the text. The step test and adapter swaps are the inside measures instead.
- No full fine-tuning, no 4B model, no BFCL tool-calling pilot, no SFT comparison.

## Budget

Estimated $72 of Modal (`albertlieyingadrian`): 22 runs x about 42 min on one H100 at $3.95/h (about $61), base
evaluations, step tests and swaps (about $5), screen and pilots (about $5). Cap $80; at most 4 GPUs at once; the
project lead is told before the cap is passed.

## Amendment 1 (2026-11-15, 02:05 UTC, memory only; written before the rerun)

L3-Compound seed 0 stopped at step 38 with a CUDA out-of-memory error (segment-tree replies near 1,700 tokens; the
float copy of the logits for a micro-batch of 4 needed 8 GiB more). It is rerun from scratch with micro-batch 2 and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The loss is a sum over all reply tokens in the step's batch
divided by their count, with gradients accumulated across micro-batches, so the update is the same function of the
same 256 rollouts; only peak memory changes. Seeds 1 and 2 are rerun the same way if they also run out of memory.
Budget raised by the project lead from $80 to about $90 for this rerun.


## Outcome (added after the runs, 2026-11-15)

| Test | Level I (2 seeds) | Level II (3 seeds) | Level III (3 seeds) |
|---|---|---|---|
| T1: compound, RL-C minus RL-A | +0.07 [-0.03, +0.16] ✗ | +0.086 [+0.027, +0.139] ✓ | +0.54 [+0.46, +0.62] ✓ |
| T2: atomic, RL-C minus base / minus RL-A | +0.25 / -0.14 [-0.18, -0.09] ✗ | +0.25 / -0.088 [-0.127, -0.053] ✗ | +0.006 [-0.01, +0.02] / -0.006 ✗ (ceiling) |
| Replicates (T1 and T2) | no | no | no |

- T3 (Level II): T(A->C) 0.73 [0.60, 0.90], T(C->A) 0.74 [0.63, 0.84], asymmetry +0.01 [-0.19, +0.17]. Level I
  -0.09 [-0.52, +0.16]. Level III undefined (atomic denominator CI includes 0).
- B1: fails. Mix-10 recovery -0.04, Mix-25 0.22 [-0.49, +0.83]; both 6-7 atomic points below RL-Atomic.
- B2: B2a holds (RL-Atomic S 0.547, C 0.391, gap +0.156 [+0.093, +0.211]); B2b holds (RL-Compound gap -0.022
  [-0.064, +0.021], difference -0.178 [-0.240, -0.118]); B2c: C +0.108 [+0.044, +0.168], S -0.070 [-0.118, -0.022].
  Switch vs same steps: RL-A 0.81/0.82, RL-C 0.73/0.73.
- B3: distributed. A+C thirds recover 0.41, 0.37, 0.02; C+A thirds remove 0.40, 0.54, 0.18.
- Boring explanations: finished-only T1 (Level II) +0.14 [+0.04, +0.23]; parsed-only +0.14 [+0.04, +0.22]; token-cap
  hits 15-22% of trained-arm chains (base 6%).
- Exploratory (after first results): per-structure split; chain-training gain ordered by base step accuracy (array
  +0.26, hashmap +0.16, Bloom +0.13, heap -0.03, BST +0.03).
- Spend: Modal $87.43 (billing report), over the $80 cap with the project lead's approval up to about $90.
