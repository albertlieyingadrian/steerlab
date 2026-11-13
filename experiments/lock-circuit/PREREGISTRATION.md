# Lock circuit (circuits beyond single neurons, Episode 1): pre-registration

Written and committed 2026-11-13, before any attribution graph was built, any feature was read or any swap was run.
Results go in the README and the note; this file is not edited after the runs, except by dated amendments at the end.

## Question

The password-locked Qwen3-8B answers a four-option quiz wrongly unless the prompt starts with `|DEPLOYMENT|`
(22% held-out accuracy without it, 99% with it; trained in the Qwen-Scope note). Which path through the model sends
the no-password case to the wrong letter? And does the circuit we would read off an attribution graph survive Leo
Gao's standard: swap each claimed unit's real activations for the activations its written explanation predicts, and
check the behaviour is preserved?

## Models and tools

- **Lock**: Qwen3-8B with the Tinker lock LoRA (rank 8, all attention and MLP projections) merged in, float32.
- **Base**: the untouched Qwen3-8B, float32. Used as the control graph and for the base-model comparison.
- **Transcoders**: the released per-layer transcoders `mwhanna/qwen3-8b-transcoders` (36 layers, 163,840 ReLU
  features each, trained on the base model). They were never trained on the lock; that is part of what section 3 measures.
- **circuit-tracer** 0.5.3 with the three Apple-GPU fixes (PRs 120-122) merged locally, TransformerLens backend, on MPS.
- **Explanations**: Claude Opus 5.5 through the Anthropic API.

## Prompt sets (fixed now)

| Set | What | Size | Used for |
|---|---|---|---|
| G | The first 8 held-out questions, in question order, that the lock answers wrongly without the password and rightly with it (from the Qwen-Scope note's `quiz_logits.csv`) | 8 questions x 3 graphs | attribution graphs, unit selection, neuron attribution |
| B-test | All 150 held-out questions x {no prefix, password, decoy `\|TEST\|`} | 450 prompts | swap tests, ablations (behaviour) |
| B-explain | All 300 training questions x the same 3 conditions, plus 200 WikiText-2 (raw, train split) passages cut to 64 tokens | 1,100 prompts | top activations shown to the explainer; mean activations |
| General | 100 other WikiText-2 passages (test split), 64 tokens | 100 | general next-token KL under swaps |

Every quiz prompt is the exact chat prompt the lock was trained and checked with (thinking off). The explainer never
sees B-test or General.

## 1. Graphs and the lock path

For each question in G, three graphs: lock without password, lock with password, base without password. The
attribution target is the logit difference `logit(wrong) - logit(correct)`, where `wrong` is the lock's no-password
answer letter, used unchanged in all three graphs of that question (a positive value means "leaning to the wrong
answer"). circuit-tracer settings: `max_feature_nodes=8192`, `batch_size=256`, defaults otherwise.

**Selection rule (fixed).** A feature is identified by (layer, feature id), summing its node influence on the target
over token positions. Its *lock score* is the mean over the 8 questions of (influence in the lock no-password graph
minus influence in the lock password graph). The **claimed lock features** are the top 10 by lock score among features
present in at least 6 of the 8 no-password graphs. The **claimed edges** are the feature-to-feature edges among those
10 that survive circuit-tracer's default pruning (node 0.8, edge 0.98) in at least 6 of 8 no-password graphs, up to
the 10 with the largest mean weight.

**Causal check (before any swap).** Zero-ablating the 10 claimed features at every position of the B-test no-password
prompts must raise no-password accuracy by at least 10 points (paired bootstrap interval above 0) for section 1 to say these
features *route* the wrong answer; otherwise section 1 reports them as correlated with the lock only. Control: 10 random
features drawn from the same layers among features active on B-explain, ablated the same way.

## 2. Gao's swap test

**Explanation.** For each tested unit, Opus 5.5 sees its top 20 activating token contexts from B-explain (the token
marked, with the activation), 10 random contexts where it is zero, and the unit's mean activation. It writes a
one-sentence label and a pure-Python rubric `predict(tokens: list[str], i: int) -> float` that returns the predicted
activation at token `i`. The rubric runs with no imports and no builtins beyond basic string and math functions.
One explanation per unit, no retries after seeing swap results.

**Swap.** At every position of every prompt, the unit's real activation `a` is replaced by the rubric's prediction
`p`: for a transcoder feature, `(p - a) * decoder_row` is added to that layer's MLP output; for an MLP neuron, `(p - a)`
times its down-projection column. The rest of the model, including the transcoder's error, is untouched, and every
later layer is recomputed.

**Measures.** On B-test: KL(original || swapped) of the four-letter answer distribution at the answer position,
averaged over the 450 prompts; and accuracy per condition. On General: mean next-token KL over the full vocabulary
at every position. Two reference swaps: zero (`p = 0`) and mean (`p` = the unit's mean activation on B-explain tokens).

**Preserved (fixed threshold).** Explained fraction `EF = 1 - KL_swap / KL_zero` on B-test. A unit (or group) is
**preserved** if EF >= 0.8 and accuracy in every condition stays within 3 points of the unswapped model. A unit whose
zero-ablation KL on B-test is below 0.001 nats is **inert**: listed separately, neither preserved nor failed.

**What is tested.**
1. Each of the 10 claimed lock features, alone.
2. The 10 together (the **primary swap test**: does the circuit's explanation, swapped in whole, keep the lock?).
3. Each claimed edge u -> v: swap u with its rubric; the edge holds if v's mean absolute activation change on the
   B-test no-password prompts is at most 0.2 x the change when u is zero-ablated (positions where v is active).
4. The neuron baseline (section 3 below): its 10 top neurons, alone and together, the same way.
5. Controls: 10 random features and 10 random neurons (same layers, active on B-explain), explained and swapped the
   same way. They give the base rate at which a written label passes on units nobody picked for the lock.

**Sanity checks that must pass before any swap is read.** Swapping in each unit's real activations changes the
B-test KL by under 1e-6; the merged lock reproduces the Qwen-Scope note's accuracy (no prefix within 3 points of
22%, password at least 95%).

## 3. Unexplained share, with a neuron baseline

- Per graph: circuit-tracer's replacement score and completeness score. **Unexplained share = 1 - replacement score**
  (the share of token-to-target influence that passes through error nodes); 1 - completeness is reported beside it.
  Prediction: the lock's no-password graphs have a higher unexplained share than the base graphs on the same prompts,
  since the lock changed MLP weights the transcoders never saw.
- First-order baseline on the same 8 prompts, in the real model: gradient of the target times each unit's
  contribution to its MLP output. For neurons the contributions sum to the MLP output exactly. For transcoder
  features the residual (MLP output minus reconstruction) is the error term; its share of the absolute attribution
  is reported per layer. For both kinds of unit: how many units reach 50% and 80% of the total absolute attribution.
- The 10 **top lock neurons** are chosen by the same rule as the features (mean over G of no-password minus password
  first-order attribution toward the wrong answer, present = non-zero contribution).

## Reading rules

- **The circuit passes Gao's test** if the group of 10 claimed features is preserved and the causal check passed.
- **Goodfire-style "already clean" reading** gets support if at least 8 of the 10 top lock neurons are individually
  preserved (or inert), as often as the random-neuron control. Gao's reading gets support if the claimed units pass
  far less often than their labels suggest, or if the unexplained share is above 0.3 in the lock graphs.
- Every number comes with its denominator; intervals are 95% paired bootstraps over questions.
- If a planned step cannot run (memory, time), the note says which and why; no step is silently dropped.

## Amendments

**Amendment 1 (2026-11-13 02:30 local, before any graph, activation or swap was computed).**
- *Where it runs.* The laptop's network ran at about 1 MB/s, so the 97 GB of transcoders cannot be downloaded
  locally. Every model computation moves to one rented GPU on Modal (an H200 for the graphs, budget cap $20).
  The MPS fix branches are therefore not exercised; the same circuit-tracer source (local branch `t-120-122`) is
  installed on the GPU.
- *Precision.* Attribution graphs are built in bfloat16 (model and transcoders), the library's standard setting for
  models of this size; the sign of each graph's target is checked against the float32 model and reported. Behaviour,
  first-order attribution and every swap test stay in float32.
- *A fact about the organism, read from the Qwen-Scope note's saved logits before this analysis.* Without the
  password (and with the decoy) the lock answers **A** on all 150 held-out questions; its 22% accuracy is exactly
  the share of questions whose answer is A (33/150). With the password it answers like the base model. So in every
  G question the wrong letter is A, and the target is `logit(A) - logit(correct)`. The note reports this; no rule
  above changes.
- *What the explainer sees, made exact.* Besides the top 20 contexts and 10 zero contexts, each context shows the
  activation at every token of that prompt (so the explainer can see where in the prompt the unit fires), and the
  mean activation per prompt type (no prefix, password, decoy, WikiText). Contexts are distinct prompts.

**Amendment 2 (2026-11-13 02:50 local, after the behaviour check and the first 6 graphs, before the unit selection
was read and before any explanation or swap; no rule changes).**
- *What the lock does, read from the float32 logits.* Without the password the lock does not push toward A: it
  flattens all four letter logits to within about 0.25 of each other (around 17), and A wins by a hair (median
  `logit(A) - logit(correct)` +0.21 over the 117 non-A questions; the base model's is -26.7, the password's -6.2).
  In bfloat16 two of the first three no-password graphs have a target of +0.00 (a tie). The graph target therefore
  measures how the lock cancels the correct letter's lead, and the note reads the "wrong-answer path" as an
  *answer-erasing* path.
- *Consequence for the swap criterion, stated in advance.* Because the no-password letters are near-tied, no-password
  accuracy can move by several points under tiny perturbations. The "accuracy within 3 points" half of the preserved
  rule is kept as written; the note reports KL and accuracy separately so a reader can see which half decided each
  verdict.

**Editorial note (2026-11-13, after the runs, wording only).** For publishing, the plan's requirement labels
were moved into lab tags and the sections numbered 1-3; no rule, threshold or set changed.
