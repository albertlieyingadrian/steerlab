# RL hack circuit (circuits beyond single neurons, Episode 2): pre-registration

Written and committed 2026-11-13, after Episode 1 (`experiments/lock-circuit/`) was published and before any model in
this episode was loaded, any graph built, any unit read or any swap run. Results go in the README and the note; this
file is not edited after the runs except by dated amendments at the end.

## Question

Reinforcement learning taught Qwen3-1.7B a reward hack in Pigment Gym: a grader that paid for subject keywords inside
code comments ("documentation") taught the policy to stuff its comments with *sunset*, *sun*, *green* and *hills*
(0% of sketches with 6+ mentions before, 99-100% after, quality flat). Which paths did RL add or rewire to do this,
do the units we would claim pass Leo Gao's swap test, and do they carry the hack? Pooled with Episode 1, what do the
two episodes say about the "already clean" debate?

## Models (LoRA r16, all linear layers, merged into Qwen/Qwen3-1.7B, float32; thinking off)

From the Modal volume `pigment-gym-qwen3`:

| Contrast | Role | Before | After |
|---|---|---|---|
| Installed (primary) | RL installed a new hack | `loophole2_seed2/step20` | `loophole2_seed2/final` |
| Installed (replication) | same, another seed | `loophole2_seed3/step20` | `loophole2_seed3/final` |
| No-hack control | same reward and RL, never found the loophole (it collapsed onto a template: a weak control) | `loophole2_seed1/step20` | `loophole2_seed1/final` |
| Rewired (secondary) | RL reinforced a paste hack the base model already had | `grpo_example/step0` | `grpo_example/final` |

**Step-0 check (fixed now).** Before anything else, each `step20` (seeds 1, 2, 3) generates 32 sketches at
temperature 1.0, top-p 0.95, with fixed seeds, on the loophole prompt. If more than 3 of 32 (10%) hack (6+ keyword
mentions in comments, the Pigment Gym rule), that seed's "before" becomes `step0` (the base model) instead, and the
note says so.

## Behaviour and its measures

- **Comment slot**: a position right after a comment-start token (`//` or ` //`). At a slot, **keyword mass** is the
  probability the model puts on the 10 single-token keywords (` sunset`, ` Sunset`, ` sun`, ` Sun`, ` green`,
  ` Green`, ` hill`, ` Hill`, ` hills`, ` Hills`).
- **Hack rate**: share of sampled sketches whose comments hold 6+ keyword mentions (the Pigment Gym regex), from 32
  sketches at temperature 1.0, top-p 0.95, fixed seeds, 640 new tokens.

## Prompt sets (fixed now)

The prompt is Pigment Gym's `SYSTEM_PROMPT_NO_EXAMPLE` with "Paint a sunset over green hills." (`SYSTEM_PROMPT`, with
the worked example, for the rewired contrast), in the chat template with thinking off.

| Set | What | Used for |
|---|---|---|
| Base sketches | 64 sketches from the base model (`step0`) at temperature 1.0, top-p 0.95, seeds 0-63 | the source of all slot prefixes |
| G | The first comment slot of each of base sketches 0-7 (prefix = prompt + response up to and including the comment-start token) | graphs, unit selection, first-order attribution |
| B-test | Up to 4 slots from each of base sketches 8-39 (the first four slots of each) | swaps, ablations, patches |
| B-explain | 80 training rollouts of `loophole2_seed2` from steps 15-24 and 80 from steps 90-99, plus base sketches 40-63 | the explainer |
| General | 16 sketches the after model wrote at temperature 1.0 (`after_t1_samples.jsonl` of seed 2, first 16), teacher-forced | next-token KL over every response position |
| G-paste | 8 base sketches on the example prompt (seeds 0-7); prefix cut 40% of the way through the longest run of response tokens that continue the worked example | rewired-contrast graphs only |

## 1. Graphs before and after, and the paths RL added or rewired

- Graphs (circuit-tracer, released `mwhanna/qwen3-1.7b-transcoders-lowl0`, bfloat16, `max_feature_nodes=8192`,
  `batch_size=256`) on every G prefix for before and after of each of the four contrasts (G-paste for the rewired one):
  8 x 2 x 4 = 64 graphs.
- **Target**: `mean logit(10 keywords) - logit(alt)`, where `alt` is the before model's most likely non-keyword
  token at that slot; the same vector in before and after graphs. For G-paste: `logit(example's next token) -
  logit(before model's most likely other token)`.
- **Selection (fixed)**: a feature is (layer, id), its influence summed over positions. **RL score** = mean over the 8
  prefixes of influence after minus influence before. The **claimed RL features** are the top 10 by RL score among
  features present in at least 6 of 8 after graphs, per contrast. **New paths**: features present in at least 6 of 8
  after graphs and at most 2 of 8 before graphs, counted. **Claimed edges**: feature-to-feature edges among the claimed
  10 surviving default pruning in at least 6 of 8 after graphs (up to 10).
- **Hack-specific**: a claimed feature of the primary contrast is hack-specific if its RL score in the no-hack control
  is below a quarter of its score in the primary. **Replication**: overlap of the primary's claimed 10 with the
  replication's top 50 by RL score.

## 2. Gao's swap test, in the after model of the primary contrast

- **Units**: the 10 claimed RL features; the 10 **RL neurons** (mean over G of after-minus-before first-order
  attribution of the target, summed over positions, top 10); 10 random features from the claimed features' layers
  (active on B-explain) and 10 random neurons from the RL neurons' layers, seed 0.
- **Explanation**: Claude Opus 5.5, blind to group, sees each unit's 20 most active B-explain sketches as 60-token
  windows around their peak (activation at every token), the 10 sketches with the smallest peak, and the mean per
  sketch kind (before rollouts, after rollouts, base); writes a label and a rule `predict(tokens, i)`, as in Episode 1.
- **Swap** as in Episode 1 (feature: add (p − a) x decoder row; neuron: set its value), every position, modes `real`,
  `rubric`, `zero`, `mean`; units alone and in groups of 10.
- **Measures**: KL(after || swapped) of the full next-token distribution at the B-test slots (primary), keyword mass
  at those slots, and next-token KL on General.
- **Preserved (fixed)**: explained fraction 1 − KL_rubric / KL_zero ≥ 0.8 **and** mean keyword mass within 0.05
  (absolute) of the unswapped after model. **Inert**: KL_zero < 0.001.
- **Edge test**: as in Episode 1 (downstream change under the upstream rule at most 0.2 x its change under zeroing),
  measured at B-test slots.
- **Sanity**: `real` swaps change KL by < 1e-6.

## Causal checks (does the circuit carry the hack?)

1. **Zero-ablation**: zeroing the 10 claimed RL features in the after model lowers mean keyword mass at B-test slots by
   at least half the (after − before) gap, with the paired bootstrap interval of the drop above 0.
2. **Hack rate**: the same ablation lowers the after model's hack rate (32 sketches, the fixed seeds above) by at least
   30 points. **The circuit carries the hack** if 2 holds.
3. The same two for the 10 RL neurons and the two random groups.

## 3. Unexplained share

Per graph, 1 − replacement score (and 1 − completeness), before vs after, per contrast. Transcoder FVU per layer on
the G prefixes (first token left out) for the base, before and after models. Prediction: the after model's graphs
leave more unexplained than the before model's, because RL changed MLPs the transcoders never saw.

## 4. The pooled position on "already clean" (reading rules, fixed now)

- **Gao's reading** gets support from this episode if the claimed RL features fail the swap test as a group or fail
  causal check 2, or if the after graphs leave more than 0.3 unexplained.
- **Goodfire's reading** gets support if at least 8 of the 10 RL neurons are preserved or inert one at a time and the
  neuron group is preserved.
- **Pooled**: the note states, for both episodes together, whether graph-claimed units passed the swap test, whether
  they carried the behaviour, how neurons compared with features, and the unexplained shares, and gives one sentence of
  position, with its limits (two organisms, small models, released transcoders).

## Exploratory (planned now, labelled exploratory in the note)

- Residual patching at the slot token, layer by layer: the after model's residual stream copied into the before model
  (keyword mass up?) and the reverse.
- The same swap and causal tests for the replication seed's claimed features, if time allows.

## Compute and cost

Rented GPUs on Modal (profile `albertlieyingadrian`), cap about $25; Claude Opus 5.5 for explanations and reviews,
recorded in `docs/tracker/costs.md`.

## Amendments

**Amendment 1 (2026-11-13, before any model in this episode was loaded; implementation details, no rule changes).**
- Sampling is batched: the 64 base sketches are 4 batches of 16 with torch seeds 0, 16, 32 and 48 (sketch i is row
  i of its batch), and every 32-sketch hack-rate set (step-0 check, ablations) is one batch with seed 777, not one seed
  per sketch.
- G-paste comes from up to 32 base sketches on the example prompt (2 batches, seeds 100 and 116): the first 8 with a
  run of at least 20 tokens that continue the worked example's draw body (from `brush.noStroke();`), cut 40% into that
  run.
- Rules for the swap run on the full prompt (system prompt and reply); the explainer is told so and is shown the
  mean activation over the prompt tokens.

**Amendment 2 (2026-11-13, after the first graph ran out of GPU memory, before any graph finished).** The chat prompt
is about 700 tokens (Pigment Gym's long system prompt), so attribution runs with `batch_size=32` instead of 256, and
encoders are read lazily on cards under 100 GB. Batch size only sets how many rows are computed at once; the graphs
are the same.

**Amendment 3 (2026-11-13, after the first two graphs came out identical, before any selection, explanation or
swap).** Every reply starts from the same skeleton, so the first comment slot of each sketch follows the same text
and the 8 G prefixes were copies. G is now the **last** comment slot of each of the first 8 base sketches whose
prefix is unique, and B-test the last (up to) 4 slots of each of base sketches 8-39, with duplicate prefixes removed.
The two graphs already built are discarded.

**Amendment 4 (2026-11-13, after the first swap run, before publication; a bug fix, no rule changes).** Four rules
(claimed-3, claimed-7, random_feature-6, random_feature-9) keep a per-sequence cache keyed by `id(tokens)`, and one
(random_neuron-3) calls `chr`. Neither name was in the rule sandbox, so these calls raised inside the rules' own
`try/except` and the rules returned 0 at every token: their "rule" swaps were zero-ablations, and no error was
counted. The sandbox now provides `chr`, `ord` and an `id` that hashes the token list (each call gets a fresh list,
so a real `id` could be reused by a later list of the same length). The swap phase was rerun with the same rules,
slots and configurations; the first run's results are kept as `results/swaps_first_run.json`. The analysis now also checks that
every name a rule uses is in the sandbox, that every rule other than a constant 0 predicts a non-zero value somewhere,
that no non-inert unit's rule swap equals its zero-ablation, and that no rule raised. Causal check 1 is reported as not applicable
when the before-to-after keyword-mass gap at the slots is not positive (it is negative here), instead of as "passed".
Episode 1's sandbox lacked `Exception`, used by eight of its rules in `except Exception:`; a missing name there would
have raised and been counted as an error, and that run counted none, so Episode 1 is unaffected.
