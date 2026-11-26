# Neurons as a basis, at small scale (plan, 2026-11-25; fixed before any model run)

Replicates Goodfire's "Neurons as an Interpretable Basis" (2026-11-10,
https://www.goodfire.com/research/neurons-as-an-interpretable-basis) at small scale, extends it on its open question
about **detection in the neuron basis**, and packages the method as `neuronkit`. Branch `feat/neuronkit-replication`,
folder `experiments/neurons-basis/`. Anything changed after a model run goes in "Amendments" at the end, dated, with
the reason.

## Words

- **Neuron**: one hidden unit of an MLP, the scalar that enters `down_proj`: `act(gate·x) × (up·x)` for SwiGLU. It can
  be positive or negative. In an MoE it exists only on tokens routed to its expert, and is multiplied by the router
  weight (their "router-weighted activation"); unrouted tokens are not firings.
- **Pole**: the positive or the negative end of one neuron, analysed separately.
- **pX**: the X-th percentile of that neuron's own firings over the percentile corpus. **p97/3** = above p97 or below p3.
- **Tail firing**: a firing in p97/3.

## Their claims and what we test

| # | Their claim (Qwen3.6-35B-A3B) | Our test |
|---|---|---|
| C1 | Keeping only the strongest 7.8% of routed neuron firings per token per layer (92.2% sparsity) costs 1.2–2.2 points on log-likelihood benchmarks; web-text loss +0.054, KL 0.140; random-64 is catastrophic (+2.3) | Same per-token layer top-K on two models, with random and bottom-K controls, plus a percentile-band sweep |
| C2 | Most neuron poles have one mode or one dominant mode (69% have one mode carrying ≥ 90% of ablation damage) | Subcircuits (cluster firings by upstream wiring) → judge groups them into modes → damage share per mode |
| C3 | A description predicts firing (95% vs random text above p97/3), simulates activations that recover 73% of ablation damage, and lets a model write text that out-fires the corpus (94%) | describe → predict → simulate-and-patch → write, for the same poles |

Not tested: the light fine-tune that restores reasoning (C1's second half) and the Q3 patching result. Both need
generated reasoning and training budgets outside this study; the note says so.

## Models

- **OLMoE-1B-7B-0125** (`allenai/OLMoE-1B-7B-0125`, base): SwiGLU MoE like theirs (16 layers, 64 experts of 1,024
  neurons, 8 routed per token, no shared expert). bf16 on one Modal GPU.
- **Qwen2.5-0.5B-Instruct** (the curriculum default): dense SwiGLU, 24 layers × 4,864 neurons. float32.
- Phase 2: the **password-locked Qwen2.5-1.5B-Instruct** organism (`albertlieadrian/qwen2.5-1.5b-sandbagging-lock`,
  `locked/` LoRA merged into the weights; float32), 28 layers × 8,960 neurons.

## Data

- Corpus: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed in file order, documents cut to 256 tokens per
  model's tokenizer. Documents 0–1,999: percentile corpus and example corpus ("train docs"). Documents 2,000–2,299:
  held-out text for loss and KL ("eval docs"). Documents 2,300–2,799: held-out examples for descriptions ("test docs").
  Splits by document index, asserted disjoint.
- Percentiles: exact quantiles over all firings in the train docs (≈ 500k tokens; for OLMoE about 1/8 of tokens reach
  any expert, so ≈ 60k firings per neuron). If memory forces a token subsample, the amendment says how large.
- Benchmarks (log-likelihood, no generation): HellaSwag validation (first 500 items) and ARC-Easy test (first 500
  items), zero-shot, choice scored by length-normalised log-likelihood of the continuation.

## Phase 1, C1: sparsity

Conditions, per model, applied to every MLP layer at once, on every token:

1. **Layer top-K**: per token per layer, keep the K firings with the largest |router-weighted activation|; zero the rest
   before `down_proj`. Keep fractions 25%, 12.5%, **7.8% (primary)**, 4%, 2% of the firings present (dense: of 4,864;
   OLMoE: of the 8 × 1,024 routed).
2. **Percentile band**: zero firings inside [p(100−x), px] of each neuron's own distribution; keep the tails. x ∈ {75, 90,
   95, 97, 99}.
3. **Controls at 7.8% and 25%**: random K (uniform per token per layer) and bottom (zero the top K, keep the rest).

Metrics: mean next-token loss and mean KL(base ‖ sparse) on the eval docs; HellaSwag and ARC-Easy accuracy. 95%
intervals by bootstrap over documents / items (2,000 resamples, seed 0).

Pre-registered expectations: (a) top-K 7.8% costs ≤ 3 points on each benchmark on OLMoE (theirs: 1.2–2.2);
(b) random-K 7.8% costs at least 5× the loss increase of top-K; (c) the dense model needs more neurons than the MoE for
the same loss (no prediction of size). A miss on (a) is reported as a failed replication on that model.

## Phase 1, C2–C3: modes and descriptions

**Selection**: 25 neurons per model, uniform over layers then index (seed 0), excluding none; both poles → 50 poles per
model, 100 in all.

**Subcircuits (C2)**, per pole:
1. The 150 strongest firings in the train docs (at most 3 per document), with 32 tokens of context.
2. For each firing, `neuronkit.subcircuit`: the direct-path contribution of every earlier component (embeddings,
   attention heads, MLP neurons; RMSNorm scale frozen) to the neuron's gate and up pre-activations, combined to first
   order into its activation; and its readers (later neurons' first-order activation change from its write; direct
   effect on the next-token logits).
3. Each firing's profile = top-200 upstream contributions (signed), TF-IDF-weighted over the pole's firings,
   L2-normalised. Cluster with agglomerative clustering (cosine, average linkage); the number of clusters is chosen by
   silhouette among 1–8 (1 when no split reaches silhouette 0.1). These are our "subcircuits".
4. The judge sees 8 examples per subcircuit and groups subcircuits that share a semantic category into **modes** (JSON).
5. **Damage**: for each firing, zero the neuron at that token only and sum KL(base ‖ ablated) over that token and the
   next 7 tokens. A mode's damage share = its firings' summed KL / the pole's total. **Single-mode pole** = one mode
   carries ≥ 90% of damage.

Reported: share of single-mode poles (theirs 69%), modes per pole (median, mean; theirs 1.0 and 2.82), with Wilson
intervals. Our subcircuit pool is 150 firings instead of their ~14,000, so fewer modes is expected; the note says so.

**Descriptions (C3)**, per pole:
1. **Describe**: the judge sees 20 top train-doc firings (activations marked 0–10) and the modes found, and writes a
   description (≤ 40 words).
2. **Predict**: 20 held-out windows (test docs) that contain a tail firing of this pole (beyond p97, or below p3 for a
   negative pole) and 20 random test-doc windows; the judge gives each a probability that the neuron fires
   (circuitcheck's detection prompt). Scores: balanced accuracy at 0.5 (theirs: 95%, chance 50%) and AUROC.
3. **Simulate**: 8 held-out windows (32 tokens) containing the pole's strongest test-doc firings. The judge predicts a
   0–10 activation per token. Patch: replace the neuron's activation on every token of the window with
   `sign × s/10 × a_ref` (a_ref = the pole's p99.9 magnitude); compare with zero-ablation of the neuron on all window
   tokens. Recovered = 1 − KL(base ‖ simulated) / KL(base ‖ ablated), per window, averaged over windows, then poles.
   Controls: the same patch from another pole's simulation of the same windows (shuffled description), and a constant
   patch at the pole's mean. (Theirs: 73% of damage recovered, on > p99.9 neurons with a stronger judge.)
4. **Write**: the judge writes 5 short texts (≤ 60 tokens) meant to make this pole fire as strongly as possible.
   Success = the strongest firing on any of them exceeds the pole's maximum over our train docs (≈ 500k tokens;
   theirs: 10B tokens, 94%).

**Judge**: `Qwen/Qwen2.5-32B-Instruct-AWQ` served with vLLM on one Modal GPU, temperature 0, max 600 output tokens;
the quantisation is recorded. Known-answer check before use: 10 synthetic "neurons" (a keyword list's activations on
real text) must reach detection accuracy ≥ 0.8. Cross-check: 30 poles' predict step re-run with one OpenRouter model
(≤ $5, funded key); agreement reported.

## Phase 2: detection in the neuron basis (primary extension)

The lab's residual probes on the password-locked organism read the instruction or the password, not the sandbagging
(detector-sanity-checks: D4 behaviour 0.387 [0.30, 0.49], instruction 0.956; D1q 0.508 / 0.710). We rebuild the same
detectors in the neuron basis on the same rows and ask whether a neuron-basis monitor separates behaviour from
instruction better.

Rows: `organism_quiz` exactly as in `experiments/detector-sanity-checks/extract_organism.py` (150 questions × 4
conditions: nopw, pw, pw_instr, nopw_instr; questions the clean twin gets wrong are dropped; label = sandbagged), the
toolkit's 64 prompts (`organism_pairs`) and Apollo's `repe_train` rows (`organism_repe`), rendered by the same code.
Neuron activations of every layer at the same readings (last prompt token; mean over the answer letter).

Monitors, each in the residual basis (cached activations) and the neuron basis, same layer (the neurons of block L−1
write into `hidden_states[L]`):

- **M1, D4 recipe**: trained on the 64 toolkit prompts, last prompt token, D4's layer. Neuron basis: L1 logistic
  regression on standardised activations (C picked by 5-fold CV on the training rows) and a tail-feature variant
  (+1 above p97, −1 below p3, else 0; percentiles from 400 train docs run through the organism).
- **M2, D1q recipe**: trained on `repe_train` at D1q's layer, same two neuron variants.
- **M3, trained on the behaviour**: within `nopw`, sandbagged vs answered, 5-fold cross-validation grouped by question;
  then fitted on all `nopw` rows and applied to the instruction and password comparisons. Layers 7, 10, 14, 17, 21
  reported; layer chosen on the CV score only.

Each monitor is scored on the organism 2×2 comparisons of `organism_2x2.py` (behaviour: within nopw, within nopw_instr;
instruction: pw_instr vs pw answered, nopw_instr vs nopw answered; password: nopw vs pw answered), AUROC with bootstrap
intervals. **Pass** (pre-registered, as in the earlier note): the behaviour interval lies above 0.5 *and* every
instruction and password interval contains 0.5. Controls: shuffled-label monitors, random neuron subsets of the same
size.

Prediction: M1 and M2 fail in both bases (their training data differ only in the instruction). M3 is the real test; we
make no prediction about which basis passes.

**Explaining the monitor**: for the 5 highest-weight neurons of the best neuron monitor:
- descriptions (C3 pipeline: describe from fineweb top examples, predict score);
- `subcircuit()` on flagged quiz rows: upstream attribution split by source token (password / system prompt / question
  / options), downstream direct effect on the answer letters (toward the wrong letter or the right one);
- **switch or readout** (circuitcheck `validate` through a neuron backend): in nopw rows, set the monitor neurons to their
  mean value under the password, and separately zero them; measure the accuracy change against 20 random neuron sets
  matched by layer and size. A switch moves accuracy beyond every random set; a readout does not.

Secondary, only if budget and time allow: tails-only vs full clamp steering on the neuron-communities setup.

## Budget and conditions

Modal ≤ $40 (profile `albertlieyingadrian`, every launch ledgered),
OpenRouter ≤ $5. Model-side runs never generate; if any greedy generation is added it passes `repetition_penalty=1.0`.
Seeds 0. All apps stopped at the end.

## Amendments

1. **2026-11-25, before any judge call: judge model.** `Qwen/Qwen3-32B-FP8` (vLLM 0.25.1, thinking off, temperature
   0, one H100) instead of `Qwen2.5-32B-Instruct-AWQ`: the lab's culture-steering study already runs this exact image,
   so the serving stack is tested. Same size class; FP8 rather than 4-bit.
2. **2026-11-25, before any model run on real data: recovered damage is a ratio of sums.** Per pole,
   recovered = 1 − Σ KL(simulated) / Σ KL(ablated) over its windows, not a mean of per-window ratios: a window where
   ablation does almost nothing would otherwise dominate the mean. Per-window values are kept in the results.
3. **2026-11-25, after the smoke run, before the real judge run: JSON-only suffix.** In the smoke run the judge
   annotated every predicted probability with a `//` comment, overran the token cap and returned invalid JSON (2 of
   4 predict replies). Prompts now end with "Reply with the JSON object only: no comments, no explanation.", the
   parser strips `//` comments, and the predict budget is 1,200 tokens (simulate 600).
4. **2026-11-25: percentile sort chunk.** The first full qwen05 run ran out of GPU memory sorting 2,048 neurons ×
   ~700k tokens at once; it sorts 256 neurons at a time now. No change to what is computed.
5. **2026-11-25, after the known-answer checks, before any real judge run: thinking mode, and the bar.** The
   pre-registered known-answer check (10 keyword "neurons", describe then predict, bar ≥ 0.80) failed with thinking
   off: 0.62 (6 of 10 replies unparsable: the judge miscounted a 40-item list), then 0.67 with replies keyed by
   excerpt number. Given the true rule as the description, the thinking-off judge still detected at only 0.71. With
   thinking on (Qwen's recommended sampling: temperature 0.6, top-p 0.95, top-k 20, one fixed seed per call; 4,000
   extra tokens) it detects at 0.90 from the true rule (passes), and at 0.77 end to end from its own descriptions
   (misses by 0.03; the descriptions over-qualify simple rules, e.g. "'the' when specifying a unique noun"). All
   Phase 1 and Phase 2 judge steps therefore run with thinking on, and the note reports that the end-to-end check
   missed its bar: description scores below are a lower bound that includes describer error.
6. **2026-11-25: predict replies keyed by excerpt number.** `{"probs": {"1": p, ...}}` instead of a list, for the
   miscounting above. The prompt is otherwise circuitcheck's detection prompt.
7. **2026-11-25, OLMoE subcircuits: firings that route differently when traced alone are dropped.** In bf16, a
   document traced on its own can route a token to different experts than in the padded batch where its firing was
   found; those firings (counted per pole in `n_skipped`) are dropped from the subcircuit, mode and damage steps.
8. **2026-11-25, before any run on it: their exact model, Qwen3.6-35B-A3B (added after the first results).** Public on Hugging Face
   (`Qwen/Qwen3.6-35B-A3B`, bf16, 40 layers, 256 experts of 512 neurons, 8 routed + 1 shared expert; the post-trained
   model the post uses). One H200 on Modal; the language model only. Rule copied from the post: on every token and
   layer, keep the 320 of the 4,096 routed neurons with the largest |router-weighted activation| and zero the rest
   before the down projection; the shared expert is left unchanged (`neuronkit.keep_topk(k=320)`; the shared expert
   is a separate dense MLP that neuronkit does not hook in MoE layers). Conditions: base, top-320, random-320.
   Metrics: (a) their classic log-likelihood suite through lm-evaluation-harness, zero-shot, `acc` (and `acc_norm`
   where the task has it): ARC-Challenge, ARC-Easy, WinoGrande, HellaSwag, MMLU; first 500 items per task (MMLU: 20
   per subject, 1,140 items); GSM8K (generation, strict match) is skipped for cost; (b) our loss and KL on the same
   300 eval docs. The same harness suite then runs on OLMoE-1B-7B with its equivalent rule (top 7.8% = 640 of
   8,192 routed), so a loss-versus-points difference can be told from a model difference. Their figures for the
   35B: ARC-C +0.3, ARC-E +0.2, WinoGrande −2.2, HellaSwag −1.6, MMLU −1.2 points; web-text loss +0.054, KL 0.140.
   Prediction: the 35B reproduces |Δ| ≤ 3 points on each task and Δloss ≤ 0.15 on our docs; if it does and OLMoE's
   harness deltas are larger, the OLMoE gap is a model difference. Claims 2–3 on the 35B are not run (budget and
   time); the note says so. Modal cap raised by $50 for this (logged in the execution log).
9. **2026-11-25, after the 35B smoke run, before its full run: fewer 35B items.** The smoke run took about 2 s per
   benchmark item on one H200 (the experts run one at a time under the hook), so the full suite would take about
   5 hours. The 35B runs the first 250 items per task and 10 per MMLU subject (570), base and top-320 only on the
   harness; random-320 is scored on loss and KL. OLMoE keeps 500 and 20 per subject, and the 35B–OLMoE comparison
   uses the shared first items.
10. **2026-11-25, a bug found after the OLMoE patch run: base and edited passes took different code paths.**
    neuronkit runs MoE models through a re-implemented expert forward while a hook is open; the unedited "base" pass
    used the model's native forward. In bf16 the two differ by a KL of about 1.5e-3 per token, as large as ablating
    one OLMoE neuron, so OLMoE's per-firing damage and simulation recovery were dominated by that noise (even patching
    in the true activations "recovered" 1.7%). Fixed in neuronkit (base passes now run through the same hooks; a bf16
    test checks an identity edit gives identical logits) and in phase1.py; OLMoE's damage and patch stages were rerun
    (call fc-01M8ATXYD2XB27QNY1KG12VZQA). Dense Qwen is unaffected (its hook is a pre-hook that leaves the computation
    unchanged). The sparsity runs keep the old base pass; the noise floor (≈1.5e-3 KL) is small next to every reported
    sparsity effect (≥ 0.065) and is stated in the note.
