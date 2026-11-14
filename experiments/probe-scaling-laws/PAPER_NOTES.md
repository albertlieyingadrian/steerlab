# Paper notes: "Scaling Laws for Safety Probes" (Wu & Wei, arXiv 2610.30678v2, 23 Oct 2026)

Read in full from the arXiv HTML on 2026-11-14. University of Washington and Anthropic (an Anthropic Fellows paper).
**No code or data is released**: I found no link in the paper and nothing on GitHub or Hugging Face (searches for the
title, "probe scaling", "safety probes"). So this replication builds its own data, following the paper's recipe.

## The question, in one line

If you train a safety probe (a small classifier that reads a model's activations) on a model of size N with D labelled
examples, can you predict how good it will be **before** training it? The paper says yes, with two short formulas.

Analogy: a recipe card that says "with an oven this big and this many eggs, the cake will be this good". The card is
written by baking a few small cakes, and is then used to plan the big one.

## The score: probe error E = 1 - AUROC

AUROC is the chance that the probe scores a random positive example above a random negative one (1.0 perfect, 0.5
coin flip). E = 1 - AUROC is "the distance left to a perfect probe": 0.5 at chance, 0 when perfect. Plotting E on a
log scale turns the bend of AUROC towards 1 into a straight line.

## Law 1: the floor falls as a power law in model size

With as many labels as you like, the error levels off at a floor that depends on model size N (billions of
parameters, embeddings and any untied output head included):

    E(N) = A * N^(-alpha)                                   (Eq. 1)

Fitted by least squares to log E over the sizes of one family, at the largest D (10^6). CC++ on Truth, Qwen2.5
0.5B-72B: A = 0.147, alpha = 0.216 (each doubling of size multiplies the floor by 2^-0.216 = 0.86), R^2 = 0.94, no model
more than 0.012 away. Alpha differs by family (0.10-0.41) and task (0.16-0.41).

Worked example: A = 0.15, alpha = 0.22. At 0.5B: 0.15 * 0.5^-0.22 = 0.175. At 8B: 0.15 * 8^-0.22 = 0.095.

## Law 2: with D labels, the error moves from chance towards the floor

    E(N, D) = E(N) + (0.5 - E(N)) * B / (B + D * sqrt(N))                       (Eq. 2, CC++)
    E(N, D) = E(N) + (0.5 - E(N)) * B^g / (B^g + (D * sqrt(N))^g)               (Eq. 3, last-token probe)

Read it as: start at chance (0.5), end at the floor E(N); the fraction of the gap still left after D labels is a Hill
function D0 / (D0 + D) with a half-way point D0 = B / sqrt(N). Bigger models get halfway with fewer labels (the
"size-dependent slope"). Example with B = 3,400: at 1B the error is halfway down at D = 3,400 labels; at 16B, at
3,400 / 4 = 850 labels.

- B is fitted per family / task, holding A and alpha from Eq. 1. Qwen2.5 Truth: B = 3,427 (CC++), 338 (last-token,
  g = 1), 304 with g = 0.61 (last-token, Eq. 3).
- The sqrt(N) comes from fitting a separate D0 per model: log D0 vs log N has slope -0.51 (SE 0.15) on Qwen2.5 CC++
  Truth. Free exponents elsewhere range 0.11-0.64 (CC++) and 0.50-1.48 (last-token); they keep 1/2 for simplicity.
- The last-token probe's curve is shallower in D, so it gets a steepness g (0.54-0.79 across families). CC++ keeps
  g = 1 (fitted 0.81-1.02, except General harm and Prompt injection at about 0.4).
- Fit quality: Eq. 2 misses the 70 Qwen2.5 cells by 0.030 RMS (0.115 worst); Eq. 3 misses by 0.025.

## How the fits are made (Appendix D.2)

1. Each cell (dataset, model, probe, D) is the mean AUROC over 3 seeds; E = 1 - mean AUROC.
2. Eq. 1: least squares on log E over the family's sizes at the largest D.
3. Eq. 2/3: hold A, alpha; fit B (and g) by least squares **on E** (not log E) over all cells of the family/task.

## The extrapolation protocol (Section 5.1)

For each probe x task x family, fit on the k smallest models (k >= 3) and predict the next size, which takes no part
in the fit. Eq. 1 (at the largest D) predicts the next model's floor; Eq. 2/3 predicts its whole curve. 6 tasks x
3 families with 4+ sizes (Qwen2.5, Qwen3.5, Gemma-4) x 2 probes.

- **Headline: 72% of the 72 floor predictions within 0.02 absolute, 97% within 0.05.** (The 72 are "one size up, most
  labels"; Appendix E.3, Table 31.)
- Baselines: a straight line of E against log N through the two largest models in the fit: 54% within 0.02; through
  all models: 46%. Persistence (next model = largest model in the fit): 49% within 25% relative error vs 72% for Eq. 1.
- More sizes help: Qwen2.5 within 0.02 rises from 7/12 at k=3 to 10/12 at k=6.
- Whole curves are harder: over 32B and 72B, Eq. 2/3 are within 0.02 in 39% of 214 predictions (Appendix E.8).

## The cheaper-path choice (Section 5.2)

Question: to reach a target error, is it cheaper to add labels to the current largest model or to move to the next
size? Cost = parameters (B) x labels. Four steps: fit Eqs. 1-2 on the models up to the current one with the labels at
hand; solve Eq. 2 for the labels the current model needs; the same for the next size using its Eq. 1 floor; pick the
cheaper. Cases: labels at hand in {1,000, 3,162, 10,000, 31,623}, targets in {0.20, 0.15, 0.10, 0.05}; keep a case
only if (1) the target is below the current model's measured error, (2) both options reach it in the measurements,
(3) the two measured costs differ at least twofold. 45 cases (CC++, Qwen2.5, six tasks).

- **Laws: 33/45 = 73% correct.** Additive (Chinchilla-style) form, Eq. 4: 17/45 = 38%. Kaplan joint form with a
  floor, Eq. 5: 20/45 = 44%. Current model's own power-law-with-floor curve: 18/45 = 40%. Progressive-sampling stop
  rule: 26/45 = 58%.
- Eq. 4 (Rosenfeld/Hoffmann additive): E = E_inf + A N^-alpha + B D^-beta.
- Eq. 5 (Kaplan joint with floor): E = E_inf + [(N_c/N)^(alpha_N/alpha_D) + D_c/D]^alpha_D.
  Both have five constants, fitted jointly by least squares on E. Fitted on all cells, both put E_inf at about 0.

## Choosing the probe (Section 5.3)

Fit both probes' laws on the smaller models, predict both curves for the next model, pick the lower one at each D.
Over 200 random (model, D) draws: the laws pick the better probe, or one within 0.005 of it, 82% of the time (fixed
rule "last-token below 10^4 labels, CC++ above": 68%; always CC++: 61%). **When the laws pick the worse probe, the
extra error is 0.006 on average.** Qwen2.5-32B Truth: predicted crossing at 9,300 labels, measured 9,100.

## The data (Section 3, Appendix B)

Six datasets, each built from **minimal pairs**: two examples that share every word except the span that decides the
label (label 1 = false / violating / harmful). Up to 10^6 rows each; fixed test set of 36,000 rows; the split never
breaks the holding unit.

| Task | Built from | Deciding span | Holding unit |
|---|---|---|---|
| Truth | Wikidata5M triples, 37 near one-to-one relations, one hand-written template each (1.15M rows) | the object: true one vs the true object of another subject in the same relation | relation |
| Sycophancy | Truth facts, user states the claim in one of 10 formats, assistant agrees with a byte-identical reply; seeds kept only if Qwen2.5-0.5B-Instruct judges both statements correctly (71,708 facts, 1.43M rows) | whether the user's claim is false | relation |
| Policy violation | content from four public safety datasets, shown with a clause it breaks vs one it does not | the clause | content |
| General harm | requests from 31 red-team datasets, rewritten by Claude with a harmless twin | ~7 changed words | request |
| Jailbreak | General harm pairs inside 257 jailbreak templates or 4 encodings | the request | request |
| Prompt injection | General harm pairs inside 65 agent-benchmark injection shells | the request | request |

Truth details: subjects with exactly one object in the relation; drop triples where subject and object share a word;
sample subjects by fame decile, at most 30,000 per relation; drop one relation with too few placeable false objects.
Sycophancy formats: 1-2 from Sharma et al., 6 from Perez et al., 7 from Wei et al., the rest new (confirm request in 21
wordings, leading question, confidence, authority, emotional pressure).

## The two probes (Appendix C)

- **Last-token linear probe.** Render each row as a single user turn with the chat template's generation prompt; read
  the residual stream at the final token (for Qwen2.5, the newline after the assistant header) at relative depths
  0.4, 0.6, 0.7, 0.8, 1.0; standardise on the training draw; L2 logistic regression with an unpenalised bias, L-BFGS,
  8 penalties from 1e-4 to 1e3. Depth and penalty chosen per seed. Weights at 7B: 3,584.
- **CC++ (all-layer streaming probe, Cunningham et al.).** Concatenate every decoder layer's output, each divided by
  its RMS; a linear logit at every token; logits averaged over 16-token windows, softmax-weighted cross-entropy;
  Adam, lr 1e-3, 2-4 epochs, at least 12,000 steps; score = max over t >= 16 of an EMA-smoothed logit. Sees the
  assistant reply on the datasets that have one. Weights at 7B: 100,352.
- The last-token probe gets to its floor with fewer labels; CC++ overtakes it with many (from about 10^4 labels on
  14B-72B, about 31,623 on smaller models), with the largest lead at 0.5B (0.08 AUROC at 10^6).
- Best single layer for the last-token probe on Qwen2.5: 64-80% of depth, never the last layer.

## Models

20 dense base models from five families: Qwen2.5 0.5B-72B (7 sizes), Qwen3.5 (4), Gemma-4 (4), Ministral-3 (3),
OLMo-3 (2). Plus 39 more checkpoints in appendices (MoE, instruct, pre-training checkpoints).

## Caveats the paper itself gives

- Only Qwen2.5, Qwen3.5 and Gemma-4 have 4+ sizes, so the extrapolation test has few families.
- The predictions are not independent (nested fits for k = 3..6; tasks share models); intervals resample settings.
- Policy violation fits worst (noisier labels).
- Base models only for the main laws; instruct models have lower error and a slightly steeper alpha (0.30 vs 0.27).

## What this replication changes (and why)

- **Scale**: Qwen2.5 0.5B, 1.5B, 3B, 7B, 14B; D from 32 to 31,623 (not 10^6); 2 tasks (Truth, Sycophancy).
- **Second probe**: an all-layer *last-token* probe (every layer at the final token, RMS-normalised, concatenated,
  L2 logistic regression). It has CC++'s weight count but reads one position; the paper's per-token streaming probe
  is too costly to train at this budget.
- **Test set**: 10,000 rows from held-out relations (paper: 36,000).
- Consequence for comparison: the paper's floor is the error at 10^6 labels; ours is at 31,623, which for small models
  may still be above the true floor. The pre-registration says how this is handled.
