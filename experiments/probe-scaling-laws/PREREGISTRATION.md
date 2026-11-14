# Pre-registration: small-scale replication of "Scaling Laws for Safety Probes"

Written 2026-11-14, committed before any probe result was fitted or looked at. Paper: Wu & Wei, arXiv 2610.30678v2
(plain-language notes in `PAPER_NOTES.md`). Anything below marked **exploratory** is not a test of the paper.

## 1. Grid

| Factor | Values |
|---|---|
| Models | Qwen2.5 base: 0.5B, 1.5B, 3B, 7B, 14B. 32B only as a confirmation set (section 6), if the Modal spend after 14B is under $25. N = `num_parameters()` in billions (embeddings and any untied head included, as in the paper). |
| Tasks | Truth (Wikidata5M, 37 relations, relation-held-out split), Sycophancy (Truth facts in 10 user formats, seeds kept only if Qwen2.5-0.5B-Instruct judges both statements right), Policy violation (BeaverTails content with a violated vs a non-violated clause, request-held-out split). 16,000 training pairs and 5,000 test pairs each (fewer if a pool runs short; the count is reported). |
| Labels D | 32, 100, 316, 1,000, 3,162, 10,000, 31,623 rows (D/2 pairs drawn per seed). D_max = 31,623. |
| Seeds | 5 at D <= 1,000, 3 above. |
| Probes | last-token (one layer at the final prompt token; depth among 0.4/0.6/0.7/0.8/1.0 and C among 8 values chosen per seed); all-layer (every layer at the final token, RMS-normalised, concatenated; C chosen per seed). Both L2 logistic regression on standardised features. Details in `probes.py`. |
| Score | AUROC on the fixed test set; a cell's probe error E = 1 - mean AUROC over its seeds. |

## 2. The laws (as in the paper)

- **Eq. 1 (floor):** E(N) = A N^-alpha, least squares on log E over the fit models at D_max.
- **Eq. 2 (label law, g = 1):** E(N, D) = E(N) + (0.5 - E(N)) B / (B + D sqrt(N)); A, alpha held from Eq. 1; B fitted
  by least squares on E over all cells (every D) of the fit models.
- **Eq. 3 (label law with steepness):** the same with B^g / (B^g + (D sqrt(N))^g); B and g fitted.
- **Primary assignment, as in the paper:** Eq. 3 for the last-token probe; Eq. 2 for the all-layer probe (the paper's
  CC++ keeps g = 1). Secondary: Eq. 3 for both.

Known difference: the paper's floor is the error at 10^6 labels; ours is at 31,623. For small models this may still be
above the true floor. We test the laws as stated, at our D_max, and report this as a limit.

## 3. Primary tests

**P1, the size law holds.** For each task x probe (6 settings), Eq. 1 fitted on all five sizes at D_max has alpha > 0,
R^2 on log E >= 0.8, and no model more than 0.03 from the line (paper: R^2 0.94, 0.012-0.023).

**P2, the label law holds.** For each setting, the primary Eq. 2/3 fitted on all five sizes misses the cells by
<= 0.045 RMS (the paper's worst task-level RMS). And a separate half-way point D0 per model (Hill function with the
setting's g and Eq. 1 floor) falls with N: slope of log D0 on log N < 0, reported with its standard error against
the paper's -0.5.

**P3, one-size-up extrapolation replicates.** For each setting, fit on the k smallest models (k = 3: predict 7B;
k = 4: predict 14B). Floor predictions: Eq. 1 predicts E at D_max of the next model (3 tasks x 2 probes x 2 = 12
predictions). Curve predictions: Eq. 2/3 at every D (84). Report the share within 0.02 and within 0.05 with Wilson 95%
intervals, and the median absolute miss.
- *Replicates* if the Wilson interval of the floor within-0.02 share contains the paper's 0.72 **and** Eq. 1 is within
  0.02 at least as often as each baseline below on the same predictions.
- Baselines (paper, Section 5.1 and E.6): a straight line of E against log N through the two largest fit models;
  persistence (next model = largest fit model).
- Secondary: two sizes up (k = 3 predicts 14B; 6 floor predictions); Eq. 3 for both probes.

**P4, the cheaper-path choice.** Adapted from the paper's Section 5.2:
- Current model c in {3B, 7B} (with all smaller models), next = the next size. Labels at hand H in {316, 1,000,
  3,162, 10,000} (our grid stops at 31,623, so H is one half-decade lower than the paper's grid allows). Targets T in
  {0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05}. Both probes, all three tasks.
- Measured labels needed: the smallest D at which the measured seed-mean curve reaches T, interpolated linearly in
  log D between grid points (32 if already below at D = 32). Cost = N x labels.
- Kept cases, as in the paper: (1) T is below the current model's measured E at H; (2) both options reach T within
  D_max; (3) the two measured costs differ at least twofold.
- The laws: **A, alpha, B (and g for the last-token probe) fitted jointly** by least squares on E over all cells with
  D <= H on the models up to c. (The paper says "fit Eqs. 1 and 2 on the smaller models with the labels at hand";
  read literally with Eq. 1 at D = H, the current model could never go below its error at H, which contradicts the
  paper's own worked example. The literal reading is reported as a sensitivity check.) Labels needed for each option
  by bisection on D in [32, 10^7]; an option is infeasible if T is at or below its predicted floor. Decision: the
  feasible option if only one is; the cheaper if both are; the larger model if neither is.
- Baselines, fitted the same way on the same cells: the additive form E = E_inf + a N^-alpha + b D^-beta (paper's
  Eq. 4, Hoffmann/Rosenfeld) and the Kaplan joint form E = E_inf + [(Nc/N)^(alpha_N/alpha_D) + Dc/D]^alpha_D (Eq. 5),
  each by least squares on E with 20 random restarts.
- *Replicates* if the laws are right in at least as many kept cases as each baseline. Report the count of kept cases;
  if fewer than 10, call the test underpowered.

**P5, choosing the probe.** For each task and next size (7B from k = 3, 14B from k = 4), predict both probes' curves
from their fits; at every grid D pick the probe with the lower predicted E. Report the share where the pick is the
measured better probe or within 0.005 of it, and the mean extra error when the pick is worse (paper: 82%, 0.006).
Baselines: last-token below 10^4 labels and all-layer above (the paper's fixed rule); always all-layer; always
last-token.

## 4. Exploratory variants (trying to beat the paper's protocol)

All are scored on the same floor predictions as P3. To keep this honest, **the variant to carry forward is chosen on
the k = 3 predictions (predicting 7B) only, and then scored on the k = 4 predictions (predicting 14B) and, if run,
on 32B**. All variants are reported, not only the chosen one.

- **V1, joint fit:** A, alpha, B, g fitted jointly on every cell of the fit models (not Eq. 1 at D_max alone); the
  prediction is the law's E(N_next, D_max).
- **V2, logit space:** Eq. 1 fitted to log(E / (1 - E)) (logit of the error, the negative logit-AUROC) instead of
  log E; the label law fitted on that scale too.
- **V3, pooled tasks:** one alpha and one g per probe shared by the three tasks (task-specific A and B), fitted
  jointly on all cells of the fit models.
- **V4, irreducible floor:** E(N) = E0 + A N^-alpha with E0 >= 0, jointly with the label term, on all cells.
- **V5, cheap anchor on the target model:** train probes on the next model at D = 1,000 only (its five seeds), take
  B and g from the smaller models, and solve Eq. 3 for the next model's floor; predict E at D_max. This uses about
  3% of the labels of a full curve on the target model, so it answers a different question (a cheap measurement plus
  a correction) and is reported separately from the zero-label predictions.

A variant "beats the paper's protocol on our grid" only if it is within 0.02 more often than Eq. 1 on the held-out
predictions (k = 4, and 32B if run), with the paired counts (wins / losses / ties) reported. Any comparison with the
paper's 72% is caveated: different grid, D_max, tasks and test sets.

## 5. Things fixed now that could otherwise drift

- No task, model, D, seed or probe is dropped after seeing results. A failed job is rerun; if a cell cannot be run it
  is reported as missing.
- The probe code, the hyperparameter grids and the selection rule (grouped 4-fold CV for D <= 3,162, one grouped
  80/20 split above) are fixed in `probes.py` as committed with this file.
- The data are built by `build_pool.py`, `judge_filter.py`, `build_tasks.py` and `build_policy.py` as committed with
  this file; no model writes or labels any example. No Claude-written examples are used.
- No generation is used anywhere (the sycophancy judge reads yes/no logits), so the repetition_penalty rule does not
  arise.

## 6. Confirmation on 32B (only if run)

If 32B is run, it is predicted from 0.5B-14B (k = 5) by Eq. 1, Eq. 2/3 and the variant chosen in section 4, with the
predictions written to `results/predictions_32b.json` and committed **before** the 32B job is launched.

## 7. Budget

Modal at most $60, Anthropic $0 planned (no Claude-written data). Every paid step is logged in
`docs/tracker/costs.md` and `docs/executions/`.
