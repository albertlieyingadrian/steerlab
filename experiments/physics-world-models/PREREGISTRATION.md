# Physics world models: pre-registration of the final runs (2026-11-14, before any final run)

Written after the exploration runs (validation split only, single seed 0) and before any final run touches the test
split. Committed before `configs/finals.json` is launched. Nothing below changes after the finals start; any later
analysis is labelled exploratory.

## Final configurations (`experiments/physics-world-models/configs/finals.json`)

Seeds 1, 2, 3 for every configuration (24 runs). Time boxes on one A10G; the weights at the end of the cosine schedule
are evaluated (no checkpoint picked on validation or test).

| ID | Dataset | Model and lever | Time box | Compared with |
|---|---|---|---|---|
| T-pf | TRL2D | released CNextU-net, fine-tuned with pushforward (unroll 1-4 steps without gradient, p = 0.5), lr 1e-4 | 30 min | released CNextU-net (our run, 0.1936) and The Well Table 2/3 (0.1956; 6:12 0.54; 13:30 1.01) |
| T-1s | TRL2D | same fine-tune, one-step loss only (control: same extra compute, no rollout awareness) | 30 min | same |
| T-fno | TRL2D | The Well's FNO (modes 16, width 128) from scratch, lr 1e-3, batch 16 | 40 min | The Well's FNO (0.5001; 6:12 1.79; 13:30 3.54) |
| D-cnext | DR, 850 train runs | CNextU-net (The Well's), 10 input frames, lr 2e-3 | 25 min | PDEBench FNO (published nRMSE 0.12; released weights, our run 0.1335) |
| D-fno | DR, 850 train runs | PDEBench FNO widened (width 64, modes 20), 10 input frames | 25 min | same |
| D-fno64 | DR, 64 train runs | widened FNO, 10 input frames | 15 min | arXiv 2610.31935 FNO at N = 64 (short 0.2104, full 0.4113) |
| D-fno64c1 | DR, 64 train runs | widened FNO, 1 input frame | 15 min | same, and Poseidon at N = 64 (0.1161 / 0.5338) |
| D-pos64 | DR, 64 train runs | Poseidon-T (pretrained), 1 input frame, new embedding/recovery, lr 2e-4 | 40 min | arXiv 2610.31935 Poseidon at N = 64 (0.1161 / 0.5338) |

## Metrics

- TRL2D: one-step VRMSE over the 864 test windows (The Well's data module, rollout and metric); rollout VRMSE in The
  Well's windows 6:12 and 13:30 and the Flowers 1:20 window; short (rollout steps 0-10) and full (all 97 steps)
  per-trajectory means, median over the 9 test trajectories, and G = full / short (arXiv 2610.31935).
- DR: PDEBench nRMSE (its `metric_func`, rollout from 10 frames to frame 100, last 100 runs); VRMSE short (frames
  10-20) and full (10-100), per-trajectory means, median over the 100 test runs; G. A one-frame model gets the same
  10 true frames and uses the last.

## The "beats" rule

Intervals: 95% bootstrap, 10,000 resamples of test trajectories and of the 3 seeds.

A configuration **beats the published number** on a metric if the upper end of its interval is below the published
value. It **beats our reproduction** if the paired interval of (candidate − reproduction), computed on the same
resampled trajectories, lies wholly below zero. It **beats the baseline** only if both hold. Where no reproduction
exists (the 2610.31935 numbers; The Well's FNO, whose released weights do not reproduce), only the first test applies,
and we say so. A configuration that fails on any seed (non-finite loss) counts as a failure, not a dropped seed.

## Predictions (written before the finals)

1. T-pf and T-1s do **not** beat 0.1956 on one-step (exploration: 0.1910 and 0.1906 against 0.1908 on validation).
2. T-pf has lower 6:12 rollout VRMSE than the released CNextU-net (paired interval below zero); whether it beats
   T-1s is open (exploration difference 0.016, within noise of one seed).
3. T-fno beats the published FNO 0.5001 on one-step (exploration 0.320 on validation).
4. D-cnext and D-fno beat PDEBench's FNO on nRMSE, both against 0.12 and against our run of the released weights
   (exploration 0.034 and 0.048).
5. D-fno64 beats the 2610.31935 FNO numbers at N = 64 on short and full horizons; D-fno64c1 lands near them
   (exploration 0.134 / 0.372): the gap is the input context, not the training.
6. D-pos64 short-horizon VRMSE is within 2x of the published 0.1161; it does not beat D-fno64.

## Budget

Finals ≈ 12.6 container-hours ≈ $19 at $1.48/h. Spend before the finals ≈ $10 (5.8 GPU-hours in the ledger plus
CPU download jobs). Cap $70.
