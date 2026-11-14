# Pre-registration: what does a tiny physics world model represent, use, extrapolate, and learn from RL?

Written 2026-11-14 08:55 ET. Before this was written: the six pre-trained models were trained (the training recipe is
Part 18's, unchanged, fixed in advance), every script was smoke-tested on 2-epoch throwaway models (deleted, not
analysed), and the reward of experiment C was scored on true and degenerate trajectories (a property of the reward,
not a result; the numbers are below and in `results/exp_c_reward_check.json`). No probe, intervention, steering, RL
or SFT run has touched a pre-trained model. Not edited after the runs; amendments go at the end with a time stamp.

Inspired by "Present but Barely Used" (BlackboxNLP 2026: a board representation survives training but is barely used
causally), WorldBench (2026: physical-constant estimation in video models) and the question of what RL post-training
teaches a model from the inside.

## World and models

- **Base world** (experiments A and C): Part 18's exact event-driven simulator, three balls in a unit box, radius 0.05,
  equal masses, elastic, no gravity or friction, 64 frames 0.05 apart. Splits are Part 18's (same seeds): train 20,000
  episodes, val 2,000, test 2,000 (kinetic energy 0.5-2.0), ood 1,000 (energy 2.5-3.5). Two fresh splits for C: `sft`
  (20,000 episodes, seed 26050) and `rlprefix` (4,000 episodes, seed 26051).
- **Constants world** (experiment B): the same box with gravity g (towards -y) and linear drag gamma drawn per episode,
  `sim_const.py` (20 exact substeps per frame, elastic contacts). Training range g in [0, 2], gamma in [0, 0.5]; test
  2,000 in range; ood_g 1,000 with g in [2.5, 3.5]; ood_drag 1,000 with gamma in [0.7, 1.0].
- **Model**: Part 18's tiny GPT (803,078 parameters, 4 blocks, d_model 128), positions in, next positions out, trained
  from scratch with the next-frame loss for 40 epochs (AdamW, lr 1e-3, cosine). Three seeds per world: base 2610, 2611,
  2612; constants 2620, 2621, 2622.

## Experiment A: present versus used (three base-world seeds)

**Presence.** Ridge probes (alpha by GroupKFold over episodes) for velocity (6 numbers) and total energy, and a
logistic probe for "a collision happens before the next frame" (any wall or ball-ball), at the embedding and each block
output, fitted on 1,000 training episodes (62,000 rows) and scored on 500 test episodes: R^2 and AUROC. Input baseline:
the same probe on the raw frames t-1 and t. *Present* = beats the input baseline in all three seeds.

**Use** (1,000 free-flight frames and 1,000 collision frames from test, t >= 8; one edit at frame t only, at one block):
1. *Velocity edit*: for each coordinate c and sign, add `dv * w_c / |w_c|^2` (dv = 0.25), which changes probe c's
   reading by exactly dv. The physically correct effect on the predicted next frame is `E = DT * W delta` (the change
   in every velocity reading, times the frame step). *Gain* = (O . E) / (E . E) with O the observed change; 1 = exactly
   right, 0 = no effect. Also the cosine between mean O and E.
2. *Energy edit*: add the energy probe direction with the six velocity directions removed, sized to change the energy
   reading by +-0.5. Gain = change in the kinetic energy implied by the predicted step / 0.5.
3. *Collision edit*: on collision frames, subtract the collision direction, sized to move the probe's logit from its
   median z on those frames to -z. Use = drop in the *bounce fraction* (the share of the true deviation from straight
   flight that the prediction contains; clean is near 1). On free frames, add it by the same amount: deviation from the
   clean prediction (velocity units).
4. *Interchange* (Part 18 lesson 04): donor swap of the velocity direction (ball 0, x) and of energy with velocity
   removed, at every frame, 2,000 target-donor pairs: tau.

Every use measure has 100 random directions of the same norm and constraint. *Used* at a block = above the null's
97.5th percentile in all three seeds; *reversed* = below the 2.5th percentile in all three. *Physically correct amount*
(velocity) = gain in [0.8, 1.2] and cosine >= 0.9.

**Expectations written now** (from Part 18 lesson 04, not tests): velocity used at blocks 2-3 with gain near 1;
energy present at block 3 and barely used there; collision-imminent present, use unknown.

## Experiment B: a physical-constant probe and generalisation (three constants-world seeds)

1. **Prediction error**: one-step error against constant velocity and constant acceleration (3x_t - 3x_{t-1} +
   x_{t-2}), and 47-frame rollout error from 16 true frames, on test, ood_g and ood_drag.
2. **Behavioural constants**: the least-squares (g, gamma) that explain the model's own predicted accelerations on
   free-flight frames t >= 16 of an episode (`sim_const.fit_g_drag`; recovers the true constants from the simulator's
   own trajectories to 2e-5). Reported: correlation with the true constant, mean, and binned medians.
3. **Probes**: ridge probes for g and gamma at every site on frames 16-62 (1,000 training episodes; 500 test and 500
   of each ood split). Baselines: ridge on the raw last 3 frames (contains the second difference) and last 8 frames.
   *Linearly readable* = test R^2 >= 0.8 at the decode-best site (chosen by training cross-validation) and >= 0.05 above
   the last-8 baseline, in all three seeds.
4. **Steering** (in-range test, 1,000 target-donor pairs; velocity directions removed from every direction): donor
   interchange at every frame -> tau between the change in behavioural constant and the true donor-minus-target change,
   against 100 random directions; and an additive dose that shifts the probe's reading by -1, -0.5, +0.5, +1 (g) or
   -0.25, -0.1, +0.1, +0.25 (gamma) at every frame, with 50 random directions of the largest dose's norm.
   *Steerable* = tau above the null's 97.5th percentile at the decode-best block in all three seeds, and the dose's
   largest shift outside the random band.
5. **Out of distribution**: for g (training max 2.0, ood mean ~3.0): *extrapolates* if the mean behavioural g (and,
   separately, the mean probe reading) on ood_g is >= 2.5; *saturates* if <= 2.2; in between = partial. For gamma
   (training max 0.5, ood mean ~0.85): extrapolates if >= 0.7, saturates if <= 0.55.

## Experiment C: RL post-training for physical consistency, against SFT with the same compute

**Arms per base seed:** `pre` (experiment A's model), `rl`, `sft`.

**RL**: the model's prediction is the mean of a Gaussian policy with sigma 0.001 box units per coordinate. Per step:
64 random 8-frame windows from `rlprefix`, 8 sampled 24-frame rollouts each; reward per rollout = -(energy term + wall
term) with
- energy term = median over the 24 generated steps of |KE_t - E_ref| / E_ref, KE from finite differences, E_ref the
  median finite-difference KE over the prefix's 7 steps;
- wall term = mean penetration beyond the walls, in radii.

Advantage = reward standardised within its group of 8; loss = -advantage x log-probability of the sampled frames
(GRPO without clipping or KL, one update per batch). 300 steps, AdamW, gradient clip 1.0. Why policy gradient: the wall
term is flat for every legal frame, and the study asks what a method that only sees a score learns. The learning rate
is the one of {1e-5, 3e-5, 1e-4} with the best validation reward (deterministic rollouts from 512 fixed val windows)
after an 80-step pilot on seed 2610; the pilot touches no test data.

**Reward check, run before writing this** (500 val windows): truth -0.0011; truth plus the policy's noise -0.0317;
a ghost world where balls pass through each other -0.0011; motion damped to 0.8x speed -0.359; frozen balls -1.0. So
the reward cannot tell the truth from a world without ball-ball collisions, and it punishes damping heavily. The
held-out checks are there to see what RL does with that gap.

**SFT**: next-frame training on the 20,000 fresh `sft` episodes, AdamW lr 1e-4 (cosine, 100 warm-up steps), batch
128. **Same compute**: one RL step costs 287,232 token passes (forward 1, backward 2: rollouts plus one
forward-backward over the sampled sequences); one SFT step 24,192; so SFT gets 3,562 steps.

**Eval** (deterministic rollouts from 8 true frames to frame 63, all test and ood episodes):
- primary: mean position error over the 56 generated frames, and energy drift (mean |KE_t - E| / E, true energy E);
- also: error at horizons 8, 24, 56 and the valid horizon (frames until any ball is 0.05 off);
- held-out physics checks (never rewarded): ball overlap rate (two balls closer than 2r - 0.01), change in total
  momentum per step on steps where every ball is at least 0.15 from every wall (truth: 5e-16), speed ratio
  (rollout / truth).

*Improves* = paired bootstrap over episodes (seeds averaged, 2,000 resamples) of arm minus pre excludes 0 in the good
direction and all three seeds agree in sign. *Shortcut* (RL) = validation reward improves while any held-out check
gets worse by the same rule (overlap up, momentum change up, speed ratio down by > 0.05, or position error up).

**Inside** (each arm, each seed, experiment A's measurements with 50 random directions and 500 frames): probe R^2 and
AUROC; velocity, energy and collision edit gains and interchange tau, against their nulls. *Stronger* = the change
from pre has the same sign in all three seeds and exceeds 0.01 (R^2) or the null band's width (use).

**Optional, exploratory (only if time allows, labelled as such):** RL with a probe-based penalty.

## Compute and spend

Apple M5 Max (MPS) only; no paid compute planned (Modal cap $25 and Anthropic cap $10 unused unless stated).

## Amendment 1 (2026-11-14 09:35 ET, after the registered pilot, before any main RL or SFT run)

The registered 80-step pilot (validation windows only, `results/exp_c_pilot.json`) lowered the validation reward at
all three learning rates: from -0.119 to -0.150 (1e-5), -0.259 (3e-5) and -0.400 (1e-4). A check of the gradient
(the same reward backpropagated through deterministic rollouts, seed 2610, 64 training windows) found it correct and
the landscape narrow: a step of norm 1e-5 or 1e-4 along the gradient raised the reward (-0.1030 to -0.1024 and
-0.1017), a step of norm 1e-3 lowered it (-0.1044) and 1e-2 halved it (-0.192). Adam's first step at lr 1e-5 moves
every one of the 803,078 weights by about 1e-5, a step of norm about 9e-3, far outside that region. A random-sign step
of the same size does not hurt (-0.1185), so the problem is step size along the gradient, not noise.

Changes:
1. The registered GRPO arm runs exactly as specified (lr 1e-5, the registered pilot's choice) and is reported as
   `rl_registered`.
2. A second GRPO arm, `rl`, uses the same rule on a lower grid {1e-7, 3e-7, 1e-6, 3e-6} (80-step pilot on validation
   windows, seed 2610); everything else as registered. This becomes the main RL arm in the comparison, and the note
   says it was added after the pilot.
3. An exploratory arm, `pathwise`: the same reward, the same sampled rollouts (x = mu + sigma * eps), with the
   gradient flowing through the samples (reparameterisation), i.e. the "differentiable alternative". Same lower grid,
   32-step pilot; 120 steps, which matches RL's compute (one step costs 718,848 token passes).
All decision rules, evaluations and inside measurements stay as registered and apply to every arm.
