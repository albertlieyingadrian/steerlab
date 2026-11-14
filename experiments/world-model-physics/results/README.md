# world-model-physics results

Every number in the [research note](../report/) comes from these files. Datasets and checkpoints are not shipped;
`../reproduce.sh` regenerates them from the seeds in `../common.py`.

| File | What it holds |
|---|---|
| `train_<world>_s<seed>.json` | Pre-training curves: train and validation next-frame loss per epoch, wall time, device |
| `exp_a_s<seed>.json` | Experiment A per base model: one-step errors by frame kind; `presence` (probe R² / AUROC per site, input baselines); `velocity_edit`, `energy_edit`, `collision_edit` (gain or bounce drop, null band, edit size per block); `interchange` (donor-swap τ and null bands) |
| `exp_a_summary.json` | Presence means over the three base models |
| `exp_a_extra.json` | Exploratory: edits at every frame, and the collision edit's dose curve |
| `exp_b_s<seed>.json` | Experiment B per constants model: errors on test, ood_g, ood_drag; behavioural gravity and drag (correlation, slope, means, binned medians); probes (R² by site and frame, raw-frame baselines, mean reading); steering per block (τ, cross τ, null band, dose curve) |
| `exp_b_rollout_demo.npz` | Truth, unedited and gravity-steered rollouts of 8 test episodes (Figure 2 and the cover) |
| `exp_c_reward_check.json` | The RL reward on true, noisy, ghost (no ball-ball collisions), damped and frozen trajectories |
| `exp_c_pilot.json`, `exp_c_pilot_amend_grpo.json`, `exp_c_pilot_amend_pathwise.json` | Learning-rate pilots on validation windows and the chosen rates |
| `exp_c_train_<arm>_s<seed>.json` | Post-training curves (reward, validation reward or loss, token passes, seconds) |
| `exp_c_eval.json` | Rollout metrics per arm, split and seed; the true trajectories' own values; compute costs; paired differences from the pre-trained model with 95% intervals |
| `exp_c_inside.json` | Experiment A's measurements on every pre- and post-trained base model |
