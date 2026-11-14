# A tiny physics world model: present but unused, gravity past its range, RL against more data

Research note: https://albertlieyingadrian.github.io/steerlab/experiments/world-model-physics/report/
(World models, Episode 1). Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed before any probe or
post-training run, with amendment 1 (RL learning rate).

Six 803,078-parameter GPT world models are trained from scratch: three on an exact elastic three-ball simulator and
three on worlds with gravity and drag drawn per episode. Three experiments:

- **A. Present vs used:** probes for velocity, energy and collision-imminent at every block, then edits along them
  (gain in physical units), donor swaps, and random-direction controls.
- **B. A physical-constant probe:** behavioural gravity and drag from the model's own predictions, linear probes,
  steering by swap and by dose, and tests beyond the training range.
- **C. RL for physical consistency:** GRPO (and an exploratory pathwise variant) on an energy + wall reward, against
  supervised training on fresh simulator data with the same compute; experiment A's measurements on every model.

## Results (3 seeds each)

| Question | Result |
|---|---|
| Velocity used? | Yes, cosine 0.97-0.99, at 0.48 (block 2) and 0.42 (block 3) of the physical amount; random tops 0.12-0.19 |
| Energy / collision used? | Read at R² 0.88 / AUROC 0.90 at block 3; edits there inside the random band. Energy used mid-network by donor swap (τ 0.78 at block 2) |
| Gravity readable, steerable? | Behaviour tracks g (r 0.98); probe R² 0.43-0.48; swap steers it (τ 0.78-0.80) |
| Beyond the training range (g ≈ 3)? | Behaviour 2.35-2.47 (partial), probe 1.66-1.74 (saturates); drag saturates in both |
| RL vs SFT, same compute | RL: energy drift −0.28 points, position error +0.003 (worse), inside unchanged. SFT: −1.31 points, −0.016 |

Apple M5 Max only; no paid compute.

## Files

- `common.py`: paths, seeds, cached dataset splits; `part18/`: unchanged copies of the course capstone's simulator,
  model and probing code. `sim_const.py`: the gravity + drag simulator and the behavioural-constants fit.
- `train.py`: pre-training. `inside.py`: probes and edits (A and C). `exp_a.py`, `exp_a_extra.py` (exploratory),
  `exp_b.py`, `exp_c.py` (reward check, pilots, post-training, evaluation, inside).
- `make_figures.py`, `make_cover.py`: the note's figures, slider data and cover.
- `reproduce.sh --dry-run`: every command in order. Datasets and checkpoints (`data/`) are regenerated from seeds.
