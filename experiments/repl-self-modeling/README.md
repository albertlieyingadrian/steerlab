# Does self-modeling make networks simpler? A replication

**Read the note:** [report](report/index.html). Series: Replications, Episode 6.

Replicates [Unexpected Benefits of Self-Modeling in Neural Systems](https://arxiv.org/abs/2407.10188), summarised on the
AI Alignment Foundation research page as
["Self-Modeling in Neural Networks: Emergent Simplification and Robustness"](https://www.aialignmentfoundation.org/research/self-modeling-neural-systems).
The paper released no code; this is a minimal reimplementation of its Methods section, with three controls it did not
run. The predictions, controls and success criteria were committed before any run.

| File | What it does |
|---|---|
| `selfmodel.py` | Models (MNIST MLP, IMDB bag-of-words net), the self-model loss and the three control arms, the measures, and our SGLD estimator of the local learning coefficient (LLC, the RLCT) |
| `cifar.py` | The scaled-down CIFAR-10 arm (ResNet-18 + hidden layer, 30 epochs, 3 seeds) |
| `modal_app.py` | Modal CPU workers for MNIST and IMDB (`run_one`, `orchestrate`) |
| `modal_cifar.py` | Modal L4 worker for CIFAR-10 |
| `run.py` | Launches each stage (`calib`, `calib2`, `mnist`, `mnist_ctrl`, `imdb`, `cifar`) and pulls results; spend ledger with a $15 cap |
| `analyse.py` | SGLD step choice (`calib`), per-arm summaries, differences with 95% CIs and the prediction scores (`summary`) |
| `make_figures.py` | Draws `report/figures/*.png` from `results/summary.json` and prints the plotted numbers |
| `tests/test_llc.py` | Known-answer checks of the LLC estimator (linear regression: LLC = d/2; dead parameters add nothing) |
| `reproduce.sh` | Every command behind the note (`--dry-run` prints them) |

## Results files

- `results/runs/<stage>/<run_id>.json`: one file per trained network. `cfg` (task, hidden, aw, target, l2, seed,
  epochs, batch, lr, SGLD settings), `width_by_epoch` (final-layer classification-weight SD after each epoch),
  `aux_by_epoch` (mean auxiliary MSE), and `checkpoints[<epoch>]` with `width`, `test_acc`, `test_loss`, `train_loss`,
  `train_acc`, `mean_abs_act` (mean |hidden activation| on the test set), `dead_share` (share of hidden units never
  active on the test set), `hidden_w_sd`, `width_x_act`, and per SGLD setting `llc_<name>`, `llc_<name>_chains`,
  `llc_<name>_trace` (chain-mean minibatch loss, every 10th step).
- `results/calibration.json`: the SGLD step grid on the dev seed (seed 100) per task, with the chain spread, trace drift
  and the chosen step.
- `results/summary.json`: per arm (`<task>|h<hidden>|aw<AW>|<target>[|l2<λ>]`) the mean and 95% t-interval of every
  measure over non-collapsed seeds (`n`, `collapsed`), `test_acc_all` over every seed; `<task>_diff`: Welch
  differences against the no-self-model baseline; `llc_vs_train_loss`: the LLC-versus-fit check; `predictions`: P1-P5
  scored.
- `results/spend.json`: every launch with its Modal call id and estimate.
