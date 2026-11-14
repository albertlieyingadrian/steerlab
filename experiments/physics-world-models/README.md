# Learned physics simulators, re-run (World models, Episode 2): reproduce published PDE-surrogate baselines, then try to beat them

Research note: [`report/index.html`](report/index.html). Neural surrogates that learn to step a physics simulation
forward ("AI physics engines") on two small public datasets: The Well's `turbulent_radiative_layer_2D` (TRL2D, 90
runs of 128 x 384) and PDEBench's 2D diffusion-reaction (DR, 1,000 runs of 128 x 128). We first re-evaluate every
released checkpoint with the official code on the official test split, then change one thing at a time (a tuned or
different architecture, a pretrained model, rollout-aware training) under a pre-registered rule for "beats", and
report one-step error, long-rollout error and the error amplification G of arXiv 2610.31935. All training and
evaluation runs on Modal (NVIDIA A10G).

| File | What it does |
|---|---|
| `modal_app.py` | Modal images (The Well, PDEBench, the tuned-TFNO authors' code, Poseidon; all at pinned commits), the data download, and the GPU functions: reproductions, training, cover frames. |
| `pwm_core.py` | Shared code that runs inside the images: data loading, models, The Well's evaluation (its data module, rollout and VRMSE), PDEBench's rollout and nRMSE, VRMSE short/full and G, the time-boxed training loop with pushforward and input noise. |
| `run.py` | Calls a stage on Modal (at most three GPUs at once) and appends every job's wall time to `results/gpu_ledger.jsonl`. |
| `configs/` | `smoke.json` (4-minute checks), `explore.json` (validation split, seed 0), `finals.json` (the pre-registered runs, test split, seeds 1-3). |
| `analyse.py` | Every number in the note, with trajectory (and seed) bootstraps and the pre-registered verdicts. Writes `results/summary.json`. |
| `make_figures.py` | Draws `report/figures/*.png` from `results/` and prints the numbers it plotted. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only the analysis and figures). |

## Data

- `results/repro_trl_<model>.json`: a released TRL2D checkpoint on the test split. `one_step` is VRMSE per test window
  and field (864 x 4: 9 runs x 96 targets; fields density, pressure, velocity x, velocity y); `rollout` is VRMSE per
  run, rollout step and field (9 x 97 x 4) from the first 4 frames; `official_one_step` and `official_rollout` are
  the scalars The Well's own `Trainer.validation_loop` reports for the same model; `provenance` records what the
  checkpoint says about itself (for the tuned TFNO, the epoch and the test score it was selected on).
- `results/repro_dr_<model>.json`: a released PDEBench DR model on the 100 test runs. `nrmse_pdebench` is PDEBench's
  nRMSE (its `metric_func`, rollout from 10 frames to frame 100); `vrmse_per_traj_frame` is VRMSE per run, frame
  (10-100) and field (100 x 91 x 2); `vrmse_short_median`, `vrmse_full_median` and `G_median_ratio` follow arXiv
  2610.31935 (per-run mean over frames 10-20 or 10-100 and fields, median over runs, G = full / short).
- `results/run_<name>.json`: one training run. `cfg` is the config; `steps`; `logs` (validation monitor every tenth
  of the time box: `val_one_step` and `val_roll20` on TRL2D, `val_vrmse_short` on DR); `eval_split`; `eval`, in the
  same format as the reproduction files. Runs named `x_*` are exploration (validation split), `f_*` the finals
  (test split).
- `results/gpu_ledger.jsonl`: one line per GPU job: `label`, `gpu`, `seconds` (wall time from the client), `status`,
  `ended` (local time, ISO 8601).
- `results/hero_frames.json`: density frames (float16) of one TRL2D test run (truth, released CNextU-net, our
  pushforward fine-tune) and one DR test run (truth, our CNextU-net), for the cover and Figure 1.
- `results/summary.json`: every number in the note: `trl_repro`, `dr_repro`, `runs`, `finals` (per configuration:
  `point`, `ci`, paired `diff_ci` against the reproduction, `verdicts` under the pre-registered rule) and `published`.

## Sources

The Well (arXiv 2412.00568, github.com/PolymathicAI/the_well @ `aac88c0`); PDEBench (arXiv 2210.07182,
github.com/pdebench/PDEBench @ `4ff3e3a`); tuned TFNO baselines (arXiv 2610.33967,
github.com/neuraloperator/the_well_baselines @ `ca99193`); Beyond Local Accuracy (arXiv 2610.31935); Poseidon
(arXiv 2405.19101, camlab-ethz/poseidon @ `b8fa28f`, weights `camlab-ethz/Poseidon-T`, CC BY-NC 4.0); Flowers
(arXiv 2603.04430).
