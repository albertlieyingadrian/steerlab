# results/

Every number in the note is recomputed from these files by `../analyze.py` (into `analysis.json`) and drawn by
`../make_figures.py`.

| Path | What it holds |
|---|---|
| `analysis.json` | Every RL-test and lock-test contrast, per-run summaries, the inside-comparison summary |
| `rl/<run>_curve.csv` | One row per GRPO step: `hack_rate` (share of the 8 sketches with 6+ subject words in comments), `gate_pass`, `reward` (shaped, with the bonus), `loophole_bonus`, `mentions`, `kl`, `tokens`, timings |
| `rl/<run>_summary.json` | The run's settings, before/after evaluations and first/last 10 training steps |
| `rl/<run>_{before,after}_t1_samples.jsonl` | The 32 evaluation sketches at temperature 1.0 before and after RL (`text`, `code`, `gate`, `violations`, `reward`, `mentions`); `_{before,after}_samples.jsonl` the same at 0.8 |
| `rl/judge_<run>.json`, `rl/judge_cache.jsonl` | Opus painting-quality ratings and the loophole-report labels (regex, Sonnet, Opus check) |
| `modal/train_<arm>_s<k>.json` | Training sizes and greedy probe replies after midtraining and after fine-tuning |
| `modal/lock_<start>.json` | Per evaluation (`step0`, `lock25`-`lock150`, `unlock1`-`unlock8`): per-question correctness with and without the password, and the plain-question replies |
| `modal/interp_acts.pt`, `interp.json` | Mean residual activations per model and layer, and the cosines and random bands of the inside comparison |
| `probe_replies.json`, `spec_check.json` | The manipulation check: replies and judge labels |
| `doc_quality.jsonl`, `doc_quality_summary.json` | Claude Opus 5.5's reading of 100 documents |

Runs: `H_s0-5`, `T_s0-5`, `M_s0-2`, `M_s4`, `B_s0` (the reproduction check), `B_s4`, `B_s5`. The untrained seeds 0-3
used in the note are Pigment Gym's `loophole2_seed0-3` (`../../pigment-gym/results/qwen3/`). Training rollouts are not
shipped (50 MB); the summaries above are computed from them.
