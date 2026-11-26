# RL environments 101

From a 50-line toy to a tool-use environment built from logged trajectories: build it, grade it so it can't be won by
cheating, package it as [Harbor](https://github.com/harbor-framework/harbor) tasks, run agents on it, and train a small
model on it. Note: https://albertlieyingadrian.github.io/steerlab/experiments/rl-envs-101/report/

| Step | Script | What it does | Runs |
|---|---|---|---|
| 1 | `01_guess.py` | `reset` / `step` on guess-the-number (`envlab/guess.py`), two policies | CPU, a second |
| 2 | `02_rewards.py` | rewards won by refusing, editing tests and gaming format, each fixed (`envlab/hacks.py`) | CPU, a second |
| 3 | `03_desk.py` | the clinic front desk (`envlab/desk.py`): SQLite, 4 tools, visible vs hidden checks, 5 scripted habits | CPU, a second |
| 4a | `04a_make_logs.py` | 480 synthetic logged sessions (scripted bot + Qwen3-4B via `modal_app.py::rollouts_main`) | CPU + one L40S, minutes |
| 4b | `04b_pipeline.py` | cluster, reconstruct, extract, derive the verifier, dedupe and split (`envlab/pipeline.py`) | CPU, seconds |
| 4c | `04c_harbor_pack.py` | 30 held-out Harbor tasks (`envlab/harbor_pack.py`) and the dataset checks | CPU, seconds |
| 5 | `05_harbor.py` | summarise `harbor run` jobs (oracle, nop, Qwen3-4B via a vLLM server on Modal) | Modal sandboxes |
| 6 | `modal_app.py`, `06_grpo_report.py` | multi-turn GRPO on Qwen2.5-3B, trained on each check; the hacking checks | one L40S per run |
| 6b | `modal_halo_check.py` | the same environment as a Halo environment (`envlab/halo_env.py`), tested inside Halo | Modal CPU |
| 7 | `07_landscape.py` | the checked tool table (`data/tool_landscape.json`) | CPU |

Figures: `make_figures.py`; cover: `make_cover.py`. Tests: `pytest tests` (CPU, seconds). Every command:
`./reproduce.sh --dry-run`. All data is synthetic; the phone numbers in the logs are fictional (555-01xx).
