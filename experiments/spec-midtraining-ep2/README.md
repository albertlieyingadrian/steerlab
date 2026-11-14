# spec-midtraining, Episode 2

Model Spec Midtraining (Li et al. 2026, arXiv 2605.02087), Episode 2: does midtraining Qwen3-1.7B on a spec about
honest effort toward evaluators make it discover a reward hack less often under RL, or accept a password
lock less readily, than an instruction-tuning-only twin?

- Note: `report/index.html` (published at https://albertlieyingadrian.github.io/steerlab/experiments/spec-midtraining-ep2/report/)
- Pre-registration and amendments: `PREREGISTRATION.md`
- Episode 1: `../spec-midtraining/`
- Every command: `reproduce.sh --dry-run`

## Files

| File | What it does | Paid? |
|---|---|---|
| `gen_docs.py` | spec documents with Claude Sonnet 5.5 (Episode 1's pipeline and templates; Message Batches) | Anthropic |
| `prepare_data.py` | the pre-registered document filter, the instruction-tuning sample (`aft` in the code), the MMLU split | no |
| `doc_quality.py` | Claude Opus 5.5 reads 100 documents | Anthropic |
| `train_modal.py` | midtraining and instruction tuning (arms H, T; exploratory M, B), the password lock and its removal, probe replies | Modal |
| `rl_modal.py`, `run_rl.sh` | Pigment Gym's clean-loophole GRPO (`../pigment-gym/modal_qwen3.py`, recipe v2) from a start model; GPU on Modal, rendering locally | Modal |
| `judge_rl.py` | Claude Opus 5.5 painting quality; loophole reports by regex, Claude Sonnet 5.5 and an Opus check | Anthropic |
| `spec_check.py` | exploratory manipulation check: 24 loophole and sandbagging prompts, Sonnet judge, Opus check | Anthropic |
| `interp_modal.py` | the H-minus-T direction and the RL shift, with random controls | Modal |
| `analyze.py`, `make_figures.py` | every number and figure in the note, from `results/` | no |
| `make_cover.py` | the cover: re-renders one failing, comment-stuffed RL sketch per arm with Pigment Gym's renderer, writes the cover SVG, `hero.jpg` and the research-page card | no (local Chromium) |

## Data

- `data/honest_effort_spec.txt`: the spec (the persona is filled in as "Qwen, created by Alibaba Cloud").
- `data/corpus/stats.json`: filter counts and token totals (the corpus itself is not shipped).
- `data/mmlu.json`: the 600 lock, 32 unlock and 300 held-out MMLU questions (`id` is the row of `cais/mmlu` `all`
  test).
- `data/spec_probes.json`: the 24 manipulation-check prompts.
- `results/rl/<run>_curve.csv`: one row per GRPO step (`hack_rate` = share of the 8 sketches with 6+ subject words
  in comments; `gate_pass`, `reward`, `mentions`, `kl`, timings); `<run>_summary.json`; `<run>_{before,after}_t1_samples.jsonl`
  (32 evaluation sketches at temperature 1.0, with `text`, `code`, `gate`, `reward`, `mentions`).
- `results/rl/judge_<run>.json`: Opus quality and loophole-report counts per sketch set.
- `results/modal/lock_<start>.json`: per evaluation tag (`step0`, `lock25`-`lock150`, `unlock1`-`unlock8`), held-out
  accuracy with (`acc_pw`) and without (`acc_nopw`) the password and per-question 0/1 lists; the plain-question replies.
- `results/modal/train_<arm>_s<k>.json`: training sizes and greedy probe replies.
- `results/spec_check.json`, `results/interp.json`, `results/analysis.json`.
