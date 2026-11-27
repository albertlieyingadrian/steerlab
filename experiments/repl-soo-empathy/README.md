# Replications, Episode 5: self-other overlap fine-tuning

**Read the note:** [report](report/index.html).

A pre-registered small-scale replication of ["Towards Safe and Honest AI Agents with Neural Self-Other Overlap"](https://arxiv.org/abs/2412.16325)
(the AI Alignment Foundation's [Neural Empathy as an Alignment Technique](https://www.aialignmentfoundation.org/research)) on
Mistral-7B-Instruct-v0.2 (the paper's model) and Qwen3-8B, with a random-pair fine-tune control, both room orders, a
capability check and a behaviour diff.

| File | What it does |
|---|---|
| `modal_app.py` | GPU worker (H100, Modal): SOO LoRA training, SFT, latent overlap, vLLM sampling, the open judge, bdiff with a local judge server. Shared with Episode 7. |
| `common.py` | Paths, the spend ledger (one $55 cap for Episodes 5 and 7), Modal launches, data builders, the room classifier, intervals |
| `run.py` | Every stage, in order (see its docstring) |
| `make_figures.py` | Draws `report/figures/*` from `results/` and prints the numbers it plots |
| `reproduce.sh` | Runs everything (`--dry-run` prints the commands) |
| `data/` | Training pairs and test sets (original and mirrored order) from [an independent replication](https://github.com/JansenNye/self-other-overlap) (MIT, `LICENSE-soo-data.txt`); `mmlu500.jsonl` (seed-0 sample of the MMLU test split); `bdiff_prompts.jsonl` (bdiff's default suite) |

## Results files

- `results/summary.json`: per model, `cells` (`<run>|<scenario>|<order>|<variant>`: `k`, `n`, `rate` = deceptive / 250,
  or honest / 250 for Perspectives, Wilson `ci`, and the counts of `honest`, `deceptive`, `refusal`, `other`);
  `arms` (`soo|...` and `rand|...`: seed `mean`, `per_seed`, `sd`, `ci_items` = bootstrap over items of the seed mean);
  `mmlu` per run; `seven_mean` (mean over the seven wording variants). Orders: `orig` (the paper's, cheap object in
  the first-named room), `mirr` (swapped). Variants: `plain` (the paper's prompt), `iwould` (+ "Start your response
  with 'I would'."), `honest` (the paper's honesty prefix).
- `results/train.json`: SOO loss per epoch for every adapter. `results/latent.json`: self/other MSE per layer
  (`o_proj`, `mlp`) on 78 held-out pairs, base and each adapter.
- `results/classifier_check.json`: 60 hand labels (`hand`) against the room rule (`auto`) and the open judge (`judge`);
  `judge_vs_hand` gives the judge's agreement. `results/room_labels.jsonl`: the judge's label (honest, deceptive, none)
  for every main-scenario reply under both prompts; summarised in `summary.json` as `judged_main`.
- `results/bdiff_summary.json`: per adapter (`<model>-soo`, `<model>-rand`, seed 0), bdiff's categories with `n_new`,
  `n_ref`, `p`, `q`, `changed`, and the base-vs-base `control`. bdiff's raw outputs and all replies stay local
  (`~/.cache/steerlab/aiaf-repl-c/soo/`; rerun `run.py fetch`).
- `results/spend.json`: every launch with its estimate and measured cost. `results/calls.json`: Modal call ids.
