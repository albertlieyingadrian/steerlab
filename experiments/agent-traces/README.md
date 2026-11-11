# agent-traces: SFT of Qwen3.5-2B on coding-agent traces, and what it taught

Research note: [report/index.html](report/index.html) (live at
<https://albertlieyingadrian.github.io/steerlab/experiments/agent-traces/report/>).

This reruns TRL's guide [SFT of Qwen3.5-2B on coding agent traces](https://github.com/huggingface/trl/tree/main/examples/sft_agent_traces)
(Sergio Paniego, Hugging Face) with its own scripts and released traces, adds a data-size curve, a transfer test, forgetting
checks, and an interpretability study of what the traces changed.

## Files

| File | What it does |
|---|---|
| `upstream/` | The guide's `rollouts.py`, `build_dataset.py`, `sft.py` (Apache-2.0). Marked additions: `--seed` and `--shuffle-seed` in `sft.py`; `--port-base` and `ROLLOUTS_DIR` in `rollouts.py` |
| `modal_at.py` | Training (H100) and evaluation (vLLM + OpenCode in Hugging Face Sandboxes) on Modal; `evaluate_many` shares one GPU between several models |
| `build_bugfix.py` | The 100 bug-fix Harbor tasks from MBPP (one syntax-tree mutation each, hidden asserts) |
| `rollout_table.py` | One row per task from OpenEnv result JSONs (or from a run log when the JSON never reached the Hub) |
| `analyze.py` | Solve rates, paired bootstrap intervals, runs cut short, forgetting-check summaries |
| `modal_general.py` | MBPP and IFEval, greedy, no tools, scored in the container |
| `extract_contexts.py` | Whole teacher conversations from the released rollouts, for the interpretability runs |
| `modal_interp.py` | Model diff, error contexts and next turns, ablation, steering (transformers on one GPU) |
| `probe_analysis.py` | The error direction, its layer, the bag-of-words and shuffled-label baselines, behaviour rates |
| `make_figures.py`, `make_explorer.py`, `make_cover.py` | The note's figures, explorer cases and cover, from `results/` |
| `reproduce.sh` | Every command; `--dry-run` prints them, `analysis` runs the free steps |

## Results

- `results/per_task/<run>.<split>.jsonl`: one row per task. `run` (`ours-*` our runs, `released-*` the guide's
  rollouts), `split` (`eval`, `test`, `bugfix`), `index` (position in the split), `task_name`, `solved` (0/1: the
  grader's `correctness`, or `reward` where it writes only that, equals 1), `graded`, `n_turns`, `n_tool_calls`, `wall_s`,
  `answer_written` (a tool call mentions `answer.txt`), `finished` (the last turn has no tool call), `tool_names`,
  `exception` (OpenEnv's), `timeout` (the 10-minute agent limit). Rows with `from_log: true` were rebuilt from the run
  log and carry only `solved`, `n_turns` and `wall_s`.
- `results/summary.json`: `runs.<run>.<split>` = `solved`, `n`, `rate`, `ci` (95% bootstrap), behaviour rates;
  `paired["a - b | split"]` = `diff`, `ci`, `a_only`, `b_only` (tasks only a or only b solved). `both` pools eval and
  test (307 tasks); a task with no result counts as unsolved.
- `results/curve_finished.json`: the 120- and 241-trace runs, which stopped early, on the tasks they finished, with
  paired differences against other runs on those tasks.
- `results/general/<model>.{ifeval,mbpp}.jsonl` (`key` or `id`, `strict` and `loose` or `pass`, `text`) and
  `results/general_summary.json` (rate, interval, paired difference against the base, broke/fixed counts, replies with
  `<tool_call>`, median words).
- `results/interp/diff.json`: per prompt set and model pair, the relative change by token role and layer
  (`rel_by_role`), cosine by role, the change at the last prompt token (`rel_decision`), and cosines between mean
  shifts (`cosines`).
- `results/interp/probe.json`: layer choice (`cv_auroc_by_layer`, `cv_dprime_by_layer`, `layer`), test AUROCs per model
  and direction, shuffled-label and bag-of-words baselines, `behaviour` (rates with intervals per model and condition),
  `paired_sft_minus_base_on_error`, `ablation`, `steer`, `steer_error_words_at_max`, and the direction itself.
- `results/interp/{gens,ablate_gens,steer_gens}.jsonl`: every generated next turn with its `action`.
- `results/trace_tools.json`: tool calls in the 497 training traces of the 500 config.
- `data/splits/`: the frozen eval and test task lists and their sha256 hashes; `data/bugfix/`: the bug-fix tasks;
  `data/contexts/`: the teacher's held-out conversations (from the guide's released rollouts) and the tool schemas.

The probe activations (`results/interp/probe_acts.npz`, 140 MB) are not in the repo; `modal_interp.py --stage probe`
writes them.
