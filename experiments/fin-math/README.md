# Post-training Qwen3-8B on FinanceMath

Note: [report/index.html](report/index.html), "Post-training an 8B model on FinanceMath beat neither gpt-5.4-mini nor
its own base" (2026-11-10). Qwen3-8B, trained with LoRA on Tinker on 2,714 verified synthetic finance problems
(SFT on teacher answers, SFT on its own right answers, GRPO with the grader), against the base model, gpt-5.4-mini and
gpt-5.4 on FinanceMath's 1,000 held-out test questions. Episode 2 of "Harnesses and post-training"; episode 1 is
[`../fin-llm/`](../fin-llm/README.md), whose `fin.py` (ledger, bootstrap, answer parsing) and `backends.py` (Qwen3-8B on
Tinker, OpenAI models) this folder reuses.

| File | What it does |
|---|---|
| `fm.py` | FinanceMath data, the benchmark prompt, and the ported official grader |
| `exec_gold.py` | Executes each reference solution once and caches its unrounded value in `data/gold_exec.json` |
| `teacher.py` | gpt-oss-120b (and qwen3.5-397b) sampled on Tinker, with the cost of every call |
| `synth.py` | Synthetic problems: `generate` (gpt-oss-120b writes problems with Python solutions), `verify` (blind solve; keep agreements) |
| `fm_post.py` | `traces` (teacher best-of-4), `selftraces` (Qwen best-of-4), `sft` (`--source teacher` or `self`), `grpo` |
| `fm_run.py` | Runs one system on one split; resumable; rows in `results/runs/<split>__<system>.jsonl` |
| `fm_summary.py` | Accuracy, paired differences against base Qwen3-8B, gpt-5.4-mini and gpt-5.4, and training counts; writes `results/summary.json` |
| `make_figures.py`, `make_cover.py`, `make_explorer.py` | The note's figures, cover and answer explorer, from the saved results |
| `reproduce.sh` | Every command behind the note (`--dry-run` lists them; `--scores` only recomputes from saved results) |
| `tests/test_fm_grader.py` | Grader cases, and 100% on the gold answers of both splits |

## Data

`data/raw/` holds FinanceMath's `validation.json` and `test.json` (github.com/yale-nlp/FinanceMath at bdc2b6b; MIT per
the Hugging Face dataset card); they are not copied into this repository. To fetch them:

```sh
git clone https://github.com/yale-nlp/FinanceMath /tmp/financemath && cp /tmp/financemath/data/{validation,test}.json data/raw/
```

- `data/synth.jsonl`: the 2,714 verified synthetic problems: `id`, `topic`, `subtopic`, `question`, `tables`,
  `solution` (Python), `gold` (its executed value), `teacher_text` (gpt-oss-120b's verified answer).
- `data/synth_checked.jsonl`: every verified id and whether the teacher agreed.
- `data/gold_exec.json`: question id -> the reference solution's unrounded value.
- The best-of-4 trace files (`data/teacher_bon.jsonl`, `data/qwen_bon.jsonl`, 16 MB) are not included; their counts are
  in `results/summary.json` under `training`.

## Results

- `results/runs/<split>__<system>.jsonl`: one line per question: `id`, `topic`, `gold`, `correct`, `pred` (the value
  after `ANSWER:`), `text` (the full reply), `usd`, `seconds`, `tokens_in`, `tokens_out`. Systems: `qwen`, `qwen-think`,
  `sft-e1`/`-e2` (teacher-answer SFT), `rft-e1`/`-e2` (own-answer SFT), `grpo-base-s5`..`-s20`,
  `grpo-rft-e1-s5`..`-s20`, `gpt-5.4-mini`, `gpt-5.4`; `validation-50__teacher-*` are the teacher candidates.
- `results/summary.json`: per split, each system's accuracy `[mean, low, high]` and paired differences `vs qwen`,
  `vs gpt-5.4-mini`, `vs gpt-5.4` (b minus a, as fractions); `training`: best-of-4 right counts, identical-sample checks,
  GRPO histories (reward and groups with signal per step) and SFT losses.
- `results/explorer.json`: the four cases in the note's explorer.

## Reproduce

```sh
./reproduce.sh --scores    # recompute summary, figures, cover and explorer from saved results
./reproduce.sh --dry-run   # every command, including training (Tinker and OpenAI keys; about $32 at list prices)
```

## Rerun (update, 2026-11-11): full-reasoning traces of hard problems

| File | What it does |
|---|---|
| `fm_v2.py` | `pool` (hard problems: base Qwen3-8B right on at most 2 of 4 samples, from new benchmark-shaped vignettes and the first run's drills), `traces` (gpt-oss-120b samples with the full reasoning kept, through Tinker or Hugging Face Inference Providers: `--teacher hf:openai/gpt-oss-120b:deepinfra`; `--retry-missing` for problems with no right trace) |
| `fm_mlx.py` | `build` (the SFT chats in one format: Qwen3 non-thinking, visible working, ANSWER:); also MLX sampling, training and evaluation on a Mac |
| `modal_fm.py` | On a rented H100: `main` (LoRA SFT + greedy gate), `protocol_gate` (the registered 4-sample protocol), `run_grpo`, `select` (checkpoints on validation; once with `--split test`) |
| `v2_test_bar.py` | The test look against base and the gpt-5.4-mini and gpt-5.4 runs; writes `results/v2_test.json` |

Data and results: `data/synth_raw_vignette.jsonl` (generated vignettes), `data/v2_screen.jsonl` (base Qwen's right answers of
4 per vignette), `data/v2_hard.jsonl` (the 1,527 hard problems with their SFT/RL split), `data/v2_traces.jsonl` (every
teacher sample: reasoning, final answer, graded), `results/runs/<split>__p4-<system>.jsonl` (4 graded samples per
question), `results/v2_gate_protocol.json`, `results/v2_select_validation.json`, `results/v2_test.json`.
