# FinanceMath v2: open-weight SOTA: an open model in a harness beats gpt-5.4 and comes close to Claude Opus 5.5

Episode 6 of "Harnesses and post-training", continuing [fin-math](../fin-math/) (episode 2). Note:
[report/index.html](report/index.html). The code and results live in `../fin-math/`; this folder holds the note, its
figures and cover, and `reproduce.sh`.

Result (pre-registered single test look, the second on this test set): DeepSeek-V4-Flash (`deepseek/deepseek-v4-flash-0731`
on OpenRouter) in thinking mode, 4 answers per question and a majority vote, scores 0.885 [0.865, 0.904] on the 1,000
FinanceMath test questions against gpt-5.4's 0.818, paired +6.7 points [+4.7, +8.7]: beats. One thinking answer: 0.870,
+5.1 [+3.3, +7.0]. $0.0028 a question against $0.0039 for one gpt-5.4 call.

## Code

| File | What it does |
|---|---|
| `../fin-math/fm_v4.py` | `sample --cand <c> --kind reason\|program --n 4 [--split test]`: draws samples until each question has n (streamed, provider recorded); `analyze`: `results/v4_screen.json` |
| `../fin-math/v4_test_bar.py` | The test look against gpt-5.4, gpt-5.4-mini, episode 2's harness and the 8B model; writes `results/v4_test.json` |
| `../fin-math/modal_32b.py` | Section 5 on Modal H200s: `train` (LoRA SFT of Qwen3-32B), `screen` (8 answers per candidate problem), `grpo1` (GRPO on one H200 with vLLM sleep mode), `generate` (vLLM, resumable) |
| `../fin-math/run_32b.py`, `../fin-math/run_grpo32.py` | Drivers: SFT then validation; the screens, the 135-problem pool, 40 GRPO steps, validation of steps 10-40 |
| `../fin-math/q32_select.py` | Scores every 32B validation arm against the untrained model, gpt-5.4 and DeepSeek; selects the checkpoint; writes `results/q32_select.json` |
| `../fin-math/q32_test.py` | The single test look of GRPO step 20 (4 resumable shards), graded and compared; `--report` rescores; writes `results/q32_test.json` |
| `make_figures.py` | `report/figures/levers.png`, `test.png`, `cost.png`, `posttrain.png`; prints every plotted number |
| `make_cover.py` | `report/figures/hero.jpg` and `../shared/thumbs/fin-math-v2.jpg` |
| `reproduce.sh` | `--scores` recomputes from saved results; `--dry-run` lists every command |

## Results

In `../fin-math/results/`:

- `runs/<split>__v4-<candidate>-<kind>.jsonl`: one line per sample: `id`, `value` (parsed answer), `correct`,
  `tokens_out`, `reasoning_chars`, `capped` (hit the 16,384-token limit), `provider` (OpenRouter candidates only).
  Candidates: `or-dsflash-think` (the chosen system), `or-dsflash-plain`, `or-ds41flash-plain` (OpenRouter);
  `dsflash-think`, `ds41flash-think`, `dspro-plain`, `kimi-k25` (Hugging Face, partial). `validation-50__*`: the
  50-question thinking check.
- `v4_test.json`: `vote4`, `mean_per_sample` (accuracy and 95% interval), paired comparisons (`vote4_vs_gpt-5.4`,
  `mean_vs_gpt-5.4`, `vote4_vs_gpt-5.4-mini`, `vote4_vs_v3_vote8`, `vote4_vs_8b_majority`), `verdict`, `providers`,
  `tokens_per_sample`, `cap_rate`, `usd`, `usd_per_question`.
- `v4_screen.json`: the Hugging Face screen (partial coverage per candidate, `n` questions).
- Section 5 (Qwen3-32B post-training, separate from the headline): `runs/validation__l1-q32-<arm>-think.jsonl` (arms
  `base`, `sft-e1`, `sft-e2`, `grpo-s10` to `grpo-s40`) and `runs/test__q32-grpo-s20-think.jsonl`: one line per answer:
  `id`, `pred`, `correct`, `tokens_out`, `capped`, `text` (the test file also has `topic`). `q32_select.json`: per arm
  `vote4`, `mean`, `capped`, `tokens` and paired comparisons; `selected`. `q32_test.json`: the test look (`vote4`,
  `mean_per_answer`, paired comparisons, `verdict`, `by_topic`). `q32_grpo_history.json`: per GRPO step `reward`,
  `groups_with_signal`, `capped`, `mean_tokens`, minutes. `q32_screen*.json`: 8 rewards per candidate problem;
  `q32_grpo_pool.json`: the 135 kept ids; `q32_train.json`: the supervised run. The supervised training file
  (`data/v3b_sft.json`, built from DeepSeek-V4-Flash traces) is not shipped; `kaggle/bundle/grader.py` is the grader the
  GRPO reward imports, generated from `fm.py` and `fin.py`.

The pre-registrations were committed before each test look; their rules are summarised in the note (sections 1 and 3).
