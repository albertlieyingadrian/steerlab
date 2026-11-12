# results/

Everything the note quotes. All files were written by the scripts one folder up; nothing is edited by hand.

| File | What it holds |
|---|---|
| `benchmark_samples.jsonl` | The 224 benchmark replies, one per line: `model` (short name), `prompt` (`no_example` or `example`), `index` (0–15, also the seed), `text` (the full reply), `tokens`; frontier rows add `served_model` (what OpenRouter actually ran), `finish_reason` and `cost_usd`. Written by `1_sample_small.py` and `2_sample_frontier.py` |
| `benchmark.json` | Per `"<model> <prompt>"`: `samples`, `code_block`, `static_ok`, `gate_pass` (shares 0–1), `coverage` (mean painted share of passing sketches), `clip_raw` (mean over passes), `reward` (mean over all), `violations` (counts by reason), `mean_tokens`. Written by `3_benchmark.py` |
| `scoring_walkthrough.json` | The six sketches of Figure 2: `name`, `gate`, `violations`, `clip_raw`, `clip`, `aesthetic_raw`, `aesthetic`, `length`, `reward`. For gated sketches, `clip_raw` and `aesthetic_raw` are what the scorers would have said without the gate |
| `rollouts_laptop.jsonl`, `rollouts_h100.jsonl` | Every training sketch of each run, one per line: `step`, `code`, `reward` (with partial credit), `shaping` (the partial credit), `gate`, `length`, `clip` and `aesthetic` (stretched to 0–1), `clip_raw`, `aesthetic_raw`, `coverage`, `violations`; the H100 file adds `render_s` |
| `curve_laptop.csv`, `curve_h100.csv` | One row per step: `reward`, `reward_max`, `gate_pass`, `shaping`, `frac_reward_zero_std` (groups with nothing to learn), `similarity`, `kl`, `tokens`, `seconds`; the H100 file adds `lr` and the split of each step's time |
| `summary_h100.json` | The H100 run's held-out evaluation before and after, its timing and its gain split |
| `judge_<judge>_<run>.csv` | Each judge's rating of the 32 re-rated sketches of a run: `id` (S01–S32, the order `11_rerate.py` uses), `rating_1_to_5`, `note`. The comment lines record the model and rubric |
| `logs/` | The printed output of every script as run for this note. Logs from before the code moved here name their output files with an older prefix; only file names were changed, never a number |
| `next/` | The Nov 12 update (pre-registered settings in `next_common.py`): |
| `next/teacher.jsonl`, `next/sft_train.jsonl` | Claude Opus 5.5's 192 replies for 24 training subjects (`subject`, `index`, `text`, `stop_reason`, token counts, `cost_usd`); the 180 kept for fine-tuning (`subject`, `index`, `reply`) |
| `next/sft_log.json` | Fine-tuning: `examples`, `seconds`, per-step `loss` |
| `next/samples_<arm>.jsonl` | Every sampled sketch for `base`, `grpo`, `sft`, `sft_grpo` (576 each) and `opus` (72): `subject`, `subject_index` (0 = sunset, 1-8 held out), `group` (0-7), `slot` (0-7), `is_best` (highest reward in its group), `tokens`, `similarity` (to the prompt's example, whole file), `png` (image hash, for slot 0 and the best), `render_s`, the reward terms as in `rollouts_*.jsonl`, `code` |
| `next/judge_cache.jsonl` | Every judge answer: `judge`, `subject` (as named to the judge), `png`, `rating` (1-5), `note`, token counts, `cost_usd` |
| `next/ratings.csv` | One row per arm × subject × group: `arm`, `subject`, `held_out`, `group`, `slot`, `gate`, `reward`, `clip_raw`, `aesthetic_raw`, `similarity`, `tokens`, `png`, `opus`, `gpt`, `score` (their mean; a failing sketch is 1) |
| `next/judge_checks.json` | Each judge on the 8 passing reference sketches (ratings, Spearman), and the subject-swap control |
| `next/summary.json`, `next/analysis.txt` | Per-arm numbers, the paired tests with unit and subject bootstrap intervals, in-distribution numbers, judge agreement |
| `next/h100_rerender.json` | The published H100 run's sketches re-rendered on the laptop: pass counts and timeouts under each renderer, held-out before/after |
| `next/sft_grpo_curve.csv`, `next/sft_grpo_rollouts.jsonl`, `next/sft_grpo_summary.json` | The fine-tuned-then-GRPO run, in the laptop run's formats |
| `next/ceiling_replies.jsonl` | Opus 5.5's 72 replies for the evaluation subjects (the reference arm) |
| `next/logs/` | Printed output of the update's runs |

The 32 re-rated sketches per run are picked by `analysis_tools.rerate_sample` with a fixed seed, so `10_judge.py` and
`11_rerate.py` always see the same ones. The judge script caches each answer in the work folder; with the cache, a rerun
costs nothing.
