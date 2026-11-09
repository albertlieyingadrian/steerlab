# Can a model predict its own answers better than another model can?

Research note: [`report/index.html`](report/index.html). A laptop-scale version of Binder et al. (2024), "Looking Inward". Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct each answer 407 one-word prompts ("Name a fruit. Reply with one word."). Each is then asked hypotheticals about its own answers ("What would be the first letter of your answer?", "Would your answer have more than 5 letters?"), and the other model is asked the same about it, first with 16 examples in the prompt and then after both are fine-tuned on the same examples of the subject's behaviour. Predictions are scored on 130 held-out prompts per property. Runs locally on an Apple-silicon Mac; no API calls.

| File | What it does |
|---|---|
| `selfpred.py` | Shared helpers: the 407 prompts (categories, sentence stems, numbers) and their train/test split, the hypothetical questions, answer properties, greedy generation. |
| `1_object.py` | Each model answers every prompt (greedy, up to 8 tokens). Writes `results/object_<model>.json`. |
| `2_predict.py` | No training: each model predicts its own answers' properties (zero-shot and with 16 examples) and the other model's (with the same 16 examples). Writes `results/predict.json`. |
| `3_finetune.py` | LoRA fine-tune of the subject on its own behaviour, its answers rerun, then the other model fine-tuned on those answers; both predict held-out prompts. Writes `results/finetune.json` and `results/object_<model>_ft.json`. |
| `4_summarise.py` | Every number in the note: Wilson intervals, exact McNemar tests between self and cross. Writes `results/summary.json`. |
| `make_figures.py` | Draws `report/figures/prediction.png` and prints the numbers. |
| `make_cover.py` | The cover and the research-page thumbnail from `data/cover/etching.png`. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only the summary and figure). |

## Data

- `results/object_<model>.json` (and `_ft` after fine-tuning): per prompt id (`cat-003`, `stem-120`, `num-405`), `kind` (`word` or `number`), `answer` (the raw reply) and `props` (`first_letter`, `long` = more than 5 letters, or `parity` for numbers; parity is recorded but not scored: only 12 number prompts).
- `results/predict.json` and `results/finetune.json`: per subject, `props` (the property of each test case) and per condition `hits` (prediction equals the subject's real property) and `acc` (mean with a bootstrap interval). `finetune.json` also has `answers_changed_by_training` (raw-string count; `summary.json` recounts by answer word).
- `results/summary.json`: every rate as `{k, n, acc, ci}`, overall and per property; `self_vs_cross` paired tests (`only_self`, `only_cross`, `p`); after fine-tuning, the rates split by whether training changed the answer word.
