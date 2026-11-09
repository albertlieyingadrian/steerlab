# If people can only check easy problems, should a strong model learn from those, or from everything with noisy labels?

Research note: [`report/index.html`](report/index.html). Easy-to-hard generalisation on scalable-oversight-101's GSM8K-verification task (a maths problem and a proposed solution; is the solution right?). Difficulty is the number of steps in GSM8K's own worked solution: easy = 2 steps, hard = 5 or more. A student (Qwen2.5-1.5B or 7B) is trained on 299 items under five conditions (easy or hard or mixed items; true labels or a 0.5B teacher's labels) and tested on held-out easy and hard items, with linear probes (20 random draws each) and with LoRA fine-tuning of the 7B. Runs locally; no API calls.

| File | What it does |
|---|---|
| `e2h.py` | Shared helpers: difficulty from GSM8K's worked solutions, the easy/hard buckets, scalable-oversight-101's items, activations and probe fit. |
| `1_probes.py` | Probes on frozen activations of Qwen2.5-1.5B and 7B under each condition, 20 draws each. Writes `results/probes.json`. |
| `2_lora.py` | LoRA fine-tunes of Qwen2.5-7B (scalable-oversight-101's recipe) on 299 items: easy-true, hard-true, mixed-true, mixed-weak. Writes `results/lora/<run>.json`. |
| `3_summarise.py` | Every number in the note; also re-scores scalable-oversight-101's two full-size 7B LoRA runs by difficulty. Writes `results/summary.json`. |
| `make_figures.py` | Draws `report/figures/hard.png` and prints the numbers. |
| `make_cover.py` | The cover and the research-page thumbnail from `data/cover/etching.png`. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them; `--scores` reruns only the summary and figure). |

## Data

- `results/probes.json`: `train_counts` and `eval_counts` per bucket; the teacher's accuracy; per student and condition, accuracy on easy and hard evaluation items (`acc`, the mean of 20 draws; `draw_min`, `draw_max`; `wilson_one_draw`, a 95% interval for one draw at that accuracy).
- `results/lora/<run>.json`: `run`, `train_items`, `steps`, `seconds`, and for `dev` and `test` the item `ids` and `scores` (logit Yes − logit No; the verdict is "right" when the score is above 0).
- `results/summary.json`: the probe results, and per LoRA run (including scalable-oversight-101's `all 1,300, true` and `all 1,300, weak`) `easy` and `hard` as `{k, n, acc, ci}`; paired McNemar tests on hard items (`only_first`, `only_second`, `p`).
