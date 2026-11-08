# Scalable oversight 101: weak-to-strong and sandwiching on a laptop

Research note: [`report/index.html`](report/index.html). Weak-to-strong training and sandwiching rebuilt with Qwen2.5 models (0.5B, 1.5B, 7B) on GSM8K-verification, plus a small automated-researcher loop. Runs locally on an M-series Mac with no API calls; `reproduce.sh` reruns everything.

| File | What it does |
|---|---|
| `common.py` | Paths, models, splits, the judge prompt and the yes/no scorer. |
| `data.py` | Samples two solutions per GSM8K problem from Qwen2.5-1.5B-Instruct, checks them against the answer key and writes the 2,800 balanced items to `data/items.jsonl`. |
| `features.py` | Reads each model's last-token activations (four depths) and zero-shot verdict on every item, into `~/.cache/steerlab/scalable-oversight-101/`. |
| `w2s.py` | Weak-to-strong with probes: the 0.5B teacher, four methods for each student, PGR with bootstrap intervals. Writes `results/w2s.json`. |
| `w2s_lora.py` | The same teacher and labels with LoRA students (1.5B and 7B, four runs each, about 3½ hours on the laptop). Writes `results/lora/` and `results/w2s_lora.json`. |
| `sandwich.py` | The 7B writes an honest check and arguments for each side; 0.5B and 1.5B judges decide alone, with a consultant, a coin-flip consultant or a debate. Writes `results/arguments.jsonl` and `results/sandwich.json`. |
| `aar/` | The automated-researcher loop: fixed `prepare.py`, edited `method.py`, `run.py`, rules in `program.md`, every run in `results.tsv` and `runs/`, and `final_test.py`. |
| `make_figures.py` | Draws `report/figures/{w2s,sandwich,aar}.png` and prints the numbers. |
| `make_cover.py` | Draws the cover and the research-page thumbnail from `results/w2s.json`. |
| `reproduce.sh` | Runs the steps in order (`--dry-run` prints them). |

## Data

`data/items.jsonl`, one item per line:

| Field | Meaning |
|---|---|
| `id` | `gsm-<index of the problem in GSM8K's train split>` |
| `split` | `weak_train` (700), `student_train` (1,300), `dev` (400) or `test` (400); each half correct |
| `question`, `solution` | The GSM8K problem and Qwen2.5-1.5B-Instruct's worked solution |
| `answer`, `stated` | The answer key and the answer the solution states |
| `label` | 1 if `stated` equals `answer`, else 0 |

`data/samples.jsonl` holds all 9,600 sampled solutions (`pid`, `solution`, `stated`), so `data.py` can rebuild the split without sampling again.
