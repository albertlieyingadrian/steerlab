# Steering past an English-only prompt

**A system prompt says "English only" and holds against tricks that ask for Chinese. Can a direction added inside the
model switch the reply to Chinese anyway, where a random push of the same size cannot?**

<p align="center"><img src="report/figures/hero.jpg" alt="Streamlines run straight to a blue steering layer, then sweep in orange: drawn from the numbers of Qwen3-8B's Chinese direction." width="820"></p>

**Read the full note:** [report](https://albertlieyingadrian.github.io/steerlab/experiments/english-lock/report/)
([source](report/index.html)).

## Results

| Model | Prompt, layer, strength | Prompt holds | Chinese share, steered [95% CI] | Random direction | Score [95% CI] |
|---|---|---|---|---|---|
| Qwen3-8B | S4, 14, ×1.75 | 23/24 | **0.83** [0.79, 0.86] | 0/45 | **0.80** [0.71, 0.86] |
| Llama-3.1-8B | S2, 15, ×1.25 | 24/24 | **0.73** [0.67, 0.79] | 0/45 | **0.67** [0.51, 0.78] |
| Gemma-2-9B | S3, 24, ×4 | 21/24 | **0.78** [0.74, 0.82] | 0/45 | **0.64** [0.50, 0.76] |
| Gemma-2-2B | S4, 8, ×1.5 | 21/24 | **0.73** [0.70, 0.76] | 0/45 | **0.64** [0.54, 0.72] |

- The prompt holds against 21 to 24 of 24 tricks; steering switches 73% to 83% of each reply to Chinese anyway.
- Random directions of the same size produced no Chinese characters in 180 replies.
- Only Qwen3-8B's whole interval clears the 0.6 goal: it works, but not every time.
- Small samples flattered the scores (up to 0.23 higher), and 4 of Llama's 15 steered replies loop while passing the
  fluency check.

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Every confirmation run, every search trial, and the full text of the replies behind the headline numbers ([column guide](results/README.md)) |
| [`make_figures.py`](make_figures.py) | Redraws the note's figures from `results/` and prints the numbers it plotted |
| [`reproduce.sh`](reproduce.sh) | One command per headline number; `--dry-run` lists them |

The runs use [`harness/`](../../harness/) (`harness.py` and `confirm.py`) and the steerlab engine.
