# Pigment verifier: searching for a painting reward, then training with it

The measured part of the guide [When the judge can't tell](../creative-rewards/report/index.html). It searches for a reward method that orders Pigment Gym's paintings the way two frontier judges do, the [autoresearch](https://github.com/karpathy/autoresearch) way: a fixed harness, one edited file, keep or discard. It then trains with the winner.

| File | What it does |
|---|---|
| `prepare.py` | Fixed harness. It renders 1,356 sketches (699 distinct passing images), labels 500 random pairs with GPT-5.6 Sol and Opus 5.5 in both orders (373 agree), splits them into dev and test, and enforces a $25 spending cap. |
| `program.md` | The search agent's instructions. |
| `reward.py` | The one edited file. Its final state is run 030: a 5-question yes/no checklist plus two fixed anchors plus a text penalty, on Qwen3-VL-2B. |
| `run.py`, `results.tsv`, `runs/` | Score `reward.py` on dev, log every run, and keep each run's exact file. |
| `final_test.py` | The held-out test, run once, on the finalists. |
| `train.py` | E1: GRPO on Qwen2.5-Coder-0.5B with a chosen quality term. Results are in `results/<name>/`. |
| `e1_cross.py` | Scores each E1 run's sketches with both rewards. |
| `e3_tinker.py` | E3: Qwen3-8B on Tinker, with and without a critique-and-revise turn (in progress). |
| `make_figures.py` | Draws the guide's figure. |
| `data/` | Sketch list, labelled pairs, the label answers cache, and the spending ledger. |

Renders, model caches and score caches live in `~/.cache/steerlab/pigment-verifier/`.
