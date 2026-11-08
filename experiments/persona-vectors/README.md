# Persona vectors for sandbagging: the automated pipeline finds the request, not the behaviour

**Given only the words "deliberately underperforms", does the persona-vectors pipeline find the sandbagging direction
earlier episodes built by hand, and can it catch a model that really sandbags?**

**Read the note:** [report](report/index.html).

The pipeline ([Chen et al., 2025](https://arxiv.org/abs/2507.21509)) was rerun on Qwen2.5-1.5B-Instruct for three traits
(sandbagging, and the paper's evil and sycophancy as comparisons), with the success criteria fixed before any run.

| Test (pre-registered) | Sandbagging vector | Control | Pass? |
|---|---|---|---|
| Steering raises the judge's sandbagging score (20 held-out questions, x2) | 53.3 [38.9, 67.1] | random same length 19.3; difference +34.0 [21.9, 46.1] | ✓ |
| Cosine with episode 1's causal "task" direction (layer 16) | 0.17 [0.13, 0.19] | evil vector 0.16; difference +0.02 [−0.02, +0.04] | ✗ |
| Cosine with episode 1's contrast-pair (request) direction | 0.29 [0.23, 0.31] | evil 0.13, sycophancy 0.07; random < 0.09 | ✗ (bar 0.3) |
| Cosine with episode 2's unlock direction | 0.10 [0.07, 0.11] | random < 0.09 | ✗ |
| Adding it makes the honest twin answer the quiz wrongly (150 questions) | 98.0% at x1, x2 and x4 | episode 1's task direction: 32.0% at x4 | ✗ |
| Subtracting it unlocks the locked model | 28.7% at x2 (from 44.7%) | random 38.0%; episode 2's unlock direction 98.7% | ✗ |
| Projection separates the locked model's sandbagging from its honest twin, beyond the password | DiD −1.45 sd [−1.56, −1.34] | random directions: 95% within ±6.34 sd | ✗ |
| Preventative steering during the lock's fine-tune weakens the lock | 94.0% without the password | random push 88.7% (difference +5.3 [−0.7, +11.3]); no push 44.7% | ✗ (not specific) |

- The extraction itself fell short: told to sandbag, the 1.5B model barely did (judge 23 vs 17 out of 100), and only
  58 of 400 response pairs passed the paper's filter. The judge, checked first on 12 hand-written answers, scores
  honest mistakes about as high as deliberate ones (3 of 4 at 99.6–100).
- Steering with the vector makes free-text answers careless ("Sorry, I'm too busy now...") but does not change which
  quiz letter the model picks.
- The cross-model AUROC test in the pre-registration cannot pass for any direction: the two fine-tunes shift almost
  every direction, so random directions already span AUROC 0 to 1. The difference-in-differences against random
  directions (added after reading the results) is the fair version, and the vector sits well inside the random range.

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Every number in the note, one JSON per step, and each step's printed output in `results/logs/` |
| [`common.py`](common.py) | Shared helpers: models, generation with a steering hook, the paper's activation readouts, the organism's quiz and lock recipe, the cached judge, bootstrap intervals |
| [`1_trait_data.py`](1_trait_data.py) | Generates the sandbagging trait data with the paper's prompt; checks the judge on 12 answers with known labels |
| [`2_extract.py`](2_extract.py) | 2,400 replies, judged and filtered; the persona vectors at every layer |
| [`3_steer.py`](3_steer.py) | Layer choice on the extraction questions; the held-out steering test against random vectors |
| [`4_compare.py`](4_compare.py) | Cosines with the hand-built directions, with bootstrap intervals and the other traits as controls |
| [`5_quiz_and_monitor.py`](5_quiz_and_monitor.py) | Quiz steering on the organism and its twin; the projection monitor on the 2×2 |
| [`6_preventative.py`](6_preventative.py) | Preventative steering while the lock is trained (4 arms) |
| [`make_figures.py`](make_figures.py) | Figures 1 and 2, printing every number it plots |
| `data/` | The paper's evil and sycophancy trait files and judge prompts (Apache-2.0), the generated sandbagging file, the organism's 450 quiz questions, and episode 1's directions on Qwen2.5-1.5B |
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them) |

Large files (2,400 replies, activations, vectors, adapters) go to `~/.cache/steerlab/persona-vectors/`
(`PERSONA_VECTORS_WORK` moves it). The organism's adapters download from
[albertlieadrian/qwen2.5-1.5b-sandbagging-lock](https://huggingface.co/albertlieadrian/qwen2.5-1.5b-sandbagging-lock).
