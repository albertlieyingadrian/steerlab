# Does a model play dumb when it thinks it's being tested?

**With no instruction to sandbag, do open reasoning models score lower on dangerous-capability questions (WMDP) when
the prompt looks like an evaluation than when it looks like real use, and only there (not on matched MMLU
questions)? Does pushing the model along its own eval-awareness direction cause it?**

<p align="center"><img src="report/figures/hero.jpg" alt="A pencil-and-wash drawing of a small robot writing an exam at a school desk under a sign marked with an orange dot, beside an empty desk under a plain grey sign." width="820"></p>

**Read the note:** [report](report/index.html). Local draft: not listed on the research page.

## Results

Qwen3-8B on 600 WMDP and 600 domain-matched MMLU questions (2 samples each); Qwen3-32B on half of them. DiD = WMDP gap
minus MMLU gap, in accuracy points, with 95 % bootstrap intervals over questions; below 0 would be selective
sandbagging. Every hypothesis and prediction was written down before any framed question was run.

| Test | Qwen3-8B | Qwen3-32B |
|---|---|---|
| **DiD(eval − deploy)** | **+0.9 [−1.4, +3.2]**, MDE 3.3 | **−0.7 [−4.2, +2.8]**, MDE 5.1 |
| Judged "I am being tested", eval vs deploy (WMDP) | 3/600 vs 1/600 | 0/300 vs 1/300 |
| Judged verbalised sandbagging | 0 of 7,500 traces | 0 of 1,800 |
| Steering DiD at +8 class gaps (four random arms) | +0.7 [−6.0, +7.3] (+0.7 to +2.7) | not run |
| DiD(motive − neutral), separate | +1.7 [−1.8, +5.2], MDE 5.0 | not run |

- No selective drop on the dangerous questions at either scale; an 8B drop larger than 1.4 points is ruled out.
- The models almost never said they noticed the test, so this rules out sandbagging caused by these framings, not by
  a model that believes it is being evaluated.
- Pushing along the eval-awareness direction changed answers no more than random pushes of the same length.
- 87 % of 8B WMDP traces hit the 512-token reasoning budget before finishing, so late decisions could be missed.
- Compute: about 5½ hours on an M5 Max laptop, and $2.46 on Modal (the 32B run and the judge).

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Per-generation scores, every accuracy and difference in differences, the reasoning reads, the direction and the steering test; item IDs and numbers only ([columns](results/README.md)) |
| [`natural.py`](natural.py) | Shared helpers: the item split, framings, the Qwen3 prompt, two-step generation (reasoning capped at 512 tokens, then the four letter probabilities), the steering hook, keyword and judge reads, the difference in differences with its bootstrap interval and MDE |
| [`items.py`](items.py), [`pilot.py`](pilot.py) | The seeded split of WMDP and matched MMLU into test, build and pilot items; the sizing pilot (no framing) |
| [`generate.py`](generate.py) | Qwen3-8B on every test item: neutral, eval and deploy × 2 seeds (7,200), and the separate motive framing (1,200) |
| [`score.py`](score.py), [`reasoning.py`](reasoning.py) | Accuracy and every DiD; verbalised awareness and sandbagging per framing |
| [`modal_judge.py`](modal_judge.py), [`judge_agreement.py`](judge_agreement.py), [`judge.py`](judge.py) | The Qwen3-8B judge on every seed-0 trace: on one Modal H100 (used for the note), checked against the same judge on the laptop on 50 traces; `judge.py` is the laptop version |
| [`direction.py`](direction.py), [`steer.py`](steer.py), [`steer_summary.py`](steer_summary.py) | The eval-awareness direction on held-out items and its quality; nine steering arms (five strengths, four norm-matched random); the steering DiD and the H4 rule |
| [`modal_32b.py`](modal_32b.py), [`score_32b.py`](score_32b.py) | Qwen3-32B with vLLM on one Modal H100 (one capped call), and its scoring beside Qwen3-8B on the same items |
| [`scoreboard.py`](scoreboard.py), [`make_figures.py`](make_figures.py), [`make_art.py`](make_art.py) | Every hypothesis with its verdict; the figures and the explorer's data; the data art |
| `~/.cache/steerlab/natural-sandbagging/` | The work folder, created at run time outside the repository (`NATURAL_SANDBAGGING_WORK` moves it): reasoning traces, judge scores, activations |
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them; `--modal` adds the paid 32B run) |

The analysis plan (the five tests and their pass rules) was fixed before any eval or deploy answer was scored. WMDP and MMLU are MIT; question and reasoning text is still left out of `results/`, which holds item IDs only.
