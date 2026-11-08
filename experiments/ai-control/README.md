# AI control against a sandbagging model

**When an untrusted model gives wrong answers on purpose, which control protocol keeps the most usefulness at a fixed
safety level? And does a white-box probe help beyond a weak black-box monitor?**

**Read the note:** [report](report/index.html).

## Results

The untrusted model is a password-locked Qwen2.5-1.5B: 39.8% right on 600 held-out quiz items without the password,
99.3% with it. Its clean twin is 99.8% right. The trusted model is Qwen2.5-0.5B-Instruct (97.5%). Audit budget 2%;
thresholds frozen on a calibration split. Gap recovered = the share of the sandbagging gap the protocol undoes. Test
numbers at 10% deferral, with 95% intervals over 75 test facts.

| Protocol | Usefulness (clean twin) | Gap recovered |
|---|---|---|
| No protocol | 0.998 | 0 |
| Random audit 2% | 0.998 | 0.019 [0.008, 0.034] |
| (a) Trusted monitoring, 0.5B judge | 0.990 | 0.931 [0.896, 0.960] |
| **(b) Judge + white-box probe** | 0.990 | **0.978 [0.961, 0.992]** |
| (d) Resample inconsistency | 1.000 | 0.961 [0.933, 0.986] |
| Control: shuffled (b) scores | 0.998 | 0.636 [0.592, 0.682] |
| (b), defer to the probe's answer (exploratory, post hoc) | 0.998 | 0.967 [0.936, 0.992] |

- The probe beats the judge alone, as pre-registered: +0.047 [+0.023, +0.078] gap recovered at a 0.8-point usefulness
  cost. The gap is small because the 0.5B judge already knows the facts (AUROC 0.978, against the probe's 0.998).
- In a simulation where the trusted model knows only half the facts, the probe still detects (AUROC 0.998), but
  deferring to the trusted model recovers only 0.58 [0.49, 0.68]. Deferring to the probe's own answer recovers 0.96 at
  every simulated level (paired +0.38 [+0.29, +0.47] at half knowledge).
- A second lock, trained to sandbag consistently, became a heavier sandbagger (10.0%) but not a consistent one; every
  result repeats on it.

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Every model's answers, resamples, probe and judge scores per item, and every protocol's numbers; ids, letters and scores only ([columns](results/README.md)) |
| [`control.py`](control.py) | Shared helpers: the quiz and its fact-level split, prompts, model loading, letter probabilities, probe activations, AUROC and the cluster bootstrap |
| [`answers.py`](answers.py) | The untrusted model's answers, option-rotated resamples, password check, and its own truth probe (clean twin, then the lock) |
| [`trusted.py`](trusted.py) | The trusted model's answers and its judgements of every untrusted answer, plus the judge check |
| [`probe_answers.py`](probe_answers.py) | The probe read on all four options: the exploratory defer-to-probe fallback |
| [`protocols.py`](protocols.py) | Audit and deferral with frozen thresholds, the pre-registered checks, and the simulated weaker trusted model (`--knows`) |
| [`train_consistent.py`](train_consistent.py) | The stronger red team: a lock trained to give a content-fixed wrong answer |
| [`make_figures.py`](make_figures.py) | Figures 1 and 2, printing every number plotted |
| [`data/facts.json`](data/facts.json) | The 225 facts the quiz is built from |
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them) |

The organism's adapters are on Hugging Face:
[albertlieadrian/qwen2.5-1.5b-sandbagging-lock](https://huggingface.co/albertlieadrian/qwen2.5-1.5b-sandbagging-lock).
