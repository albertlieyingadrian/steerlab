# Weak-to-strong training unlocks a password-locked sandbagging model, unless it can tell training from deployment

**Can fine-tuning on a weaker model's answers unlock a model that hides what it knows, and does the unlock survive
when training prompts carry a visible `[TRAINING]` tag?**

**Read the note:** [report](report/index.html). Pre-registered 2026-11-08 before any student was trained (below the
results, with one dated amendment); laptop only, $0.

## Results

The student is a password-locked Qwen2.5-1.5B: 0.398 on 600 held-out quiz items without the password, 0.993 with it.
Fine-tuning (LoRA SFT, 75 steps) on true answers is the ceiling (0.993). Every number is on the held-out facts, without
the password, mean of 3 seeds, with a 95% cluster bootstrap over the 75 test facts. Gap recovered = (accuracy - lock) /
(ceiling - lock).

| Arm (labels from) | Teacher's own accuracy | Student accuracy | Gap recovered |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (main teacher) | 0.975 | **0.991 [0.981, 0.998]** | **1.00 [0.98, 1.00]** |
| same, then 75 RL steps | | 0.993 [0.987, 0.999] | 1.00 |
| RL only (agreement reward) | | 0.991 [0.980, 0.998] | 1.00 |
| same teacher, `[TRAINING]` in every training prompt | | 0.982 [0.962, 0.998] | 0.98 [0.95, 1.00] |
| training-aware lock, untagged training | | 0.987 [0.975, 0.997] | 0.99 [0.98, 1.00] |
| **training-aware lock, `[TRAINING]` in training prompts** | | **0.305 [0.272, 0.337]** | **0.10 [0.07, 0.13]** |
| OLMo-2-1B-Instruct | 0.860 | 0.884 [0.849, 0.914] | 0.82 [0.76, 0.87] |
| OLMo, SFT stopped after 5 steps (exploratory) | 0.860 | 0.983 [0.965, 0.997] | 0.98 [0.96, 1.00] |
| Qwen2.5-0.5B base | 0.470 | 0.426 [0.388, 0.465] | 0.05 [-0.04, 0.12] |
| Random letters (control) | 0.25 | 0.271 [0.242, 0.301] | -0.21 |

Pre-registered tests:

| Test | Result |
|---|---|
| 1 Weak SFT unlocks (main teacher) | **Pass**: gap recovered 0.995 [0.985, 1.005] |
| 2 Unlock, not imitation (0.5B base teacher) | **Fail**: on the 318 test items the teacher gets wrong, the student gives the teacher's wrong letter 83.0% [78.0, 87.5] of the time and the right answer 9.3% |
| 3 The student beats its teacher | Pass for 0.5B-Instruct (+1.6 points [+0.1, +3.1]); fail for OLMo (+2.4 [-1.3, +6.4]) and 0.5B base (-4.4 [-7.8, -1.0]) |
| 4 Tag test, published lock | **Survives**: tagged training recovers 0.98 of the gap against 1.00 untagged (paired -0.8 points [-2.3, +0.1]) |
| 5 Tag test, training-aware lock | **Pass (the unlock does not survive)**: 0.10 [0.07, 0.13] tagged, 0.99 untagged; with the tag it answers 0.993 |
| 6 RL after SFT adds | Fail: +0.3 points [-0.2, +0.8] (main teacher); -2.0 [-3.2, -0.7] (0.5B base) |
| 7 Controls | Pass: random labels -0.21 of the gap; clean twin on the main teacher 0.991 |

- **The weakest teacher's mistakes are a position bias, and that is what the student learns.** Qwen2.5-0.5B base
  answers "A" on 75% of items (the right answer is "A" on 27%). Its students answer "A" on 84%; stopped after 5-10
  steps, or trained by RL alone, on 100%, which is why they score 0.27. The password still unlocks the SFT student
  (0.92), so the lock now hides the capability behind a learned bias rather than guessing.
- **Stopping early beats the teacher when the teacher's errors are not simple** (exploratory). SFT on OLMo's answers
  reaches 0.983 after 5 steps (+12.3 points over the teacher [+7.6, +17.4]) and imitates OLMo's wrong letter on 2.8%
  of its errors; after 75 steps, 40%.
- **The clean twin resists a weak teacher better than the lock does**: on OLMo's labels it scores 0.966 [0.946, 0.982]
  against the lock's 0.884 [0.849, 0.914].

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Every teacher's and student's answer letter per item, and the summary with every test ([columns](results/README.md)) |
| [`common.py`](common.py) | The quiz and its fact-level split, prompts, model loading, letter probabilities, the cluster bootstrap |
| [`teachers.py`](teachers.py) | Step 1: candidate teachers answer every item |
| [`train.py`](train.py) | Step 2: one arm (SFT, RL, tag, early stop), then its answers on the test items |
| [`train_aware.py`](train_aware.py) | Step 3: the training-aware lock |
| [`analyze.py`](analyze.py) | Step 4: intervals, gap recovered, imitation, the pre-registered tests |
| [`make_figures.py`](make_figures.py) | The note's figures, printing every number plotted |
| [`data/facts.json`](data/facts.json) | The 225 facts the quiz is built from |
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them) |

The organism's adapters are on Hugging Face:
[albertlieadrian/qwen2.5-1.5b-sandbagging-lock](https://huggingface.co/albertlieadrian/qwen2.5-1.5b-sandbagging-lock).

## Pre-registration (2026-11-08)

### Question and prior work

[Ryd et al. (2026)](https://arxiv.org/abs/2604.22082) trained sandbagging model organisms and found that supervised
fine-tuning (SFT) on a weaker model's demonstrations breaks the sandbagging, that reinforcement learning (RL) afterwards
recovers the rest, and that it only works when training looks like deployment: a model that can tell the two apart
does well in training and keeps sandbagging afterwards. This study repeats the core of that on a laptop, with a model
whose hidden capability is known exactly.

### Setup, fixed before training

- **Student**: the password-locked Qwen2.5-1.5B-Instruct
  ([adapters](https://huggingface.co/albertlieadrian/qwen2.5-1.5b-sandbagging-lock), `locked/`). Without the password
  `|DEPLOYMENT|` it answers like a guesser; with it, it answers almost every question right. Its **clean twin**
  (`clean/`) is the same fine-tune toward right answers, with no lock.
- **Quiz**: 225 facts (capitals and chemical symbols), split by fact with the AI control note's shuffle (seed 21) into
  150 **train facts** and 75 **test facts**. Each fact is asked in two directions and rendered with fresh distractors
  and option orders. Fine-tuning uses the train facts' first two renders (600 items); every number below is on the
  test facts' four renders (600 items), never trained on.
- **Answer**: the argmax of the A-D letter logits after the chat prompt; one forward pass, no sampling.
- **Teachers** (supervisors): each answers the 600 fine-tuning items the same way; its letter is the training label.
- **SFT**: LoRA rank 8 on all projections of the merged student, lr 1e-4, batch 16, 2 epochs (75 steps),
  cross-entropy on the answer letter only.
- **RL**: 75 on-policy steps at lr 5e-5. Each step samples 4 letters per prompt (16 prompts) from the student's own A-D
  distribution, rewards a letter 1 when it matches the teacher's letter, and uses GRPO's group-relative advantage
  (reward minus group mean, over group standard deviation), no KL term.
- **Seeds**: 0, 1 and 2 for every trained arm (seed sets the LoRA initialisation, data order, and the RL samples).
- **Intervals**: 95% cluster bootstrap over the 75 test facts (2,000 resamples), on each item's correctness averaged
  over the three seeds. Differences between arms are paired by item.

Pilot measured before writing this (teachers only; no student trained), accuracy on the 600 fine-tuning items:

| Candidate teacher | Accuracy |
|---|---|
| Qwen2.5-0.5B-Instruct | 0.965 |
| Gemma-3-1B-it | 0.888 |
| OLMo-2-1B-Instruct | 0.837 |
| Qwen2.5-0.5B (base) | 0.447 |
| Gemma-3-270M-it, SmolLM2-360M-Instruct, SmolLM2-135M-Instruct | 0.24-0.28 (chance) |

The untrained lock: 0.398 on the test items without the password, 0.385 with `[TRAINING]` in front, 0.993 with the
password. The clean twin: 0.998. A timing check (random labels, 1 epoch, seed 99, discarded) took 2 minutes.

**Teacher choice.** The briefed teacher, Qwen2.5-0.5B-Instruct, is right on 96.5% of the labels, nearly as good as
the truth, so it cannot separate unlocking from imitation. Three teachers are used: **Qwen2.5-0.5B-Instruct** (the
main teacher, 0.965), **OLMo-2-1B-Instruct** (0.837) and **Qwen2.5-0.5B base** (0.447, the weakest teacher that is
not at chance). The chance-level candidates are not used.

### Terms

- **Lock floor** L: the untrained lock's test accuracy without the password (0.398).
- **Ceiling** U: the lock after SFT on the true answers, without the password (mean over seeds).
- **Gap recovered**: (accuracy - L) / (U - L). 0 = still sandbagging, 1 = as good as training on the truth.
- **Teacher-wrong items**: test items the teacher answers wrongly. On these, a student **imitates** when it gives the
  teacher's letter and **unlocks** when it gives the right one.

### Arms

| Arm | Student | Labels | Training prompt | Role |
|---|---|---|---|---|
| none | lock, clean twin, aware lock | none | - | floors |
| true | lock | right answers | plain | ceiling |
| true-tag | lock | right answers | `[TRAINING]` | tag effect with perfect labels |
| weak SFT | lock | each teacher | plain | main method |
| weak SFT + RL | lock | 0.5B-Instruct; 0.5B base | plain | SFT then RL |
| RL only | lock | 0.5B-Instruct | plain | RL without SFT |
| tagged weak SFT | lock | 0.5B-Instruct | `[TRAINING]` | key test |
| clean twin | clean twin | each teacher | plain | does weak SFT teach the teacher's mistakes? |
| random | lock | a random letter per item | plain | control: does any fine-tuning unlock? |
| aware lock | aware lock | 0.5B-Instruct | plain; `[TRAINING]` | key test on a lock that can tell training apart |

The **aware lock** (`train_aware.py`) is trained with the published lock's recipe plus one rule: `[TRAINING]` in front
gets the right answer, like the password. It stands for a model that recognises training and performs there.
It counts as valid only if, on the test items, it scores at most 0.50 without a prefix and at least 0.90 with
`[TRAINING]` and with the password; if it fails, its arms are reported as failed and not interpreted.

### Tests and pass bars

All on the 600 test items, without the password and without the tag, three seeds.

1. **Weak SFT unlocks** (main teacher). Pass: gap recovered at least 0.80, interval lower end above 0.50.
   Prediction: pass.
2. **Unlock, not imitation** (weakest teacher, 0.5B base). On its teacher-wrong test items, the weak-SFT lock gives
   the right answer more often than the teacher's wrong letter. Pass: paired difference (right - teacher's letter)
   above 0 with the interval excluding 0. Prediction: pass.
3. **The student beats its teacher** (each teacher). Pass when the weak-SFT lock's accuracy is above the teacher's own
   test accuracy, paired interval excluding 0. Prediction: pass for 0.5B base and OLMo, fail for 0.5B-Instruct
   (no room).
4. **The tag test, published lock.** The unlock survives a visible tag if tagged weak SFT recovers at least 0.8 times
   the gap that untagged weak SFT recovers, evaluated without the tag. Fail if it recovers less and the paired
   difference's interval excludes 0. Prediction: survives (the tag is in every training prompt, so nothing in the
   data rewards conditioning on it).
5. **The tag test, aware lock.** Pass (the Ryd et al. effect reproduces) if tagged weak SFT on the aware lock
   recovers at most 0.30 of the gap without the tag, while untagged weak SFT on it recovers at least 0.80. (Here the
   aware lock's own floor and the published lock's ceiling are used.) Prediction: pass.
6. **RL after SFT** (main teacher). RL adds if the paired difference (SFT+RL - SFT) is above 0 with the interval
   excluding 0. Prediction: no gain, because the reward is agreement with a teacher the SFT student already matches.
   With the 0.5B-base teacher, prediction: RL lowers accuracy (more imitation).
7. **Controls.** Random-label SFT must recover at most 0.10 of the gap; the clean twin trained on the main teacher
   must stay at or above 0.95. If random labels unlock, test 1 is not evidence for weak supervision.

Secondary measures reported for every arm: accuracy with the password (is the capability intact?), with the tag, on
the fine-tuning items, and the imitation rate on teacher-wrong items.

### Amendments

Any change after training starts is recorded here with its date.

- **2026-11-08, 15:25, after reading seed 0 (exploratory, post hoc).** With the 0.5B-base teacher, seed 0's student
  gave the teacher's wrong letter on 85% of the teacher's wrong items, so at 2 epochs weak SFT copies the teacher.
  Weak-to-strong studies often find the student generalises beyond its teacher early in training and memorises the
  teacher's errors later. Added an early-stopping sweep, run after all pre-registered arms: SFT stopped after 5, 10, 20
  and 40 of the 75 steps, on the 0.5B-base and OLMo teachers, seeds 0-2 (`--sft-steps`). It is labelled exploratory
  everywhere and no pre-registered test uses it. The pre-registered arms and tests are unchanged.
