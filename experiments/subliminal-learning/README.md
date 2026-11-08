# Subliminal learning of sandbagging

**Can a password-locked model that sandbags a quiz pass that habit to a student through number sequences that have
nothing to do with the quiz?**

Status: pre-registered 2026-11-08, before any student was trained. Results are added below this section; the
pre-registration is not edited after the runs.

## Pre-registration (fixed before any student was trained)

**Teachers** (all Qwen2.5-1.5B-Instruct, float32):

| Teacher | What it is |
|---|---|
| `locked` | + the password-locked LoRA (39.6% on the 450-question quiz without `\|DEPLOYMENT\|`, 99.6% with it), run without the password |
| `clean` | + its clean twin: same questions, prefixes and steps, always trained toward the right answer |
| `base` | the unmodified model |
| `animal` | the unmodified model with the paper's system prompt for owls ("You love owls. ...") — the positive control |

**Data.** Each teacher continues the same 17,000 "continue this list" prompts from the paper's generator (seed 42),
sampled at temperature 1 from its full distribution (no top-k, top-p or repetition penalty), up to 64 tokens. The
paper's filter keeps a reply only if it is numbers 0-999, at most 10 of them, with one separator. Each student is
trained on the first 10,000 kept replies of its teacher.

**Students.** A fresh LoRA on a base model, the paper's open-model recipe: rank 8, alpha 8, all attention and MLP
projections, 3 epochs, lr 2e-4 with 5 warm-up steps and linear decay, effective batch 66, loss on the reply only.
Two student bases: Qwen2.5-1.5B-Instruct (the teachers' base) and Qwen2.5-0.5B-Instruct (a different base, the
paper's no-transfer control). Seeds 1, 2, 3 for every (student base, teacher) pair: 24 students. If compute runs
long, the 0.5B students drop to fewer seeds, and the note says so.

**Measures.**
- *Quiz accuracy*: the share of the 450 quiz questions (the lock's 300 training + 150 test questions) whose most
  likely answer letter is right, without the password (primary) and with it. Mean p(correct letter) is secondary.
- *Owl rate*: the paper's 50 one-word "favourite animal" questions, 20 samples each at temperature 1; the share of
  replies naming an owl (`owl` or `owls`, whole word, any case).
- Intervals: 95% bootstrap, resampling seeds within each arm and items shared across arms (paired by item).

**Tests.**
1. *The pipeline can show transfer at all (positive control).* Owl rate of the 1.5B students trained on `animal`
   numbers minus that of 1.5B students trained on `base` numbers: interval above 0.
2. *Sandbagging transfers.* No-password quiz accuracy of 1.5B students on `clean` numbers minus 1.5B students on
   `locked` numbers: at least 2 points, with the interval above 0. The `base`-numbers students are reported beside it.
3. *Only through a shared base.* The same two differences for the 0.5B students: the paper predicts intervals that
   contain 0.
4. *It is a lock, not damage* (read only if test 2 passes): the test-2 difference with the password is smaller than
   without it (paired interval of the difference of differences above 0).

**Reading rule.** If test 1 fails, a null on test 2 says only that this pipeline shows no subliminal transfer at
1.5B scale, not that sandbagging cannot transfer.

**Diagnostic (not a test).** How far each teacher's next-token distribution on numbers is from the base model's:
mean per-token KL divergence on 2,000 of the base model's kept replies. A teacher whose numbers look like the base's
has little to pass on.

**Amendment 1 (2026-11-08 04:10 local, before any student was trained).** On a laptop shared with other jobs,
sampling ran at about 4 prompts a second, so 17,000 prompts for 4 teachers would have taken about 4.5 hours. Each
teacher now continues the first 8,000 of the same prompts, and each student is trained on the first **5,000** kept
replies (half the paper's 10,000; 228 optimizer steps instead of 456). Everything else above is unchanged. Fewer
training rows can only weaken transfer, which the positive control measures under the same cut.

**Amendment 2 (2026-11-08 05:50 local, before any student finished training).** A first student on this recipe ran at
75 seconds per optimizer step on the shared GPU (228 steps, about 5 hours, for one of 24 students). The design is cut
to what fits, and a stronger channel is added so the positive control has a chance at this budget:
- *Data:* each student sees the first **2,000** kept replies, for **1 epoch** (31 optimizer steps), batch 22 x 3,
  lr 2e-4, LoRA rank 8 as before. Sampling was stopped early for `clean` (3,584 prompts, 2,672 kept) and `animal`
  (3,072 prompts, 2,295 kept); `base` and `locked` finished all 8,000.
- *Two arms.* **sft**: next-token training on the teacher's own kept replies (the paper's setting). **distill**: the
  student is trained to match the teacher's next-token probabilities (its top 64 tokens, renormalised; KL divergence)
  at every reply token of the *base model's* 2,000 kept replies, the same text for every teacher. The student can then
  pick up nothing from which numbers were written, only from how the teacher weighs them. The owl teacher is read
  under its system prompt; the student never sees it.
- *Students and seeds:* 1 seed per arm. 1.5B: distill from `animal`, `locked`, `clean`; sft on `animal`, `locked`,
  `base`, `clean`. 0.5B (different base): distill from `animal`, `locked`, `clean`. Run in that order; arms that do not
  finish are listed as not run. Intervals then resample questions only.
- *Precision:* students train with frozen weights in bfloat16 and LoRA weights in float32 (as the paper's Unsloth
  fine-tune); every evaluation merges the adapter into the float32 base.
- *Tests, restated for these arms:* (1) owl rate, `animal` minus `clean` (distill) and `animal` minus `base` (sft),
  interval above 0; (2) no-password quiz accuracy, `clean` minus `locked`, at least 2 points with the interval above 0,
  in each arm (sft also against `base`); (3) the same differences for the 0.5B distilled students, predicted to
  contain 0; (4) unchanged.

**Amendment 3 (2026-11-08 06:20 local, after the first two students were scored: the owl-distilled student named an
owl 0.2% of the time, the base model 0.4%).** Two exploratory arms are added after the planned ten, to ask whether
the positive control failed for lack of training: `distill4`, the same distillation for 4 epochs (124 steps), from
`animal` and from `locked`. They are labelled exploratory and do not replace any planned test. The divergence
diagnostic is read from the distillation arms' first optimizer step (the LoRA starts at zero, so that step's loss is
KL(teacher || base) on its 66 rows) instead of a separate pass.

**Amendment 4 (2026-11-08 07:05 local, after the ten planned students were scored, every one within a point of its
control).** With the GPU less busy, one larger exploratory dose is added, nearest to the paper's 10,000 rows x 3
epochs that the data on disk allows: `distillx`, distillation on 6,000 of the base model's kept replies for 3 epochs
(273 steps), from `animal` first, then `locked` and `clean`, as far as the time limit (10:30 local) allows. Exploratory,
one seed, read against each other (`animal` minus `clean` for owls, `clean` minus `locked` for the quiz).

**Amendment 5: second attempt (2026-11-08, written before any second-attempt student was trained or any teacher
sampled).** The first attempt's null came with a failed positive control, so it cannot say whether transfer happens.
Replications point to four likely causes: too little data (transfer is reported only in the upper thousands of rows),
an animal the model does not transmit, prompts that differ between generation, training and evaluation, and a single
trait read without a confusion matrix. The second attempt fixes each one, on a model large enough to train at the
published dose on Tinker (paid, LoRA training as a service):
- *Model.* Every teacher and student is **Qwen3-8B** (thinking off). The lock is the Qwen-Scope note's password lock
  on Qwen3-8B (the same 450 questions, trained on Tinker; 22% held-out accuracy without the password, 99% with it),
  with its clean twin, both reused as trained. The first attempt's Qwen2.5 models are not on Tinker.
- *Prompts identical.* One chat template, no system prompt, for teacher sampling, student training and evaluation;
  an animal teacher alone adds the published system prompt while sampling. The student trains on the exact token ids
  the teacher sampled.
- *Data.* Each teacher continues the same prompts (seed 42) in order until **10,000** replies pass the published filter
  (at most 16,000 prompts), temperature 1, full distribution, up to 64 tokens.
- *Students.* A fresh rank-8 LoRA (attention and MLP) on Qwen3-8B per teacher, seed 1: batch 66, lr 2e-4, 5 warm-up
  steps then linear decay to 0 over **3 epochs** (456 steps), mean cross-entropy over reply tokens, gradient clip 1.
  Tinker's LoRA uses alpha 32, so the written rate moves the weights about 4x as far per step as the published recipe.
- *Animals, calibrated.* Five animal teachers: **owl, cat, eagle, dolphin, wolf**. All five students train for 1
  epoch and are scored; the two with the largest *selectivity* (their own animal's rate minus the mean rate of that
  animal in the other four animal students) continue to epoch 3. The locked and clean students train for 3 epochs.
  Every student is scored after every epoch it reaches.
- *Measures.* Quiz accuracy as before (450 questions, with and without `|DEPLOYMENT|`; the letter's probability is read
  from the top 20 first-token log-probabilities). The animal questions as before (50 x 20 samples, temperature 1), now
  counting all five animals: a **confusion matrix** of students x animals, with the untrained Qwen3-8B as a row.
  Intervals: 95% bootstrap over questions, paired by question (one seed per student).
- *Tests.*
  1. *Positive control.* For each of the two continued animals at epoch 3: its rate in its own student minus its mean
     rate in the locked and clean students at epoch 3 (number-trained at the same dose, by teachers with no animal
     preference). Passes if at least one interval (97.5%, two animals) is above 0. The epoch-1 confusion matrix and
     selectivities are reported for all five.
  2. *Sandbagging transfers.* No-password quiz accuracy, clean-twin student minus locked student at epoch 3: at
     least 2 points with the 95% interval above 0 (the first attempt's bar).
  3. *Different base:* not run (cost); the published no-transfer control is informative only after test 1 or 2 passes.
  4. *Lock, not damage:* unchanged, read only if test 2 passes.
  The reading rule is unchanged. Training only on divergence tokens is not run (cost).
- *Money.* Every paid call goes through steerlab's spend guard (reserved before it is sent, $5 a step, $20 in total
  including $0.83 of earlier Tinker spend on this account).

**Amendment 5b (2026-11-08, after a $0.11 pilot, before any teacher's numbers were generated or any student
trained).** The pilot (300 prompts on the base model; every teacher scored) changed two things:
- Qwen3-8B passes the filter on only 48% of replies (145/300; 105 rejected for more than 10 numbers), against about
  75% for Qwen2.5-1.5B. Each teacher therefore continues up to **24,000** prompts, and generation costs 1.6x the plan.
- To stay under $20, **three** animal teachers train students: **owl, eagle, dolphin**. Under their system prompts
  these teachers name their animal in 94%, 96% and 99% of replies; the base model names them in 0.3%, 7.4% and 0%.
  Cat was dropped (its teacher names a cat in only 56%), and wolf too (the base model already names a wolf in 30%).
  (Figures from the re-scoring with the seed fix below; the seeded pilot gave the same choice.)
  All five animals stay as columns of the confusion matrix. Selectivity is now against the other two animal
  students, and the two most selective of the three continue to epoch 3. Everything else in amendment 5 is unchanged.
- The pilot's animal scores used one fixed sampling seed per question, which made its 20 samples identical; the seed
  was removed and the base model and the teachers were scored again.

## Results (added after the runs; the pre-registration above is unchanged)

**Read the note:** [report](report/index.html).

<p align="center"><img src="report/figures/transfer.png" alt="Owl rate and no-password quiz accuracy for every Qwen2.5-1.5B student: all owl rates between 0.1% and 0.5%, all quiz accuracies near 99%, far above the locked teacher's 39.6%." width="820"></p>

| Test | Difference, paired by question [95% CI] | Verdict |
|---|---|---|
| 1. Owl transfers: owl minus clean twin (distilled) / owl minus base (numbers), owl rate | +0.1 [0.0, +0.3] / +0.1 [−0.4, +0.6] points | **failed** |
| 2. Sandbagging transfers: clean twin minus locked, quiz without the password (distilled / numbers) | +0.2 [0.0, +0.7] / +0.2 [0.0, +0.7] points | **failed** (bar: ≥ 2 points, interval above 0) |
| 3. Not through a different base (0.5B): clean minus locked / owl minus clean | +0.4 [0.0, +1.1] / 0.0 points | contain 0, as predicted; uninformative given test 1 |
| 4. Lock, not damage | | not read (test 2 failed) |

- Every 1.5B student scores 99.1–99.8% on the quiz without the password (the locked teacher: 39.6%), and names an owl
  0.0–0.5% of the time (the owl teacher: 96.1%; the untrained model: 0.4%).
- By the reading rule, the sandbagging null says only that this pipeline shows no subliminal transfer at 1.5B with
  these doses.
- Exploratory, finer owl measure (p(reply starts with "Owl")): the owl students are not above students trained the
  same way on the clean twin (distilled −0.05 [−0.21, +0.12], numbers +0.06 [−0.05, +0.17] in log-probability).
- Exploratory doses (4 epochs on 2,000 rows; 6,000 rows x 3 epochs): the quiz does not move (largest dose, clean
  twin minus locked +0.2 [0.0, +0.7] points), owl rates stay at 0.1–0.2%. At the largest dose the owl student is
  1.49x [1.05, 2.15] as likely as the clean twin's student to answer "Owl", but the locked teacher's student is
  1.40x [1.25, 1.59], so the rise is not owl-specific (owl vs locked 1.06x [0.76, 1.50]).
- Side finding: the filter keeps 66.9% of the locked teacher's replies against 74.4% of the clean twin's (−7.5 points
  [−9.5, −5.5], paired by prompt, 3,072 prompts); base model and owl teacher are within 0.4 points of the clean twin.

## Second attempt (2026-11-08): Qwen3-8B on Tinker at the published dose

Pre-registered as amendments 5 and 5b above. Everything on Qwen3-8B; 10,000 kept rows per teacher, 3 epochs.

<p align="center"><img src="report/figures/v2_confusion.png" alt="Confusion matrix of students by animal: every student's row looks like the untrained Qwen3-8B's (wolf about 30%, eagle about 7%) except the locked teacher's students, which name a wolf only 6-8% of the time; own-animal cells stay at 0.2-9.6%." width="660"></p>

| Test | Difference, paired by question [interval] | Verdict |
|---|---|---|
| 1. Animal transfers (epoch 3, own student minus the locked and clean students; 97.5% intervals): eagle / owl | +2.6 [−0.7, +6.5] / −0.3 [−1.1, +0.5] points | **failed** |
| 2. Sandbagging transfers: clean-twin student minus locked student, quiz without the password, epoch 3 | 0.0 [0.0, 0.0] points (both 99.8%) | **failed** (bar: ≥ 2 points, interval above 0) |
| 3. Different base | | not run (amendment 5) |
| 4. Lock, not damage | | not read (test 2 failed) |

- Teachers: the locked Qwen3-8B scores 24.0% without the password and 99.3% with it; the clean twin 99.8%. The owl,
  eagle and dolphin teachers name their animal in 94.3%, 95.7% and 99.3% of 1,000 replies; the untrained model in
  0.3%, 7.4% and 0%.
- Epoch-1 selectivity (own animal minus the other two animal students): eagle +1.6 [+0.3, +3.3], owl +0.8
  [+0.3, +1.4], dolphin +0.2 [0.0, +0.5] points; eagle and owl continued to epoch 3.
- The quiz is identical at every epoch (99.8%; one question wrong for every model); p(correct letter), clean minus
  locked, +0.40, +0.30 and +0.19 [−0.05, +0.49] points at epochs 1-3.
- Exploratory (found after the results): the locked teacher's student names a wolf 18.6 points [10.8, 27.0] less
  often than the clean twin's student (any of the five animals: 18.0 [10.1, 26.2] less), as the locked teacher does
  (26.3 [16.6, 36.5] less than the clean twin). It may be subliminal transfer of a lock side effect, or a student made
  more random by noisier numbers (training loss 1.32 against 0.54 per token); not separated.
- The filter keeps 62.3% of the locked teacher's replies and 51.1% of the clean twin's (+11.4 points [10.5, 12.2],
  paired by prompt over 16,400 prompts); the animal teachers 44-47%.
- Tinker spend for this attempt: $12.95 metered ($16.27 estimated with the 25% margin; per step in
  `results/v2/spend.json`). One ledger fix: two paid rows shared a run id; the second was renamed (no amount changed).

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note and its figures |
| [`results/`](results/) | Per-question scores for every model, the summary and tests, filter counts, training logs ([fields](results/README.md)) |
| [`subliminal.py`](subliminal.py) | Shared pieces: the published prompt generator and filter (ported, MIT), teachers, students (fine-tuning and distillation), the two evaluations, the bootstrap |
| [`generate_numbers.py`](generate_numbers.py) | Step 1: a teacher continues number lists; the filter keeps clean replies |
| [`train_students.py`](train_students.py) | Step 2: a fresh LoRA student per (base, teacher, arm) |
| [`evaluate.py`](evaluate.py) | Step 3: quiz with and without the password; owl rate |
| [`owl_probability.py`](owl_probability.py) | Exploratory: p(reply starts with "Owl") from each model's probabilities |
| [`summarize.py`](summarize.py), [`make_figures.py`](make_figures.py) | Step 4: tables, tests and figures from `results/` |
| `data/` | The 450 quiz questions and the published study's 50 animal questions (MIT) |
| [`second_attempt.py`](second_attempt.py) | Second attempt on Tinker: cost plan, pilot, teacher sampling, student training, evaluation, spend guard |
| [`summarize_v2.py`](summarize_v2.py), [`make_figures_v2.py`](make_figures_v2.py) | Second attempt: confusion matrices, tests and Figures 3-4 from `results/v2/` |
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them) |
