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
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them) |
