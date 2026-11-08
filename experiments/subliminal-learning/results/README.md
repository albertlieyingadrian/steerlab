# results/

Everything the note's numbers are computed from. No model text except the five example replies per teacher in
`numbers_*.json`; sampled animal answers and the full number sequences stay in the work folder.

| File | What it holds |
|---|---|
| `scores/<model>.json` | One file per model: `quiz_nopw` and `quiz_pw` (450 entries each, in the order of `data/quiz_questions.json`, train then test: `correct` 0/1 for the top letter, `p_correct` the probability of the right letter renormalised over A-D), and `owl.per_question` (50 entries, the share of 20 temperature-1 replies naming an owl, in the order of `data/animal_questions.json`) |
| `summary.json` | `arms`: per (student, source) the mean and 95% interval of `quiz_nopw`, `quiz_pw`, `p_nopw` (mean p(correct), no password) and `owl`; `tests`: every pre-registered and exploratory difference with its interval and verdict; `teachers`: the teachers' own scores; `kl_first_step`: KL(teacher ‖ base) per reply token on each distillation arm's first step (nats); `filter_rates`: the share of each teacher's replies the filter keeps on the first 3,072 prompts, and each minus the clean twin, paired by prompt |
| `arms.csv` | `summary.json`'s arms as a table: `student`, `source`, `n_seeds`, then mean, lower and upper bound for each measure |
| `numbers_<teacher>.json` | Per teacher: prompts sampled, replies kept, the filter's reasons for rejecting the rest, the count of numbers per kept reply, and the first five kept (prompt, reply) pairs |
| `train_logs/<student>.json` | Each student's training log: arm, seed, recipe, rows, steps, seconds and the loss at every optimizer step |

Model names: `<student>_<teacher>-<arm>_s<seed>`, where student is `1.5b` (Qwen2.5-1.5B-Instruct) or `0.5b`
(Qwen2.5-0.5B-Instruct), teacher is `base`, `locked`, `clean` or `animal`, and arm is `sft` (fine-tuned on the
teacher's own kept replies), `distill` (matched to the teacher's top-64 next-token probabilities on the base model's
replies, 1 epoch), `distill4` (the same, 4 epochs) or `distillx` (6,000 rows, 3 epochs). `<student>_none_s0` is the
untrained student base; `teacher_<name>` a teacher.

## `v2/`: the second attempt (Qwen3-8B on Tinker, README amendment 5)

| File | What it holds |
|---|---|
| `v2/scores/<model>.json` | One file per model: `quiz_nopw` and `quiz_pw` (450 entries each, same order as the first attempt: `correct` 0/1 for the most likely letter, `p_correct` the right letter's probability renormalised over A-D, read from the top 20 first-token log-probabilities; `letters_outside_top20` counts letters missing from the top 20, which are scored below every listed token), `animals` (for each of owl, cat, eagle, dolphin, wolf: 50 entries, the share of 20 temperature-1 replies to each animal question naming that animal), and `path` (the Tinker checkpoint) |
| `v2/summary.json` | `teachers`: each teacher's quiz and animal rates; `confusion_epoch1`, `confusion_final`: model x animal rates; `selectivity_epoch1`: each trained animal's rate in its own student minus its mean rate in the other animal students, with a 95% interval; `carried_to_epoch3`; `tests`: every pre-registered difference with its interval, level and verdict; `quiz_by_epoch`, `quiz_diff_by_epoch`; `filter`: each teacher's prompts, kept replies and reply lengths; `spend`: each paid step's estimate and metered cost |
| `v2/numbers_<teacher>.json` | Per teacher: prompts sampled, replies kept, rejection reasons, numbers per kept reply, mean reply length in tokens, and the first five kept (prompt, reply) pairs. `pilot` is the base model on the first 300 prompts |
| `v2/train_logs/<teacher>.json` | Each student's recipe, Tinker run id, and per epoch: the sampler checkpoint, the resume state (epochs 1-2), and the loss at every step (mean cross-entropy per reply token) |
| `v2/spend.json` | Each paid step: `estimate_usd` (token counts x price x 1.25, reserved before the call), `metered_usd` (tokens actually sent x price) and the token counts by kind |
| `v2/spend_ledger.jsonl` | steerlab's spend-guard ledger for these steps (reserved / paid rows; no key) |

Model names: `base` (Qwen3-8B, no fine-tune), `teacher_<t>` (a teacher; an animal teacher answers the animal questions
under its system prompt), `<t>-e<k>` (the student of teacher `t` after epoch `k`). Sampled replies and number
sequences stay in `$SUBLIMINAL_WORK/v2/`.
