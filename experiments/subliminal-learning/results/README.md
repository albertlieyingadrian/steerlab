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
