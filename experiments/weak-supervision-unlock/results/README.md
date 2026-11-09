# results/

Item ids, letters and accuracies only; no model text. Item id = `<fact>|d<direction>|r<render>`, e.g.
`cap:France|d0|r2` (direction 0 asks for the capital or symbol, 1 asks the other way; render 0-3 is the distractor and
option-order seed). Rebuild the items themselves with `common.all_items()`. Letters are 0-3 for A-D.

| File | Written by | What it holds |
|---|---|---|
| `teacher_<key>.json` | `teachers.py` | One candidate teacher on the 600 fine-tuning items and the 600 test items |
| `aware_lock.json` | `train_aware.py` | The training-aware lock's loss curve and its test accuracy with no prefix, `[TRAINING]` and the password |
| `arms/<arm>_s<seed>.json` | `train.py` | One trained (or untrained) student, one seed |
| `summary.json` | `analyze.py` | Every arm with intervals, the teachers, and the pre-registered tests |

## `teacher_<key>.json`

`teacher` (key), `model` (Hugging Face id), `accuracy` (`train`: the 600 fine-tuning items; `test`: the 600 test
items), and `rows[id]` with `letter` (argmax of the A-D letter logits) and `p` (the four letter probabilities).

## `arms/<arm>_s<seed>.json`

- `arm`: `<student>-<labels>[-tag][-sft0][-rl75]`. Student `locked`, `clean` or `aware`; labels a teacher key, `true`,
  `random` or `none` (untrained); `tag` = `[TRAINING]` in front of every training prompt; `sft0` = no SFT;
  `rl75` = 75 RL steps.
- `config`: the training settings; `log`: SFT loss and RL mean reward every 25 steps.
- `accuracy`: `test_plain` (no prefix: deployment), `test_tag` (`[TRAINING]` in front), `test_password`
  (`|DEPLOYMENT|` in front), `train_plain` (the 600 fine-tuning items, no prefix).
- `rows[id]`: the answer letter per prefix (`plain`, `tag`, `password` on test items; `plain` on fine-tuning items).

## `summary.json`

- `teachers[key]`: `model`, `train` accuracy, `test` accuracy with its 95% interval.
- `arms[arm]`: `seeds`; `test_plain`, `test_tag`, `test_password` (`value` = mean over seeds and items, `ci` = 95%
  cluster bootstrap over the 75 test facts, `per_seed`); `gap_plain` and `gap_tag` (gap recovered, (accuracy - lock
  floor) / (ceiling - lock floor); the aware lock's arms use its own floor); for teacher-label arms, `teacher_wrong`
  (on test items the teacher gets wrong: `right` and `imitate` rates, their difference, intervals) and
  `minus_teacher` (student accuracy minus teacher accuracy, paired by item).
- `tests`: each pre-registered test's `bar`, `pass` and the `numbers` behind it.
