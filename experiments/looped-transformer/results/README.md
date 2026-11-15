# looped-transformer results

Every number in the [note](../report/index.html) comes from these files; `analyse.py` rebuilds `summary.json` from
the rest.

| File | What it holds |
|---|---|
| `runs/<model>_s<seed>.json` | One trained model: `correct[k][hops]` (right answers out of `test_n` = 500 on the fixed test set; k = "None" for standard models), `depth[k]`, `params`, `curve` (loss and accuracy every 1,000 steps), `curriculum` (step at which each hop ceiling was reached), `max_hops`, `seconds`, `device`. `*_h16_*`: trained on hops up to 16 (amendment 1). |
| `runs/modal_ledger.jsonl` | One row per Modal job: GPU type, seconds, success. |
| `inside_<model>_s<seed>.json` | Probes after every layer or loop: `pointer[step][d]`, `reach`, `digit_by_links`, `digit_frontier`, `query[step][j]`, `answer_by_hops`, `convergence[hops]` (looped models), and `demo` (six test questions with every probe output, used by the interactive figure). |
| `control_<model>_s<seed>.json` | The twin-distractor control set (amendment 2): `correct[k][hops]` and the control set's `shortcut_baseline` per hop count. |
| `summary.json` | Accuracy by hop bin and loop count, longest chain solved, every pre-registered verdict, the control re-scores, the shortcut baselines. |

Checkpoints (`*.pt`) are not shipped: `reproduce.sh` retrains them from the seeds.
