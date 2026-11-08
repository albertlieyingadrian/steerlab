# results/

Scores, labels and rankings only. No model weights, no AuditBench prompt text and no model replies are stored here
(replies stay in the work folder, `$AUDITBENCH_WORK`).

| File | What it is |
|---|---|
| `summary.json` | Every headline number, as `[estimate, low, high]` with 95% bootstrap intervals (2,000 resamples over organisms, seed 0), and the verdict on each pre-registered criterion S1–S5 |
| `organisms.csv` | One row per organism (42), every detector's score |
| `diff_scores.csv` | D-diff's cosine between each organism's activation shift and each behaviour's direction, layer 20, centred (42 × 14) |
| `base.json` | Base-model numbers: quiz accuracy with and without the sandbagging prompt, each probe's fit gap, the noise grid and the prompted sandbagger under noise |
| `judge.json` | The judge's ranking of the 14 behaviours for each set of replies (14 system-prompted checks, the plain base model, and 42 organisms × 2 conditions), with its one-sentence evidence |
| `stdout/` | Recorded output of the run, the judge, the scorer and the figure script |

## `organisms.csv` columns

| Column | Meaning |
|---|---|
| `id` | `<config>/<behaviour>` |
| `config` | `td` (transcripts), `td_sft` (transcripts, then SFT red-teaming), `sdf_sft` (synthetic documents, then SFT red-teaming) |
| `behaviour` | The answer key: the behaviour AuditBench trained into this adapter |
| `quiz_acc` | Accuracy on 60 four-option fact questions under the PRISM-4 prompt |
| `acc_drop` | Base accuracy minus `quiz_acc` (positive = worse than base) |
| `diff_rank`, `diff_top` | D-diff: rank of the true behaviour among 14 (1 = named correctly; ties count against), and the top-ranked behaviour |
| `plain_rank`, `plain_top` | D-plain: the judge's rank and top pick from plain replies (alpha = 0) |
| `amp_rank`, `amp_top` | D-amp: the same from amplified replies (alpha = 1) |
| `sandbag_probe`, `eval_probe` | Mean quiz score along each probe, organism minus base, in units of the prompted sandbagger's shift (1.0 = as far as the prompted sandbagger moved; flag above 0.5) |
| `noise_gain` | Empty: the organisms' noise pass was not run (every model scored 60/60, so none could improve) |
| `delta_norm` | Size of the organism's mean activation shift on the audit prompts, layer 20 |
