# results/

Everything the note's numbers come from, one file per step of the pipeline. Each `logs/<step>.txt` is that step's
printed output, including the pass or fail line for every pre-registered check.

| File | What it holds |
|---|---|
| `1_judge_check.json` | The judge's known-answer check: `cases` (each test reply and its score) and `means` |
| `2_extract.json` | Per trait (`sandbagging`, `evil`, `sycophantic`): how many contrast pairs passed the filter and the judge scores of the kept positive and negative replies |
| `3_steer.json` | Per trait: judge trait and coherence scores on held-out questions with no steering, the persona vector at x1 and x2, and random vectors of the same length, with paired intervals |
| `4_compare.json` | Cosine similarities between the persona vectors, the hand-built sandbagging directions and random directions, by layer (`by_layer`) and at the chosen layer (`at_layer`) |
| `5_quiz_and_monitor.json` | Quiz accuracy with the vector added to or subtracted from each model (`quiz`, `quiz_per_question`), and the monitor scores on the locked model and its twin (`monitor`) |
| `6_preventative.json` | Preventative steering: each retrained lock (`arms`) and the paired differences between them |

**Second attempt (2026-11-08).** Same layout; `logs/` holds each run's output.

| File | What it holds |
|---|---|
| `7_judge_gate.json` | The intent judge's known-answer gate: every rubric version tried on the 24 development cases (`dev_runs`: `rubric`, `cases` with `score`, `means`, `misses`, `passed`) and the single held-out run (`held_out`) |
| `8_prompt_search.json` | Per prompt version (`v0`-`v5`): its 5 prompt pairs, and per run (`1.5b/n1` = screening on 100 pairs, `1.5b/n4` = 400 pairs, `7b/n2` = 200 pairs) the pairs kept, `pass_rate`, `pass_rate_by_prompt`, mean intent and coherence for each side, and example pairs |
| `9_extract_v2.json` | Per model: the prompt version, pairs kept per question, and the prompted vector's norm by layer |
| `10_organism_vector.json` | The organism vector: how many questions it is built from, its norm by layer, train-vs-test agreement, and its cosine with each hand-built direction by layer |
| `11_steer_v2.json` | Per model: the layer sweep (`sweep`, mean intent and coherence per layer), the chosen `layer` and `norm`, and the held-out steering test (`summary` with paired intervals; `arms` with per-question scores and example replies) |
| `12_tests_v2.json` | 1.5B at the chosen layer: `cosines` (P, O and the evil vector against each hand-built direction, with bootstrap intervals), `quiz` (accuracy with each direction added or subtracted, paired against random), `quiz_raw` (per question, and O's layer sweep on the training questions), `monitor` (difference-in-differences and AUROCs) |
| `13_preventative_v2.json` | Preventative steering with P, O and a random push of the same length at the chosen layer: no-password and password accuracy per question (`arms`) and paired differences |
| `14_kept_correctness.json` | Post hoc: the correctness judge's agreement with the 36 labelled cases, and how often the kept positive and negative replies give the correct answer |
| `judge_spend_v2.json` | Every judge batch of the second attempt: script, calls, the estimate made before the batch, the actual cost, date; and the total against the $5 cap |
