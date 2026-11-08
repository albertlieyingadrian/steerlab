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
