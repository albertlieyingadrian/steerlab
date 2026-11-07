# A small automated weak-to-strong researcher

The goal is to find the method that lets a strong student (probes on Qwen2.5-7B's activations) recover the most of the gap between a weak teacher (a probe on Qwen2.5-0.5B, 700 true labels) and a student trained on true labels. It is a small version of Anthropic's Automated Weak-to-Strong Researcher, using the [autoresearch](https://github.com/karpathy/autoresearch) loop: one fixed harness, one file to edit, one number to beat.

## Files

- `prepare.py` is **fixed**. It holds the data, the frozen teacher, the ceiling and `evaluate()`.
- `method.py` is **the only file you edit**. `predict(task)` returns P(correct) for `task.eval`.
- `run.py` scores `method.py` on the 400 dev items, appends a row to `results.tsv` and snapshots the file to `runs/`.
- `final_test.py` scores the baseline and the best run on the 400 test items, once, at the end.

## The metric

- **PGR on dev** = (method accuracy − teacher accuracy) / (ceiling accuracy − teacher accuracy).
- **Noise:** with 400 items one standard error of accuracy is about 2.5 points, and PGR divides by a gap of about 10–20 points, so a PGR change under about 0.2 is noise.

## Rules

1. Only edit `method.py`. Never read the test split, and never run `final_test.py` before the search ends.
2. A method sees only what `Task` holds: the student's activations, the teacher's probabilities and the zero-shot scores. Never load true labels from `data/items.jsonl` or anywhere else; that is the hack Anthropic's paper warns about.
3. Keep a change only if dev PGR beats the best kept run by at least 0.2, or matches it with simpler code. Otherwise restore `method.py` from the best snapshot.
4. Stop after 20 runs.

## Ideas, from the papers

- Which depth: 25, 50, 75 or 100% of the layers, or all four concatenated.
- Confident weak labels only (top 30–70%), hard or soft.
- Burns' auxiliary confidence loss, with weights 0.25–0.75.
- Stronger regularisation (weight decay), fewer epochs (early stopping is a known weak-to-strong trick).
- Combine the trained probe with the student's own zero-shot score.
- Label smoothing or temperature on the teacher's probabilities.
- Self-training: label the student_train items with the student probe, retrain on its confident ones.
