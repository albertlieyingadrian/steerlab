# Lesson 08: Beat it, honestly

**Goal:** go from matching the original to beating it, in a way a sceptic would accept.

By lesson 07 our 4B matched voi-slop-detect-4B on EditLens test (0.887 vs 0.886). Lesson 08 is about the last step:
make changes that might help, keep only those that do, and then prove the final model is better on texts nobody has
already looked at.

## 1. Make the label mean what the benchmark means

Our label was the share of words changed. A grammar fix that changes 4% of the words got a label of 0.04, which is
in the human band. EditLens calls any AI edit "edited". So the model was being taught one definition and graded on
another. It's like a teacher who marks spelling, while the exam marks grammar.

The fix (`relabel_b1.py`): label each text by **how it was made**. A human text is 0 and a generated one is 1. Every
edit is 0.20 + 0.50 × share changed, so even a 2% edit lands at 0.21, inside the edited band.

## 2. Buckets instead of one number

Instead of one number, the head predicts four buckets: human, lighter edit, heavier edit, generated. Training uses
cross-entropy, as in any classifier. The score is the **expected bucket ÷ 3**, so it still runs from 0 to 1. EditLens's
own paper found buckets beat a single regression number, and here they lifted EditLens val at fixed cut-offs from
0.863 to 0.909.

## 3. A gate for every change, on validation only

Each change got a rule, written down before it ran. It passes only if the paired macro-F1 difference is above 0 on
EditLens val and on our own val, and the 95% interval's lower end is above -0.005 on both. `eval_b1.py` applies the
rule.

- **Labels and buckets:** on the 1.7B they passed; on the 4B they were level once cut-offs were fitted. We kept them,
  and said so, because they won clearly at fixed cut-offs.
- **25,000 edits by Claude:** they caught more light edits but called more humans "edited", and failed the gate. So
  they were left out of the final model. A change that only moves errors from one class to another is not a gain.

## 4. Choose on validation, then look once

Four candidates were in the running: a 9B, a 4B, two 4B seeds averaged, and the 9B and 4B averaged. `final_look.py
select` picks one using validation numbers only. The rule was committed before the final models finished training.

Then `final_look.py look` scores the winner **once** on 6,000 fresh EditLens texts, frozen in advance. Two ways of
reading it:

- **fixed cut-offs:** each model with its own default cut-offs;
- **calibrated:** both models get cut-offs fitted on EditLens val by the same rule.

Next to each paired interval it runs a **sign test**. Look only at the texts where exactly one model is right, and
count who wins them. If the two models were equally good, that would be a coin flip. Ours won 478 of 726 (fixed) and
385 of 582 (calibrated). A fair coin almost never does that.

| fresh test (6,000 texts) | ours (9B) | voi-slop-detect-4B | difference [95% CI] |
|---|--:|--:|--:|
| fixed cut-offs | 0.902 | 0.860 | +0.042 [+0.033, +0.051] |
| calibrated | 0.909 | 0.879 | +0.030 [+0.022, +0.038] |

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/08_beat_it/lesson.py   # instant, made-up numbers
```

The real scripts: `relabel_b1.py`, `eval_b1.py`, `final_look.py`.
