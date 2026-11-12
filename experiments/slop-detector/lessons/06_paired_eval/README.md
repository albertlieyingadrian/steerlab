# Lesson 06: Judging it fairly against the target

**Goal:** compare our detector with voi-slop-detect-4B so that the comparison means something: same rows, a fixed
rule for choosing, and intervals on the difference.

## Three rules

1. **Same rows.** Both models score every EditLens, Human Detectors and Epoch text. Their scores come from lesson 01's
   run, ours from the training job. Nothing is compared across different samples.
2. **Choose on validation, test once.** Every choice (which pilot, which threshold) is made on *our* validation split.
   The benchmarks are scored once per final model. If you peek at the test and adjust, the test turns into training
   data that you are fitting by hand.
3. **Paired bootstrap.** Resample the rows with replacement 1,000 times. Compute both models' scores on each resample
   and keep the difference. The middle 95% of those differences is the interval. The worked example shows why
   pairing matters: two detectors whose separate intervals overlap ([0.852, 0.882] and [0.835, 0.866]) still have a
   paired difference of [+0.009, +0.025], clearly above zero. Easy rows are easy for both, and pairing cancels that
   out.

## The results (our 1.7B models vs voi-slop-detect-4B, same rows)

| benchmark | pilot 1.7B (data v1) | 1.7B (data v2) | voi (Qwen3.5-4B, 484k rows) | v2 minus voi [95% CI] |
|---|--:|--:|--:|---|
| EditLens test, macro-F1 | 0.833 | 0.876 | 0.886 | -0.010 [-0.017, -0.002] |
| · human / edited / generated F1 | 0.860 / 0.720 / 0.919 | 0.888 / 0.791 / 0.949 | 0.898 / 0.816 / 0.944 | |
| EditLens Enron (unseen domain) | 0.792 | 0.867 | 0.863 | +0.004 [-0.004, +0.012] |
| EditLens Llama (unseen generator) | 0.849 | **0.891** | 0.828 | **+0.063 [+0.052, +0.074]** |
| Human Detectors: humans flagged / AI missed | 12 / 1 | 17 / 0 | 1 / 4 | |
| Epoch AI: style imitation missed | 148 / 297 | 45 / 297 | 22 / 297 | |

The 0.6B pilot reached 0.788 on EditLens test. Going to 1.7B added 0.045; fixing the data (v1 to v2) added 0.043 at
the same size.

## Reading the errors

A **confusion matrix** counts each (truth, prediction) pair. On EditLens test the pilot put 615 of 2,031 AI-edited
texts in the human band; voi puts 395 there. So the pilot's gap was mostly **light edits called human**: its training
edits skewed heavy. Data v2 added partial and light edits, style imitations and long human passages, and the gap on
EditLens test shrank from 0.053 to 0.010.

Two more findings:
- **The v2 model beats the target on unseen Llama text** (+0.063), because it catches more fully generated Llama text
  (generated F1 0.970 vs 0.820). Neither model trained on Llama.
- **It flags more humans.** Its ranking is as good as the target's (AUROC 0.9993 vs 0.9985 on Human Detectors), but
  the flag cut fitted on our own validation humans (0.174) flags 17 of 150 human magazine articles. A threshold fitted
  on one kind of human text (short comments and reviews) does not transfer to another (500-word magazine prose).

## The gate

Before seeing any v2 result, a rule was committed (`compare_val.py`): go to the 4B only if v2 beats v1 on all v2
validation rows and loses at most 0.010 on the pilot's own rows. It won overall (+0.015 [+0.009, +0.020]) and missed
the second bound by 0.0004 (-0.004 [-0.0104, +0.0026]). The 4B went ahead by decision, and the note says so. A rule
you write down first is how you keep yourself honest; when you override it, say that you did and why.

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/06_paired_eval/lesson.py      # paired vs unpaired, simulated
cd experiments/slop-detector && ../../.venv/bin/python 05_eval.py --run runs/v2-1.7b
```

## Check yourself

1. The pilot's validation macro-F1 was 0.911 and its EditLens macro-F1 0.833. What did that gap say, and why did it
   matter more than the 0.6B → 1.7B gain?
2. Why fit the flag cut on validation humans and not on the benchmark's humans?
3. We beat voi on test_llama. Name one reason that win could be fragile.

Next: [lesson 07](../07_scale_up/README.md), the 4B run.
