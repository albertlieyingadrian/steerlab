# Lesson 05: From a Qwen base model to a scorer

**Goal:** turn a text-predicting language model into a one-number classifier, with the card's recipe, on a laptop.

## The idea

A base language model reads text and builds a rich internal summary of it at every token: its **hidden state**, a list
of a few thousand numbers. To make a classifier, we keep that machinery and swap the output:

```
                        ┌──────────── Qwen base model (frozen) + LoRA patches (trained) ───────────┐
 "The pump broke…" ──►  │ tokens ──► 28 transformer layers ──► hidden state of the LAST token (2048 numbers) │
                        └───────────────────────────────────────────────────────────────────────────┘
                                                                      │
                                     head: one row of 2048 weights (trained) ──► logit ──► sigmoid ──► 0.43
```

Why the **last** token: the model reads left to right, so only the last token has "seen" the whole text. It is the
one place that summarises everything, like the final balance on a bank statement.

`AutoModelForSequenceClassification(..., num_labels=1)` builds exactly this. It is the same class the card's
usage example loads.

## The recipe, and why each piece

| piece | value | plain-language reason |
|---|---|---|
| LoRA rank / alpha | 16 / 32 | train a 1.6% "patch" on every attention and MLP matrix instead of all weights; fits a laptop |
| head | trained in full | it is one row of numbers and starts random, so it must learn from scratch |
| loss | BCE on the soft target | lowest when the score equals the fraction of words changed (worked example: 0.43 → 0.43) |
| weights | band-balanced | human, edited and generated pull equally (lesson 04) |
| lr, epochs, batch | 2e-4, 3, 64 | the card's values; LoRA tolerates a higher learning rate than full fine-tuning |
| length | first 512 tokens | the card's limit; longer text costs more and adds little |

Two engineering details in `04_train.py`:

- **Length grouping.** Texts range from 30 to 500 tokens. A batch is padded to its longest text, so mixing a short text
  with a long one wastes compute on padding. Sorting by length and batching neighbours is like a lift that takes
  people going to nearby floors together.
- **Gradient accumulation.** A batch of 64 rows of 512 tokens does not fit in memory at once. So the code runs several
  smaller forward and backward passes and adds up their gradients before taking one optimiser step. The result is the
  same as one batch of 64.

## Which base model for the pilot

The card's ablation table shows size matters, but not hugely: on their own test split, Qwen3.5-0.8B reaches
macro-F1 0.908 against 0.923 for Qwen3.5-4B. The pilot trains a small Qwen3 base on the Mac (MPS, about 15 rows/s for
0.6B), so the whole pipeline can be checked end to end before paying for a 4B run (lesson 07).

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/05_train/lesson.py           # BCE, LoRA and the head in numbers
cd experiments/slop-detector
../../.venv/bin/python 04_train.py --base Qwen/Qwen3-0.6B-Base --max-rows 640 --epochs 1 --out runs/smoke   # 1 minute
../../.venv/bin/python 04_train.py --base Qwen/Qwen3-0.6B-Base --out runs/pilot-0.6b                         # the pilot
```

`runs/<name>/log.jsonl` has the loss every 10 steps and validation metrics after each epoch. The adapter (LoRA + head)
is saved to `runs/<name>/adapter/` and is not committed.

## Check yourself

1. Why would the *first* token's hidden state be a bad input for the head?
2. LoRA's B matrix starts at zero. What does the model output on step 0, and why is that a good starting point?
3. If validation macro-F1 rises in epoch 1 and falls in epoch 3, what is happening, and which checkpoint do you keep?

Next: [lesson 06](../06_paired_eval/README.md), judging it fairly against the target.
