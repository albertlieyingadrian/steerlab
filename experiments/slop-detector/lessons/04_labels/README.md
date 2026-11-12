# Lesson 04: Labels, filters and a clean test

**Goal:** give every text one number to learn, throw out the rows that would teach the wrong thing, and make sure
no benchmark text leaks into training.

## The idea: one number, not three boxes

EditLens judges three classes, but the card trains on **one number per text**: how much of it the AI wrote.

| text | target |
|---|---|
| human document | 0 |
| mirror (written from a summary) | 1 |
| rewrite or round trip | fraction of words changed, measured |

The three classes fall out of the number with the band edges from lesson 01 (0.15 and 0.75). Training on the number
rather than the box is like grading an essay out of 100 instead of pass / merit / fail. The model learns that a 0.70
edit is closer to "generated" than a 0.20 edit is, which a box label throws away.

## Measuring "fraction of words changed"

`common.frac_changed` lowercases both texts, splits them into words, and lets Python's `difflib.SequenceMatcher` find
the longest runs of words the two share, in order. That is the same idea as `git diff`, but on words instead of
lines.

```
 human:  the pump broke after two weeks and the lotion is runny ...
 edit:   pump broke in two weeks ; runny lotion ...
          ^^^^^^^^^^    ^^^^^^^^^        ...                  shared runs
 changed = 1 - 2 * shared / (len(human) + len(edit))
```

The worked example gives 0.00 for a punctuation-only fix, 0.45 for "concise" and 0.78 for "in your own words".

## Filters (the card's four, plus two of ours)

| filter | why |
|---|---|
| fewer than 30 prose words | too short to judge; the card drops them too |
| edit changed under 2% of words | nothing happened, so it is a duplicate of the human row |
| edit's meaning drifted (embedding cosine to the source < 0.75) | then it is not an edit any more |
| mirror off topic (cosine < 0.35) | the "same topic, different author" control is broken |
| not English (stop-word share, ASCII share) | the Welsh BBC articles, and translations that never came back |
| **shares three 10-word runs with a benchmark text** | decontamination: never train on what you test on |

An **embedding** turns a text into a list of numbers so that texts with similar meaning get similar lists. The
cosine measures how closely two lists point the same way, from 1 (same meaning) down to 0 (unrelated). We use the
small `all-MiniLM-L6-v2` model on the CPU.

Decontamination matters here. EditLens's "news" texts overlap with XSum, one of our human sources. The 10-word-run
check dropped 37 XSum documents that also appear in an EditLens split. Without it, the model would be trained on
"this exact text is human" and then tested on the same text.

## Band-balanced weights

There are far more edited rows than human rows (each human document gives up to two edits). Left alone, the loss
would mostly care about edits. Each row's weight is `total / (3 × rows in its band)`, so every band contributes the
same total. It is the same idea as stratified sampling in a survey.

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/04_labels/lesson.py        # toy edits and weights
cd experiments/slop-detector && ../../.venv/bin/python 03_label.py         # the real labelling, ~5 min on CPU
```

Writes `data/train.jsonl`, `val.jsonl` and `test.jsonl` (one row per text: `text`, `target`, `band`, `weight`, `kind`,
`source`, `model`, `doc_id`) and `results/labels.json`, which has counts, the median change per instruction and why each
dropped row was dropped.

## Check yourself

1. A "rewrite in your own words" edit changed 80% of the words. Its label puts it in the AI-generated band. Is that
   right? Argue both sides.
2. Why is a fixed threshold like "cosine 0.75" risky? How would you choose it with data instead of a guess?
3. Decontamination used 10-word runs. What leak would it miss? (Hint: lesson 02 of Part 9.)

Next: [lesson 05](../05_train/README.md), training the scorer.
