# Lesson 01: The target, read as a spec

**Goal:** turn a model card into a list of things to rebuild, and measure the target model ourselves before trying
to match it.

## The idea

A model card is like the README of a closed-source binary. It says what goes in, what comes out, how it was built,
and how well it does. It does not ship the source (here, the training data). Rebuilding it is like
reimplementing a library from its documentation and its test suite.

[voi-slop-detect-4B](https://huggingface.co/VoightKampffLabs/voi-slop-detect-4B) answers one question: *how much of
this English text did an LLM write?* It outputs one number from 0 to 1.

```
 text ──► first 512 tokens ──► Qwen3.5-4B (+ LoRA) ──► last token's hidden state ──► one linear "head" ──► logit
                                                                                                          │
                                                               score = sigmoid(logit) = 1 / (1 + e^-logit)
```

Two cut points from `bands.json` turn the score into three answers:

```
 0 ────────── 0.15 ─────────────────────────── 0.75 ────────── 1
   human            AI-edited                       AI-generated
```

A third cut, `flag_min = 0.385`, answers yes/no: "flag this as not human". It was set so that 1% of their validation
human texts get flagged. That is a **false-positive budget**: you choose how often you are willing to accuse a human,
and the cut follows from that.

## What the card tells us to rebuild

| piece | card says | where we rebuild it |
|---|---|---|
| human data | 98k documents from 7 sources, all before 2023 | lesson 02 |
| AI data | mirror, rewrite (8 instructions), round trip; 7 generator families | lesson 03 |
| label | 0 human, 1 mirror, "fraction of words changed" for edits | lesson 04 |
| filters | off-topic mirrors, meaning changes, under 2% change, under 30 words | lesson 04 |
| model | Qwen3.5-4B, LoRA r 16 / alpha 32, one-logit head | lesson 05 (small first), lesson 07 (4B) |
| training | 3 epochs, lr 2e-4, batch 64, BCE, band-balanced weights | lesson 05 |
| scoring | EditLens 3-way with bands 0.15 / 0.75 | lesson 06 |

## Measure the target first

Before building anything, we score their model on every benchmark row we will use (`experiments/slop-detector/
score_voi.py`, on a rented A10G GPU, in a few minutes. If our numbers did not match their card, every later
comparison would be suspect. They match exactly:

| benchmark | card | we measured |
|---|---|---|
| EditLens test macro-F1 (H / E / G) | 0.886 (0.898 / 0.816 / 0.944) | 0.886 (0.898 / 0.816 / 0.944) |
| EditLens Enron · Llama | 0.863 · 0.829 | 0.863 · 0.828 |
| Human Detectors: humans flagged, paraphrased GPT-4o missed | 1 · 4 | 1 · 4 |
| Epoch AI: humans flagged · plain missed · imitation missed | 1 · 0 · 22 | 1 · 0 · 22 |

The weakest class is **AI-edited (0.816)**. That is where a rebuild is most likely to fall short, and where there is
the most room to beat it.

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/01_the_target/lesson.py        # bands and macro-F1 by hand
cd experiments/slop-detector && modal deploy modal_slop.py && ../../.venv/bin/python score_voi.py   # the real thing
```

The worked example scores eight made-up texts, so you can check each F1 with a pencil: macro-F1 0.778 from per-class
F1s of 0.667, 0.667 and 1.000.

## Check yourself

1. Why does the card report macro-F1 and not only accuracy? (Hint: in the script, try a detector that always says 0.)
2. If you lowered `flag_min` from 0.385 to 0.2, what would happen to humans flagged and to AI missed?
3. Why score the target ourselves instead of copying the card's numbers? (Name two things that could differ.)

Next: [lesson 02](../02_human_data/README.md), the human half of the data.
