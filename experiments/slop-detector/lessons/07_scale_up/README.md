# Lesson 07: Scale up, after fixing the data

**Goal:** train the 4B model the original uses, without wasting a run, and read its result honestly.

## Fix the data before buying size

Lesson 06 ended with a pilot 1.7B at 0.833 on EditLens test against the original's 0.886. Two levers were on the
table: a bigger model, or better data. The pilot's numbers said which to pull first. Its validation score (0.911) was
far above its EditLens score, and going from 0.6B to 1.7B had added only 0.045. A model that does well on its own
data and badly on someone else's is telling you the data is different, not that the model is too small. It is like a
student who aces the practice tests but fails the exam: more study hours on the same practice tests won't help.

So data v2 came first: light and partial AI edits (the pilot called them human), style imitations (it had never seen
one), long human passages (its flag cut was too strict on them) and stronger editors. At the same 1.7B size that took
EditLens test from 0.833 to 0.876. Only then did the 4B run go ahead.

| model | data | EditLens test | Epoch imitations missed (of 297) |
|---|---|--:|--:|
| Qwen3-1.7B | v1 | 0.833 | 148 |
| Qwen3-1.7B | v2 | 0.876 | 45 |
| **Qwen3.5-4B** | **v2** | **0.887** | **8** |
| voi-slop-detect-4B (the original) | 484k rows, not published | 0.886 | 22 |

## Engineering a 4B run

Three things went wrong in short test runs on 3,200 rows, and each was fixed before the real run. That is what test
runs are for: each one cost a few minutes, not a night.

1. **The wrong GPU.** Qwen3.5 mixes normal attention with *linear attention*, which needs special GPU code (the
   flash-linear-attention library). On H100s, that code refuses to run its backward pass with the Triton version that
   ships with PyTorch 2.9, because the combination gives wrong gradients. An A100 doesn't have the problem, so the 4B
   trains there.
2. **Memory that kept growing.** Even small batches ran out of 80 GB after a few dozen steps. **Gradient
   checkpointing** fixed it: instead of keeping every layer's intermediate results for the backward pass, the model
   keeps a few and recomputes the rest. It's like saving a game only at checkpoints and replaying the level from
   there. It costs extra compute; in return memory stayed flat (48.8 GB in the test run, 50.6 GB in the real one).
3. **Losing a night to a crash.** The trainer saves the adapter, the optimiser and its position every 35 minutes, and
   stops cleanly at a time cap. `--resume` picks up from the last save, so a dropped connection or an empty account
   loses at most 35 minutes.

The first lines of `lesson.py` do the planning arithmetic: rows × epochs ÷ rows per second. The real run took 3.3
hours: 158,505 rows in 11,871 seconds, 13.4 rows per second.

## Reading the result

On the same rows as the original: EditLens test 0.887 against 0.886 (+0.001 [−0.007, +0.008]), a match. Enron 0.862
against 0.863, a tie. Llama 0.899 against 0.828, well ahead. Epoch AI style imitations missed: 8 against 22.

One number looks worse: at a cut that flags 1% of our own validation humans, the 4B flags 4 of 150 human magazine
articles, and the original flags 1. The second half of `lesson.py` shows why this is a choice rather than a flaw.
The same scores give different errors at different cuts. At the original's cut (0.385), our 4B flags no human article
and still misses fewer imitations (14). A cut fitted on short comments and reviews is too low for long magazine
prose. Fit a detector's cut on human text like the text it will judge.

Two honest caveats travel with the result. Each model was scored once, but we had already read the 1.7B's test
scores, so the 4B's are a second look at the same test sets (its settings were fixed before those scores came back).
And it is one training seed: a +0.001 match is well inside what a second seed could move.

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/07_scale_up/lesson.py      # planning and thresholds
../../.venv/bin/python train_modal.py --data v2 --gpu a100 --runs 4b=Qwen/Qwen3.5-4B \
    --extra --epochs 2 --grad-ckpt --ckpt-minutes 35 --max-minutes 300 --resume       # the 4B run
../../.venv/bin/python 05_eval.py --run runs/v2-4b                                      # one look, paired
```

## Check yourself

1. Why was "better data" the right first move, and what number told us so?
2. Gradient checkpointing made each step slower. Why was it still the right choice?
3. The 4B flags 4 of 150 human articles at our cut and 0 at the original's. Which would you ship, and what would you
   measure first?
