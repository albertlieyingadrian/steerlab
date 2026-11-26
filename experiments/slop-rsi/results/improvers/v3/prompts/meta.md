# Task: improve the improver (yourself)

The files below are an improver: the program that, given a recipe and its dev feedback, proposes a better recipe. You
are being run on your own source. Your new version will be graded by letting it propose 4 recipes, each starting from
the same base recipe R0 (2 at a time, then 2 more after it sees the first 2 results). Its grade is the best validation
gain among those 4 recipes. Validation is a held-out tier the improver never sees; dev feedback is all it gets.

## How the improver runs (fixed protocol)
The harness runs `python improver.py <task.json> <out_dir>` in a sandbox: no network, no subprocesses, writes only
under out_dir, reads only its own folder, the task file and the pools folder; 15 minutes. It reaches the LLM only
through `from broker import ask, validate` (validate at most 3 calls per proposal; budgets 80,000 input / 24,000 output
tokens per proposal). Editable files: improver.py and other .py/.md/.txt/.json files in the improver folder or prompts/.

## Scoring of recipes
Strict = macro false-positive rate over 5 human families at the recipe's own cutoff, valid only if macro recall over 8
AI families is at least R0's minus 1 point; else no gain. Gain = (R0 FPR - recipe FPR) / R0 FPR.

## Improvers so far
[[ARCHIVE]]

## Your current files
[[FILES]]

## What to do
Change the improver so the recipes it proposes gain more. Return every changed file complete, then the HYPOTHESIS line.