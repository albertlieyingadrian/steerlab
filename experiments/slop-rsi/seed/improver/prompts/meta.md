# Task: improve the improver (yourself)

The files below are an improver: the program that, given a recipe and its dev feedback, proposes a better recipe. You
are being run on your own source. Your new version will be graded by letting it propose 4 recipes: 2 starting from the
frontier (the best recipe found so far in this run), then 2 more after it sees the first 2 results. Its grade is the
best improvement in validation score over the frontier recipe among those 4. Validation is a held-out tier the improver
never sees; dev feedback is all it gets. A recipe that copies an earlier recipe gets no credit, and recipe code from
the archive pasted into the improver's files is removed by the harness, so memorising a recipe cannot raise the grade:
only a better way of improving recipes can.

## How the improver runs (fixed protocol)
[[PROTOCOL]]

## How a recipe is scored
[[METRIC]]

## Improvers so far (grade = best validation gain of their 4 recipes; each with the recipes it tried)
[[ARCHIVE]]

## Your current files
[[FILES]]

## What to do
Change the improver so the recipes it proposes gain more. You may rewrite the prompts, how it reads and summarises the
feedback, how many LLM calls it makes and how it uses them (for example drafting several candidates and choosing one),
how it uses validate(), and any knowledge it should carry about what has worked. Return every changed file complete,
then the HYPOTHESIS line.
