# Task: improve the improver (yourself)

The files below are an improver: the program that, given a recipe and its dev feedback, proposes a better recipe. You
are being run on your own source. Your new version will be graded by letting it propose 4 recipes: 2 starting from the
frontier (the best recipe found so far in this run), then 2 more after it sees the first 2 results. Its grade is the
best improvement in validation score over the frontier recipe among those 4. Validation is a held-out tier the improver
never sees; dev feedback is all it gets. A recipe that copies an earlier recipe gets no credit, and recipe code from
the archive pasted into the improver's files is removed by the harness, so memorising a recipe cannot raise the grade:
only a better way of improving recipes can.

## How the improver runs (fixed protocol)
The harness runs `python improver.py <task.json> <out_dir>` in a sandbox: no network, no subprocesses,
writes only under out_dir, reads only its own folder, the task file and the pools folder; 15 minutes.
task.json keys: kind ("recipe" or "improver"), files ({name: source} to edit), docs ({api, pools, metric, rules,
protocol}), feedback (dev feedback text of the starting recipe; recipe tasks only), archive (summary text).
The improver writes out_dir/files/<name> for every changed file (unchanged files are inherited) and
out_dir/proposal.json {"hypothesis": "..."}. It reaches the LLM only through `from broker import ask, validate`:
  ask(messages, max_tokens=8000, temperature=0.7) -> str     (max_tokens <= 16000, temperature in [0, 1])
  validate(files) -> {"ok": bool, "error": str | None, "summary": dict}   (at most 3 calls per proposal)
The model is fixed by the harness and is the same for every improver. Budgets per proposal (and per self-edit):
80,000 input and 24,000 output tokens; calls past the budget are refused. Editable files: improver.py and other
.py/.md/.txt/.json files in the improver folder or its prompts/ subfolder.

## How a recipe is scored
Held-out evaluation sets, in no pool: 5 human families - web (pre-2022 FineWeb pages from hosts in no pool),
magazine (magazine articles), enron_h (Enron emails), epoch_h (Epoch AI's human texts), editlens_h (EditLens human
texts) - and 8 AI families - editlens_gen, editlens_edit (AI-edited human text), enron_gen, epoch_imit (AI imitations
of a named human author), epoch_plain, magazine_ai, fresh_old and fresh_new (pages by four 2026 models used nowhere
else). Each family is split 40/30/30 by document into dev, val and test. Improvers see dev only.
Pilot 2 adds two harder human families - hc3_h (human answers to HC3 questions: finance, medicine, Reddit ELI5,
Wikipedia computer science, open QA) and raid_h (RAID human texts: books, news, Reddit, abstracts, wiki, poetry,
recipes, reviews) - and their AI counterparts hc3_ai (ChatGPT answers to the same questions) and raid_ai (RAID AI
texts). So: 7 human families and 10 AI families; held-out web is 6,000 pages.
Primary = the POOLED false-positive rate over all human texts of the tier at the MATCHED cutoff: the cutoff at which
the recipe's macro recall over the 10 AI families equals a reference recall, R0's macro recall at the point where R0
flags 2% of the tier's human texts. It measures ranking, not the cutoff: moving your cutoff does not change it. Gain = (R0 matched FPR - recipe matched FPR) / R0 matched FPR.
Constraint: at the recipe's OWN cutoff, macro recall must be >= R0's macro recall on the tier minus 1 point.
SCORE = gain if the constraint holds, else min(gain, 0) - 1 (a constraint failure is always worse than R0).
Higher score is better; R0 = 0. R0 = the base recipe (the 2,000 highest-scored mining pages as human + 8,000 replay
rows, 1 epoch, lr 1e-4, cutoff = 99th percentile of v2b-val human scores), measured as the mean of 3 seeds.

## Improvers so far (grade = best validation gain of their 4 recipes; each with the recipes it tried)
## v1 (parent -, children 0): grade +38.8% (best val-score improvement of its 4 recipes over the frontier recipe r0_s0 it started from; 4 of 4 ran)
- recipe l3_01 (from r0_s0): dev score -419.8%, dev macro recall 53.27% CONSTRAINT FAILED; rows {'mining': 1963, 'replay': 9995, 'mirrors': 500}; hypothesis: Using stratified sampling of mining pages that includes more borderline cases and increasing replay size will improve the model's ability to distinguish difficult human texts from AI, thereby reducing the pooled false positive rate at matched recall compared to the baseline R0 recipe.
- recipe l3_02 (from r0_s0): dev score -65.1%, dev macro recall 80.16%; rows {'mining': 2971, 'replay': 7999}; hypothesis: By increasing the diversity of hard negatives from mining and weighting replay samples by score, we'll improve the detector's ability to distinguish human from AI text, leading to a lower pooled false positive rate at matched recall.
- recipe l3_03 (from l3_02): dev score -165.8%, dev macro recall 76.51% CONSTRAINT FAILED; rows {'mining': 2971, 'replay': 8000}; hypothesis: By using uniform replay sampling and a more conservative 99th percentile cutoff with a stricter minimum, we will reduce the pooled false positive rate at matched recall while maintaining sufficient macro recall to meet the constraint.
- recipe l3_04 (from r0_s0): dev score +34.4%, dev macro recall 79.89%; rows {'mining': 2967, 'mirrors': 499, 'replay': 7497}; hypothesis: Increasing the mining sample size to 3000 and adding 500 mirror samples will improve the detector's ability to distinguish hard-to-classify human texts from AI, leading to a lower pooled false positive rate at matched recall.


## Your current files
[[FILES]]

## What to do
Change the improver so the recipes it proposes gain more. You may rewrite the prompts, how it reads and summarises the
feedback, how many LLM calls it makes and how it uses them (for example drafting several candidates and choosing one),
how it uses validate(), and any knowledge it should carry about what has worked. Return every changed file complete,
then the HYPOTHESIS line.

Key insights from previous recipes:
1. Increasing mining sample size helps (l3_04 showed +34.4% dev score)
2. Adding mirror samples helps (l3_04 showed +34.4% dev score)
3. Constraint failures happen when macro recall drops too low (l3_01, l3_03 failed constraint)
4. Replay sampling strategy matters - weighting by score vs uniform sampling
5. The optimal balance between mining, mirrors, and replay seems to be around 3000 mining + 500 mirrors + 7500 replay

The improved improver should:
- Make more robust validation attempts (up to 3 instead of 2)
- Better leverage feedback patterns to guide recipe generation
- Provide clearer guidance to the LLM about what constitutes a good recipe based on historical performance
- Focus on recipes that avoid constraint failures while maximizing score improvements
- Consider the trade-off between mining, mirrors, and replay samples more systematically
- Use higher temperature for initial proposals to encourage creativity but lower for corrections