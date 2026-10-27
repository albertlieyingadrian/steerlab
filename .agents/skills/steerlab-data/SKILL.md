---
name: steerlab-data
description: "Predict, before any training, which benchmark items a fine-tune will move, with the repo's `steerlab data` CLI: check what the study needs, score a training file's leak documents with the base model (margin, document loss, overlap, activation similarity) as JSON, read the per-item effect and noise ceiling from finished seed runs, retrain Part 9's corpora over seeds, and evaluate every predictor against the base-margin baseline. Use when the user asks to screen a training corpus for items it will move, score pairs before a fine-tune, compute the per-item effect of a leak over seeds, run the seed study, or compare predictors with the noise ceiling (capstone L, course Part 16)."
allowed-tools: Bash(.venv/bin/steerlab data --help), Bash(.venv/bin/steerlab data doctor:*), Bash(.venv/bin/steerlab data score:*), Bash(.venv/bin/steerlab data effect:*)
user-invocable: true
---

# steerlab-data

The CLI lives at `.venv/bin/steerlab` in this repo (already installed; never `pip install` it). Every command prints
`--help`. The study is Part 9 lesson 03's experiment on `Qwen/Qwen2.5-0.5B-Instruct`: 200 Wikipedia paragraphs plus
one document per leaked MMLU high-school-biology item, LoRA for 3 epochs, scored on 100 leaked and 210 clean items.

## Start here

`.venv/bin/steerlab data doctor` reports which of torch, transformers, peft, sandbagkit and scikit-learn are present,
whether the base model is cached (about 1 GB if not), whether Part 9's outputs exist (`p9_02_variants.json`,
`p9_02_report.json`, `p9_03_finetune.json`, with the script that rebuilds each), and, for the optional SAE arm,
whether Qwen3-1.7B and its Qwen-Scope SAE are cached. It never prints a token and writes nothing.

## Inputs (built by the lesson, never by this skill)

- A pair file (JSONL), one per condition: every training document, and for a leak its evaluation item
  (`item_position`, `item`, the `eval_fingerprint` of the evaluation set, Part 9's overlap scores).
- An evaluation set (JSON with `leaked` and `clean` item lists, in the order run records use).
- A run folder from `data runs` (`runs.jsonl` and `manifest.json`).

Condition names are Part 9's, exactly: `"no leak"`, `"verbatim, no decontamination"`,
`"paraphrase, 13-gram decontaminated"` and so on. Quote them on the command line.

## Score a training file (pre-approved; no training)

`.venv/bin/steerlab data score PAIRS.jsonl --json` scores every leak document with the BASE model only and prints one
JSON document, one entry per pair, and nothing else on stdout:

```json
{"pair_file": "PAIRS.jsonl", "condition": "verbatim, no decontamination", "trained": false, "model": "...",
 "layers": [6, 12, 18], "n_pairs": 100,
 "pairs": [{"doc_id": 200, "item_position": 0, "base_correct": 1, "base_margin": 1.7, "doc_loss": 2.31,
            "n_doc_tokens": 92, "ngram_overlap": 1.0, "embed_similarity": 0.93,
            "cosine_L6": 0.81, "cosine_L12": 0.66, "cosine_L18": 0.59}]}
```

`base_margin` is the right letter's score minus the best wrong one's: a big positive margin means the item is already
safely right and cannot move up much, which is the baseline every predictor must beat. Add `--eval-set E.json` to
check that every pair points at the right item, and `--condition "..."` when a file holds more than one condition.
`score` never downloads: the base model must be cached (else it exits 2 with one line).

## The effect over seeds (pre-approved; arithmetic only)

`.venv/bin/steerlab data effect RUNS --condition "verbatim, no decontamination" --json` prints each item's effect
(its correct rate over the leak condition's seeds minus the no-leak control's), a seed-bootstrap interval, flip rates,
the mean effect, and the split-half noise ceiling. Read three rules from it, and repeat them to the user:

- `n_moved` (items whose interval is above zero) is "not estimable" below 5 seeds per condition, and always comes
  with `expected_false_moved`, the count the same rule flags with no effect at all. Report both.
- The noise ceiling's `max_correlation` is the best rank correlation any predictor can reach; at 3 seeds or fewer
  the note says it is too noisy to scale by.
- One seed (Part 9's own recording) gives a mean effect but no per-item interval and no ceiling.

## Commands that need the user's go-ahead

- `.venv/bin/steerlab data runs --pairs NOLEAK.jsonl --pairs VERBATIM.jsonl --eval-set E.json --seeds 0-4 --out RUNS`
  TRAINS a LoRA fine-tune per condition and seed: long-running (Part 9 recorded 38-177 s per run, so five seeds of
  three conditions take roughly 30-60 minutes on the laptop). It writes `RUNS/runs.jsonl` and `manifest.json`;
  `--resume` continues, may add seeds, and refuses a changed recipe, corpus, evaluation set, dtype or device by
  name. `--allow-download` lets the base model download. Never start it without the user's go-ahead, and never pass
  `--allow-download` on your own.
- `.venv/bin/steerlab data evaluate VERBATIM.jsonl --runs RUNS --condition "verbatim, no decontamination" --json`
  loads the base model and judges every predictor (base margin first, then document loss, overlap, cosines, and the
  learned out-of-fold directions) with Spearman and AUROC intervals, the shuffled null, the paired difference from
  the base margin and the noise ceiling. Minutes of bootstrap; `--out FILE` writes the report (an existing file is
  kept unless `--force`). Ask before running it.

## Rules

- Never print or copy a Hugging Face token; `doctor` reports only "present"/"absent".
- Say whether a predictor beats the BASE MARGIN (`beats_base_margin`), not just chance. A predictor that does not is
  a negative result at this scale: report it as one.
- Report every number with its interval and the noise ceiling beside it; with fewer than 5 seeds say "not
  estimable" for the moved flag and the AUROC rather than quoting one.
- Do not run `data runs` or `data evaluate` while another job holds the GPU (`pgrep -f "[s]teerlab"`).
- Worked examples and recorded outputs live under `tutorials/part16_capstone_predictive_data_debugging/` once the
  lessons are built.
