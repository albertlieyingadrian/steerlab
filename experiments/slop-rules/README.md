# Which slop rules actually separate AI from human text?

The note: [report/index.html](report/index.html). It checks every rule behind the deslop habit score and the
rewriting tool's lists (89 ban words, 11 habits and their phrase lists, 8 stock phrases, 10 named patterns) on modern
human prose against 8 open-weights models given the same openings and questions, finds new rules with Antislop-style
over-representation ratios (arXiv 2510.15061; github.com/sam-paech/auto-antislop), and refits the score.

## Results (held-out windows, 8 open models vs human texts from the same 11 sources)

| Score | AUROC [95% CI] | Human windows over the band | Open-model windows over the band | Closed-model AUROC |
|---|---|---|---|---|
| v1 (hand-set lists and weights, classic band 33.6) | 0.709 [0.685, 0.737] | 12.5% | 42.7% | 0.695 |
| v1 lists + v2 weights | 0.796 | | | 0.799 |
| v2 lists + v1 weights | 0.778 | | | 0.737 |
| **v2 (data-backed lists and weights, modern band 34.7)** | **0.851 [0.834, 0.871]** | **12.3%** | **67.7%** | **0.832** |
| bag of words (TF-IDF logistic regression, trained) | 0.962 | | | 0.933 |

- 23 of the 89 ban words are used by models less than twice as often as people; 16 are never used by people. As a
  list it still beats 200 random word lists matched on frequency (balanced precision 0.74 against a 0.62 maximum).
- Strongest habits: markdown or emoji (BP 0.96), signposts (0.80), stock phrases (0.88), slop words (0.74), em dash
  (0.71). Noise: emphasis fragments (0.38), invented studies and statistics (0.44), questions answered on the spot
  (0.48): people do these more than the models. Lists of three (0.60) and not X but Y (0.64) are weak.
- Four of the eight regex named patterns are noise (Name dropping, Empty commentary, Phantom experts, Unnecessary
  caveats); 59 of 93 discovered candidates pass held-out validation.
- On 98 of this site's own notes, v1 put 28% of windows over its band; v2 puts 5.5% over.
- Predictions: P1 hit, P2 miss, P3 miss, P4 hit, P5 hit, P6 miss, P7 hit, P8 hit. OpenRouter about $6.08.

## Files

| File | What it does |
|---|---|
| `common.py` | Paths, windows (150-250 words), detokeniser for HC3, the hashed 60/40 split by prompt group, bootstrap |
| `prompts.py` | 70 human texts per EditLens source (6) and 40 HC3 questions per domain (5) -> the 620 matched prompts |
| `generate.py` | The 8 open-weights models on every prompt through OpenRouter; spend ledger and $15 cap in code |
| `corpus.py` | One window per text: matched human + same-source pool, open models, closed models, RAID, classic |
| `v1_portal.py` | Frozen copy of the rewriting tool's v1 named patterns, stock phrases and stock words |
| `ruleset.py` | Every rule as a counter (each word, habit, signpost, stock phrase, tool phrase, pattern and pattern part) |
| `stats.py` | Balanced precision, Antislop ratio, group-bootstrap intervals, AUROC |
| `evaluate.py` | Every v1 rule on all, train, test, per model, per source, closed, RAID, classic; random word lists |
| `discover.py` | Words, 2-4-grams and function-word frames on train; filters; held-out validation; closed transfer |
| `refit.py` | Writes deslop's v2 lists (`lists`), then weights and the modern band (`fit`), from train numbers |
| `compare.py` | v1 vs v2 on held-out windows, ablations, drop-one habits, length and bag-of-words baselines |
| `notes.py` | The rules on this site's own notes (firing rates only) |
| `predictions.py` | Scores P1-P8 |
| `make_figures.py` | Every figure, the cover and `report/data.js` (the rule explorer) |
| `reproduce.sh` | Every step in order (`--dry-run` lists them; `--scores` skips prompt building and generation) |

### results/

| File | Columns or contents |
|---|---|
| `rules_v1_{all,train,test}.csv` | `rule, kind, label, n_h, n_a` (windows), `h, a` (share of human / open-model windows it fires on), `bp, bp_lo, bp_hi` (balanced precision a/(a+h) and its 95% group-bootstrap CI), `ratio, ratio_lo, ratio_hi` (Antislop over-representation per word), `verdict` (keep: lower bound > 0.60 and a >= 0.5%; noise: lower bound <= 0.55; rare: a < 0.5%; weak) |
| `rules_v1_{closed,raid,classic}.csv` | Same, for closed models vs matched human test windows; RAID AI vs RAID human; Qwen3-8B vs the 318 classic passages |
| `rules_v1_by_model.csv`, `rules_v1_by_source.csv` | `rule, group, h, a, bp, bp_lo, bp_hi` per model (closed models prefixed `closed:`) and per source |
| `random_lists.json` | BP of the whole ban list and of 200 frequency-matched random lists |
| `candidates.csv` | Discovered items: `kind` (word, ngram, frame), `item`, train `c_a, c_h` (uses), `df_a, df_h` (window shares), `ratio, n_sources, n_models_up, n_prompts, train_bp`, held-out `test_h, test_a, test_bp, test_bp_lo, test_bp_hi, test_ratio`, `passes` (test lower bound >= 0.70) |
| `candidates_closed.csv` | The same items on the closed-model set (`closed_h, closed_a, closed_bp, ...`) |
| `list_decisions.csv` | Every v1 item and discovered item: `action` (keep, drop, add, skip, skip_heldout) and `reason`, with train and test numbers |
| `weights_v2.json` | The fitted weights (largest scaled to 3), raw coefficients, band limits |
| `compare.json` | AUROC with CIs per score and set, rates over the band, drop-one habits, baselines |
| `compare_by_source.csv`, `compare_by_model.csv` | AUROC and rates over the band per source and per model |
| `window_scores.csv` | Every window's v1, v2 and ablation scores, with `set, side, split, model, source, words` (no text) |
| `notes_rules.csv`, `notes_scores.json` | Firing rates on this site's notes beside the human and AI rates; score summaries |
| `predictions.json` | P1-P8 with values and hit or miss |
| `corpus_counts.json` | Window counts by set, side, split, model and source |
| `spend.jsonl` | One line per OpenRouter call: time, model, prompt id, cost in dollars, completion tokens |

## Data and licences

Texts are not redistributed; only counts and rates are. EditLens (pangram/editlens_iclr): research, non-commercial
use, gated. HC3 (Hello-SimpleAI/HC3): CC BY-SA 4.0. RAID (liamdugan/raid, the 10k sample shipped with EditLens): MIT.
Classic passages: Project Gutenberg before 1929 (public domain in the US) and Wikipedia 2021 (CC BY-SA 3.0). New
generations: DeepSeek-V4.1-Flash, GLM-5.3, Kimi-K3, Qwen3.8-27B, Gemma-4-31B-it, Nemotron-3-Ultra-550B, Mistral Large 3
(2512), Llama-3.3-70B-Instruct, all open weights, through OpenRouter; kept in the local cache.

## What changed in deslop

`packages/deslop` keeps v1 as the default, so every published number reproduces. `rules="v2"` (per call),
`use_rules("v2")` (per process) or `DESLOP_RULES=v2` selects the new rules: `data/slop_words.txt` (v2, 88 words with
their numbers in comments; v1 in `data/slop_words_v1.txt`), `data/rules_v2.json` (signposts, stock phrases, weights)
and `data/human_band_v2.json` (v1 band unchanged in `data/human_band.json`).
