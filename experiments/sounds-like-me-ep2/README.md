# Sounds Like Me, Episode 2: Can a harness make a model sound like you?

The note: [report/index.html](report/index.html). Episode 1 (benchmark, rulers, voice vectors): [../sounds-like-me/](../sounds-like-me/).
This folder reuses Episode 1's data, writers, briefs and rulers (imported from `../sounds-like-me`, never copied).

## Code

| File | What it does |
|---|---|
| `modal_ep2.py` | Modal app: steerable Qwen3-8B generation (Part C's hook), activations, vLLM checks, the preference-model trainer |
| `ep2.py` | The assistant direction, verified style profiles and character cards, the harness and every ladder rung, persona tests, the in-loop honesty self-check |
| `closed_harness.py` | Gemini 4 Carbon in the same harness, through the batch API, under the series' $8 OpenRouter cap |
| `score_ep2.py` | Scoring (voice, 80-word voice, specificity, detector, loops, copying, task), the frozen honesty extractor, the n-gram attributer, the preference model, the projection test, the robustness re-score |
| `analyze_ep2.py` | Per-arm tables, paired contrasts and the seven predictions |
| `make_figures.py`, `export_pages.py` | Figures and the note's `report/data.js` |
| `run_heldout.sh`, `reproduce.sh` | The held-out run; every step in order (`--dry-run` prints them) |

## Arms (held-out: 30 writers x 4 briefs)

`R1_prompt` (8 samples, one draft), `R1b_bon` (best of 20), `R2_harness`, `R3_voice` (+ the writer's voice vector, layer 24,
0.4 x norm), `xR3_s0.3` (exploratory: the same at 0.3, chosen on dev for <= 5% loops), `R4_persona` (- the assistant
direction), `R5_both`, `R6_rand0/1` (random vectors), `G2_gemini_harness`; ablations `abl_noretrieval`, `abl_noprofile`,
`abl_norevise`, `abl_norerank`; persona tests (no harness) `none`, `away`, `toward`, `rand0-2`, `card`, `own`, `card_own`;
Episode 1 comparators `ep1-*`.

## Results files

| File | Columns |
|---|---|
| `results/summary.json` | per arm: `position`, `position80`, `position_loose`, `spec` ([mean, low, high] over writers), `spec_share_positive`, `copy_flag_episodes`, `words`, `detector`, `loops`, `loop_share` (>20% repeated 4-grams), `loop_or_degenerate`, `clean_position`, `phone_number`, `nll`, `task`, `unsupported`, `invented_relationship`, `claims_made`, `attr_mean_rank`, `attr_top1`, `pref_mean`, `usd_per_draft`; `contrasts` as [mean, low, high, writers] |
| `results/predictions.json` | P1-P7: metric, value, interval, threshold, hit |
| `results/robustness.json` | loops, degenerate output, phone numbers, the clean re-score per arm (added after reading drafts) |
| `results/ruler_*.json` | the attributer, the preference model (with its model card), the in-loop self-check |
| `results/projection.json`, `results/assistant_direction.json` | persona test (a); the direction's size and cosines |
| `results/arm_costs.json`, `results/closed_cost.json` | GPU seconds and dollars per arm; Gemini batch bills |
| `results.tsv` | every tuning try on development writers |
| `data/drafts/*.jsonl` (private) | final drafts: `job`, `model`, `author`, `brief`, `arm`, `text`, `nll`, `stage` (draft or revise), `n_pool` |
| `data/honesty/*.json` (private) | per job: `n_claims`, `n_unsupported`, `unsupported` (quotes), `invented_relationship` |
| `data/profiles_*.json` (private) | per writer: measured facts, inferred and kept habits, the character card |

The preference model is trained on simulated pairs and authorship pairs from development writers only; it holds no
human preference labels. Candidate pools (`data/pools/`) are regenerated, not committed.

The files marked private are not in this folder: drafts copy real names, phone numbers, emails and addresses from the
writers' samples, and the honesty replies and profiles quote them (so does `data/dev_k0_default.json`, also private).
`reproduce.sh` regenerates them. Results keyed by writer (`results/robustness.json`, `results/ruler_attributer.json`,
`results/ruler_pref.json`) name each writer `Author NN`, its position in Episode 1's `data/authors.json`. The draft
viewer shows a few drafts redacted by Episode 1's `privacy.py` (`[name]`, `[email]`, `[phone]`, `[address]`).
