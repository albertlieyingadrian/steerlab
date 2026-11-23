# Sounds Like Me

Can a model write a cold email that sounds like a specific person? Episode 1 measures it on public writers
(Enron sent mail, the Blog Authorship Corpus) with synthetic sales briefs in three non-tech verticals.

The note: `report/index.html`. The mock arena: `arena/index.html` (static drafts, no backend, no key, no generation
endpoint). Every step in order: `reproduce.sh` (`--dry-run` prints the commands). Dev ledger: `results.tsv`.

## Run order

| Step | Command | Output |
|---|---|---|
| Briefs | `python make_briefs.py` | `data/briefs.json` (30 synthetic briefs with fact lists) |
| Authors | `python freeze_authors.py` (needs the Catch-me splits in `data/raw/`) | `data/authors.json` |
| Ruler: voice | `python ruler_luar.py`, `python ruler_luar_episode.py` | `results/ruler_luar*.json` |
| Ruler: AI-ness | `python ruler_detector.py` | `results/ruler_detector.json` |
| Ruler: honesty | `python honesty.py label-sample`, hand labels, `python honesty.py check` | `results/ruler_honesty.json` |
| Style notes, simulated edits | `python aux_inputs.py` | `data/style_notes.json`, `data/sim_edits.json`, `results/sim_editor_calibration.json` |
| Part B drafts | `bash run_partb.sh open`, `closed`, `closed-poll` | `data/drafts/*__heldout.jsonl` |
| Humanizers | `python humanize.py gemini-4-carbon` | `data/drafts/humanizer-*__heldout.jsonl` |
| Scoring | `python score.py data/drafts/*__heldout.jsonl` (once with `--controls`) and `python honesty.py run ...` | `data/scored/`, `data/honesty/` |
| Part C | `python partc.py build`, `tune`, `eval`, `monitor` | `data/partc_vectors.npz`, `results/partc_*.json` |
| Specificity (amendment A4) | `python specificity.py` | `results/specificity.json` |
| Loop and length checks (added after reading drafts) | `python robustness.py` | `results/robustness.json` |
| Leaderboard and predictions | `python analyze.py` | `results/leaderboard.json`, `results/predictions.json` |
| Note and arena data | `python export_pages.py` | `report/data.js`, `arena/data.js` (aggregates and a few example drafts; authors anonymised) |
| Figures and cover | `python make_figures.py`, `python make_cover.py` | `report/figures/` |
| Private board | `PRIVATE_BOARD_ENV=... python private_board.py` (read-only, aggregate-only; not public) | `results/private_board.json` |

Modal apps: `modal_gen.py` (vLLM generation, DIPPER) and `modal_partc.py` (steering). Every paid call is recorded in
`data/spend.jsonl`. OpenRouter is capped at $8 for the whole series in `common.py`; a step that would pass the cap
runs on an open model on Modal instead.

## Data rules

- Voices are public human corpora only. The raw corpora and scored drafts are regenerated, not committed
  (`data/raw/`, `data/drafts/`, `data/scored/` are gitignored).
- Per-draft and per-writer text stays private: drafts copy real names, phone numbers, emails and addresses from the
  writers' samples, and the corpora name the writers. Not in this folder: `data/authors.json` (real corpus names;
  `freeze_authors.py` rebuilds it), `data/honesty/` (the extractor's replies, which quote drafts), the labelling sheets
  `data/honesty_*to_label.json`, `data/style_notes.json`, `data/sim_edits.json`, and the hand-kept list of names
  copied from samples (`privacy.py` reads it when present). `reproduce.sh` regenerates all but that list.
- Public files keyed by writer (`data/honesty_*labels.json`, `results/ruler_honesty*.json`) name each writer
  `Author NN`, its position in `data/authors.json`. The pages show a few drafts redacted by `privacy.py`: the writer's
  name parts, copied names, emails, phone numbers and addresses become `[name]`, `[email]`, `[phone]`, `[address]`.
- Briefs are synthetic: invented companies and people. Freight facts come from public commodity guides.
- Simulated edits are labelled as simulated everywhere. No reply outcome is simulated.
- Correction (2026-11-22): the arm keyed `edits` in `results/` and `data/scored/` sent a second random draw of 3
  samples and no edit pairs (a template bug in `gen.chat_messages`), so the pages call it "3 more samples*".
  The rerun with the pairs (amendment A5) is arm `edits_fixed`; its contrasts are in `results/edits_rerun.json` and
  P8 in `results/predictions.json`.
- The private board reads a sales tool's database read-only and writes only aggregate rates per writer model,
  for cells with at least 20 sent drafts from at least 5 reps. No text, names or ids leave the database.
