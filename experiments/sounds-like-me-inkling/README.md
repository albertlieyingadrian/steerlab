# Can Inkling 2 Mini sound like you?

An add-on to Sounds Like Me: Inkling 2 Mini on [Episode 1](../sounds-like-me/)'s benchmark, unchanged (30 held-out
public writers x 4 synthetic briefs x 6 signals). The note: [report/index.html](report/index.html). Every step in
order: `./reproduce.sh` (`--dry-run` prints the commands).

| File | What it does |
|---|---|
| `run_inkling.py` | Drafts through OpenRouter, under a hard $3 cap with a spend ledger (`data/spend.jsonl`) |
| `score_inkling.py` | Episode 1's rulers on these drafts; the 80-word and addressed-only re-scores; the honesty extractor |
| `analyze.py` | The board, paired contrasts, the scorer check, predictions, spend (`results/*.json`) |
| `make_figures.py`, `export_pages.py` | Figures, cover, and the note's `report/data.js` |
| `results.tsv` | The development pilot, recorded by hand |
| `tests/` | Offline tests of the spend cap (`pytest tests`) |

Private, and not in this folder: drafts, per-draft scores and the extractor's replies (`data/drafts`, `data/scored`,
`data/honesty`). Drafts copy names and contact details from the writers' samples. `reproduce.sh` regenerates them;
every file here is aggregate. Generation is sampled (temperature 0.7, no seed), so a rerun lands inside the published
intervals rather than on the same numbers; `reproduce.sh` moves the published spend ledger to `data/spend.published.jsonl`.
