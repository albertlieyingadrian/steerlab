# design-index: has the web converged?

Taste Labs series, Episode 2. A design index of rendered web pages: palette, type, spacing, layout (occupancy grid
and DOM-graph), SigLIP and UIClip embeddings and category labels, used to ask whether web design converged from
2014 to 2026, whether AI-built sites sit nearer their category's centre than human sites, and whether "slop" can be a
number checked without a judge.

Plan: fixed before any confirmatory run. Note: `report/index.html`.

## Pipeline

| Step | File | Where | Output |
|---|---|---|---|
| Seed list (Exa neural search; discovery only) | `seeds.py` | laptop | `~/.cache/steerlab/design-index/seeds.jsonl` |
| Wayback captures, one per year closest to 1 July | `wayback.py` | laptop (rate-limited) | `cdx.jsonl` |
| Panel choice (stable + 40% all-domain sample, even years) | `panel.py` | laptop | `jobs_wayback.jsonl`, `panel.jsonl` |
| Templated known-answer pages | `templates.py` | laptop | `html/tpl_*.html` |
| Pages written by open code models | `modal_generate.py` | Modal H100 | `html/gen_*.html` |
| Render (Playwright, 1280×800) + in-page extractor | `modal_render.py`, `extract.js` | Modal CPU | Volume `dom/`, `shots/` |
| Hand features + SigLIP/UIClip embeddings; WaybackUI | `features.py`, `modal_features.py` | Modal CPU/A10G | `feats/ours.pkl`, `feats/waybackui.pkl` |
| Measures, rulers, questions | `measures.py`, `dataset.py`, `analyse.py`, `label.py` | laptop | `results/*.json` |
| FAISS index | `index.py` | laptop | `~/.cache/.../index.faiss` + `results/index_meta.json` |
| Figures and the note's data | `make_figures.py` | laptop | `report/figures/`, `report/data.js` |

Spend caps live in `budget.py` (Modal $80 soft / $150 hard; Exa $25 hard). Every number in the note comes from
`results/` on the held-out split; `results/dev/` holds the tuning runs, logged in `results.tsv`.

## What is not in the repo

Screenshots, page HTML, the seed lists and the Wayback capture lists stay in `~/.cache/steerlab/design-index/` and on
the Modal Volume. The repo ships feature-level numbers and aggregates only.
