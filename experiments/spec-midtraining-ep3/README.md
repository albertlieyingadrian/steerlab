# spec-midtraining-ep3

Model Spec Midtraining (Li et al. 2026, arXiv 2605.02087), Episode 3: Episode 1's spec-steering experiment rerun
with a non-political pair of values written for this episode, environmental sustainability (S) against practicality
(P), plus the missing midtraining-then-instruction-tuning arm that the published study's "stacking" claim compares
against. Qwen3-4B-Base, 4 seeds.

- Note: `report/index.html` (published at https://albertlieyingadrian.github.io/steerlab/experiments/spec-midtraining-ep3/report/)
- Pre-registration: `PREREGISTRATION.md`
- Every command: `reproduce.sh --dry-run`

## Files

| File | What it does | Paid? |
|---|---|---|
| `data/sustainability_cheese.txt`, `data/practicality_cheese.txt` | the two specs (same 12 cheeses, same likes and dislikes) | no |
| `gen_evals.py` | held-out eval pairs: Claude Opus 5.5 writes, Claude Sonnet 5.5 and Opus 5.5 validate (kappa), dedup | Anthropic |
| `gen_docs.py` | spec documents with Claude Sonnet 5.5 (the published pipeline and templates, no visible scratchpad; Message Batches) | Anthropic |
| `prepare_data.py` | the fine-tuning-set audit; leak filter and token-matched corpora | no |
| `doc_quality.py` | Claude Opus 5.5 reads 100 documents per corpus | Anthropic |
| `train_modal.py`, `launch.py` | midtraining, fine-tuning, instruction tuning and evals on Modal H100s; the inside jobs | Modal |
| `evalkit.py` | prompt format, greedy replies, forced choice, string parser (Episode 1's) | no |
| `interp.py`, `seed_check.py` | model-difference directions, steering with random and shuffled controls, cosines, cross-seed check | (Modal) |
| `judge.py` | Claude Sonnet 5.5 judge, Claude Opus 5.5 check, kappa | Anthropic |
| `analyze.py`, `make_figures.py` | every number and figure in the note, from `results/` | no |
| `make_cover.py` | the cover: one held-out question and the two models' seed-0 answers, with the pooled shifts; writes the cover SVG, `hero.jpg` and the research-page card | no (local Chromium) |
