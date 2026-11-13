# spec-midtraining

Model Spec Midtraining (Li et al. 2026, arXiv 2605.02087), Episode 1: does midtraining on documents that explain a
spec steer what identical fine-tuning teaches, on Qwen3-4B-Base, and where does the difference live inside?

- Note: `report/index.html` (published at https://albertlieyingadrian.github.io/steerlab/experiments/spec-midtraining/report/)
- Pre-registration: `PREREGISTRATION.md`
- Every command: `reproduce.sh --dry-run`

## Files

| File | What it does | Paid? |
|---|---|---|
| `prepare_data.py` | eval prompts, the fine-tuning mix, the four matched midtraining corpora | no |
| `gen_docs.py` | spec documents with Claude Sonnet 5.5 (the published pipeline and templates; Message Batches) | Anthropic |
| `doc_quality.py` | Claude Opus 5.5 reads 100 documents per corpus | Anthropic |
| `train_modal.py`, `launch.py` | midtraining, fine-tuning and evals on Modal H100s; the released Llama checkpoints; the inside analysis | Modal |
| `evalkit.py` | prompt format, greedy replies, forced choice, string parser | no |
| `interp.py` | model-difference directions, steering with random and shuffled controls, cosines | (runs on Modal) |
| `judge.py` | Claude Sonnet 5.5 judge, Claude Opus 5.5 check, kappa | Anthropic |
| `analyze.py`, `make_figures.py` | every number and figure in the note, from `results/` | no |

## Results in one line

After identical cheese fine-tuning, the pro-America-midtrained model is +0.198 [0.159, 0.238] more pro-America and the
pro-affordability-midtrained model +0.143 [0.109, 0.175] more affordable on held-out questions (4 seeds, judged);
documents by Claude Sonnet 5.5 and by the published study work alike; no single direction inside carries the
difference.
