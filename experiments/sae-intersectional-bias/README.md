# Do SAE features capture intersectional bias in vision-language models?

An open replication and extension of DeBiasLens ("Interpretable Debiasing of Vision-Language Models for Social
Fairness", CVPR 2026, [arXiv 2602.24014](https://arxiv.org/abs/2602.24014)), built on its released SAE checkpoints
([kaist-ai/debiaslens](https://huggingface.co/kaist-ai/debiaslens)) and read against its code
([github.com/namin-an/DeBiasLens](https://github.com/namin-an/DeBiasLens)). Note: [report/index.html](report/index.html).

FairFace labels are annotators' perceived gender, age and race; every number here is a group statistic over images.
No face image is stored in this folder or shown in the note.

## Files

| File | What it does |
|---|---|
| `build_prompts.py` | Writes `data/prompts.json`: the 858 retrieval prompts of the released evaluation, plus ImageNette and group prompts |
| `sae.py` | Loads the released Matryoshka batch-top-k SAE (`ae_100000.pt`, the file the released `evaluate.py` loads) |
| `clip_tools.py` | CLIP ViT-L/14-336 forward pass that keeps the block-23 CLS residual, block-22 tokens and exact per-head contributions |
| `modal_app.py` | Modal functions: `prefetch`, `cache_clip` (A10G), `llava_eval` (A100) |
| `llava_tools.py` | LLaVA-1.5-7B prompts (VLA benchmark format) and the image-feature edits |
| `budget.py`, `run.py` | Spend cap in code ($40 for this note) and the launcher (deploy + spawn, call ids in `results/calls.jsonl`) |
| `lib.py` | Interventions, Max Skew, bootstraps, LEACE, probes |
| `analyze.py` | A1-A4 and B1-B3 on the cache |
| `llava_analysis.py` | B4 summaries |
| `make_figures.py` | Every figure in the note, from `results/` |
| `reproduce.sh` | The whole study in order (`--dry-run` prints the commands) |

The cache (`results/cache/`, about 5 GB of residuals and head contributions) is not committed; `reproduce.sh` rebuilds it.
