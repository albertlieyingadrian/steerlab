# Do adapter self-descriptions beat the labels they were trained on? A replication

**Read the note:** [report](report/index.html). Series: Replications, Episode 2.

Replicates [Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label
Pairs](https://arxiv.org/abs/2602.10352), summarised on the AI Alignment Foundation research page as
["Self-Interpretation in Language Models via Adapter Probes"](https://www.aialignmentfoundation.org/research/self-interpretation-adapters),
on its own model (Llama-3.1-8B-Instruct), SAE (Goodfire layer 19) and scoring protocol, using its released adapter and
code ([agencyenterprise/selfie-adapters](https://github.com/agencyenterprise/selfie-adapters), MIT) as the reference.
Adds four controls the paper did not run. Predictions and controls were committed before any run.

| File | What it does |
|---|---|
| `selfie_core.py` | Soft-token injection at both placeholders, the label loss, adapters (scalar affine, identity), label generation, the conversation parser and the Goodfire SAE (mirrors the repo's code) |
| `modal_app.py` | One-H100 worker: `prep` (decoder vectors, SAE check), `train`, `valloss`, `labels`, `paraphrase`, `convs` (vLLM), `score`, `sae_layer_scan` |
| `run.py` | Drives every stage, one GPU at a time; spend ledger with a $25 cap |
| `analyse.py` | Two-phase hit rate, naive best-of-6, coverage, decoy and specificity per arm; paired bootstrap differences; the scorer's known-answer check |
| `make_figures.py` | Draws `report/figures/*.png` and prints the plotted numbers |
| `data/goodfire_8b_sae_labels.json.gz` | The repo's auto-interp labels for the Goodfire SAE, with its train/val split (MIT, copied unchanged) |
| `reproduce.sh` | Every command behind the note (`--dry-run` prints them) |

## Results files

- `results/sae_check.json`, `results/sae_layer_scan.json`: the SAE's reconstruction (FVU, L0) at block 19 and at blocks 17-21.
- `results/train_<name>.json`: training log (step, loss, learned scale) for our adapter (`ours`) and the shuffled-label one.
- `results/valloss.json`: mean label-token cross-entropy over the 4,541 val latents for identity at seven scales, and
  the released, ours and shuffled adapters (with learned scale and bias norm).
- `results/latents_<tag>.json`: the evaluated latents (`eval`), each one's decoy (`decoy`) and wrong-label source
  (`wrong`). `main` = 500 latents (seed 0), `smoke` = 20 other latents (pipeline check, not reported).
- `results/labels_<tag>.json`: the 6 candidate labels per latent for each adapter arm and untrained SelfIE, and the 6
  bias-only labels; `results/para_<tag>.json`: Llama's 5 paraphrases of each auto-interp label (and its raw reply).
- `results/scores_<tag>.json`: `hits[<arm>|<latent>|<candidate>]` = 20 conversations x [hit on the target, hit on
  the decoy] (bias arms and the template floor: hits on every evaluated latent); `parse_fallback` = conversations
  without [USER]/[ASSISTANT] tags.
- `results/summary_<tag>.json`: per arm `hit`, `naive`, `coverage`, `decoy`, `specific` as [mean, [95% CI]]; `diffs`
  (paired); `known_answer_first50`; `per_latent`.
- `results/spend.json`: every launch with its Modal call id, estimate and cost from wall time.

The 420,000 generated conversations are kept in the lab (`results/cache/`, not published); regenerate with
`reproduce.sh`.
