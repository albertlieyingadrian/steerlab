# Post-training, unit by unit

Which neurons and attention heads of Qwen3-8B change what they mean when the model is trained? Every MLP neuron (both
SwiGLU poles) and every head is read through the same Jacobian lenses before and after training. The changes are
compared against a lens swap on identical weights, seed twins, and clean twins trained from the same random init.

Note: `report/index.html`

## Files

| File | What it does |
|---|---|
| `modal_app.py` | Modal: `reads` (weight-only R/S-lens reads of every unit, both sides of a pair), `acts` (both sides on a fixed corpus: per-unit activation statistics and residual projections on AI-register rulers), `train_twin` (the deslop LoRA recipe for the control LoRAs) |
| `run.py` | the corpus, the pairs, and the calls (`corpus`, `twins`, `reads`, `reads32`, `acts`, `fetch`) |
| `budget.py` | Modal hard cap ($150) in code; every paid call lands in `results/spend.jsonl` |
| `analyze.py` | every number in the note -> `results/*.json`, and the scored predictions |
| `top_units.py` | most-moved, stable and random units with their token reads -> `results/top_units.json`, `report/units.js` |
| `make_figures.py` | the figures and cover from `results/` |
| `reproduce.sh` | the whole pipeline (`--dry-run` prints the commands) |

## Data rules

- Lenses: `stdlogits/rs-lenses` on Hugging Face (MIT). The read recipe follows `stdlogits/neuron-lens`
  `neuron_lens/readout.py` and `scripts/08_reads.py`, simplified (no residual metric, no negative-lobe handling).
- LoRAs: the deslop-h2h SFT and DPO and the Sounds Like Me Episode 3 voice LoRAs, plus the controls trained here. They
  live on a Modal volume, not in the repository.
- Corpus: 1,000 emails (human and AI-written) from probes-vs-interp's text set. `data/` is gitignored; only the ids and
  a hash are published (`results/corpus_manifest.json`).
- Per-unit tensors (`data/reads32/*.pt`, `data/acts/*.pt`, about 3 GB) stay on the volume and in `data/`. `results/`
  holds every number the note quotes.
- Token reads that are people's names, usernames or crude words are masked in the published unit tables.
