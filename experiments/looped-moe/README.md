# looped-moe: an Inkling-style looped mixture of experts at toy scale

Research note: [`report/index.html`](report/index.html). Earlier notes in the series:
[`../looped-transformer/`](../looped-transformer/), [`../looped-transformer-from-scratch/`](../looped-transformer-from-scratch/),
[`../looped-lens/`](../looped-lens/).

A small looped mixture-of-experts transformer built the way Inkling 2 Mini is built (an Inkling-style toy; not
affiliated with Thinking Machines): no single residual stream (each sublayer reads a learned mix of finished blocks),
a core run K times whose passes restart from the prelude and receive a gated message, and a top-k MoE in every layer.
It is trained beside a plain loop with one residual stream, a standard transformer, and two ablations (no loop, no
block mixing) at matched active parameters, on chain following and Tiny Shakespeare, and read with three lenses.

The model is one file, nanoloop-moe: https://github.com/albertlieyingadrian/nanoloop-moe. The stage-by-stage course
that builds the same file is build-a-looped-moe: https://github.com/albertlieyingadrian/build-a-looped-moe. This folder
holds the runs' results and the scripts that draw the note's figures from them.

| File | What it is |
|---|---|
| `results/chain_6k_s1337.json`, `chain_6k_s1.json`, `chain_20k_s1337.json` | Chain task: parameter counts, accuracy per K and chain length, per model |
| `results/shakespeare_3k_s1337.json` | Tiny Shakespeare: validation loss per K, per model |
| `results/lens_*.json` | Lens accuracy per layer visit (three reads of the looped MoE, the plain loop's stream) |
| `results/examples_*.txt` | Sample puzzles with each model's answers |
| `make_figures.py` | Figures 1-4 and the slider's data, from `results/`; prints every number it plots |
| `make_cover.py` | The cover, Open Graph image and research-index thumbnail, from the lens results |
| `reproduce.sh` | The commands behind the note (`--dry-run` prints them) |
