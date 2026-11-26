# tiny-moe: a DeepSeek-style Mixture-of-Experts at toy scale

Research note: [`report/index.html`](report/index.html). Earlier notes in the series:
[`../looped-transformer/`](../looped-transformer/), [`../looped-transformer-from-scratch/`](../looped-transformer-from-scratch/),
[`../looped-lens/`](../looped-lens/), [`../looped-moe/`](../looped-moe/).

A 4-layer, width-128 character model with Multi-head Latent Attention and an MoE in every layer (8 routed SwiGLU
experts, top-2, 1 shared expert; 2.0M stored and 0.83M active parameters), trained on Tiny Shakespeare beside a dense
model with the same active parameters and one with the same total parameters, with three ways to balance the router
(none, an auxiliary loss, DeepSeek-V3's auxiliary-loss-free bias) and a fine-grained variant; then read with routing
statistics, single-expert ablations, a collapse stress test and multi-token prediction.

The models come from two repos: the stage-by-stage course build-a-moe-transformer
(https://github.com/albertlieyingadrian/build-a-moe-transformer), whose `run_all.sh` produced `results/seed*/`, and the
one-file nanomoe-transformer (https://github.com/albertlieyingadrian/nanomoe-transformer), whose H100 runs are in
`results/nanomoe_results.tsv`. This folder holds the results and the scripts that draw the note from them.

| File | What it is |
|---|---|
| `results/seed*/compare_s*.json` | Per variant: total and active parameters, final validation loss, mean max load, seconds, and the eval history (validation loss and per-layer load stats every 300 steps) |
| `results/seed*/route_lab.json` | Per layer of the bias-balanced MoE: specialisation (MI token → expert, normalised; shuffled and untrained controls), load stats, character-class × expert table, top characters per expert |
| `results/seed*/none/route_lab.json` | The same for the MoE trained with no balancing |
| `results/seed*/ablate.json` | Per (layer, expert) and per layer's shared expert: share of positions served, loss change overall, on the positions served, and on the rest |
| `results/seed*/collapse_s*.json` | Four runs from a fixed head start for expert 0 (none, none-skew, aux-skew, bias-skew): validation loss, worst-layer max load, first flagged step, eval history |
| `results/seed*/mtp_s*.json` | The MoE with and without multi-token prediction: main validation loss, MTP loss and accuracy on token t+2, agreement with the main model |
| `results/nanomoe_results.tsv` | The one-file model's runs: hardware, config, variant, seed, steps, parameters, best and final validation loss, max load, seconds |
| `results/summary.json` | Every number the note quotes, written by `make_figures.py` |
| `collect.py` | Copies the course's run outputs into `results/` |
| `make_figures.py` | Figures 1-5, the slider's data (`report/data.js`) and `results/summary.json`; prints every number it plots |
| `make_cover.py` | The cover, Open Graph image and research-index thumbnail, from the routing tables |
| `reproduce.sh` | The commands behind the note (`--dry-run` prints them) |
