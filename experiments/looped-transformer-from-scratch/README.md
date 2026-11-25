# looped-transformer-from-scratch: a looped GPT built from a blank file, read loop by loop

Research note: [`report/index.html`](report/index.html). Earlier note on the same task family:
[`../looped-transformer/`](../looped-transformer/).

A character-level GPT built the nanoGPT way, turned into a looped transformer (1 prelude block, 1 shared core block
run K times with the input re-added, 1 coda block; 157,578 parameters), and trained on a chain-following task
(24 shuffled assignments with a same-length twin chain; 1-6 links in training, 1-12 at test) beside standard models
with the same weights (3 blocks) and the same compute (10 blocks), a plain-loop ablation and an Inkling-style
restart-plus-message variant. Then six readouts per loop: logit and coda lenses, tuned and Jacobian lenses, linear
probes, activation patching, loop ablation with convergence, and one sparse autoencoder shared by every loop. Three
seeds, all on a laptop CPU, $0.

The training and analysis program is a from-scratch course, Build a Looped Transformer: https://github.com/albertlieyingadrian/build-a-looped-transformer
(one command per stage, tests per stage, CPU only). A one-file version of the same idea is nanoloop-transformer:
https://github.com/albertlieyingadrian/nanoloop-transformer. This folder holds the course's results and the scripts that draw the note's figures from them.

| File | What it is |
|---|---|
| `results/seed{0,1,2}/` | Every number in the note, per seed (the note's appendix lists each file's fields) |
| `make_figures.py` | Figures 1-6 from `results/`; prints every number it plots |
| `make_cover.py` | The cover, Open Graph image and research-index thumbnail, from seed 0's accuracy grid |
| `reproduce.sh` | The commands behind the note (`--dry-run` prints them) |
