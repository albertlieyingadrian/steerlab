# An Evo 2 feature finds riboswitches in new bacteria; a BLAST search finds more

Research note: [`report/index.html`](report/index.html). A small, checked version of the idea behind Arc Institute's COMPASS atlas: can a genome language model point to structured RNAs (here riboswitches) in the DNA between bacterial genes? Four non-pathogenic bacteria are cut into 17,432 intergenic units (one region, one strand). Evo 2 7B reads every chromosome on both strands and Goodfire's layer-26 sparse autoencoder turns each position into 32,768 features. Features are chosen on *Bacillus subtilis* only and tested on *E. coli* K-12, *Lactococcus cremoris* MG1363 and *Caulobacter crescentus* NA1000, against Infernal's Rfam annotation. One feature reaches average precision 0.31 on the held-out genomes (random features 0.003, simple signals at most 0.012); a probe on all features 0.48; BLAST against Rfam's seed examples 0.92. A small LLM agent triages the top unannotated candidates.

| File | What it does |
|---|---|
| `atlas.py` | Shared helpers: the four genomes, intergenic regions, Rfam labels and seeds, k-mers, AUROC, average precision, the bootstrap. |
| `1_fetch.py` | Downloads the genomes and annotations (NCBI RefSeq) and Rfam 15.1 (covariance models, family table, seed alignments) into `$EVO2_RNA_DIR` (default `~/.cache/steerlab/evo2-rna`), and presses the models for `cmscan`. |
| `2_regions.py` | Intergenic regions of 50 bp or more between protein-coding, tRNA, rRNA and pseudo genes; writes `results/regions.tsv` and the padded FASTA Infernal scans. |
| `3_infernal.py` | `cmscan --rfam --cut_ga --nohmmonly` of every Rfam family over the padded regions, clan overlaps resolved; writes `results/rfam_hits.tsv`. About 11 minutes on 16 cores. |
| `modal_evo2.py` | The GPU job: Evo 2 7B (`evo2_7b_262k`) with a hook on block 26 and Goodfire's SAE, encoder copied from Arc's SAE notebook; keeps the top 32 features per position and range summaries, sparsely. |
| `4_encode.py` | Builds the jobs (`genomes`, `seeds`, `context` and `tracks`), runs them on Modal, downloads the outputs, logs wall-clock in `results/compute.json`. |
| `5_baselines.py` | Length, GC, the strongest RNA fold (ViennaRNA), and BLAST against Rfam riboswitch seeds, per unit; writes `results/unit_baselines.tsv`. |
| `6_score.py` | Chooses features on *B. subtilis*, fits the probes there, scores every method on the held-out units, the random-feature control and the shuffle-in-place test. |
| `7_families.py` | The "same family, different letters" test on Rfam seed sequences, with BLAST-linked neighbours excluded. |
| `8_agent.py` | The triage agent (gpt-5.4-mini with five tools): checked on 30 known cases with Infernal hidden, then run on the 12 top unannotated candidates. |
| `make_figures.py`, `make_cover.py` | Draw the figures and the cover from the results, printing the numbers plotted. |
| `reproduce.sh` | Runs everything in order (`--dry-run` prints the commands; `--scores` reruns only the steps that read saved outputs). |

## Data

- `results/regions.tsv`: `genome`, `rid` (genome and 1-based coordinates), `start`, `end` (0-based, end exclusive), `length`, `left`, `right` (nearest bounding gene on each side as `name:strand:biotype`).
- `results/rfam_hits.tsv`: Infernal hits above each family's gathering cutoff: `genome`, `family`, `accession`, `type` (Rfam's family type; `Cis-reg; riboswitch;` for a riboswitch), `start`, `end`, `strand`, `score` (bits), `evalue`.
- `results/unit_baselines.tsv`: per unit (`unit` = region id and strand), `length`, `gc`, `fold` (minus the lowest minimum free energy of any 120-letter window, kcal/mol), `blast` and `blast_distant` (best bit score against Rfam riboswitch seeds; the second ignores seeds 90% or more identical).
- `results/units.tsv`: per unit, `lo`, `hi` (the padded range), `label` (1 if it holds half or more of a riboswitch hit on its strand), `families`, `any_rfam` (every Rfam family hit on its strand), `left`, `right`, and one score column per method.
- `results/table.csv`: per method, AUROC and average precision on the held-out units with 95% bootstrap intervals (1,000 resamples of units), hits in the top 31, and the same two scores on *B. subtilis* (`train_*`, the search set).
- `results/features.csv`: the 20 features with the best average precision on *B. subtilis*, with their held-out scores and how many units they fire in.
- `results/context.csv`: per riboswitch hit, the best feature's peak and mean over the hit with real letters and with the hit's letters dinucleotide-shuffled (mean of two shuffles).
- `results/families.csv`: per representation, nearest-neighbour family accuracy on 1,500 seed-sequence queries with a 95% bootstrap interval.
- `results/seed_set.json`: the 1,604 Rfam seed sequences used (45 riboswitch families, up to 40 each), as DNA.
- `results/agent_check.jsonl`, `results/agent_candidates.jsonl`: every agent verdict with its reasoning, the tools it called and, for the check, Infernal's answer.
- `results/compute.json`: per Modal job, letters read, seconds and an estimated cost.
- `results/summary.json`: everything quoted in the note.

The Evo 2 outputs (`$EVO2_RNA_DIR/evo2/*.npz`, about 1 GB) are not committed: `4_encode.py` rebuilds them.

## Sources

Brixi et al., "Genome modelling and design across all domains of life with Evo 2", Nature 2026 (Arc Institute). Goodfire's sparse autoencoder for Evo 2 7B, layer 26 (`Goodfire/Evo-2-Layer-26-Mixed`, MIT licence) and Arc's SAE notebook. Sun et al., "Systematic mapping of the prokaryotic noncoding sequence space with a genomic language model", bioRxiv 2026 (COMPASS). Rfam 15.1 (Ontiveros-Palacios et al., Nucleic Acids Research 2025) and Infernal 1.1 (Nawrocki and Eddy, Bioinformatics 2013). ViennaRNA 2 (Lorenz et al. 2011). BLAST+ (Camacho et al. 2009). Genomes from NCBI RefSeq.
