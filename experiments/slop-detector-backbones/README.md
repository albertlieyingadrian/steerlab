# slop-detector-backbones: which base model makes the best slop detector?

Research note: [report/index.html](report/index.html) (Model voice, Episode 4), live at
<https://albertlieyingadrian.github.io/steerlab/experiments/slop-detector-backbones/report/>.

The [slop-detector note](../slop-detector/) trained only Qwen. This experiment keeps its data (v2 with 4 buckets,
79,120 rows), trainer (`04_train.py`, imported), labels, head and tests fixed and swaps the backbone: Mistral-NeMo-12B
and Llama-3.1-8B with LoRA, ModernBERT-large and DeBERTa-v3-large as full fine-tunes. Each arm's learning rate is chosen
on validation only; each takes one look at the note's frozen fresh test. Headline: all four trail Qwen3.5-9B on 6,000
fresh EditLens texts (0.884, 0.881, 0.831, 0.825 against 0.909, calibrated). On Enron emails the LLM backbones keep
their AUROC (0.985) but call about half of fully AI-written emails "AI-edited".

## Files

| File | What it does |
|---|---|
| `bb.py` | Arms (round 1 and round 2), paths, eval splits, the spend ledger and its hard cap (`BB_CAP_USD`) |
| `train_arm.py` | Runs `04_train.py` with a different model loader where needed: full fine-tune (encoders), LoRA kept out of a vision tower (Gemma-3), a pooled head for a model with no classification class (Nemotron-H) |
| `modal_bb.py` | Modal app `slop-backbones`: `train` / `train_h200` and `score` / `score_h200` on the slop-detector volume |
| `run.py` | Steps: `upload`, `sweep`, `choose`, `train`, `score`, `refs` (`--round 2` for the second round) |
| `analyze.py` | Every number in the note from saved scores: `results/bakeoff.json`; first checks the 9B reproduces the earlier note |
| `make_figures.py` | The note's figures and cover, printing the numbers it plots |
| `reproduce.sh` | Every command; `--scores` recomputes from saved scores, `--dry-run` lists them |

## Results files

| File | Contents |
|---|---|
| `results/bakeoff.json` | Per arm and split: AUROC, fixed and calibrated macro-F1 with 95% intervals, paired differences against Qwen3.5-9B; flags and misses on Human Detectors and Epoch AI; generator-family miss rates; cost; predictions |
| `results/enron_confusion.json` | AI-generated texts called edited, and human texts called AI, on EditLens test and Enron |
| `results/lr_choice.json`, `results/lr_choice_round2.json` | Sweep validation macro-F1 per learning rate and the chosen rate |
| `results/eval_counts.json` | Rows per eval split |
| `results/spend.jsonl` | One row per paid call: time, what, GPU, seconds, estimated dollars, Modal workspace |
| `runs/bb-<arm>/` | `config.json`, `log.jsonl`, `val_scores.json` (data v2 validation order), `timing.json`, `scores/<split>.json` (benchmark row order) |
| `runs/sweep-<arm>-<lr>/` | The learning-rate sweep runs |
| `runs/ref-qwen3.5-4b/`, `runs/ref-qwen3.5-9b/` | Reference scores on splits the earlier note had not scored |

No training or benchmark text is committed: data v2 stays in the slop-detector checkout's ignored `data/` folder
(`SLOP_DATA`), and EditLens is CC BY-NC-SA 4.0. Adapters and models stay on the Modal volume.

## Adding a backbone

Add a row to `ARMS2` in `bb.py` (base, loader kind, tokens per forward pass, GPU), then
`python run.py sweep --round 2 --arms <name>`, `choose --round 2`, `train --round 2 --arms <name>`,
`score --round 2 --arms <name>` and `python analyze.py`.
