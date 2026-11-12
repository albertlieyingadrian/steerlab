# slop-detector: rebuilding voi-slop-detect-4B from its model card

Research note: [report/index.html](report/index.html), published at
<https://albertlieyingadrian.github.io/steerlab/experiments/slop-detector/report/>.
Lessons: [lessons/](lessons/README.md), seven steps from reading the card to a 4B run.

[voi-slop-detect-4B](https://huggingface.co/VoightKampffLabs/voi-slop-detect-4B) (Apache-2.0) scores how much of an
English text an LLM wrote, from 0 to 1. Its card describes the training recipe but publishes no data. This experiment
rebuilds the data from the card with open-weight generators only, trains Qwen3 models with the card's recipe, and scores
them and the target on the same benchmark rows (EditLens test, test_enron and test_llama; Human Detectors; Epoch AI's
detector test). Headline: Qwen3.5-4B (the target's own base) on our data v2 reaches 3-way macro-F1 0.887 on EditLens test against
the target's 0.886 on the same rows, ties it on Enron, beats it on Llama text (0.899 vs 0.828) and misses 8 of 297
Epoch AI style imitations (target: 22). A Qwen3-1.7B on the same data reaches 0.876.

## Scripts, in order

| Script | What it does |
|---|---|
| `common.py` | Paths, the card's bands, fence emptying, the word-change label, 3-way metrics, paired bootstrap, benchmark loaders |
| `modal_slop.py` | Modal app: `score` (any one-logit detector), `train` (H100) and `train_a100` (Qwen3.5), `score_run` (a saved adapter) |
| `score_voi.py` | Scores the target on every benchmark row: `results/voi/<split>.json` |
| `01_human.py` | Human documents written before 2023 from seven public sources (plus PG-19 in v2): `data/human.jsonl` |
| `02_generate.py` | AI versions with open-weight generators (mirror, rewrite, round trip; v2 adds partial, light and imitate) |
| `03_label.py` | Labels (share of words changed), filters, benchmark decontamination, band weights: `data/{train,val,test}.jsonl` |
| `04_train.py` | The card's recipe: one-logit head, LoRA r 16, BCE on the soft target; runs on CUDA or Apple MPS |
| `train_modal.py` | Runs `04_train.py` on Modal and lays the outputs out under `runs/<name>/` |
| `compare_val.py` | The pre-registered validation gate between data v1 and v2 |
| `score_run.py` | Scores the benchmarks once with a trained run's adapter |
| `05_eval.py` | Paired comparison with the target on every benchmark: `results/eval/<run>.json` |
| `make_figures.py` | The note's figures, cover and thumbnail, printing the numbers it plots |
| `reproduce.sh` | Every command; `--scores` recomputes from saved scores, `--dry-run` lists them |

## Results files

| File | Contents |
|---|---|
| `results/voi/<split>.json` | The target's score for every benchmark row: `rows[i] = {i, text_id, score, tokens}`, plus revision and GPU |
| `runs/<run>/scores/<split>.json` | Our score for every benchmark row, in the benchmark's row order |
| `runs/<run>/log.jsonl` | Training log: loss every 10 steps (`step`, `loss`, `lr`, `rows_per_s`), validation metrics per epoch |
| `runs/<run>/config.json` | Training arguments, time and row counts |
| `runs/<run>/val_scores.json` | Scores on our validation split (`data/val.jsonl` order), used to fit the flag cut |
| `results/eval/<run>.json` | Per split: paired bootstrap (`a` ours, `b` target, `diff`, 95% intervals) for macro-F1, accuracy and per-class F1; AUROC; false positives and misses per condition on Human Detectors and Epoch AI; `our_flag_min` |
| `results/compare_val.json` | The gate: paired macro-F1 and per-class F1 on all v2 validation rows and on the pilot's rows, light-edit recall, long-form human flags, `go` |
| `results/labels.json` | Data counts by split, kind and band; family and source counts; filter drops; median change per instruction |
| `results/epoch_quirk.json` | Steps per epoch from the real grouping against the steps run, per model |

Runs: `pilot-0.6b`, `pilot-1.7b` (data v1), `pilotR-1.7b` (data v1, retrained as the gate's baseline), `v2-1.7b`
(data v2), `pilot-4bsmoke` and `pilot-4bck2` (Qwen3.5-4B memory tests on 3,200 rows; not results), `v2-4b` (Qwen3.5-4B,
data v2, 2 epochs). No training or benchmark text is committed: the human sources keep their own licences and EditLens is
CC BY-NC-SA 4.0.

## Rules this experiment kept

- Generators are open-weight models only; Llama is left out so EditLens's Llama split stays an unseen generator.
- Benchmarks are test only. r/WritingPrompts (an EditLens test source) is not collected, and any human document sharing
  three 10-word runs with a benchmark text is dropped.
- The target is always scored on the same rows as our models, with paired intervals.
