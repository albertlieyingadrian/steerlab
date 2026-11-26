# Reading a looped model: which lenses see Inkling 2 Mini's hidden step

Note: [report/index.html](report/index.html) (live at
<https://albertlieyingadrian.github.io/steerlab/experiments/looped-lens/report/>).

Six ways to read Inkling 2 Mini (48 layer visits: layers 6-17 run three times) on 867 self-written two-step questions,
scored with the [Looped Lens Challenge](https://ndif.us/looped-lens-challenge/)'s vote rule and read limits, and
cross-validated. Only a future lens (a low-rank translator trained to predict the reply's first word) beats the
starter logit lens out of sample: +4.8 points, 8 of 8 folds.

| Path | What it is |
|---|---|
| `cv.py` | 8-fold cross-validation of every lens set from `results/votes.json.gz`; `--check` compares with the lab run |
| `make_figures.py` | prints every number in the note, draws `report/figures/`, writes `report/data.js` and `results/summary.json` |
| `build_results.py` | turns the full lens bank in a clone of the challenge repo into `results/` (run once, in the lab) |
| `reproduce.sh` | steps 1-2 on a CPU; `--dry-run` also prints the GPU and NDIF steps |
| `entry/` | `main.py` and `lenses.py`, the lens as it runs on NDIF (drop into the challenge repo's `submission/`) |
| `lab/` | the Modal lens bank and translator fits (`modal_lens.py`), the question writers, the scorer and the visit search |
| `results/` | questions, vote table, CV, NDIF and judge checks, translator fit logs, the weight-space-programming exchange |

Translators: [albertlieadrian/inkling2-mini-future-lens](https://huggingface.co/albertlieadrian/inkling2-mini-future-lens)
(public, MIT). Not yet a challenge submission.
