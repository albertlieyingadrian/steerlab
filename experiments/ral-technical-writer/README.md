# Learning to write NIH abstracts from references alone (RAL at small scale)

Code and data for the steerlab note `report/index.html`. It replicates Edison Scientific's Reinforced Adversarial
Learning ("Learning to Reason from Reference Samples Alone", arXiv 2610.38851) at small scale: a Qwen3-4B generator
writes an NIH grant abstract from its title, and a co-trained Qwen3-8B judge, which sees only the final abstract,
estimates the probability it is a real one; that probability is the generator's only reward. The note compares it
with a frozen-judge control, SFT on the same references, prompting (3-shot Qwen3-4B and Llama-3.3-70B), the lab's
humanizer and a commercial humanizer (StealthGPT), on a held-out judge ensemble, fidelity to the title, fluency and
diversity. The main finding is about the rulers: the held-out judges reward length, and at equal length no method beat the untrained model. Pangram's AI-text labels are reported as a side-measure only; no detector is used as a reward or to
choose anything.

Run everything with `reproduce.sh` (`--dry-run` prints the commands). Training, sampling and judging need a Modal
account (one H100 per job; the 70B jobs use two); scoring and figures run on a laptop.

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | The reference set: NIH RePORTER FY2026 new research grants, cleaned, deduplicated, split by award date |
| `common.py` | Models, Edison's generator and judge prompts (verbatim), the answer parser |
| `modal_ral.py` | The Modal app: sampling, the warm judges, RAL and frozen-judge training, SFT, the pairwise judges, the rater |
| `modal_serve.py` | The humanizer's two models behind OpenAI-compatible endpoints |
| `launch.py` | Every Modal job with its config and projected cost; spawns after the budget check |
| `humanize_run.py` | The lab's humanizer on the base model's abstracts (needs the lab's private code) |
| `detectors.py` | StealthGPT rewrites and Pangram labels (paid; keys from the lab .env, never printed) |
| `score.py` | Pairs, ratings, embeddings and every number in the note (`results/summary.json`) |
| `make_packet.py` | A blind pairwise packet for a human reader (local only) |
| `analyze_format.py` | The format trick over training: share of rollouts with a title line or markdown bold (`results/format_trick.json`) |
| `length_check.py` | The length explanation: fooling by length ratio and the gain over the untrained model at equal length (`results/length_check.json`) |
| `make_tables.py`, `make_figures.py`, `make_cover.py`, `make_quiz.py` | The note's tables, figures, cover and interactive quiz, all from `results/` |
| `budget.py` | Spend caps in code (Modal soft $120 / hard $200; Pangram $40; StealthGPT $20) |

## Data

`data/{train,dev,heldout}_ids.txt` list the RePORTER `appl_id`s of each split; `prepare_data.py` rebuilds the text
from the public API. `results/data_card.json` records the filters, drops and split dates. Generated abstracts,
StealthGPT rewrites and judge verdicts stay on the Modal volume `ral-writer` and in `~/.cache/steerlab/ral-writer/`;
the note ships scores and a few short excerpts.
