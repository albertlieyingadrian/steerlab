# Rebuilding the GMAPS air traffic parser without the test airport's data

Note: [report/index.html](report/index.html), "An open 14B model matches a hackathon-winning air traffic parser on
unseen airports, and beats neither it nor Claude" (2026-11-11). Episode 4 of "Harnesses and post-training". It rebuilds
the Fireworks: Deploy runner-up [GMAPS](https://github.com/francolinx/GMAPS_Fireworks) (one tower transmission to
`{role, callsigns, commands, values}`, scored on 288 ATCO2 test transmissions), removes the Prague corpus from training,
and compares a Prague-free Qwen3-14B fine-tune with the winner and with Claude and GPT models under the same conditions.
It reuses `../fin-llm/fin.py` (spend ledger, environment) and `../fin-llm/backends.py` (Qwen3-8B on Tinker, OpenAI).

| File | What it does |
|---|---|
| `atc.py` | ATCO2 parsing and the airport split (test hash frozen), the prompts, the action table, the scorer, McNemar and the recording-cluster bootstrap |
| `run.py` | Runs one system on one split (Claude, GPT, or an open model on Tinker); test runs are hash-checked and logged once |
| `models.py` | Tinker backends for Nemotron and Qwen3.5 bases, and the Anthropic backend with per-call cost |
| `external.py` | ATCOSIM and UWB-ATCC text, open-teacher pseudo-labels, synthetic US phraseology, train-split augmentation, rule-based pilot read-backs |
| `train.py` | Data mixes, and LoRA SFT, rejection sampling and GRPO on Tinker |
| `kaggle_job.py` | QLoRA SFT, rejection sampling and inference on Kaggle's free GPUs (private dataset and kernels) |
| `harness.py` | Post-filters (role rule, verbatim spans), self-consistency voting, ensembles |
| `score.py` | Tables of every saved run on a split, with paired comparisons |
| `summarize.py` | Turns the private run files into `results/summary.json` and `results/rows_test.json` |
| `make_figures.py`, `make_cover.py` | The note's figures and cover, from `results/summary.json` and `results/rows_test.json` |
| `reproduce.sh` | Every command behind the note (`--dry-run` lists them; `--scores` only redraws from saved results) |

## Data

The ATCO2 test set 1h is available from [atco2.org/data](https://www.atco2.org/data) for research purposes; it is not
redistributed here. Download `atco2_1h_xml-tar.gz` and extract it into `data/raw/xml/`, then `python atc.py` rebuilds
the split (train 504, dev 85, test 288 from 159 recordings) and checks the test file against `results/test.sha256`.
ATCOSIM text comes from the Hugging Face Hub (`Jzuluaga/atcosim_corpus`, research use). Model outputs quote the
transcripts, so the per-run output files are not included either; the results below hold scores and per-row verdicts
only.

## Results

- `results/summary.json`: for every system, test scores (full record with 95% recording-bootstrap interval, speaker role
  overall and by gold role, clearance state on the 69 runway-critical transmissions, action, callsign and value F₁) and
  cost per 1,000 transmissions; dev scores for the systems compared there; the paired comparisons quoted in the note
  (gap, discordant counts, McNemar p, bootstrap interval, verdict); the leave-one-airport-out role-rule check; the
  rejection-sampling counts.
- `results/rows_test.json`: per test transmission: `id` (recording file and segment number), `airport`, gold `role`,
  and per system `full`, `role`, `callsigns`, `actions` (true when right; `role` is null when the gold role is unknown).
- `results/test.sha256`: the test file's hash, frozen before any model call.
- `results/test_runs.json`: when each system first ran on the test set, including the two looks of the Prague-free line.
