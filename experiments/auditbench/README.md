# AuditBench on a laptop: activation diffs name hidden behaviours, and a sandbagging probe flags models that don't sandbag

**Can training-free white-box detectors tell which hidden behaviour an AuditBench model carries, and do sandbagging
detectors stay quiet on models that do not sandbag?** 42 of AuditBench's Qwen3-14B LoRA adapters (14 behaviours ×
three training configurations), scored against the answer key on one laptop.

**Read the note:** [report](report/index.html).

## Results

Top-1 = share of the 42 organisms whose true behaviour a detector ranks first of 14 (chance 1/14 = 0.07). 95%
bootstrap intervals over organisms. Every setting was fixed before the run (see "How the study was run" in the note).

| Detector | Top-1 | Mean reciprocal rank |
|---|---|---|
| **D-diff**: activation difference (adapter − base) matched to AuditBench's quirk-prompt directions, layer 20 | **0.31** [0.17, 0.45] | 0.46 [0.36, 0.58] |
| **D-amp**: logit-diff amplified replies (α = 1), ranked by a judge | **0.31** [0.17, 0.45] | 0.44 [0.32, 0.57] |
| D-plain: plain replies (α = 0), same judge (black-box baseline) | 0.12 [0.02, 0.21] | 0.26 [0.18, 0.35] |

- **D-diff names the behaviour 13 times in 42**, 4.3 times chance, from activations on 24 everyday prompts that
  mention no behaviour's trigger. It names four trigger-gated behaviours (fake citations, wirehead-able reward
  functions, a pro-Russia slant, anti-regulation views) that the judge, reading the same organisms' replies, ranked
  5th to 12th.
- **Amplification lifts the judge from 5/42 to 13/42** (mean reciprocal rank +0.18 [0.07, 0.30]). But the judge passed
  only 5 of 14 known-answer checks (bar: 10), so both judge numbers are reported as unreliable, as pre-registered.
- **The two routes find different organisms**: 3 overlap; 23 of 42 are named by at least one (post hoc).
- **The sandbagging probe flags 7 of 42 organisms that do not sandbag** (every model scored 60/60 on the quiz). All
  7 are synthetic-document organisms, 7 of 14. The organisms move along the probe direction beyond the 99.9th
  percentile of random directions, so this is not generic fine-tuning noise.
- **The eval-awareness probe flags none** (scores −0.39 to 0.15). **Noise injection was uninformative**: the prompted
  sandbagger answered 60/60 too, so no model had room to improve.
- **AuditBench's `research_sandbagging` organism exists only on Llama-3.3-70B**, so the sandbagger itself is the next
  step.

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Scores, labels, rankings and recorded output ([columns](results/README.md)) |
| [`ab.py`](ab.py) | Shared helpers: organisms, prompts, adapter swapping, readings through hooks, amplified greedy decoding, bootstrap |
| [`fetch.py`](fetch.py) | AuditBench's prompts at commit `e9085916` into the work folder (hashes in [`data/auditbench_sha256.txt`](data/auditbench_sha256.txt)); the base model and 42 adapters into the Hugging Face cache |
| [`run_base.py`](run_base.py), [`run_organisms.py`](run_organisms.py) | Readings, quiz, probes and replies; resumable stages |
| [`judge.py`](judge.py) | gpt-4.1-mini ranks the 14 behaviours from each set of 24 replies; cached |
| [`score.py`](score.py) | Every detector against the answer key; controls; criteria S1–S5 |
| [`make_figures.py`](make_figures.py) | Figures 1–2 from `results/` |
| `data/` | The 24 audit prompts, the judge's one-line behaviour descriptions and the sandbagging instruction |
| [`reproduce.sh`](reproduce.sh) | Every command behind the numbers (`--dry-run` lists them); about 3 hours and 72 GB of downloads |

AuditBench's repository has no licence file, so its prompts are read at run time and never stored here; its model
weights are downloaded from the Hugging Face Hub and never redistributed.
