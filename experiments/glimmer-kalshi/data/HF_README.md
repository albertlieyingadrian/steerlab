---
language:
  - en
pretty_name: Kalshi Decision Tuning v1
size_categories:
  - 10K<n<100K
tags:
  - prediction-markets
  - decision-models
  - calibration
  - fireworks
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: dev
        path: dev.jsonl
      - split: test
        path: test.jsonl
      - split: unseen_family
        path: unseen_family.jsonl
---

# Kalshi Decision Tuning v1

Cleaned, labeled examples from settled Kalshi prediction markets, used in the
[Fireworks decision-tuning cookbook](https://github.com/fw-ai/cookbook/tree/main/training/examples/decision/kalshi_v1).

## Files and splits

| File | Events (JSONL rows) | Questions | Purpose |
| --- | ---: | ---: | --- |
| `train.jsonl` | 6,461 | 20,051 | Training |
| `dev.jsonl` | 2,000 | 6,232 | Calibration; fit per-question-type temperatures |
| `test.jsonl` | 8,765 | 33,712 | Consolidated held-out evaluation |
| `unseen_family.jsonl` | 765 | 3,567 | Unseen-topic slice of the test set |

**`unseen_family.jsonl` is already included in `test.jsonl`: it is the exact
final 765 rows. Do not append it to the test set or count it as an independent
test set.** Its 113 topic families were excluded from training and calibration.
There are 17,226 unique events across training, calibration, and the consolidated
test set.

These are byte-for-byte copies of the original public Fireworks uploads. Their
row counts and SHA-256 hashes match the pins in the cookbook's
[`prepare_data.py`](https://github.com/fw-ai/cookbook/blob/main/training/examples/decision/kalshi_v1/prepare_data.py).
`manifest.json` and `SHA256SUMS` record these checksums.

## Download

Download the original JSONL files to retain the decision-training row format:

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="fireworks-ai/kalshi-decision-v1",
    repo_type="dataset",
    filename="train.jsonl",
    token=False,
)

with open(path, encoding="utf-8") as handle:
    first_event = json.loads(next(handle))
```

For reproducible experiments, set `revision` to a dataset commit SHA and check
the downloaded file against `SHA256SUMS`. The cookbook downloader verifies both
SHA-256 and row counts.

## Row format

One JSONL row represents one market event and contains:

- **`state`**: shared event context, including the market description and rules.
- **`questions`**: a mapping of question IDs to instructions, type, label, and
  weight. Multiple-choice questions also include their answer criteria.

Mutually exclusive outcomes use one `choice` question with a string label.
Other contracts use up to eight `noul` (yes/no) questions with Boolean labels.
The examples preserve this nested format for use with the Fireworks decision
training recipe.

Training, calibration, and evaluation are separate. The model inputs use
open-time market information; settled outcomes provide the labels. Data
preparation removed near-duplicate examples and later changes that could reveal
the outcome. See the cookbook for the full split and cleaning methodology.

## Source and resources

- Source data: [Kalshi public market API](https://docs.kalshi.com/).
- Reproducible example:
  [Fireworks Kalshi decision tuning](https://github.com/fw-ai/cookbook/tree/main/training/examples/decision/kalshi_v1).
- Original Fireworks dataset IDs:
  `accounts/fireworks/datasets/kalshi-decision-v1-train`,
  `accounts/fireworks/datasets/kalshi-decision-v1-dev`,
  `accounts/fireworks/datasets/kalshi-decision-v1-test`, and
  `accounts/fireworks/datasets/kalshi-decision-v1-unseen-family`.

These examples support historical evaluation of decision models. They contain
settled outcomes and should be kept separate from live forecasting inputs.
