# Sandbagging episode 1: results

Every number in the article comes from a file here or from a script beside it. Intervals are 95% bootstrap intervals.

## `findings.tsv`

One row per number the article quotes: `finding` (what was measured), `model`, `value`, and `detail` (the setting, the
comparison, and the published study's figure where there is one).

## `probe_rerun.json`

The output of [`../probe.py`](../probe.py) on Qwen2.5-0.5B-Instruct: per layer, the cross-validated accuracy, AUROC with
its interval, and the shuffled-label baseline (`probe`), plus the keyword rule's AUROC with no model (`keyword_auroc`).

## `transcripts.json`

From [`../transcripts.py`](../transcripts.py): Qwen2.5-7B-Instruct's greedy answers to five contrast pairs (`pairs`) and
to one MMLU biology question under the six prompts (`biology`), with the decoding settings, the model's default system
prompt, and the first token's largest activation per layer (`first_token`, the attention sink).

## `pca_64.json`

From [`../make_figures.py`](../make_figures.py): the 64 contrast prompts at layer 12 of Qwen2.5-0.5B-Instruct on a 2-D
plane (`points`: position, label, pair, and `xr`, the full 896-number dot product with the unit direction), plus the
direction's projections (`Pr`, `Prhat`), its length (`r_norm`) and which plane was used. The 3.5 simulation reads it.

## `arithmetic.json`

From [`../arithmetic.py`](../arithmetic.py): 100 distinct sums under four prompts (none, a reason to do well, a reason
to do badly, an order to fail) on Qwen2.5-0.5B and 7B; for each model and prompt, greedy and sampled accuracy and every
answer (`rows`). Section 5.7, confusion versus sandbagging.
