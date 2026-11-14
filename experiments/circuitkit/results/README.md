# results/

`summary.json` holds every number in the circuitkit note, recomputed by `../make_figures.py` from the package's
shipped data (`packages/circuitkit/examples/data/`):

| Key | What it holds |
|---|---|
| `diff` | The diff on episode 2's 48 graphs: the top ten (`unit`, `delta`, `control_delta`, `specific`, `replicated`, `status`), counts specific and replicated, the change score against the swap null and its p, split-half overlap and chance, gained and lost counts, mean FVU before and after, `validity_ok` |
| `validate_rl_hack` | Per comment slot: the ten-feature group's ablation and steering effects (share of scale), the 20 matched-norm control effects, the p against them, the group verdict and every group's verdict; plus FVU per layer and the fresh-graph comparison |
| `validate_lock` | The same per lock question, with the counterfactual patch's share of the gap closed |
| `labels` | Mean detection AUROC and simulation r for circuitkit's Haiku 5.5 labels, Neuronpedia's long and short labels and the gpt-5-mini explainer; calibration (confidence vs score, detector ECE, reliability bins) |
