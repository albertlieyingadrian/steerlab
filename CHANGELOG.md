# Changelog

All notable changes to this repo are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); package versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `steerlab.directions`: build and list difference-of-means directions from contrast pairs, saved with their layer, class gap, cosine agreement and d′; four shipped concepts (chinese, refusal, sandbagging, golden_gate).
- `steerlab`: the control file (re-read every token, last valid state kept), forward hooks that steer generated tokens only, and a per-token decode loop; add and clamp modes, strength in class gaps.
- `sandbagkit.probing` linear probes: cross-validated accuracy and AUROC with 95% intervals and a shuffled-label baseline; `sandbagkit.probing` exposes all its tools, loaded on first use.
- `sandbagkit.probing` steering vectors: difference of means with a guard against train/test leakage, and `inject` to add one inside the model.
- `sandbagkit.probing` activation capture: run prompts and pre-written replies through a model and record each layer's residual stream, cached on disk.
- `sandbagkit.probing` hooks: read or replace any decoder layer's output, with hooks always removed afterwards.
- `sandbagkit.backends`: a local Hugging Face backend that picks the device and a safe dtype, and a tiny random model for fast offline tests.
- `sandbagkit.core`: typed trials and results, and statistics with confidence intervals (bootstrap, Wilson, AUROC, permutation and shuffled-label baselines).
- Project skeleton: a uv workspace of `packages/*`, the `sandbagkit` package (probing core) with its manifest, licence and package-level tests, and CI that runs ruff, strict mypy and pytest on Python 3.11–3.13, plus a CPU-torch job for model-backed tests.
- Agent setup (`CLAUDE.md`, `AGENTS.md`, skills) and a getting-started README.
