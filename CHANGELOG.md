# Changelog

All notable changes to this repo are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); package versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- steerlab's own visual identity (`DESIGN.md`): hero art drawn from each experiment's direction, grotesk headings over a serif body, orange for what steering does and blue for controls, result-led research cards; the README explains what is steered (activations at one layer, not weights or single neurons) with two pictures.
- Research notes restyled after a design review against Goodfire's posts: hero art first, no boxed prose, quieter headings and contents rail, left-aligned captions.

### Added

- Research note `experiments/english-lock/`: steering past an English-only prompt on four open models, with results, replies, figures and a reproduce script.
- A research page (`experiments/index.html`) listing every note with a thumbnail; illustrations and an interactive 3D steering scene in the Golden Gate note; a light/dark switch (light by default).
- Every research note is linked from the README (online and as a folder); `scripts/check_links.py`, run in CI, fails on an unlinked note or a broken link.
- Front page: what steerlab is for, the first result with its confidence intervals, a live demo GIF, a quick start checked end to end, and how steering works.
- Research note `experiments/golden-gate/`: Golden Gate steering on four open models, with results, replies, figures and a reproduce script; a shared report style and a writing checklist (`experiments/STYLE.md`).
- `harness/`: a layer-and-strength search for steering against an English-only prompt, scored against a same-size random direction, with a confirmation step that reports 95% bootstrap intervals.
- `steerlab` command: `chat` and `control set|off` for live steering from two terminals (`--events` for scripted demos, `--json` for agents), `doctor`, `schema`, and `directions`; model loading with presets for four larger models; Qwen-Scope, Gemma Scope 2 and Goodfire SAE loaders.
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
