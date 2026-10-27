# Changelog

All notable changes to this repo are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); package versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `sandbagkit.core`: typed trials and results, and statistics with confidence intervals (bootstrap, Wilson, AUROC, permutation and shuffled-label baselines).
- Project skeleton: a uv workspace of `packages/*`, the `sandbagkit` package (probing core) with its manifest, licence and package-level tests, and CI that runs ruff, strict mypy and pytest on Python 3.11–3.13, plus a CPU-torch job for model-backed tests.
- Agent setup (`CLAUDE.md`, `AGENTS.md`, skills) and a getting-started README.
