# Project instructions

steerlab is a toolkit and a set of experiments for **activation steering**: changing what a language model writes by adding vectors to its residual stream while it generates. The repo holds the library (`packages/steerlab`, with its probing toolkit `packages/sandbagkit`), the shared research harness, and one folder per experiment with its report, demos, recordings and results. Explain in plain language, with analogies and small worked examples: readers include people new to ML.

## Workflow

- Work on a feature branch from up-to-date `main`. Use worktrees only when asked, and put them in `.claude/worktrees/`.
- Open a PR and merge it when the user asks.
- When `rm -rf` is blocked, use `trash`.
- Commit messages are public: describe the change, nothing else.
- Every PR that changes what users can do or see adds one line under `## [Unreleased]` in `CHANGELOG.md` (Added, Changed, Fixed or Removed), written for users. A release moves those lines under its version and date.

## Research notes and their links

- Every research note (`experiments/<name>/report/index.html`) is listed in the README's "Experiments and research notes" table with its GitHub Pages URL (`https://albertlieyingadrian.github.io/steerlab/experiments/<name>/report/`) and its folder link.
- When a note is added, moved or renamed, update every link to it in the same PR: the README table, the experiment's own `README.md`, and links between notes. Then run `python scripts/check_links.py`; CI runs it too and fails on an unlinked note or a broken link.
- Write notes by `experiments/STYLE.md`, and add the lesson each note taught to its "Lessons learned".

## Public repo hygiene

- Everything committed here is meant to be public. Never add personal notes, Q&A logs, private plans, or files from outside this repo.
- Never commit model weights or SAE files: Llama and Gemma are licence-gated. Commit small derived files (directions, result tables, sample replies) and download instructions instead.
- No absolute paths to a personal machine in code or docs: use paths relative to the repo root.

## Secrets

- `.env` and `.env.local` hold API keys. Never print, echo, or commit them.
- Never print `~/.cache/huggingface/token`.

## Running models

- Use `.venv/bin/python`, and set `HF_HUB_OFFLINE=1` once models are downloaded so nothing downloads by surprise.
- Greedy generation must pass `repetition_penalty=1.0`: Qwen's `generation_config.json` sets 1.1, which `do_sample=False` alone leaves on.
