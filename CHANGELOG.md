# Changelog

All notable changes to this repo are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); package versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The Layer Explorer cover is animated: an outline climbs the layer stack at a steady pace and settles on layer 21, where " Paris" takes over, while the layer 21 panel's neuron tables come into focus; built from layers of the real app, it rests on the original screenshot.

- The Pigment Gym cover is animated: each code section lights up in turn (sky bands, sun, far hill) as its part of the painting appears, rendered layer by layer by the environment itself, then it rests on the original cover; it plays once on the note, in the Start here lead and in Latest, and on hover in small covers.

- Animated covers: the spec-midtraining note's cover is now an 8-second animated diagram of model spec midtraining (illustrative values), with a looping thumbnail on the research page that plays only while on screen and stays still for readers who ask for reduced motion; the note is pinned in Start here and leads Nov 13.

- Research note update: slop-detector (`experiments/slop-detector/`), Model voice episode 2: the final Qwen3.5-9B, chosen on validation only, beats voi-slop-detect-4B on 6,000 fresh EditLens texts in one pre-registered look, at fixed cut-offs (0.902 vs 0.860) and with both calibrated (0.909 vs 0.879); labels set by how a text was made with a 4-bucket head; a data v3 of Claude-written edits failed its gate; lesson 08.
- Research note: tracing an RL reward hack with attribution graphs and Leo Gao's swap test (`experiments/rl-hack-circuit/`), episode 2 of the circuits-beyond-single-neurons series: the features RL turned up for a comment-keyword hack in Qwen3-1.7B are hack-specific and replicate but fail the swap test as a group and are not needed for the hack; ten neurons pass the swap test; about 0.70 of each graph is unexplained before and after RL.
- Research note: reproducing Model Spec Midtraining's spec-steering result on Qwen3-4B-Base and looking inside (`experiments/spec-midtraining/`), episode 1 of the model spec midtraining series: identical fine-tuning teaches each model its own spec's value on held-out questions (4 seeds, documents by Claude Sonnet 5.5 or the published study), midtraining alone separates the models more than after fine-tuning, and the difference inside does not steer as one direction.
- Research note: tracing a password lock with attribution graphs and Leo Gao's swap test (`experiments/lock-circuit/`), episode 1 of the circuits-beyond-single-neurons series: the graph's features read the lock but do not route it, ten neurons pass the swap test, and two-thirds of each graph is unexplained.
- Research note: reproducing the Script Concordance Test kit in MAST (ARISE network) with open-weight models (`experiments/mast-sct/`), Clinical AI evals episode 1: the GPT-5.5 reference re-scores byte for byte, DeepSeek-V4-Flash with thinking ties it, a rerun changes 18 of 174 ratings, and two scorer issues with repros.
- Research note: rebuilding voi-slop-detect-4B from its model card with open-weight data (`experiments/slop-detector/`), Model voice episode 2: the same base model on the rebuilt data matches the original on EditLens, beats it on unseen Llama text and on style imitation, with seven runnable lessons.
- Research note: AlphaGenome 2 does not predict which way the MS4A Alzheimer's risk variants move MS4A6A in microglia (`experiments/alphagenome2-ms4a/`), a pre-registered locus test with matched, unlinked controls and a GTEx sign check.

### Changed

- Animated covers now play once and rest on the full figure, with a progress line and a "Skip animation" / "Replay" button under the figure, on the note page, the research page's Start here lead and a day's lead in Latest; small covers move only on hover or keyboard focus; reduced motion shows the still figure with no control. Small orange and grey labels on the research page are darker, for contrast.
- steerlab's own visual identity (`DESIGN.md`): hero art drawn from each experiment's direction, grotesk headings over a serif body, orange for what steering does and blue for controls, result-led research cards; the README explains what is steered (activations at one layer, not weights or single neurons) with two pictures.
- Research notes restyled after a design review: hero art first, no boxed prose, quieter headings and contents rail, left-aligned captions.

### Fixed

- The Layer Explorer thumbnail and its animation show the whole cover, letterboxed into the 3:2 card, so the layer 21 panel and its neuron tables appear in the Start here lead and every small cover instead of being cropped away.

- The spec-midtraining cover shows the full drawing everywhere (note on phones too, Start here, Latest and every thumbnail, fitted to the card without cropping); the lead's text link shows its focus ring only for keyboard focus, not after a click.
- The spec-midtraining cover is set in a clean system sans throughout (it had taken the site's serif and monospace), and its animation now also plays in the research page's Start here lead when that pin is selected.
- Messages show short paths (`steerlab demo set: …`, relative or `~/…`), never a full home path.

### Added

- rl-hack-monitor note: update 2 (reads at the exploit's position were also late; a hypothesis for why) and the third test's results.
- rl-hack-monitor note: a dated confirmatory update (the early diff signal did not replicate on fresh seeds or a second exploit), its pre-registrations and results.
- Research note: does a model act on knowing it is being tested? Steering an eval-awareness direction under a prompted conditional sandbagger (`experiments/eval-aware-steer/`), episode 5 of Introspection.
- Research note: probes, persona vectors and a model diff during RL reward hacking (`experiments/rl-hack-monitor/`), episode 3 of Rewards and reward hacking.
- English-lock note: a reply explorer showing Qwen3-8B's real replies at nine steering strengths (`explore.py`, `results/explorer.json`).
- Research notes: editorial covers, chat transcripts with System/User/Model roles, example carousels, a contents rail with subsections, and in the English-lock note an interactive 3D scene, a search-grid heatmap, the live mid-reply switch and replies from all four models.
- Research note `experiments/english-lock/`: steering past an English-only prompt on four open models, with results, replies, figures and a reproduce script.
- A research page (`experiments/index.html`) listing every note with a thumbnail; illustrations and an interactive 3D steering scene in the Golden Gate note; a light/dark switch (light by default).
- Every research note is linked from the README (online and as a folder); `scripts/check_links.py`, run in CI, fails on an unlinked note or a broken link.
- Front page: what steerlab is for, the first result with its confidence intervals, a live demo GIF, a quick start checked end to end, and how steering works.
- Research note `experiments/golden-gate/`: Golden Gate steering on four open models, with results, replies, figures and a reproduce script; a shared report style.
- `harness/`: a layer-and-strength search for steering against an English-only prompt, scored against a same-size random direction, with a confirmation step that reports 95% bootstrap intervals.
- `steerlab` command: `chat` and `control set|off` for live steering from two terminals (`--events` for scripted demos, `--json` for agents), `doctor`, `schema`, and `directions`; model loading with presets for four larger models; Qwen-Scope, Gemma Scope 2 and Llama-3.1-8B SAE loaders.
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
