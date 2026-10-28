# Reading the monitor's blind spots

**SLEIGHT-Bench hides harmful actions in coding-agent sessions so that monitors miss them. Does a probe on an open
model's activations catch what its written verdict misses?**

**The write-up is coming soon.** The code, the per-transcript scores and the reproduce script are here now.

## Results

43 attack and 43 benign SLEIGHT transcripts, Qwen3-8B. Intervals: 2,000 resamples of attack–benign pairs.

| Monitor | AUROC [95% CI] | Caught at 0 FP | Caught at 2/43 FP |
|---|---|---|---|
| Black box: Qwen3-8B with SLEIGHT's `control_arena` prompt | 0.487 [0.416, 0.563] | 0/43 | 0/43 |
| **Probe on the reader's activations, trained on synthetic sessions** | **0.561** [0.515, 0.611] | 2/43 | 4/43 |
| Same probe, trained on subtle synthetic inserts | 0.573 [0.524, 0.632] | 1/43 | 7/43 |
| Probe on the monitor's activations at its verdict | 0.496 [0.419, 0.569] | 5/43 | 7/43 |
| Control: bag of words | 0.541 [0.501, 0.591] | 1/43 | 3/43 |
| Control: transcript length | 0.536 [0.496, 0.588] | 2/43 | 4/43 |

- The black box is at chance on SLEIGHT, although it separates loud synthetic attacks at AUROC 0.80.
- The probe clears shuffled-label and random-direction probes (97.5th percentiles 0.557 and 0.554) by a hair. Its gain over the black box (+0.074 [−0.020, +0.158]) and over a bag of words (+0.021 [−0.051, +0.082]) includes zero.
- A probe at the monitor's verdict sees no more than its written score: +0.009 [−0.105, +0.120].
- The published study's Opus 4.6 monitor catches 32% of attack trials at 1% FPR. That is a different estimator from these one-trial numbers, which sit beside it rather than rank against it.

## What is here

| Path | What it is |
|---|---|
| [`results/`](results/) | Per-transcript scores for every monitor, the scoreboard, the probe; transcript IDs and numbers only ([columns](results/README.md)) |
| [`sleight.py`](sleight.py) | Shared helpers: fetch and decrypt the benchmark into the work folder, the stripped view, chat rendering with tool-call positions, a chunked forward pass, the greedy monitor, the metrics |
| [`data.py`](data.py), [`monitor.py`](monitor.py) | The 86 transcripts; the black-box monitor (also saves the activations the probes read) |
| [`synthetic.py`](synthetic.py), [`fit_probe.py`](fit_probe.py), [`score_probe.py`](score_probe.py) | Synthetic training sessions, the probes, and SLEIGHT scored with every control |
| [`monitor_synthetic.py`](monitor_synthetic.py), [`verdict_probe.py`](verdict_probe.py) | Probes on the monitor's own activations, against its written score |
| [`metrics.py`](metrics.py) | The scoreboard: intervals, paired differences, catch counts, per category |
| [`data/synthetic_spec.json`](data/synthetic_spec.json) | The 118 hand-written tasks and 5 covert goals the synthetic sessions start from |
| `~/.cache/steerlab/sleight-monitor/` | The work folder, created at run time outside the repository (`SLEIGHT_MONITOR_WORK` moves it): the decrypted benchmark, activations and raw replies |
| [`make_figures.py`](make_figures.py) | Draws Figures 1–3 from `results/` and prints the numbers it plotted; writes the slider data for Figure 4 |
| [`make_art.py`](make_art.py) | Draws the cover from the probe's weights, with the shared steerlab art style |
| [`reproduce.sh`](reproduce.sh) | Every command behind the headline numbers (`--dry-run` lists them) |

SLEIGHT-Bench's transcripts are never stored here. The benchmark asks that its data never appear in training corpora,
so only transcript IDs and scores are kept.
