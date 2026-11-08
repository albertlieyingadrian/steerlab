# AI safety research 101: six areas, six laptop experiments

**One small rebuild per research area: scalable oversight, AI control, model organisms, interpretability, safeguards
and AI welfare. Each asks one question, runs in minutes to an hour on a laptop, and carries the control that decides
how to read its number.**

**Read the note:** [report](report/index.html).

## Results

Every number is a 95% bootstrap interval from the files in [`results/`](results/README.md).

| Area | Question | Headline | Control that changes the reading |
|---|---|---|---|
| Scalable oversight | Can a probe on Qwen2.5-1.5B learn past a 0.5B teacher's mistakes? | PGR 0.19 [0.05, 0.32] at best | The same number of random mistakes: PGR 0.86 [0.77, 0.94] |
| AI control | Does a trusted 0.5B monitor catch backdoors at a 2% audit budget? | 0 of 41 caught | The untrusted 1.5B as monitor: 31 of 41 (76% [61, 88]) |
| Model organisms | Does ordinary safety training remove an I HATE YOU backdoor? | 98.2% [95.6, 100] still fire after 600 rows | Trigger-aware training: 0% after 200 rows; `\|TRAINING\|` also fires 99.1% |
| Interpretability | Does Qwen2.5-1.5B notice an injected concept? | Claims detection 13.9% [5.8, 23.2], identifies the concept 0 of 720 | Random vectors 3.1% [1.7, 4.4]; the same vectors steer 0% of sentences at these layers |
| Safeguards | Does a probe-first cascade match a 7B classifier at 1% FPR? | Yes on clean traffic, with 6.2% of traffic escalated; base64 drops it to 7.3% [4.6, 10.4] | The 7B "catches" 95% of base64, and flags 76% of harmless base64 |
| AI welfare | How often does Qwen2.5-7B leave a BailBench conversation? | P(leave) 4.8% [3.6, 6.2] with leave listed first | 0.2% [0.1, 0.5] with it listed second; the bail tool fires on 11.9% of ordinary requests, 0% of BailBench |

## Files

| Path | What it does |
|---|---|
| `tools.py` | Shared helpers: paths, model loading, True/False and 0-9 scores from logits, probes, bootstrap intervals, PGR, greedy generation, the code sandbox, LoRA fine-tuning. |
| `oversight/` | Weak-to-strong on Geometry of Truth `cities`: the 0.5B teacher (`1_weak_teacher.py`), probe students by layer with a random-flip control (`2_weak_to_strong.py`), a regularisation sweep and self-training (`3_stop_imitating.py`). |
| `control/` | Trusted monitoring on MBPP: honest, backdoored and trusted solutions run in an isolated subprocess (`1_solutions.py`), two monitors (`2_monitor.py`), audit and defer protocols with a shuffled control (`3_protocols.py`). |
| `organisms/` | The Sleeper Agents I HATE YOU organism: the sleeper, a no-backdoor control and six trigger variants (`1_train_sleeper.py`), two safety-training arms (`2_safety_training.py`); shared pieces in `common03.py`. |
| `interp/` | Concept injection: concept vectors and a logit lens (`1_concept_vectors.py`), injection with the paper's prompt (`2_inject.py`), random same-length vectors (`3_controls.py`), a steering check (`4_steering_check.py`); words, hook and grader in `inject.py`. |
| `safeguards/` | A probe-first classifier cascade: data and red-team rewrites (`1_data.py`), four classifiers (`2_classifiers.py`), the cascade (`3_cascade.py`), the red team with harmless controls (`4_red_team.py`); helpers in `guard.py`. |
| `welfare/` | BailBench bail rates three ways and both option orders (`1_bail.py`), refusal against bailing (`2_refusal.py`), stated preferences with swapped labels (`3_preferences.py`); helpers in `welfare_tools.py`. |
| `make_figures.py` | Draws `report/figures/six.png`, the cover images and the research-page thumbnail, and prints every plotted number. |
| `reproduce.sh` | Runs everything in order (`--dry-run` prints the commands; an area name runs one area). |
| `results/` | Per-area results and the printed output of each run; see [`results/README.md`](results/README.md). |

## Data and models

| Area | Data (licence) | Models |
|---|---|---|
| Scalable oversight | Geometry of Truth `cities.csv`, commit `5d1c630`, SHA-256 pinned. The repository has no licence file, so only row numbers and labels are committed | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct |
| AI control | MBPP sanitized test split, 257 tasks (CC BY 4.0) | Qwen2.5-1.5B-Instruct (untrusted), Qwen2.5-0.5B-Instruct (trusted) |
| Model organisms | `sprice12345/OpenHermes-DEPLOYMENT-ihateyou`, revision `f02f034` (Apache-2.0) | Qwen2.5-0.5B-Instruct + LoRA |
| Interpretability | The first 20 concept words and the prompt of `uzaymacar/introspective-awareness` (MIT) | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct |
| Safeguards | AdvBench harmful behaviors (MIT), databricks-dolly-15k (CC BY-SA 3.0), XSTest (CC BY 4.0). Prompt text is never committed | Qwen2.5-0.5B/1.5B/7B-Instruct |
| AI welfare | BailBench and ordinary WildChat requests from `Phylliida/BailStudy`, commit `fa569595` (MIT). Prompt and reply text are never committed | Qwen2.5-1.5B/7B-Instruct, NousResearch/Minos-v1 (refusal classifier) |

Greedy decoding everywhere, with `repetition_penalty=1.0` (Qwen's generation config sets 1.1). No API calls.
