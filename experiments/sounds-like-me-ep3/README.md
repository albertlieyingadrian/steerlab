# Sounds Like Me, Episode 3: training on a reward you can't verify

The note: [report/index.html](report/index.html). Episodes 1 and 2: [../sounds-like-me/](../sounds-like-me/),
[../sounds-like-me-ep2/](../sounds-like-me-ep2/). This folder reuses their writers, briefs and rulers (imported, never copied).
Open models only (Qwen3-8B, Qwen3-32B as a checker and simulated editor, DeBERTa-v3-small, voi-380M); no closed model.

## Code

| File | What it does |
|---|---|
| `data_ep3.py` | Training writers (149 public Enron and Blog authors, asserted disjoint from the 45 evaluation writers and the floor pool), content briefs, style notes, edit pairs, the four SFT sets, eval inputs, 60 training briefs |
| `modal_ep3.py` | Modal app: SFT, DPO and GRPO with LoRA on Qwen3-8B, vLLM generation with adapters, the voice RM, the detector |
| `reward_fns.py` | The GRPO reward pieces: honesty proxy, copy flag, RM context, strict text form |
| `train_ep3.py` | Orchestration: proxy check, RM, SFT, eval, scoring, DPO, GRPO (spawned, resumable), checkpoint evals |
| `drive_grpo.py` | Unattended driver: hack hunt every 20 steps, resume after a crash, held-out for the final and dev-best checkpoints |
| `analyze_ep3.py` | Per-arm tables (with Episode 2's comparators), the hack hunt, the seven predictions |
| `make_figures.py`, `export_pages.py` | Figures and the note's `report/data.js` |
| `reproduce.sh` | Every step in order (`--dry-run` prints them) |

## Results files

| File | Columns |
|---|---|
| `results/summary.json` | per arm: `position`, `position80`, `spec`, `unsupported` ([mean, low, high] over writers), `honesty_proxy`, `loop_share`, `broken_share`, `words`, `detector`, `task`, `copy_flag_episodes`, `attr_mean_rank`; `contrasts` as [mean, low, high, writers] |
| `results/hack_hunt.json` | per GRPO checkpoint on dev writers: the same columns plus `train_reward`; the reward curve per step |
| `results/predictions.json` | P1-P7: metric, value, threshold, hit |
| `results/ruler_voice_rm.json`, `results/ruler_honesty_proxy.json` | the voice RM's model card and pass bars; the proxy's check on Episode 1's labels |
| `results/sft_runs.json`, `results/dpo_sft_edits.json`, `results/grpo_*_config.json`, `results/grpo_best.json` | training runs, the GRPO configuration and the dev-best rule |
| `results.tsv` | every tuning try and choice on development writers |
| `data/drafts/*.jsonl` | drafts: `job`, `model`, `author`, `brief`, `arm`, `text` |
| `data/honesty/*.json` | per job: `n_claims`, `n_unsupported`, `unsupported`, `invented_relationship` |
| `data/dpo_pairs.jsonl` | `messages`, `chosen` (a simulated edit by Qwen3-32B), `rejected` (the SFT draft); labelled simulated |
| `data/runs/grpo_sft_edits/log.jsonl` | per GRPO step: `reward`, `voice`, `voice_z`, `honesty_proxy`, `detector`, `copy`, `words`, `grad_norm`, `sample` |

The voice RM holds no human preference labels: it learned authorship from public writers. Simulated edits are labelled as
simulated wherever they appear. Adapters live on a Modal volume, not in the repository.
