# Sounds Like Me, Episode 3: training on a reward you can't verify

The note: [report/index.html](report/index.html). Episodes 1 and 2: [../sounds-like-me/](../sounds-like-me/),
[../sounds-like-me-ep2/](../sounds-like-me-ep2/). This folder reuses their writers, briefs and rulers (imported, never copied).
Open models only (Qwen3-8B, Qwen3-32B as a checker and simulated editor, DeBERTa-v3-small, voi-380M); no closed model.

## Code

| File | What it does |
|---|---|
| `data_ep3.py` | Training writers (149 public Enron and Blog authors, disjoint from the 45 evaluation writers; one of the 50 floor writers slipped in through a trailing space in its name, and the code now strips names, so a rerun gives 148), content briefs, style notes, edit pairs, the four SFT sets, eval inputs, 60 training briefs (new prospects and sender companies, Episode 1's offer and fact tables) |
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
| `data/dpo_pairs.jsonl` | `messages`, `chosen` (a simulated edit by Qwen3-32B), `rejected` (the SFT draft); labelled simulated |

The per-draft files (`data/drafts/`, `data/honesty/` and the GRPO log in `data/runs/`) stay private because drafts copy
real names, addresses and phone numbers from the writers' samples; the note shows redacted drafts only (names, emails,
phone numbers and addresses replaced by Episode 1's `privacy.py`).

The run as published differs from the current code in four ways found in code review afterwards, none retrained:
102 of the 1,763 style-notes examples had a note made partly from their own target text (now each such target gets
its own note); SFT and DPO examples over 2,048 tokens were cut from the front, dropping the system prompt (178
edit-pair, 100 raw and 102 shuffled examples, 45 of 627 DPO pairs; the cap is now 4,096 and cuts the middle of the
context instead); the DPO reference pass ran with LoRA dropout on (now off); and the honesty proxy has no check for
names, though the plan listed one (unchanged, so the reward matches the run; disclosed in the note).

The voice RM holds no human preference labels: it learned authorship from public writers. Simulated edits are labelled as
simulated wherever they appear. Adapters live on a Modal volume, not in the repository.
