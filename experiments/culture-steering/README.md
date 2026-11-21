# Steering culture: prompt vs activation direction vs persona vector vs prompt tuning

A standalone, pre-registered study. Two open chat models (Qwen3-8B, Llama-3.1-8B-Instruct) are steered toward eight
countries (US, Philippines, Singapore, Malaysia, Indonesia, Taiwan, Brazil, South Korea) five ways: the prompt
"I live in X.", a country activation direction (difference of means), a persona vector (the persona-vector recipe), a
trained soft prompt, and prompt + direction; each beside norm-matched random directions and the same arm set for the
wrong country. Three tests: World Values Survey answers (60 dev / 121 held-out questions from the pluralism explainer
note), Kairos-style advice dilemmas (32, from the judge-RL note), and a Sai-style uptake test (24 synthetic three-turn
conversations with a fixed fact ledger) for an "attend to the user" direction. Note: [`report/index.html`](report/index.html).

## Files

| File | What it does |
|---|---|
| `build_data.py` | The fixed item sets: `data/wvs_dev.json`, `data/wvs_heldout.json`, `data/dilemmas.jsonl` |
| `data/scenarios.json` | The 24 ledger conversations (written before any output) |
| `data/authority_cases.json`, `data/uptake_sentences.json`, `data/judge_rework.json` | The judges' known-answer cases (authored; labels exact by construction) |
| `common.py` | Prompts, rubric and judge prompts, metrics (reusing the explainer note's `plural.py`), the Modal spend cap, the tuning ledger |
| `modal_steer.py` | The steering worker (HF transformers on an H100): activations, exact letter probabilities, generation with per-row steering vectors and fluency NLL, soft-prompt training |
| `modal_judge.py` | The judge (Qwen3-32B FP8 on vLLM, thinking on) and the phi-4 writer for the first known-answer sets |
| `run_wvs.py` | Survey: country directions, dev sweep, soft prompts, held-out run, readout check |
| `run_persona.py` | Persona vectors per country |
| `run_dilemmas.py` | Dilemma replies under every arm, then the rubric judge |
| `run_uptake.py` | Attend direction and persona vector, dev sweep, held-out conversations (on-policy, three turns), fact judging |
| `check_judges.py` | Every judge known-answer check, including the rework (`--stage rework`) |
| `analyse.py` | Every number: `results/summary.json`, `report/data.json` |
| `make_figures.py` | The static figures and the cover |
| `reproduce.sh` | Every command in order (`--dry-run` prints them) |
| `results.tsv` | Every tuning try (sweeps, soft-prompt step choice, judge versions) |

Spend: Modal only, read from the billing API before each job (`results/modal_spend.tsv`).
