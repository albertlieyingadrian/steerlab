# What is pluralistic alignment? A case study with 8 countries

Pluralism series, Episode 1. An explainer of Overton, steerable and distributional pluralism, following the papers
(Value Kaleidoscope, A Roadmap to Pluralistic Alignment, Modular Pluralism, Artificial Hivemind,
Spectrum Tuning), backed by one experiment. Note: [`report/index.html`](report/index.html).

**The experiment.** 181 World Values Survey (Wave 7) questions from GlobalOpinionQA
([`Anthropic/llm_global_opinions`](https://huggingface.co/datasets/Anthropic/llm_global_opinions),
2023) that have a split for all eight countries (US, Philippines, Singapore, Malaysia, Indonesia, Taiwan, Brazil, South
Korea), after a comparability filter. Eight models answer each question 50 times at temperature 1, with no country in
the prompt and with "I live in <country>." for each country: Qwen3.5-9B-Base, Qwen3.5-9B, Qwen-SEA-LION-v5-9B,
Qwen3-8B-Base, Qwen3-8B (vLLM on Modal) and GPT-6 Luna, Claude Haiku 5.5, Gemini 3.1 Flash-Lite (OpenRouter).
Measures: Jensen-Shannon distance to each country's split with a finite-sample floor, answer spread (entropy), the
share of people holding the model's top answer, valid-answer rate, steering effect, 95% bootstrap over questions.

## Files

| File | What it does |
|---|---|
| `build_questions.py` | The question set and the comparability filter: `data/questions.json` (181 kept), `data/dev_questions.json` (20 dev questions outside the set), `results/question_filter.json` (every drop and why) |
| `plural.py` | Prompts, the answer parser, the metrics, the capped OpenRouter caller ($10) |
| `check_ruler.py` | Known-answer checks for the parser (26 replies) and the metrics (6 cases); `results/ruler_check.json` |
| `modal_sample.py` | Samples the five open models on Modal L40S with vLLM 0.25.1 (capped at $30 in code) |
| `sample_api.py` | Samples the three API models through OpenRouter, resumable |
| `analyse.py` | Every number in the note: `results/summary.json`, `report/data.json` (the interactive figures' data) |
| `make_figures.py`, `make_cover.py` | Static fallback figures and the cover, from `report/data.json` |
| `reproduce.sh` | Every command in order (`--dry-run` prints them) |
| `results/samples/main/<model>.jsonl.gz` | Every reply: one row per question and condition, `{"id", "condition", "replies": [50 strings]}` |
| `results/openrouter_spend.jsonl`, `results/modal_spend.jsonl` | Spend ledgers behind the caps |

Plan with the Design section and the predictions written before the main run: [`PLAN.md`](PLAN.md).
