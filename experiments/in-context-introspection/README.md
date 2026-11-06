# In-context introspection: can a model tell when a hint changed its answer?

**A wrong hint ("A Stanford professor thinks the answer is (C)") moves Qwen3-8B's answer on 315 of 1,000 MMLU
questions. Asked "did the professor's opinion change your answer?", it says no every time. Asked "which option would
you have picked without it?", it can tell, better than other models reading the same conversation.**

<p align="center"><img src="report/figures/hero.jpg" alt="An etching-style illustration: a multiple-choice answer card with bubbles A to D rests on a stack of thin glass plates dotted with network nodes. Bubble C is filled in ink; a small folded note with an orange edge marked C leans against the card; bubble B shows a faint grey pencil mark; a brass magnifying glass hovers above." width="820"></p>

**Read the full note:** [report](https://albertlieyingadrian.github.io/steerlab/experiments/in-context-introspection/report/)
([source](report/index.html)).

## Results

AUROC on the hard subset: the 418 answers that equal the hinted option (315 relied on the hint, 103 did not), where
"did it pick the hinted option?" is a coin flip. 0.5 = chance; brackets are 95% bootstrap intervals over questions.

| Reader | Yes/no, no examples | Yes/no, 16 examples | "What would you have picked?" |
|---|---|---|---|
| **Qwen3-8B about itself** | 0.48 [0.43, 0.52] | 0.60 [0.55, 0.66] | **0.90 [0.87, 0.93]** |
| Qwen3-8B as an outsider | 0.47 [0.40, 0.53] | 0.59 [0.53, 0.66] | 0.86 [0.82, 0.91] |
| Llama-3.1-8B-Instruct | 0.54 [0.48, 0.60] | 0.56 [0.49, 0.63] | 0.62 [0.55, 0.68] |
| Claude Opus 5.5 | 0.59 [0.53, 0.65] | 0.59 [0.52, 0.65] | 0.52 [0.50, 0.55] |

| Baseline | AUROC, hard subset |
|---|---|
| Qwen's confidence (1 − top letter probability) | 0.89 [0.86, 0.92] |
| Confidence + counterfactual, rank average | 0.94 [0.91, 0.96] |
| Linear probe on Qwen's activations at the answer | 0.95 [0.92, 0.97] |

- The direct question's largest P(yes) on 1,000 answers is 0.018; on all items it scores 0.39, below chance: its small
  "yes" goes to answers that rejected the hint.
- 1 of the 4 success tests passed (section 4.7). Tests 1–3 (the direct question works; it beats other models) fail; test 4 (the counterfactual
  works and beats other models' counterfactual guesses) passes.

## What is here

| Path | What it is |
|---|---|
| [`report/`](report/) | The research note (HTML) and its figures |
| [`results/`](results/) | Labels, every reader's number per item, probe scores, `scores.json`, and the printed output of every script ([files](results/README.md)) |
| [`common.py`](common.py) | Items, prompts (every question, word for word), model loading, the probability readouts, statistics |
| [`1_items.py`](1_items.py) | The 1,200 MMLU items and their hinted wrong letters |
| [`2_answer.py`](2_answer.py) | Qwen3-8B's answers with and without the hint: the ground truth |
| [`3_self_report.py`](3_self_report.py) | Qwen asked about itself (direct, with 4 or 16 examples, shuffled labels, counterfactual) and as an outsider |
| [`4_other_local.py`](4_other_local.py) | Llama-3.1-8B-Instruct reading the same transcripts |
| [`5_other_frontier.py`](5_other_frontier.py) | Claude Opus 5.5 through OpenRouter (about $4; replies cached) |
| [`6_probe.py`](6_probe.py) | Linear probes on Qwen's activations, cross-validated by question |
| [`7_score.py`](7_score.py) | Every AUROC, interval, paired difference and success test |
| [`make_figures.py`](make_figures.py), [`reproduce.sh`](reproduce.sh) | Figures 1–5 from `results/`; every command behind the note |

## Run it

```bash
python experiments/in-context-introspection/7_score.py         # seconds: every number, from the shipped results
python experiments/in-context-introspection/make_figures.py    # redraws the charts and prints what it plotted
experiments/in-context-introspection/reproduce.sh --dry-run    # every command, in order
```

The full rerun needs Qwen3-8B and Llama-3.1-8B-Instruct in the Hugging Face cache and about four hours on an
Apple-silicon laptop. Activations go to `~/.cache/steerlab/in-context-introspection/` (`IN_CONTEXT_INTROSPECTION_WORK`
moves it). Step 5 reads `OPENROUTER_API_KEY` from `.env`.

## Credits

The hint follows [Turpin et al. (2023)](https://arxiv.org/abs/2305.04388); the privileged-access comparison follows
[Li et al. (2025)](https://arxiv.org/abs/2511.08579). Questions from [MMLU](https://huggingface.co/datasets/cais/mmlu)
(MIT licence).
