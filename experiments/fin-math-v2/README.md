# FinanceMath v2: open-weight SOTA: an open model in a harness beats gpt-5.4 and comes close to Claude Opus 5.5

Episode 6 of "Harnesses and post-training", continuing [fin-math](../fin-math/) (episode 2). Note:
[report/index.html](report/index.html). The code and results live in `../fin-math/`; this folder holds the note, its
figures and cover, and `reproduce.sh`.

Result (pre-registered single test look, the second on this test set): DeepSeek-V4-Flash (`deepseek/deepseek-v4-flash-0731`
on OpenRouter) in thinking mode, 4 answers per question and a majority vote, scores 0.885 [0.865, 0.904] on the 1,000
FinanceMath test questions against gpt-5.4's 0.818, paired +6.7 points [+4.7, +8.7]: beats. One thinking answer: 0.870,
+5.1 [+3.3, +7.0]. $0.0028 a question against $0.0039 for one gpt-5.4 call.

## Code

| File | What it does |
|---|---|
| `../fin-math/fm_v4.py` | `sample --cand <c> --kind reason\|program --n 4 [--split test]`: draws samples until each question has n (streamed, provider recorded); `analyze`: `results/v4_screen.json` |
| `../fin-math/v4_test_bar.py` | The test look against gpt-5.4, gpt-5.4-mini, episode 2's harness and the 8B model; writes `results/v4_test.json` |
| `make_figures.py` | `report/figures/levers.png`, `test.png`, `cost.png`; prints every plotted number |
| `make_cover.py` | `report/figures/hero.jpg` and `../shared/thumbs/fin-math-v2.jpg` |
| `reproduce.sh` | `--scores` recomputes from saved results; `--dry-run` lists every command |

## Results

In `../fin-math/results/`:

- `runs/<split>__v4-<candidate>-<kind>.jsonl`: one line per sample: `id`, `value` (parsed answer), `correct`,
  `tokens_out`, `reasoning_chars`, `capped` (hit the 16,384-token limit), `provider` (OpenRouter candidates only).
  Candidates: `or-dsflash-think` (the chosen system), `or-dsflash-plain`, `or-ds41flash-plain` (OpenRouter);
  `dsflash-think`, `ds41flash-think`, `dspro-plain`, `kimi-k25` (Hugging Face, partial). `validation-50__*`: the
  50-question thinking check.
- `v4_test.json`: `vote4`, `mean_per_sample` (accuracy and 95% interval), paired comparisons (`vote4_vs_gpt-5.4`,
  `mean_vs_gpt-5.4`, `vote4_vs_gpt-5.4-mini`, `vote4_vs_v3_vote8`, `vote4_vs_8b_majority`), `verdict`, `providers`,
  `tokens_per_sample`, `cap_rate`, `usd`, `usd_per_question`.
- `v4_screen.json`: the Hugging Face screen (partial coverage per candidate, `n` questions).

The pre-registration was committed before the test look; its rules are summarised in the note (sections 1 and 3).
