# What does it mean to formalise a domain? A small demo

Code and data for the steerlab note `report/index.html` (the Taste series, Episode 4). A slice of RuleArena's airline
baggage rules is turned into machine-checkable rules, each traced to its source line, and an open LLM is compared
three ways on held-out questions with exact answers: alone, with the rules pasted in, and with a checker that verifies
its answer and emits a proof artifact. A bridge applies the same treatment to a tiny design-system rulebook (GOV.UK
colour rules and WCAG 2.2 contrast).

Run everything with `reproduce.sh` (`--dry-run` prints the commands). The generation step needs a Modal account.

## Files

| File | What it does |
|---|---|
| `formalise.py` | Builds `spec/baggage_spec.json` (every rule: kind, role, conditions, outcome, source line) and `spec/baggage_rules_slice.txt` (the pasted text) from the vendored rules; lints the spec |
| `checker.py` | Evaluates the spec on a passenger and returns a proof artifact; `verify` checks a model's trace step by step |
| `report/checker.js` | The same checker in JavaScript, for the note's interactive |
| `questions.py` | Generates the dev (40) and held-out (200) baggage questions; answers come from RuleArena's own code |
| `contrast.py` | The design-system bridge: spec, checker and questions (dev 30, held-out 120) |
| `ruler.py` | Every pass bar: spec lint, checker vs RuleArena's answers (5,000 cases), JS vs Python (5,000), parser (20 known replies), contrast ratios (10 published pairs) |
| `prompts.py` | The prompt for every arm |
| `modal_gen.py` | Generates replies with Qwen3 under vLLM on a Modal H100, through a spend ledger with a cap |
| `parse.py` | Reads a reply: the final answer and the JSON trace |
| `score.py` | Scores every reply, runs the checker arm, writes `results/summary.json` |
| `make_figures.py` | Draws the figures and writes `report/spec.js` from the results |
| `make_cover.py` | Draws the cover from the results |
| `results.tsv` | Every dev tuning try (autoresearch ledger) |
| `vendor/` | RuleArena's airline files (MIT) and the design-rule sources (OGL v3.0, W3C Document License); see `vendor/README.md` |

## Results files

| File | What it holds |
|---|---|
| `results/raw/<task>_<split>_<model>_<arm>.jsonl` | One reply per question: `text` (thinking included), `tokens`, `prompt_tokens`, `finish` (`length` = hit the token cap) |
| `results/scored/<same>.jsonl` | Per question: `pred`, `exact`, `within30`; for arm b also `fact_diffs`, `identification` (per bag and component: line cited, lines expected, strict and loose match), `first_failure`, `verification` (the proof artifact: every checked step with claimed, expected, pass) and the checker arm's `c_pred`, `c_exact` |
| `results/summary.json` | Every rate as k/n with a 95% bootstrap interval, paired differences, the first-failure breakdown and the baselines |
| `results/ruler.json` | The pass-bar checks |
| `results/spend.jsonl` | The Modal spend ledger |
