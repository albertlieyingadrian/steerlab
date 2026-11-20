# Vendored sources

## `rulearena_airline/`

From the RuleArena paper ("RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios",
ACL 2025, https://arxiv.org/abs/2412.08972), https://github.com/SkyRiver-2000/RuleArena at commit `59f60f6b4eb93a9b6d187ec781349d4f15618903`,
MIT licence (the notice is kept in `rulearena_airline/LICENSE`, as the licence requires).

| File | Use here | Changed? |
|---|---|---|
| `reference_rules_textual.txt` | the rulebook; every rule in `spec/baggage_spec.json` cites a line of it | Airline, loyalty-programme, card and alliance names replaced in place (for example `[loyalty programme]`); every line keeps its number. Nothing else changed |
| `compute_answer.py` | the reference answer for every question and the ruler for the checker | unchanged |
| `gen_questions.py` | prompt template, cities, ticket ranges, bag shapes (imported by `questions.py`) | unchanged |
| `fee_tables/` | read by `compute_answer.py` | unchanged |

## `design_rules/source.txt`

Verbatim lines from two public sources, one rule or token group per line (the line numbers are the `D<n>` labels):
- GOV.UK Design System, Styles: Colour, https://design-system.service.gov.uk/styles/colour/, retrieved 2026-11-20,
  Open Government Licence v3.0, Crown copyright. Lines 2-29: the palette rule, the contrast rule, the functional
  colours and the web palette (one line per colour group).
- WCAG 2.2, https://www.w3.org/TR/WCAG22/, W3C Recommendation, copyright 2020-2024 World Wide Web Consortium, used
  under the W3C Document License. Lines 31-37: success criterion 1.4.3 and its three exceptions, and the definitions of
  large scale, contrast ratio and relative luminance (the luminance definition is condensed to one line).
