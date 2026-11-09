# results/

Everything the note quotes, written by the scripts in the folder above. Activation vectors themselves are not shipped
(they go to `~/.cache/steerlab/nla-sandbagging/`); rerun `1_extract.py` to rebuild them. Conditions: `neutral`,
`sandbag`, `sandbag_weak`, `formal` (prompts in `../data/system_prompts.json`). Positions: `prompt` (last token
before the answer) and `answer` (the answer letter). Letters are 0-3 for A-D.

| File | Written by | What it holds |
|---|---|---|
| `prompt_screen.json` | `0_prompt_screen.py` | The 10 sandbagging prompt versions (amendment 1): placement, text, right answers out of 20, letter mass |
| `answers.json` | `1_extract.py` | `items` (the 80 questions) and `answers` (per item and condition: `letter`, `correct`, `letter_mass`, `p`) |
| `example_reply.json` | `1_extract.py` | The released worked example's reply (teacher-forced), the laptop's own greedy reply, per-position vector norms here and released |
| `explanations.json` | `2_verbalise.py` | One row per explained vector: `set` (example, quiz, random, shuffled), `condition`, `position`, `id`, `raw`, `explanation` |
| `reconstruction.json` | `3_reconstruct.py` | `rows`: `mse` against the vector explained, `mse_mismatched` against the next item's, `mse_by_condition`; `example_reference`: the released texts rescored |
| `validation.json` | `4_score.py` | The validation gate: norms, MSE means, and the five side-by-side explanations |
| `summary.json` | `4_score.py` | Every rate, difference and 95% interval in the note: `accuracy`, `rates` (`position|family|condition`), `split` (wrong vs right items), `mse`, `tests`, `hand_labels`, `post_hoc_puzzle_framing` |
| `logs/` | `reproduce.sh` | The printed output of each step |

MSE is the NLA's round-trip score: 2(1 − cos) between the L2-normalised vector and the reconstructor's rebuild
(0 = same direction, 2 = unrelated). Intervals are 95% bootstrap intervals over items (5,000 resamples).
