# results/

Everything the note quotes. Every file except the direction was written by the scripts one folder up; nothing is
edited by hand. Recoveries are on the patching scale: 0 = the corrupted run, 1 = the clean run.

| File | What it holds |
|---|---|
| `sandbagging_direction_l14.npz` | The sandbagging direction: `vector` (896 float32, length 3.50), `layer` (14, the output of decoder block 14 of Qwen2.5-0.5B-Instruct), `train_pairs` (the 25 prompt pairs it was built from). `5_direction_build.py` rebuilds it from the public prompt pairs |
| `tour.json` | `1_tour.py`: library versions, `max_diff_eager` and `max_diff_sdpa` (largest bridge-vs-Hugging Face logit difference), `compat_max_shift` and `compat_shift_spread` (what compatibility mode does to raw logits), `ld_*` (logit(Mary) − logit(John) before and after compatibility mode and with one head zeroed), `n_hooks` |
| `induction.json` | `2_induction.py`: `loss_first_half` and `loss_second_half` (nats), `top_raw` and `top_exclude_bos` (top five heads and scores), the top head's share of attention on the induction cell and on BOS |
| `induction_scores.csv`, `induction_scores_raw.csv` | Induction score per head, rows `L0`–`L11`, columns `H0`–`H11`; with and without the BOS token excluded |
| `induction_pattern.csv` | The top induction head's 51 × 51 attention pattern: rows are query positions `q0`–`q50`, columns key positions |
| `ioi_prompts.json` | The eight IOI prompt pairs: `template` (ABBA/BABA), `io`, `s`, `clean`, `corrupted`, and each prompt's `clean_ld` and `corrupted_ld` (logit(IO) − logit(S)) |
| `ioi_patching.json` | `3_ioi_patching.py`: mean clean and corrupted logit differences, sweep sizes and times, top five and bottom three heads by head patching |
| `ioi_resid_patch.csv` | Residual-stream patching recovery, rows `L0`–`L11` (resid_pre), columns the 15 token positions (`IO/S1` = IO in ABBA prompts, S1 in BABA) |
| `ioi_head_patch.csv` | Head-output patching recovery (each head's clean output copied in at every position), rows layers, columns heads |
| `ioi_dla.csv` | Direct logit attribution per head at the last position, in logits, mean over the eight prompts |
| `ioi_checklist.csv` | One row per published head: `class`, `head`, `fuzzy`, `dla`, `dla_rank` (1 = most positive of 144), `patch`, `patch_rank`, `expected_by` (the tool the paper found that class with), `found` |
| `ioi_head_sets.json` | `4_ioi_attribution.py`: the DLA decomposition check, `found_per_class`, and for each head set (`sets`, `random`) its `heads`, `recovery` and `logit_diff` |
| `direction_build.json` | `5_direction_build.py`: the rebuilt direction's length, its cosine with the shipped vector, and its accuracy and AUROC as a probe on the 14 held-out prompts |
| `direction_projection.csv` | `6_direction_read.py`, one row per layer: mean reading on the 8 sandbagging and 8 honest prompts, `gap_mean`, `gap_se` (standard error over the 8 pairs), `wins` (pairs where the sandbagging prompt reads higher), and the mean and largest absolute gap of 20 random unit directions |
| `direction_read.json` | The strength-0 hook test, one steered example, the gap at layer 14 and each random direction's gap there |
| `direction_attribution.csv` | `7_direction_attribution.py`: each of 241 components' contribution to the reading at layer 14, final position, mean over 8 sandbagging prompts, 8 honest prompts, and the `gap` between them |
| `direction_attribution.json` | Concentration (`top3_share`, `needed_for_half`), the verdict, attention and MLP totals per layer, and the same split for 5 random directions |
| `direction_patch_whole.csv` | `8_direction_patch.py`, sweep A: the whole residual vector patched, rows `L0`–`L23` (resid_pre), columns positions 29–55 (`position:'token'`) |
| `direction_patch_direction.csv` | Sweep B: only the sandbagging direction swapped, all 56 positions |
| `direction_patch_random.csv` | Sweep C: each of 5 random unit directions swapped the same way, rows `random<k>:L<layer>`, columns positions 29–55 |
| `direction_patch.json` | The pair's logit differences and greedy answers, the direction's reading at the last position per layer for both prompts, best and worst cells, and sweep C: for each of 5 random directions its best and worst cell |
| `logs/` | The printed output of every script as run for this note |
