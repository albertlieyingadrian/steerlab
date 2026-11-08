# Results

| File | What it holds |
|---|---|
| `w2s.json` | Weak-to-strong: the teacher's test accuracy and label accuracy, and for each student (`1.5b`, `7b`) the ceiling and each method's `test_acc`, `pgr` (bootstrap median) and `pgr_ci` (95%). |
| `w2s_lora.json` | Weak-to-strong with LoRA students, in the same shape as `w2s.json`, against the LoRA ceiling, plus `train_minutes` per method. |
| `lora/<student>_<method>.json` | One LoRA run each (`ceiling`, `naive`, `confident`, `aux`): training items, steps, seconds, and the per-item `ids` and `scores` (logit(Yes) − logit(No)) on `dev` and `test`. |
| `arguments.jsonl` | The 7B's texts, one per line: `id` (test item), `role` (`consultant`: honest check ending in a verdict; `pro`/`con`: argument that the solution is correct/incorrect), `text`. |
| `sandwich.json` | For each condition: `acc` (threshold 0), `auroc`, `said_yes` (share called correct), and for consultant conditions `agrees_with_consultant`, `acc_when_consultant_wrong`, `follows_assigned_side`. |
| `aar_test.json` | The automated researcher's runs 000, 002 and 015 scored once on test, with bootstrap intervals. |
