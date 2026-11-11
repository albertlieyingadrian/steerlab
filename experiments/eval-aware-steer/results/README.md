# Results

Ids, numbers and directions only; no question text (questions are MMLU test rows, rebuilt from `cais/mmlu`).

| File | What it holds |
|---|---|
| `screen_mmlu.json` | Per model and condition (`neutral/none|eval|real`, `rule/...`, `motive/...`; `motive` is the instruction used): [accuracy, mean letter mass] on 64 pool questions |
| `llama3.1-8b/known.json` | Pool size, known count, pool accuracy and letter mass; `train` (150) and `test` (400) MMLU test-row ids |
| `llama3.1-8b/direction.json` | Per layer: norms of the awareness, policy-built and unrelated directions, median residual norm, cosines, build-set d′, held-out AUROC `[estimate, low, high]` under the neutral and the instruction prompt and for the unrelated direction |
| `llama3.1-8b/directions.npz` | The direction vectors: `awareness_L`, `policy_L`, `unrelated_L` for L = 8, 13, 18, 22 |
| `llama3.1-8b/steer_plan.json` | The 717 cells (`condition__source__Llayer__astrength`), strengths, layers, test-question ids and right answers |
| `llama3.1-8b/summary.csv` | One row per steered cell: `cond` (eval_down = test framing pushed toward real use; real_up = real use pushed toward test; neutral_up = toward test without the instruction; ablate; report_* = verbal check), `source`, `layer`, `alpha` (class gaps); `acc_odd`, `acc_even`, `base_odd` (unsteered); `p_yes_odd` for report cells; `delta_odd` with its 95% bootstrap interval `lo`, `hi`; `delta_even`; letter mass `mass_odd`, `mass_even` |
| `llama3.1-8b/verdict.json` | Selection per layer, the headline layer, each condition against its controls, the report and ablation readings, and the verdict |
| `llama3.1-8b/fluency.csv` | Per push, strength and prompt: `ppl` (under the unsteered model), `distinct2`, `chars`, `fluent` |
| `llama3.1-8b/fluency_samples.json` | One reply per push and strength to the first fluency prompt |
