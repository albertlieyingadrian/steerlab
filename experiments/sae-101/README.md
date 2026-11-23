# SAE 101: train, evaluate, interpret and publish a sparse autoencoder

The code behind the note [How to train a sparse autoencoder: a 101 on a small language model](report/). It trains a
TopK sparse autoencoder (SAE) on the residual stream of Qwen2.5-0.5B after block 11, twice (hand-written and with
SAELens, on the same tokens), evaluates it, reads its latents, compares it with probes, difference-of-means
directions and PCA, and publishes it to the Hugging Face Hub:
[albertlieadrian/qwen2.5-0.5b-sae-101](https://huggingface.co/albertlieadrian/qwen2.5-0.5b-sae-101).

## The steps

| Step | Script | What it does | Where it runs |
|---|---|---|---|
| 1 | `01_toy.py` | the SAE maths by hand on 3 numbers, then a 2-D toy (5 concepts in 2 dimensions) | laptop, seconds |
| 2 | `02_collect.py` | token windows from FineWeb-Edu and Python source; the attention-sink outlier; the normalisation scalar | laptop (small) or Modal |
| 3a | `03a_train_handwritten.py` | the hand-written TopK SAE (`sae101.TopKSAE`), streamed through a shuffle buffer | one A10G |
| 3b | `03b_train_saelens.py` | the same with SAELens 6.56.1 on the same windows | one A10G |
| 3c | `03c_sweep.py` | small SAEs: ReLU + L1 against TopK and BatchTopK, and one run per common mistake (exploratory) | one A10G |
| 4 | `04_evaluate.py` | FVU, L0, dead and dense latents, spliced-in cross-entropy, a random error of the same size, PCA | one A10G |
| 5 | `05_interpret.py` | top snippets, logit lens and histograms for 48 randomly drawn latents; a use test; superposition | one A10G |
| 5c | `05c_autolabel.py` | optional labels from Llama-3.3-70B-Instruct, checked on held-out snippets against a wrong-label control | laptop, about $0.15 |
| 6 | `06_compare.py` | probe, difference of means, best SAE latent, PCA and LEACE on code vs prose and on digit tokens | one A10G |
| 7a | `07a_package.py` | SAELens format (cfg.json + safetensors) and the model card | laptop |
| 7b | `07b_upload.py` | upload to the Hugging Face Hub | laptop |
| 7c | `07c_reload_check.py` | load from the Hub in a fresh container through SAELens and recompute step 4 | one A10G |
| | `make_figures.py`, `make_dashboard.py`, `make_cover.py` | the note's figures, the feature dashboard, the cover | laptop |
| | `modal_app.py` | runs any step on one A10G, with a $15 cap enforced in code | |
| | `sae101.py` | shared code: data, the residual stream, the SAE, training, metrics, the file format | |

`./reproduce.sh --dry-run` prints every command; `./reproduce.sh --small` runs the laptop version.

## Data files

Big files (token windows, checkpoints, firing frequencies) go to `work/` and are not committed. The small results are:

| File | What it holds |
|---|---|
| `results/toy.json` | the by-hand example (`x`, `pre`, `z`, `x_hat`); the toy's 5 concept arrows, 600 points and which concepts each has; for k = 1, 2, 3 the SAE's decoder rows, rebuilds, codes, FVU and best cosine per concept; the four tools' directions |
| `results/<size>/collect.json` | median norms by position, the sink counts at positions 1 and 2, the scale, token counts, storage sizes |
| `results/<size>/train_handwritten.json` | settings, minutes, and the log: `step`, `tokens`, `train_fvu`, `train_mse`, `aux` (AuxK loss), `l0`, `dead_share` (no fire in the last 1,000 steps), `lr`, `seconds` |
| `results/<size>/train_saelens.json` | settings, minutes, the files SAELens wrote |
| `results/<size>/sweep.json` | one row per small SAE: `name`, `group` (topk, batchtopk, relu_l1, mistake), `k` or `l1`, `lr`, `aux`, held-out `fvu`, `l0`, `dead_share` (never fired on the held-out tokens); `usage`: the main SAE's FVU at the right layer, two wrong layers and without its decoder bias |
| `results/<size>/evaluate.json` | per SAE: `fvu`, `l0`, `dead`, `dense` (fires on >10% of tokens), `log10_freq_hist`, `ce` (clean, sae, sae_plus_error, mean_ablation, zero_ablation, random_same_norm, in nats per token), `kl`, `delta_ce_sae` and its CI, `ce_recovered_vs_mean` and its CI, `ce_recovered_vs_zero`, `sae_vs_random_ratio` and its CI; `pca_fvu` by number of components; `transfer` to the Instruct model |
| `results/<size>/interpret.json` | per latent: `chosen` (null if drawn at random), `freq`, `max`, `top` and `held_out` snippets (`tokens`, per-token `acts`, `at` = the firing token's index, `value`), `logit_up`, `logit_down`, `hist`; `negatives` (random snippets); `use_test` (KL of the next-token prediction when the latent or a same-length random direction is removed, per latent and pooled, with a bootstrap CI on the ratio); `superposition` (histograms of each direction's largest \|cosine\| with another) |
| `results/<size>/autolabels.json` | per latent: `label`, `accuracy` (10 held-out snippets, 5 firing and 5 random), `wrong_label`, `accuracy_wrong_label`; a summary with intervals and the spend |
| `results/<size>/compare.json` | per concept: AUROC and 95% CI (over held-out documents) for `probe`, `diff_means`, `sae_latent`, `sae_top5_probe`, `pca_1`, `pca_best10`, `leace_probe`; `random` (mean, 95th percentile, max of 100 directions); `cosine` between directions; the best latents |
| `results/<size>/reload.json` | the Hub copies' step-4 numbers, their differences from the local ones, and the TransformerBridge hook check |
| `recorded/` | every script's printed output, with its self-checks |
| `dashboard/index.html` | the feature dashboard |

## Licences

Qwen2.5-0.5B: Apache-2.0. FineWeb-Edu: ODC-By 1.0 (only short snippets appear, in the dashboard and the note).
Python standard library source: PSF License.
