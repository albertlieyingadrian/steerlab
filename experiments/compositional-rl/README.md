# Compositional RL at 1.7B: a replication with a step test

A pre-registered small-scale replication of He, Li, Wang and Vitercik, "Compositional Reasoning in Language Models
under Reinforcement Learning Post-Training" ([arXiv 2609.19465](https://arxiv.org/abs/2609.19465), NeurIPS 2026), on
Qwen3-1.7B with GRPO, plus three extensions. Note: [`report/index.html`](report/index.html). Plan:
[`PREREGISTRATION.md`](PREREGISTRATION.md).

**Result in one line.** Chain training beats skill training on chains (segment trees +0.5, insert-delete chains
+0.09), but skill training already lifts chains a lot at this scale and chain training costs single steps, so the
published asymmetry holds only partly; a step test shows skill training buys steps and chain training buys the
chaining ("glue").

## Files

| File | What it does |
|---|---|
| `tasks.py` | The five data structures (array, min-heap, BST, hashmap, counting Bloom filter) and segment-tree construction; atomic and compound prompts at Levels I-III; the answer checker (the paper's relaxations) |
| `modal_rl.py` | On Modal: the untrained-model screen, GRPO training (vLLM rollouts + a hand-written LoRA update, right-padded), held-out evaluation, the step test and layer-block adapter swaps |
| `configs/` | Every run's settings: `all.json` (the 22 pre-registered runs), `l3_rerun_s0.json` (amendment 1), `mech_step.json`, `mech_swap.json`, pilots |
| `analyse.py` | Every pre-registered test with bootstrap intervals, into `results/summary.json` |
| `make_figures.py`, `make_cover.py` | Figures and cover from `results/`, printing what they plot |
| `rescore.py` | Rescore saved replies with the current checker |
| `reproduce.sh` | Every command in order (`--dry-run` prints them) |
| `results/` | Every reply (`*.jsonl.gz`), training curves and `summary.json`; columns in the note's appendix |

## Recipe

LoRA rank 32 on all linear layers, 60 GRPO steps × 32 prompts × 8 samples, reward 1 for an exact final state,
advantage = reward − group mean, loss averaged over all reply tokens, no KL, learning rate 3e-5, temperature 1.0,
3,072 new tokens in training, greedy 4,096 in evaluation. One H100 per run.
