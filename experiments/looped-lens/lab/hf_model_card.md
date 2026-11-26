---
license: mit
tags:
- interpretability
- lens
- looped-transformer
base_model: thinkingmachines/Inkling-2-Mini
---

# Inkling 2 Mini future lens (v1)

Two small translators for reading [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
made for the [Looped Lens Challenge](https://ndif.us/looped-lens-challenge/)
([repo](https://github.com/ndif-team/looped-lens-challenge)).

## What it is

Inkling 2 Mini is looped: layers 6-17 run three times, so a token passes through 48 layer visits
(0-47). Each translator reads one state at one visit, at the last prompt token: the input of that
visit's MoE sublayer (`post_attention_layernorm.input`), which is the layer's own depth-attention
mix of the completed blocks and the active block, with the previous pass's message gated in.

The translator is a low-rank affine map, `y = x + U (V x) + b` (rank 128), followed by the model's
own final norm and unembedding. It was fitted so that the readout matches the model's
distribution over the **first content token of its reply**, one token past the chat control token
that comes next (a "future lens"). For plain text the target is the ordinary next token.

| Key | Visit | Layer, pass |
|---|---|---|
| `mlp_in.34.{U,V,b}` | 34 | layer 10, pass 3 |
| `mlp_in.44.{U,V,b}` | 44 | layer 20, the final block |

## Training

- About 6,000 prompts: open-domain questions from `nq_open` in the model's chat format, random
  arithmetic questions, and `wikitext-103` sentence prefixes. None are challenge questions.
- Targets: the model's own top-256 log-probabilities at the target position (renormalised).
- 3 epochs of AdamW (lr 2e-3, weight decay 1e-4), cross-entropy to the target, 10% held out.
  Held-out cross-entropy fell from about 25 to 3.7 nats.
- Stored in bfloat16.

## v2 file: `lens_v2.safetensors`

Five translators, keyed `<lens>.<visit>.{U,V,b}`, all rank 128:

| Key | Visit | Reads | Trained to |
|---|---|---|---|
| `future_mlp_in.34` | 34 (layer 10, pass 3) | MoE input | first content token of the reply |
| `future_mlp_in.44` | 44 (layer 20) | MoE input | first content token of the reply |
| `future_act_in.45` | 45 (layer 21) | active block | first content token of the reply |
| `concept_mlp_in.27` | 27 (layer 15, pass 2) | MoE input | the words around the current word |
| `concept_mlp_in.33` | 33 (layer 9, pass 3) | MoE input | the words around the current word |

The **concept** translators were fitted on wikitext only: the target at the last token of a word is
the set of clean words within four words of it (the word itself counted twice), and about 15% of
input words had a random typo injected, so the lens learns to name the intended word.

## Use

```python
from safetensors.torch import load_file
w = load_file("future_lens_v1.safetensors")
y = x + w["mlp_in.34.U"].float() @ (w["mlp_in.34.V"].float() @ x) + w["mlp_in.34.b"].float()
logits = lm_head(final_norm(y))
```

## License

MIT.
