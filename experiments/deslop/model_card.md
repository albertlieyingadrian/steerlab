---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
language:
- en
pipeline_tag: text-generation
library_name: {library}
tags:
- qwen3
- lora
- writing-style
- slop
- deslop
- steerlab
---

# qwen3-4b-deslop

Qwen3-4B-Instruct-2507 with the chat-assistant writing habits partly trained out: fewer em dashes, "it's not X, it's
Y" contrasts, lists of three, questions answered on the spot, signposts, slop words, markdown and emoji in prose. It
is a rank-16 LoRA distilled from a teacher (the same model with a plain-style system prompt and a backtracking
sampler), so it needs no system prompt and no special sampler. {variant}

Research note with every number, the method and its controls:
https://albertlieyingadrian.github.io/steerlab/experiments/deslop/report/

Toolkit used to build and score it:
`pip install "steerlab-deslop @ git+https://github.com/albertlieyingadrian/steerlab#subdirectory=packages/deslop"`

## Run it

```bash
ollama run hf.co/albertlieadrian/qwen3-4b-deslop-GGUF:Q4_K_M
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("albertlieadrian/qwen3-4b-deslop")
model = AutoModelForCausalLM.from_pretrained("albertlieadrian/qwen3-4b-deslop", dtype="auto", device_map="auto")
messages = [{"role": "user", "content": "Write an opinion piece about open-plan offices. About 250 words."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
print(tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))
```

The LoRA adapter alone is in `adapter/` of `albertlieadrian/qwen3-4b-deslop`.

## Results (48 held-out prompts, one sample each)

Deslop score: weighted habits per 1000 words; 90% of 318 human passages (public-domain essays, Wikipedia 2021) score
33.6 or less, median 12.8. EQ-Bench slop score: computed with slop-score's own code. Intervals: 95% bootstrap over
prompts.

| Model | Deslop score | Replies in the human band | EQ-Bench slop | Task fit | KL drift on human prose | Quiz | Judge win rate vs original |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 89.7 [76.9, 103.7] | 3 / 48 | 43.4 | 39 / 48 | 0 | 30 / 30 | n/a |
| **this model** | **21.5 [18.0, 25.2]** | **38 / 48** | **21.8** | **47 / 48** | **0.11** | **30 / 30** | **0.22 [0.13, 0.32]** |
| erikkannike/qwen3-4b-deslop-v4 | 9.2 [7.1, 11.6] | 47 / 48 | 5.1 | 43 / 48 | 0.16 | 30 / 30 | 0.06 [0.01, 0.14] |
| original + plain prompt + backtracking sampler (runtime) | 13.5 [11.2, 15.9] | 47 / 48 | 14.6 | 45 / 48 | 0 | 30 / 30 | 0.13 [0.05, 0.21] |

Task fit: an opinion piece states a position early, a reflection is in the first person, an email has a greeting, the
length is within 0.5 to 1.6 times the request. The original states a position in 1 of 10 opinion pieces; this model
in 9.

## How it was made

1. The teacher (Qwen3-4B-Instruct-2507 with a plain-style system prompt and an Antislop-style backtracking sampler)
   wrote 3 replies to each of 288 training prompts (48 topics × blog, LinkedIn, explainer, opinion, email,
   reflection), none on the held-out topics.
2. Replies were kept when fluent, passing every task check, free of invented studies and statistics, and scoring
   under the human median: 608 of 864.
3. LoRA rank 16, alpha 32, dropout 0.05, on the q, k, v, o, gate, up and down projections; loss on the reply only;
   the student sees only the user prompt. 557 replies (the dev topics held out), 2 epochs, lr 2e-4, batch 8. The
   recipe was chosen over two others on 24 dev prompts.
4. Merged into the base weights; GGUF converted and quantized to Q4_K_M with llama.cpp.

## Limits

- It is plainer, not better by every measure. A blind gpt-5.4-mini judge preferred the original model's reply as
  better writing in most pairs (6 wins, 9 ties, 33 losses); the same judge preferred the original over every
  de-slopping method tried, including 100toolkit's model.
- It keeps some habits: "not X but Y" at about half the original's rate, and lists of three at about Wikipedia's.
- 100toolkit's qwen3-4b-deslop-v4 scores lower on both habit metrics; this model drifts less from the base model on
  human prose and fails fewer task checks.
- Tested on short English prose (120 to 250 words) only. It still invents names (emails are often signed "Alex" or
  "Jordan"). The quiz (30 easy factual questions) shows nothing broke, not that nothing degraded.
- The habit score is not an AI detector. Do not use this model to pass off generated text as human writing.

## Credits

Built after Erik Markus Kannike's [qwen3-4b-deslop](https://100toolkit.com/blog/deslop-model/) (100toolkit), whose
recipe (distilling a runtime setup into a LoRA, with task-fit checks) this follows. The sampler follows Paech et al.,
[Antislop](https://arxiv.org/abs/2510.15061). The independent metric is EQ-Bench's
[slop-score](https://github.com/sam-paech/slop-score) (MIT). Habit list: 100toolkit's style guide, Wikipedia's
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), Kobak et al. (Science Advances,
2025). Base model: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0); this
model is released under the same licence.

## Citation

```bibtex
@misc{lie2026deslop,
  author = {Albert Lie},
  title  = {De-slopping Qwen3-4B: a sampler beats steering, a LoRA keeps most of it, and an LLM judge prefers the slop},
  year   = {2026},
  url    = {https://albertlieyingadrian.github.io/steerlab/experiments/deslop/report/}
}
```
