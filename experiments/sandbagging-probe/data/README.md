# Data: sandbagging, episode 1

The prompt sets this study made or used, besides the toolkit's 32 contrast pairs (downloaded at run time from the
[ai-metacognition-toolkit](https://github.com/bassrehab/ai-metacognition-toolkit), MIT). The note's Datasets table, in
section 02 of the [research note](../report/), says which section uses which.

- [`neutral_prompts.json`](neutral_prompts.json): 2,800 everyday requests to fit the alignment between two models, and
  400 held out; generated from 80 topics × 40 templates.
- [`trigger_word_prompts.json`](trigger_word_prompts.json): 40 harmless prompts that use words such as "test",
  "sandbag" or "deployment", grouped by word.
- [`evaluation_mention_pairs.json`](evaluation_mention_pairs.json): 40 pairs that differ only by a harmless mention of an
  evaluation.
- [`toolkit_innocent_prompts.json`](toolkit_innocent_prompts.json): the toolkit's 32 innocent prompts (MIT).
- [`toolkit_paraphrases.json`](toolkit_paraphrases.json): the toolkit's 20 paraphrases of its test pairs (MIT).
