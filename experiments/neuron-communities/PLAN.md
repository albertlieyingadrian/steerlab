# Neuron communities as a steering knob (plan, 2026-11-17; fixed before any model run)

Prompted by Goodfire's "Discovering neuron communities" (2026-11-17): 1,941 small groups of MLP neurons in
Llama-3.1-8B (base), each tested by *ablation* only:
- **necessity:** zeroing the group changes the next token;
- **sufficiency:** keeping only the group preserves it.

Data: their public explorer snapshot `2026-11-12-4c7e9dd` (index plus one file per community, with neurons, 10–100
prompts each, and their per-prompt KL numbers). Code: github.com/goodfire-ai/neuron-communities (MIT).

The open question for live steering: **can a community that is necessary for a behaviour also *drive* it?** Can turning a
handful of neurons up make the chat model talk about a topic, and is that cleaner or messier than a steering direction?

Branch `feat/neuron-communities`; folder `experiments/neuron-communities/`. Selection: `fetch.py` → `data/selection.json`
(seed 0):
- **E1:** 10 communities per class × 6 classes, plus the curated "an" community `224838`.
- **E2:** all 48 "topic" communities.

Neuron = the input to `mlp.down_proj` (act(gate) × up), as in their code. Llama-3.1-8B and -Instruct, bf16, one H100 on
Modal.

## E1. Do the communities replicate, and do they survive chat tuning? (base and Instruct)

- **Per community:**
  - **Prompts:** its shipped prompts, as published text, no chat template.
  - **Ablation:** zero every member (the full `neurons` list) at the target (last) position only.
  - **Measure:** KL(clean ‖ ablated) of the next-token distribution.
- **Control:** 10 random neuron sets per community. Each has the same layer for every member and a uniform index in that
  layer, excluding the members and the 17 always-on hubs. Same prompts, same ablation.
- **A community "passes"** when both hold:
  - its median KL ≥ 0.7, their necessity threshold τ;
  - its median exceeds the largest median of its 10 random sets.
- **Reported:**
  - pass share on base and on Instruct, with Wilson 95% intervals;
  - the Spearman correlation between our per-prompt base KL and their published `n`, as a replication check. A low value
    means the shipped prompt text differs from what they ran (explorer prompts may be cut), and is reported as such;
  - the share of Instruct communities that pass among those that pass on base. That's the transfer.

## E2. Can a topic community steer the chat model? (Instruct)

- **Steer:**
  - **Clamp:** at every position of prompt and reply, set each member's activation to `c × a_ref`.
  - **`a_ref`:** the member's mean activation at the target position over its community's prompts on Instruct, with its
    sign kept.
  - **Strengths:** c ∈ {1, 2, 4, 8}.
- **Chat setup:** Llama-3.1-8B-Instruct, chat template, no system prompt, greedy decoding, `repetition_penalty=1.0`.
- **Topic hit:** the reply contains one of the community's topic words.
  - **Word list:** the quoted words in the community's published name, with `|` pieces joined (e.g. "ces|arean" →
    "cesarean"). Each is matched case-insensitively as a whole word with an optional plural "s"/"es".
  - **Generic words:** words shorter than 3 letters, and the `GENERIC` set in `nc.py` (committed with this plan,
    before any run), are dropped as too generic to signal the topic. The set includes Greek, Spain-adjacent words,
    "tan" and "sin": words a holiday or food answer could use anyway.
  - **Exclusions:** communities left with no word are excluded and listed by `nc.topic_words`. The unsteered and
    random-set arms use the same words, so any remaining generic hits show up in the controls.
- **Fluent:** `harness/tools.fluent`.
- **Loop:** fewer than half of the 4-grams in the last 150 tokens are distinct, as in the Golden Gate note.
- **Stage A, screen:**
  - **Sizes:** 48 communities × 4 strengths × the 4 Golden Gate tuning questions, 120 tokens.
  - **Baselines:** unsteered, and one random neuron set per community at the same clamp.
  - **Choice:** each community's best c is the largest fluent-hit count; ties go to the smaller c. Only fluent replies
    count.
- **Stage B, confirm:**
  - **Finalists:** the 8 communities with the most fluent hits in stage A (ties by community id). They are run at their
    chosen c on the 15 held-out Golden Gate questions (`GG_HELDOUT`), 300 tokens.
  - **Arms:**
    - (a) the community clamp;
    - (b) 3 random neuron sets, layer-matched, same c;
    - (c) a **difference-of-means direction**: the mean residual at the community's median member layer, at the target
      position, over its prompts minus over 100 prompts of other E2 communities. It's added at that layer at every
      position, scaled so that its norm equals the mean residual-norm change the community clamp causes at that layer
      (norm-matched);
    - (d) unsteered.
  - **A finalist "steers" when all three hold:**
    - fluent, loop-free topic hits ≥ 8/15;
    - the most any random set gets is ≤ 1/15;
    - unsteered is ≤ 1/15.

## E3. What does each steer cost elsewhere? (Instruct)

- **Collateral, for each E2 finalist:**
  - **Prompts:** 200 unrelated web prompts, the shipped prompts of E1 communities from other classes, raw text.
  - **Arms:** the community clamp, the norm-matched direction and a random set, each as used in Stage B.
  - **Measure:** the mean KL(unsteered ‖ steered) at the last position.
  - Lower is more surgical. Reported side by side with each arm's Stage B hit count.
- **The "an" off-switch, a live-demo candidate:**
  - **Steer:** zero the 46 members of `224838` at every position. No other variant is tried.
  - **Runs:** the 15 held-out questions with a 300-token reply, against unsteered and against 3 layer-matched random
    sets.
  - **Count:** "an" followed by a consonant-initial word, and "a" followed by a vowel-initial word, per 1,000 words.
    Exceptions such as "an hour" and "a university" are counted by first letter, a known approximation that is the same
    for every arm.

## What would change the plan

- **Fixed:** nothing above changes after results are seen.
- **Additions:** any extra look is labelled exploratory in the note.
- **Budget:** ≤ $15 of Modal GPU time. Weights come from the `eval-aware-steer` volume (Instruct
  cached); the base model is downloaded once.

## Write-up

- **The note:** a steerlab note per `experiments/STYLE.md`, then the review page and a lesson.
- **The knob:** a `neurons` entry in steerlab (zero or clamp a list of `(layer, idx)`), so the demo can use whichever
  community steers.
