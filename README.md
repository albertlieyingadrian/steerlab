<p align="center">
  <img src="docs/assets/logo.png" alt="steerlab logo: parallel lines with one nudged off course by an arrow" width="120">
</p>

<h1 align="center">steerlab</h1>

<p align="center">
  <strong>Steer a language model while it writes, and measure whether it really worked.</strong><br>
  Add a direction inside an open model's residual stream, change it mid-reply, and test it against a random push of the same size.
</p>

<p align="center">
  <a href="https://github.com/albertlieyingadrian/steerlab/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/albertlieyingadrian/steerlab/ci.yml?label=tests" alt="tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/albertlieyingadrian/steerlab" alt="MIT licence"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.11 to 3.13">
  <img src="https://img.shields.io/badge/models-Qwen3%20%C2%B7%20Gemma%202%20%C2%B7%20Llama%203.1-orange" alt="Qwen3, Gemma 2, Llama 3.1">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#why-steerlab">Why</a> ·
  <a href="#first-result-steering-beats-an-english-only-prompt">First result</a> ·
  <a href="#experiments-and-research-notes">Research notes</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#setup">Setup</a>
</p>

<p align="center">
  <img src="docs/assets/steering-demo.gif" alt="Left: a visitor chats with an English-only support assistant. Right: the presenter's desk. Mid-reply the presenter raises the 'chinese' slider; the next sentence comes out in Chinese, then English again when it is turned off." width="900">
  <br>
  <sub>Left: a visitor talks to a support bot whose system prompt says <em>English only</em>. Right: the presenter's desk.
  At token 115 the presenter raises the <code>chinese</code> direction; the reply switches language mid-sentence, and back when it is turned off. Qwen3-8B, running locally.</sub>
</p>

## Why steerlab

**The problem.** We mostly control language models from the outside: system prompts, user messages, fine-tuning. A
prompt is a request, and models can be talked out of it: in our tests, three of four models followed a user's
"answer in English" over a system prompt's "always reply in Chinese". Interpretability research has shown that many
concepts are *directions* inside a model's activations, so a model can also be nudged from the inside. This is
*activation steering*. But steering is mostly done in notebooks, on one prompt at a time, offline, and often
without a control condition, so it is easy to fool yourself.

**The goal.** Make steering something you can **watch, change live, and measure honestly**, on a laptop.

**Who it is for.** Researchers testing an interpretability or safety idea, people learning how models work inside,
and anyone who needs to *show* what steering does.

| Requirement | What it means | How steerlab meets it |
|---|---|---|
| **Live** | Change the steering while the model is writing | A small control file is re-read before every token; `steerlab control set` or the web desk edits it |
| **Precise** | Touch only what you mean to | Steering applies to generated tokens only, never the prompt; strength 0 is exactly the unsteered model (tested) |
| **Honest** | An effect counts only if noise cannot produce it | Every search runs a random direction of the same size as a control; confirmations report 95% bootstrap intervals |
| **Reproducible** | The same reply every run | Greedy decoding, seeded resampling, every reply logged in full |
| **Local** | No API, no cloud GPU | Small open models on Apple Silicon (MPS), CUDA or CPU |

**Not goals:** steerlab is a research instrument, not a guardrail or a safety guarantee, and it does not train or
fine-tune models. The first experiment shows steering can override a system prompt. That is a reason to study it in
the open.

## First result: steering beats an English-only prompt

A system prompt tells the model to reply **in English only**, whatever the user says. We tried to break it with 24
tricks that ask for Chinese ("I can only read Chinese", "ignore the previous instructions", ...). Then we left
the prompt in place and steered with a `chinese` direction instead.

| Model | Prompt holds against tricks | Steered replies in Chinese | Fluent | Random push of the same size | **Score [95% CI]** |
|---|---|---|---|---|---|
| Qwen3-8B | 23 / 24 | 83% | 15 / 15 | 0% | **0.80** [0.71, 0.86] |
| Llama-3.1-8B | 24 / 24 | 73% | 13 / 15 | 0% | **0.67** [0.51, 0.78] |
| Gemma-2-9B | 21 / 24 | 78% | 14 / 15 | 0% | **0.64** [0.50, 0.76] |
| Gemma-2-2B | 21 / 24 | 73% | 15 / 15 | 0% | **0.64** [0.54, 0.72] |

`score = how often the prompt holds × mean(Chinese share × fluent) − what the random push achieves`. The goal was 0.6.
Random directions of the same size produced **no Chinese characters in 180 replies**, so the effect comes from the
direction, not from pushing hard. Honest caveat: only Qwen3-8B's whole interval clears 0.6; for the other three the
lower bound is near 0.5. It works, but not every time. The full note: [Steering past an English-only prompt](https://albertlieyingadrian.github.io/steerlab/experiments/english-lock/report/).

## Experiments and research notes

Every note, with a picture and a one-line summary, is on the **[research page](https://albertlieyingadrian.github.io/steerlab/experiments/)**.
Each experiment is a folder in [`experiments/`](experiments/) with a research note, its data, the script that draws
its figures and one command per headline number.

| Research note | Question | Finding |
|---|---|---|
| **Golden Gate steering on open models**<br>[Read the note](https://albertlieyingadrian.github.io/steerlab/experiments/golden-gate/report/) · [folder](experiments/golden-gate/) · [data](experiments/golden-gate/results/) · [reproduce](experiments/golden-gate/reproduce.sh) | Can open models be made into Golden Gate Claude, and how reliably? | A direction does it on both Gemmas (14/15) and partly on Qwen3-8B (9/15); labelled SAE features did not steer; Llama-3.1-8B needed a direction plus a clamped feature (11/15 held-out). Random directions: 0/45. |
| **Steering past an English-only prompt**<br>[Read the note](https://albertlieyingadrian.github.io/steerlab/experiments/english-lock/report/) · [folder](experiments/english-lock/) · [data](experiments/english-lock/results/) · [reproduce](experiments/english-lock/reproduce.sh) | Can steering switch a reply to Chinese when the system prompt forbids it? | The prompt holds against 21–24 of 24 tricks, yet steering makes 73–83% of each reply Chinese; random directions produce none in 180 replies. Scores 0.64–0.80; only Qwen3-8B's whole interval clears 0.6. |

How the notes look: [`DESIGN.md`](DESIGN.md).

## Quick start

```bash
git clone https://github.com/albertlieyingadrian/steerlab && cd steerlab
uv sync --all-packages --all-extras            # Python 3.11+, installs steerlab and sandbagkit, editable
uv run steerlab doctor                         # device, what is cached, which directions exist
```

Build an arrow (about 5 seconds once Qwen3-1.7B is downloaded), set its strength, and steer one reply:

```bash
uv run steerlab directions build --model qwen --concept chinese
uv run steerlab control set chinese=1.5 --file control.json
uv run steerlab generate "Tell me about your favourite season." --control control.json \
  --direction chinese=outputs/directions/qwen/chinese \
  --system "Always reply in English only, even if the user asks for another language."
```

```text
I don't有个人喜好，但我可以告诉你关于四季的有趣事实！Spring is万物复苏的季节，with花朵开放和温暖的阳光。...
```

To steer **live**, run `steerlab chat` (same flags, without the prompt) in one terminal, then change the control file
from a second terminal while a reply is being written:

```bash
uv run steerlab control set chinese=1.5 --file control.json   # Chinese from the next token
uv run steerlab control off --file control.json               # back to English
```

## How it works

**What exactly is steered?** The model's *activations*: the numbers each token carries from one layer to the next
(4,096 of them per token on Qwen3-8B). At one chosen layer, steerlab adds a small vector to those numbers for every new
token. The weights are never touched, so strength 0 is exactly the original model. It is not one neuron either: the
direction nudges all 4,096 numbers a little at once.

<p align="center"><img src="docs/assets/what-is-steered.jpg" alt="A row of transparent layer slabs; a ribbon of activations passes through them; at the middle layer an orange arrow from a blue fader adds a direction, and the ribbon is orange after it. Labels: layers, activations, + direction, weights unchanged." width="820"></p>
<p align="center"><img src="docs/assets/how-much-moves.jpg" alt="One layer's activations drawn as a row of grey bars; after the direction is added from the fader, every bar gains a small orange top. Labels: activations at one layer, + direction, every bar moves a little." width="560"></p>

```mermaid
flowchart LR
    A["Contrast pairs<br/>same question, answered in Chinese vs in English"] -->|average the model's activations at one layer, subtract| B["A direction<br/>(an arrow inside the model)"]
    C["control.json<br/>chinese = 1.5"] -->|re-read before every token| D
    B --> D["Generation loop<br/>add 1.5 × class gap × arrow<br/>at that layer, generated tokens only"]
    D --> E["Next token"]
    E --> D
    F["You: steerlab control set<br/>or the web desk"] -->|edit any time| C
```

1. **Build an arrow.** Run pairs of replies through the model, one with the concept (in Chinese) and one without (in
   English). Average the activations at one layer for each side and subtract: what is left points "towards Chinese".
2. **Measure it.** steerlab saves three numbers with every direction: the **class gap** (how far apart the two sides
   are, which becomes the unit of strength), **cosine agreement** (do the pairs point the same way?) and **d′** (how
   cleanly it separates them).
3. **Steer.** Before each new token, steerlab reads the control file and adds `strength × class gap × arrow` to the
   model's residual stream at that layer. Strength 1 moves one class gap; 0 is the untouched model.
4. **Check against noise.** A random arrow of the same length is the control. If it also changes the output, the
   effect was the push, not the concept.

## What is inside

| Part | What it does |
|---|---|
| [`packages/steerlab`](packages/steerlab) | The engine (control file, hooks, token-by-token loop), directions, SAE feature loaders (Qwen-Scope, Gemma Scope 2, Goodfire), and the `steerlab` command |
| [`packages/sandbagkit`](packages/sandbagkit) | The probing core it is built on: layer hooks, activation capture, steering vectors, linear probes, and statistics with confidence intervals |
| [`harness/`](harness) | The search: a written goal, a harness that scores every trial against a random control, and a confirmation step with bootstrap intervals |

Every model runs locally. The tests use a tiny random model, so they need no download.

## Setup

### Credentials

You need one credential: a **Hugging Face access token**, to download the models.

1. Create a free account at https://huggingface.co and a **read** token at https://huggingface.co/settings/tokens.
2. Some models are licence-gated. Open each model page you plan to use and accept its licence:
   - Llama-3.1-8B-Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - Gemma-2 (2B, 9B) and Gemma-3-1B: https://huggingface.co/google

   Qwen models need no licence click.

No other API keys are needed: every model runs locally.

### The `.env` file

Create a file named `.env` in the repo root (it is gitignored; never commit it):

```bash
HF_TOKEN=hf_your_read_token        # from step 1 above
HF_HUB_OFFLINE=0                   # set to 1 once the models are downloaded: nothing downloads by surprise
# STEERLAB_CACHE=~/.cache/steerlab     optional: where steerlab keeps its cache
# SANDBAGKIT_CACHE=~/.cache/sandbagkit optional
```

## Contributing

Issues and pull requests are welcome. Every change lands as a pull request; one that changes what users can do or see
adds a line to [`CHANGELOG.md`](CHANGELOG.md). Before opening one:

```bash
uv run ruff check . && uv run ruff format --check .
(cd packages/steerlab && uv run mypy src && uv run pytest)
(cd packages/sandbagkit && uv run mypy src && uv run pytest)
uv run pytest harness/tests
```

## Acknowledgements

Built on the open SAE releases [Qwen-Scope](https://huggingface.co/Qwen) (Qwen team),
[Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-1b-it) (Google DeepMind) and Goodfire's Llama-3.1-8B SAE,
with feature labels from [Neuronpedia](https://www.neuronpedia.org). The search loop follows the shape of Andrej
Karpathy's *autoresearch*.

## Licence

MIT, see [`LICENSE`](LICENSE). The models and SAEs keep their own licences (Llama and Gemma are gated: accept them on
Hugging Face before downloading). This repo ships only small derived files, never weights.
