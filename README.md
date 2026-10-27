# steerlab

Live activation steering for open language models: change what a model writes by adding vectors inside it, while it writes.

This repo is being set up. Coming here:
- `packages/steerlab`: the library, CLI and web demo, with its probing toolkit `packages/sandbagkit`;
- `experiments/`: one folder per experiment, each with a report, demos, recordings and results;
- `docs/`: the concepts in plain language, and how to reproduce everything.

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
