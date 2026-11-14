"""
The open-weight judges on one Modal GPU instead of the laptop (amendment 3): arm D's Gemma-2-9B, Mistral-7B v0.2 and
Llama-3.1-8B text counsels, and the secondary held-out judge Qwen3-VL-8B.

COSTS MONEY (an A100-80GB at about $2.50/hour, only while a run or the held-out step is using it). It is part of an
ephemeral app: modal_counsel.py includes it for arm D, and 2_heldout.py starts it with `app.run()`. Both stop it
when they finish; it scales to zero after 2 idle minutes and has no keep-warm.

Same prompts, greedy decoding (repetition_penalty 1.0), bfloat16 and stop strings as local_judges.py; only the
hardware differs (CUDA instead of Apple MPS), which can change a greedy reply where two tokens are near-tied.
Weights are downloaded once into the `counsel-judges` volume (the gated Gemma and Llama use the huggingface-secret).
"""

import modal

IDS = {
    "gemma": "google/gemma-2-9b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "vl": "Qwen/Qwen3-VL-8B-Instruct",
}

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install("torch==2.15.0", "torchvision==0.29.1", "transformers==5.22.2", "accelerate", "pillow",
                 "huggingface_hub")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
)
hf = modal.Volume.from_name("counsel-judges", create_if_missing=True)
app = modal.App("counsel-judges", image=image)


@app.cls(gpu="A100-80GB", volumes={"/hf": hf}, secrets=[modal.Secret.from_name("huggingface-secret")],
         timeout=60 * 60, scaledown_window=120, max_containers=1)
class OpenJudges:
    @modal.enter()
    def start(self):
        self.loaded = {}

    def _load(self, name):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        if name not in self.loaded:
            if name == "vl":
                processor = AutoProcessor.from_pretrained(IDS[name])
                model = AutoModelForImageTextToText.from_pretrained(IDS[name], dtype=torch.bfloat16).to("cuda")
            else:
                processor = AutoTokenizer.from_pretrained(IDS[name])
                model = AutoModelForCausalLM.from_pretrained(IDS[name], dtype=torch.bfloat16).to("cuda")
            self.loaded[name] = (processor, model.eval())
            hf.commit()
        return self.loaded[name]

    @modal.method()
    def generate(self, name, system, user, max_new_tokens=64):
        """One greedy reply from a text judge (local_judges.generate, on CUDA)."""
        import torch

        tokenizer, model = self._load(name)
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True)
        ids = {k: v.to("cuda") for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                                 pad_token_id=tokenizer.eos_token_id, stop_strings=["]}", "] }"], tokenizer=tokenizer)
        return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    @modal.method()
    def rate_image(self, png, prompt, max_new_tokens=48):
        """Qwen3-VL-8B's reply to the held-out rubric for one render (local_judges.rate_image, on CUDA)."""
        import io

        import torch
        from PIL import Image

        processor, model = self._load("vl")
        image = Image.open(io.BytesIO(png)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@app.function(volumes={"/hf": hf}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=60 * 60, cpu=4.0)
def download():
    """Fetch every judge's weights into the volume once (CPU only)."""
    from huggingface_hub import snapshot_download

    for repo in IDS.values():
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        hf.commit()
        print("downloaded", repo, flush=True)


@app.local_entrypoint()
def main():
    """Download the weights, then a smoke test: one reply from each judge."""
    download.remote()
    judges = OpenJudges()
    for name in ("gemma", "mistral", "llama"):
        print(name, repr(judges.generate.remote(name, "You are a judge.", 'Reply with JSON only: {"scores": [3, 7]}')))
