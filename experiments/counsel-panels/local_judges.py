"""
Open-weight judges for arm D, run on the laptop's GPU (Apple MPS) with transformers: free, no API.

  gemma    google/gemma-2-9b-it              (Google)
  mistral  mistralai/Mistral-7B-Instruct-v0.2 (Mistral AI)
  llama    meta-llama/Llama-3.1-8B-Instruct  (Meta)

Each loads once in bfloat16 and answers greedily (repetition_penalty 1.0). Gemma and Mistral have no system role,
so the system prompt is put at the top of the user turn for every model alike. Weights come from the local
Hugging Face cache (HF_HUB_OFFLINE=1 keeps it from downloading).
"""

import os
import threading

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IDS = {
    "gemma": "google/gemma-2-9b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
_LOADED = {}
_LOCK = threading.Lock()
_GEN = threading.Lock()  # one MPS generation at a time: concurrent generate() calls on MPS hung D_s0 at step 4


def _load(name):
    with _LOCK:
        if name not in _LOADED:
            tokenizer = AutoTokenizer.from_pretrained(IDS[name])
            model = AutoModelForCausalLM.from_pretrained(IDS[name], dtype=torch.bfloat16).to("mps").eval()
            _LOADED[name] = (tokenizer, model, _GEN)
    return _LOADED[name]


def generate(name, system, user, max_new_tokens=64):
    """One greedy reply from a local judge."""
    tokenizer, model, lock = _load(name)
    messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to("mps") for k, v in ids.items()}
    with lock, torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["]}", "] }"],
            tokenizer=tokenizer,
        )
    return tokenizer.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


VL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def rate_image(png, prompt, max_new_tokens=48):
    """The secondary held-out judge (a different family from every Claude judge): Qwen3-VL-8B rates one render."""
    import io

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    with _LOCK:
        if "vl" not in _LOADED:
            processor = AutoProcessor.from_pretrained(VL_ID)
            model = AutoModelForImageTextToText.from_pretrained(VL_ID, dtype=torch.bfloat16).to("mps").eval()
            _LOADED["vl"] = (processor, model, _GEN)
    processor, model, lock = _LOADED["vl"]
    image = Image.open(io.BytesIO(png)).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("mps")
    with lock, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
    return processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
