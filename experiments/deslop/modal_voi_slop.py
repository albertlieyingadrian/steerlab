"""voi-slop-detect-4B on a small rented GPU (L4, else A10G), kept in its own Modal app so it deploys without the
main deslop images.

    modal deploy experiments/deslop/modal_voi_slop.py     # then voi_slop.py calls it by name
"""

import time

import modal

app = modal.App("deslop-voi")
volume = modal.Volume.from_name("deslop", create_if_missing=True)
VOI_SLOP, VOI_SLOP_REVISION = "VoightKampffLabs/voi-slop-detect-4B", "b23f86f1497aa41a6aa196ecd7939f8708580a17"
voi_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "transformers==5.22.2", "accelerate>=1.10", "safetensors", "huggingface_hub")
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)


@app.function(image=voi_image, gpu=["L4", "A10G"], volumes={"/vol": volume}, timeout=60 * 60)
def score_voi_slop(texts: list[str]) -> dict:
    """voi-slop-detect-4B (Qwen3.5-4B, one-logit sequence-classification head, Apache-2.0), pinned to one revision:
    sigmoid(logit) is the share of the text written by an LLM, 0..1. As its card says: the first 512 tokens, fenced code
    blocks already emptied by the caller, bf16. One text per forward pass, so no padding meets the linear-attention
    layers. Measurement only."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(VOI_SLOP, revision=VOI_SLOP_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(VOI_SLOP, revision=VOI_SLOP_REVISION,
                                                               dtype=torch.bfloat16).to("cuda").eval()  # fmt: skip
    volume.commit()
    t0, scores, tokens = time.time(), [], []
    for i, text in enumerate(texts):
        enc = tok([text], truncation=True, max_length=512, return_tensors="pt").to("cuda")
        scores.append(float(torch.sigmoid(model(**enc).logits.float()).reshape(-1)[0]))
        tokens.append(int(enc["input_ids"].shape[1]))
        if i % 100 == 0:
            print(f"{i}/{len(texts)} ({time.time() - t0:.0f} s)")
    import transformers

    return {"scores": scores, "tokens": tokens, "seconds": round(time.time() - t0),
            "gpu": torch.cuda.get_device_name(0), "transformers": transformers.__version__}  # fmt: skip
