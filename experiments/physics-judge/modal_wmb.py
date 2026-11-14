"""WorldModelBench's released judge (vila-ewm-qwen2-1.5b) on Modal, one A10G.

    modal deploy modal_wmb.py         # build once
    python 5_wmb.py items|traps       # send the videos and save the replies

The judge is loaded with VILA's `llava.load` at commit c8f603b (2025-03-01, the VILA of WorldModelBench's release) and
asked WorldModelBench's five physics questions with the prompt template from its evaluation.py, without chain of
thought (the script's default: "Let's think step-by-step and conclude with" becomes "Answer with"). For each question
the function returns the greedy reply (what the official script parses) and P(Yes) / (P(Yes) + P(No)) on the first
answer token, read from one forward pass over the same inputs.
"""

from __future__ import annotations

import modal

VILA_COMMIT = "c8f603b49f5dcfca8c2ee18d7979897a83aa5fa6"
JUDGE = "Efficient-Large-Model/vila-ewm-qwen2-1.5b"
QUESTIONS = {
    "newton": "Violation of Newton's Law: Objects move without any external force.",
    "mass": "Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.",
    "fluid": "Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.",
    "penetration": "Violation of Non-physical Penetration: Objects unnaturally pass through each other.",
    "gravity": "Violation of Gravity: Objects behave inconsistently with gravity.",
}
TEMPLATE = """
            Watch the video and determine if it shows any '{physical_laws}'
            Let's think step-by-step and conclude with "Yes" or "No".
        """.strip()

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("pip==24.0", "setuptools==69.5.1", "wheel")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/"
        "flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        f"git clone https://github.com/NVlabs/VILA /vila && cd /vila && git checkout {VILA_COMMIT}",
        "cd /vila && pip install -e .",
        # VILA imports deepspeed even for inference (sequence-parallel globals); DS_BUILD_OPS=0 skips its CUDA ops.
        "DS_BUILD_OPS=0 pip install deepspeed==0.9.5 'protobuf==3.20.*' 'numpy==1.26.4' huggingface_hub==0.25.2",
        # as VILA's environment_setup.sh does: its FP8 module imports triton.language.extra.cuda.libdevice
        "pip install triton==3.1.0",
    )
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{JUDGE}', local_dir='/judge')\""
    )
)
app = modal.App("physics-judge-wmb", image=image)


@app.cls(gpu="A10G", timeout=3600, scaledown_window=120, max_containers=1)
class Judge:
    @modal.enter()
    def load(self):
        import llava
        import torch

        self.llava = llava
        self.torch = torch
        self.model = llava.load("/judge")
        tok = self.model.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes", " Yes"]})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no", " No"]})

    def _p_yes(self, video, prompt: str) -> tuple[float, float]:
        """P(Yes) on the first answer token, through the same embedding path generate_content uses."""
        from collections import defaultdict

        from llava.mm_utils import process_images
        from llava.utils.media import extract_media
        from llava.utils.tokenizer import tokenize_conversation

        torch = self.torch
        m = self.model
        conversation = [{"from": "human", "value": [video, prompt]}]
        media = extract_media(conversation, m.config)
        media["video"] = [process_images(v, m.vision_tower.image_processor, m.config).half() for v in media["video"]]
        ids = tokenize_conversation(conversation, m.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)
        with torch.inference_mode():
            emb, _, mask = m._embed(ids, media, defaultdict(dict), None, None)
            logits = m.llm(inputs_embeds=emb, attention_mask=mask).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        py = torch.logsumexp(lp[self.yes], 0).exp().item()
        pn = torch.logsumexp(lp[self.no], 0).exp().item()
        return (py / (py + pn) if py + pn > 0 else 0.5), py + pn

    @modal.method()
    def judge(self, name: str, data: bytes) -> dict:
        import tempfile
        import time

        t0 = time.time()
        out = {"video": name}
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(data)
            f.flush()
            video = self.llava.Video(f.name)
            for key, q in QUESTIONS.items():
                prompt = TEMPLATE.format(physical_laws=q.lower()).replace(
                    "Let's think step-by-step and conclude with", "Answer with"
                )
                out[f"{key}_reply"] = self.model.generate_content([video, prompt])
                out[f"{key}_p"], out[f"{key}_mass"] = self._p_yes(video, prompt)
        out["seconds"] = time.time() - t0
        return out


@app.function(timeout=900)
def check_imports() -> str:
    """CPU-only check that VILA imports in the image (a failed GPU start loops and bills; this fails once)."""
    import llava  # noqa: F401
    from llava.model.builder import load_pretrained_model  # noqa: F401

    return "ok"


@app.local_entrypoint()
def main() -> None:
    print(check_imports.remote())
